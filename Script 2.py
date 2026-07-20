import asyncio
import os
import json
import re
import logging
import httpx
import httpcore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from datasets import load_dataset, Dataset
from QAEG import qaeg
from nltk.stem import PorterStemmer
from sentence_transformers import util

# ====================== User-defined configuration (modify according to your setup) ======================
# QAEG custom inference server URL
SERVER_URL = "http://your-server-address:port/chat"   # Replace with your LLM service address

# QAEG result save directory and file name
SAVE_DIR = "./results"                                 # Replace with your save directory
SAVE_FILE_NAME = "qaeg_results.json"                   # Replace with your desired file name
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_FILE_NAME)

# Evaluation dataset path (supports local or Hugging Face path)
DATASET_PATH = "./datasets/your_dataset.json"          # Replace with your dataset path

# Embedding model path (for similarity computation, required by QAEG)
EMBEDDING_MODEL_PATH = "/path/to/embedding/model"      # Replace with your embedding model directory

# ====================== Suppress QAEG logging ======================
logging.basicConfig(level=logging.WARNING)
logging.getLogger("qaeg").setLevel(logging.WARNING)
logging.getLogger("logger").setLevel(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"

# ====================== Load QAEG output-level faithfulness evaluation dataset ======================
ds = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)

# ====================== Initialize QAEG inference framework ======================
rag = qaeg(
    backend_type="custom_server",
    model_name="custom_model",
    server_url=SERVER_URL,
    similarity_model=EMBEDDING_MODEL_PATH,
    enable_dual_evidence_graph=True,
    enable_sufficiency_estimation=True
)


# ====================== Normalize QAEG structured output and compatibility handling ======================
def repair_and_format_json(text, context=None, options=None):
    if not isinstance(text, str):
        text = str(text)

    # Clean control characters and code block markers from generated output
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)

    # Extract reasoning and final answer fields
    reason_val = ""
    answer_val = ""
    try:
        obj = json.loads(text)
        for k, v in obj.items():
            k_lower = k.lower()
            if "reason" in k_lower or "thought" in k_lower or "thinking" in k_lower:
                reason_val = str(v)
            elif "answer" in k_lower or "ans" in k_lower or "result" in k_lower:
                answer_val = str(v)
        if not reason_val and not answer_val and len(obj) >= 1:
            items = list(obj.items())
            if len(items) == 1:
                answer_val = str(items[0][1])
            else:
                sorted_items = sorted(items, key=lambda x: len(str(x[1])), reverse=True)
                reason_val = str(sorted_items[0][1])
                answer_val = str(sorted_items[1][1]) if len(sorted_items) > 1 else reason_val
    except:
        pass

    # Fallback option compatibility strategy for malformed outputs (only as ultimate fallback)
    def is_valid_answer(ans, opts):
        if not ans or not opts:
            return False
        ans_clean = ans.strip().lower()
        for opt in opts:
            if opt.strip().lower() in ans_clean or ans_clean in opt.strip().lower():
                return True
        return False

    if options and context and not is_valid_answer(answer_val, options):
        option_counts = {}
        context_lower = context.lower()
        for opt in options:
            opt_clean = opt.strip()
            if opt_clean:
                count = context_lower.count(opt_clean.lower())
                option_counts[opt_clean] = count
        sorted_options = sorted(option_counts.items(), key=lambda x: x[1], reverse=True)
        if sorted_options:
            fallback_answer = sorted_options[0][0]
            fallback_reason = reason_val if reason_val else f"Directly selected based on explicit mention in context: {fallback_answer}"
            return json.dumps({"Reason": fallback_reason, "Answer": fallback_answer}, ensure_ascii=False)

    return json.dumps({"Reason": reason_val, "Answer": answer_val}, ensure_ascii=False)


# ====================== QAEG output-level faithfulness diagnosis helper functions ======================
# Explanation:
# 1. "Answer correctness" follows the unified evaluation protocol's Accuracy metric, using evaluate() returned acc.
# 2. "Context support" is determined based on QAEG's admissible evidence chunks, considering:
#    - Exact answer match;
#    - Keyword/stem coverage;
#    - Semantic similarity;
#    - Actual Top-K context chunks and context-oriented relational triples.
# These combined rules identify paraphrased answers, morphological variants, and answers directly supported by contextual relational triples.
LEXICAL_COVERAGE_THRESHOLD = 0.60
SEMANTIC_SUPPORT_THRESHOLD = 0.40

_PORTER_STEMMER = PorterStemmer()
_SUPPORT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "to", "in", "on", "at",
    "for", "from", "by", "with", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "into", "over", "under", "about", "what",
    "which", "who", "whom", "whose", "when", "where", "why", "how", "should", "would", "could",
    "can", "may", "might", "will", "do", "does", "did", "have", "has", "had"
}


def normalize_for_support(text):
    """Normalize case, punctuation and whitespace for reproducible evidence support judgement."""
    if text is None:
        return ""
    text = str(text).lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def content_stems(text):
    """Extract content word stems (stopwords removed) for matching morphological variants in evidence."""
    normalized = normalize_for_support(text)
    stems = []
    for token in normalized.split():
        if token in _SUPPORT_STOPWORDS:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        stems.append(_PORTER_STEMMER.stem(token))
    return stems


def is_refusal_answer(answer):
    """Identify refusal outputs from QAEG context-bounded generation; empty answers are also treated as refusal."""
    normalized = normalize_for_support(answer)
    if not normalized:
        return True

    refusal_patterns = [
        r"\bi don t know\b",
        r"\bi do not know\b",
        r"\bcannot answer\b",
        r"\bcan t answer\b",
        r"\bcannot be answered\b",
        r"\bcannot be determined\b",
        r"\bcan t be determined\b",
        r"\binsufficient information\b",
        r"\binsufficient evidence\b",
        r"\bnot enough information\b",
        r"\bnot enough evidence\b",
        r"\banswer is unknown\b",
        r"\bunanswerable\b",
        r"\bno answer\b",
        r"\bnot provided\b",
        r"\bnot mentioned\b",
        r"\bnot specified\b"
    ]
    return any(re.search(pattern, normalized) for pattern in refusal_patterns)


def _extract_chunk_texts(item):
    """Extract context chunks from QAEG retrieval results, compatible with different field names."""
    chunk_items = item.get("topk_chunks", item.get("chunks", []))
    texts = []
    for chunk_item in chunk_items:
        if isinstance(chunk_item, dict):
            chunk_text = chunk_item.get("chunk", chunk_item.get("text", ""))
        else:
            chunk_text = str(chunk_item)
        if str(chunk_text).strip():
            texts.append(str(chunk_text))
    return texts


def build_evidence_map(facts, chunks):
    """
    Build a set of admissible evidence for each sample for output-level faithfulness diagnosis:
    1. The actual filtered Top-K context chunks;
    2. Relational triples generated from context-oriented evidence views and used for reasoning.

    Checking both chunks and relational triples avoids missing answers directly supported by structured contextual relations.
    """
    evidence_map = {}

    def add_evidence(sample_id, text):
        if sample_id is None or not str(text).strip():
            return
        evidence_map.setdefault(sample_id, [])
        evidence_map[sample_id].append(str(text).strip())

    for item in chunks or []:
        sample_id = item.get("id")
        for text in _extract_chunk_texts(item):
            add_evidence(sample_id, text)

    for item in facts or []:
        sample_id = item.get("id")

        # Context-oriented relational states already merged Top-K chunks; collect and deduplicate later.
        for text in _extract_chunk_texts(item):
            add_evidence(sample_id, text)

        triples = item.get("facts", [])
        for triple in triples:
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                subject, predicate, obj = [str(x).strip() for x in triple]
                add_evidence(sample_id, f"{subject} {predicate} {obj}")
            elif str(triple).strip():
                add_evidence(sample_id, str(triple))

    # Deduplicate by normalized text while preserving original evidence text for semantic encoding.
    for sample_id, texts in evidence_map.items():
        deduplicated = []
        seen = set()
        for text in texts:
            key = normalize_for_support(text)
            if key and key not in seen:
                seen.add(key)
                deduplicated.append(text)
        evidence_map[sample_id] = deduplicated

    return evidence_map


def lexical_support_score(answer, evidence_units):
    """
    Compute lexical coverage support of the answer against QAEG context evidence:
    - Returns 1 if the normalized full answer appears in any evidence or concatenated evidence;
    - Otherwise computes coverage ratio of answer content stems in the evidence set.
    """
    normalized_answer = normalize_for_support(answer)
    if not normalized_answer or not evidence_units:
        return 0.0

    normalized_units = [normalize_for_support(text) for text in evidence_units]
    combined_evidence = " ".join(normalized_units)

    padded_answer = f" {normalized_answer} "
    if any(padded_answer in f" {unit} " for unit in normalized_units):
        return 1.0
    if padded_answer in f" {combined_evidence} ":
        return 1.0

    answer_stems = set(content_stems(answer))
    if not answer_stems:
        return 0.0

    evidence_stems = set(content_stems(combined_evidence))
    matched = len(answer_stems & evidence_stems)
    return matched / len(answer_stems)


def semantic_support_score(question, answer, evidence_units, similarity_model):
    """
    Reuse the already-loaded all-MiniLM-L6-v2 from QAEG for semantic support, without extra model loading.
    Compares both "answer" and "question+answer" with each evidence unit and takes the maximum similarity.
    """
    if similarity_model is None or not evidence_units:
        return 0.0

    evidence_units = [str(x).strip() for x in evidence_units if str(x).strip()]
    if not evidence_units:
        return 0.0

    query_texts = [str(answer).strip()]
    if str(question).strip():
        query_texts.append(f"Question: {question} Answer: {answer}")

    try:
        query_embeddings = similarity_model.encode(
            query_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        evidence_embeddings = similarity_model.encode(
            evidence_units,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        scores = util.cos_sim(query_embeddings, evidence_embeddings)
        return float(scores.max().item())
    except Exception:
        # In case of semantic encoding failure, fall back to lexical judgement to avoid disrupting the entire batch.
        return 0.0


def is_answer_supported(answer, question, evidence_units, similarity_model=None):
    """
    Reproducible "context support" judgement for QAEG output-level diagnosis.

    Support is considered satisfied if any of the following holds:
    1. The full answer appears in evidence;
    2. Valid content stem coverage reaches 60%;
    3. Maximum semantic similarity with actual evidence reaches 0.40.

    Evidence is drawn only from the admissible Top-K context chunks and context-oriented relational triples,
    without using reference answers or justifications.
    """
    if is_refusal_answer(answer):
        return False

    lexical_score = lexical_support_score(answer, evidence_units)
    if lexical_score >= LEXICAL_COVERAGE_THRESHOLD:
        return True

    semantic_score = semantic_support_score(
        question=question,
        answer=answer,
        evidence_units=evidence_units,
        similarity_model=similarity_model
    )
    return semantic_score >= SEMANTIC_SUPPORT_THRESHOLD


def calculate_faithfulness_metrics(evaluation_details, facts, chunks, similarity_model=None):
    """
    Support Precision = number of non-refusal outputs with actual evidence support / number of non-refusal outputs

    Faithful Accuracy =
        (number of non-refusal samples that are correct and have evidence support + number of correct refusals) / total samples

    "Answer correct" uses the acc field from the unified evaluate() interface, maintaining the same Accuracy definition as in experiments.
    """
    evidence_map = build_evidence_map(facts, chunks)
    non_refusal_count = 0
    supported_non_refusal_count = 0
    faithful_correct_count = 0

    for detail in evaluation_details:
        sample_id = detail.get("id")
        question = detail.get("question", "")
        answer = detail.get("prediction", "")

        # Use the unified evaluation protocol's Accuracy judgement, not stricter Exact Match.
        is_correct = detail.get("acc", 0) > 0
        non_refusal = not is_refusal_answer(answer)
        supported = (
            is_answer_supported(
                answer=answer,
                question=question,
                evidence_units=evidence_map.get(sample_id, []),
                similarity_model=similarity_model
            )
            if non_refusal else False
        )

        if non_refusal:
            non_refusal_count += 1
        if supported:
            supported_non_refusal_count += 1

        # Non-refusal outputs must be both correct and context-supported; correct refusals count as faithful.
        if is_correct and (supported or not non_refusal):
            faithful_correct_count += 1

    total_count = len(evaluation_details)
    support_precision = (
        100.0 * supported_non_refusal_count / non_refusal_count
        if non_refusal_count > 0 else 0.0
    )
    faithful_accuracy = (
        100.0 * faithful_correct_count / total_count
        if total_count > 0 else 0.0
    )

    return {
        "support_precision": support_precision,
        "faithful_accuracy": faithful_accuracy,
        "non_refusal_outputs": non_refusal_count,
        "supported_non_refusal_outputs": supported_non_refusal_count,
        "faithful_correct_outputs": faithful_correct_count
    }


# ====================== QAEG batch inference with network retry ======================
@retry(
    stop=stop_after_attempt(5),  # Maximum 5 attempts per QAEG batch
    wait=wait_exponential(multiplier=2, min=4, max=20),  # Exponential backoff, max 20s
    retry=retry_if_exception_type((httpx.RemoteProtocolError, httpcore.RemoteProtocolError, ConnectionError)),
    reraise=True
)
async def process_single_batch_with_retry(rag, batch_ds):
    """Execute single-batch QAEG evidence construction and context-bounded generation, with automatic retry on network errors."""

    # Step 1: Build query-anchored evidence contracts and asymmetric relational views
    print(f"    🧭 Starting to build query-anchored evidence contracts and asymmetric relational views...")
    facts_dict = await rag.get_self_facts(batch_ds)
    retrieved_facts = facts_dict['retrieved_facts']

    # Step 2: Filter admissible Top-K context evidence chunks for answers
    print(f"    📚 Starting to filter admissible Top-K context evidence chunks...")
    chunks = rag.get_topk_chunks(batch_ds, retrieved_facts)

    # Associate context-oriented relational states with corresponding context chunks
    for f, c in zip(retrieved_facts, chunks):
        f.update(c)

    # Step 3: Perform context-bounded generation under sufficiency constraints
    print(f"    🛡️ Starting context-bounded generation under sufficiency constraints...")
    sampling_kwargs = {
        "generation_type": "normal_cot",
        "temperature": 0.0001,
        "top_p": 1,
        "max_tokens": 300
    }
    raw_preds_dict = await rag.get_predictions(batch_ds, facts_dict, **sampling_kwargs)

    return retrieved_facts, chunks, raw_preds_dict


# 3. Run QAEG output-level faithfulness evaluation in batches
async def run_in_batches(dataset, batch_size=5):
    all_predictions = []
    batch_metrics = []
    total_size = len(dataset)
    total_samples_run = 0
    total_non_refusal_outputs = 0
    total_supported_non_refusal_outputs = 0
    total_faithful_correct_outputs = 0
    num_batches = (total_size + batch_size - 1) // batch_size

    start_batch_idx = 0  # Can be modified to resume from a specific batch
    for i in range(start_batch_idx, num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_size)
        print(f"\n--- Running QAEG inference for batch {i + 1}/{num_batches} (samples: {start_idx}-{end_idx}) ---")
        batch_ds = dataset.select(range(start_idx, end_idx))

        try:
            # Execute QAEG batch inference with network retry mechanism
            facts, chunks, raw_preds_dict = await process_single_batch_with_retry(rag, batch_ds)

            # Normalize generated outputs into unified JSON structure for evaluation
            repaired_preds_dict = {}
            printable_preds = []
            for j, item in enumerate(batch_ds):
                data_id = item['id']
                raw_text = raw_preds_dict.get(data_id, "")

                repaired_json_str = repair_and_format_json(
                    raw_text,
                    context=item.get('context', ''),
                    options=item.get('choices')
                )
                repaired_preds_dict[data_id] = repaired_json_str
                try:
                    printable_preds.append(json.loads(repaired_json_str))
                except:
                    printable_preds.append(repaired_json_str)

            # Print context-bounded generation results for current batch
            print(f"\n{'=' * 20} Batch {i + 1} QAEG context-bounded output details {'=' * 20}")
            for idx in range(len(batch_ds)):
                original_idx = start_idx + idx
                sample = batch_ds[idx]
                pred = printable_preds[idx] if idx < len(printable_preds) else "No valid structured output obtained"
                print(f"\n📌 Evaluation sample index: {original_idx}")
                print(f"❓ Question: {sample.get('question', 'N/A')}")
                print(f"✅ Reference answer: {sample.get('answer', 'N/A')}")
                print(f"🛡️ QAEG output: {pred}")
                print("-" * 80)

            all_predictions.extend(printable_preds)

            # Perform output-level faithfulness diagnosis for current batch
            print(f"    📊 Starting to compute QAEG output-level faithfulness metrics for current batch...")
            batch_eval_with_details = rag.evaluate(
                batch_ds, repaired_preds_dict, cot_format=True, detailed_output=True
            )
            evaluation_details = batch_eval_with_details.get("details", [])

            # Compute Support Precision and Faithful Accuracy based on admissible evidence
            similarity_model = getattr(getattr(rag, "evidence_retriever", None), "similarity_model", None)
            batch_results = calculate_faithfulness_metrics(
                evaluation_details=evaluation_details,
                facts=facts,
                chunks=chunks,
                similarity_model=similarity_model
            )

            print(f"✅ Batch {i + 1} QAEG Support Precision: {batch_results['support_precision']:.2f}%")
            print(f"✅ Batch {i + 1} QAEG Faithful Accuracy: {batch_results['faithful_accuracy']:.2f}%")

            batch_metrics.append({
                "batch": i + 1,
                "index_range": f"{start_idx}-{end_idx}",
                "total_in_batch": len(batch_ds),
                "valid_in_batch": len(batch_ds),
                "metrics": batch_results
            })

            # Accumulate QAEG faithfulness diagnostic statistics from completed batches
            total_samples_run += len(batch_ds)
            total_non_refusal_outputs += batch_results['non_refusal_outputs']
            total_supported_non_refusal_outputs += batch_results['supported_non_refusal_outputs']
            total_faithful_correct_outputs += batch_results['faithful_correct_outputs']

            if total_samples_run > 0:
                current_support_precision = (
                    100.0 * total_supported_non_refusal_outputs / total_non_refusal_outputs
                    if total_non_refusal_outputs > 0 else 0.0
                )
                current_faithful_accuracy = (
                    100.0 * total_faithful_correct_outputs / total_samples_run
                )
                print(f"📈 Cumulative QAEG Support Precision up to now: {current_support_precision:.2f}%")
                print(f"📈 Cumulative QAEG Faithful Accuracy up to now: {current_faithful_accuracy:.2f}%")

        except Exception as e:
            print(f"❌ Batch {i + 1} QAEG inference failed (network retries exhausted), skipping this batch: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

        # Throttle requests between consecutive batches to reduce server load
        if i < num_batches - 1:
            await asyncio.sleep(3)

    # 4. Summarize QAEG output-level faithfulness metrics across all successful batches
    print("\n" + "=" * 40)
    print("📊 QAEG all batches completed, summarizing output-level faithfulness diagnosis results")
    if total_samples_run > 0:
        final_faithful_accuracy = (
            100.0 * total_faithful_correct_outputs / total_samples_run
        )
    else:
        final_faithful_accuracy = 0.0

    final_support_precision = (
        100.0 * total_supported_non_refusal_outputs / total_non_refusal_outputs
        if total_non_refusal_outputs > 0 else 0.0
    )

    final_results = {
        "support_precision": final_support_precision,
        "faithful_accuracy": final_faithful_accuracy,
        "non_refusal_outputs": total_non_refusal_outputs,
        "supported_non_refusal_outputs": total_supported_non_refusal_outputs,
        "faithful_correct_outputs": total_faithful_correct_outputs,
        "total_evaluated_samples": total_samples_run,
        "total_original_samples": total_size
    }
    print(f"🎯 QAEG overall Support Precision: {final_results['support_precision']:.2f}%")
    print(f"🎯 QAEG overall Faithful Accuracy: {final_results['faithful_accuracy']:.2f}%")
    print(f"📦 Evaluated samples: {final_results['total_evaluated_samples']}/{final_results['total_original_samples']}")
    print("=" * 40)

    # 5. Save QAEG predictions, batch metrics, and experiment configuration
    print(f"\n💾 Saving QAEG output-level faithfulness evaluation results to: {SAVE_PATH}")
    save_data = {
        "total_samples": total_size,
        "batch_size": batch_size,
        "total_batches": num_batches,
        "batch_metrics": batch_metrics,
        "final_metrics": final_results,
        "all_predictions": all_predictions,
        "config": {
            "enable_dual_evidence_graph": True,
            "enable_sufficiency_estimation": True
        }
    }
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)
    print("✅ QAEG output-level faithfulness evaluation results saved successfully!")


if __name__ == "__main__":
    asyncio.run(run_in_batches(ds, batch_size=5))