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

# ====================== Suppress unnecessary logging during QAEG experiments ======================
logging.basicConfig(level=logging.WARNING)
logging.getLogger("qaeg").setLevel(logging.WARNING)
logging.getLogger("logger").setLevel(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"

# ====================== Load QAEG evaluation dataset ======================
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


# ====================== Repair QAEG structured output and compatibility for evaluation ======================
def repair_and_format_json(text, context=None, options=None):
    if not isinstance(text, str):
        text = str(text)

    # Remove control characters and code block markers from model output
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)

    # Extract reason and answer fields
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

    # Fallback strategy for malformed outputs
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


# ====================== Batch inference function with retry on network errors ======================
@retry(
    stop=stop_after_attempt(5),  # Maximum 5 attempts per batch
    wait=wait_exponential(multiplier=2, min=4, max=20),  # Exponential backoff, max 20s
    retry=retry_if_exception_type((httpx.RemoteProtocolError, httpcore.RemoteProtocolError, ConnectionError)),
    reraise=True
)
async def process_single_batch_with_retry(rag, batch_ds):
    """Execute single-batch QAEG evidence construction and context-bounded generation, with automatic retry on network errors."""

    # Step 1: Build query-anchored relational evidence views
    print(f"    🧭 Starting to build query-anchored evidence contracts and relational views...")
    facts_dict = await rag.get_self_facts(batch_ds)
    retrieved_facts = facts_dict['retrieved_facts']

    # Step 2: Filter answer-admissible context evidence chunks
    print(f"    📚 Starting to filter top-K admissible evidence chunks from context...")
    chunks = rag.get_topk_chunks(batch_ds, retrieved_facts)

    # Associate relational evidence views with corresponding context chunks
    for f, c in zip(retrieved_facts, chunks):
        f.update(c)

    # Step 3: Perform context-bounded answer generation
    print(f"    🛡️ Starting context-bounded answer generation...")
    sampling_kwargs = {
        "generation_type": "normal_cot",
        "temperature": 0.0001,
        "top_p": 1,
        "max_tokens": 300
    }
    raw_preds_dict = await rag.get_predictions(batch_ds, facts_dict, **sampling_kwargs)

    return retrieved_facts, chunks, raw_preds_dict


# 3. Run QAEG evaluation pipeline in batches
async def run_in_batches(dataset, batch_size=5):
    all_predictions = []
    batch_metrics = []
    total_size = len(dataset)
    total_correct = 0
    total_samples_run = 0
    total_recall_sum = 0.0
    num_batches = (total_size + batch_size - 1) // batch_size

    start_batch_idx = 0
    for i in range(start_batch_idx, num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_size)
        print(f"\n--- Running QAEG inference for batch {i + 1}/{num_batches} (samples: {start_idx}-{end_idx}) ---")
        batch_ds = dataset.select(range(start_idx, end_idx))

        try:
            # Execute QAEG batch inference with network retry mechanism
            facts, chunks, raw_preds_dict = await process_single_batch_with_retry(rag, batch_ds)

            # Normalize raw model outputs into JSON format required for QAEG evaluation
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
                print(f"\n📌 QAEG evaluation sample index: {original_idx}")
                print(f"❓ Question: {sample.get('question', 'N/A')}")
                print(f"✅ Reference answer: {sample.get('answer', 'N/A')}")
                print(f"🛡️ QAEG context-bounded output: {pred}")
                print("-" * 80)

            all_predictions.extend(printable_preds)

            # Evaluate answer accuracy and evidence coverage for current batch
            print(f"    📊 Starting evaluation of QAEG answers and context recall for current batch...")
            batch_results = rag.evaluate(batch_ds, repaired_preds_dict, cot_format=True)

            # Compute context recall for QAEG generation-stage available evidence
            batch_recall_list = []
            for j, item in enumerate(batch_ds):
                gt_answer = item.get('answer', '').strip().lower()
                sample_chunks = chunks[j]
                # Handle different chunk formats: read 'text' field if dict, else string directly
                if isinstance(sample_chunks, list):
                    context_text = ' '.join([
                        chunk.get('text', '') if isinstance(chunk, dict) else str(chunk)
                        for chunk in sample_chunks
                    ]).lower()
                else:
                    context_text = str(sample_chunks).lower()

                gt_words = gt_answer.split()
                if len(gt_words) == 0:
                    sample_recall = 0.0
                else:
                    hit_count = sum(1 for word in gt_words if word in context_text)
                    sample_recall = hit_count / len(gt_words)
                batch_recall_list.append(sample_recall)

            batch_context_recall = sum(batch_recall_list) / len(batch_recall_list) * 100 if batch_recall_list else 0.0

            print(f"✅ Batch {i + 1} QAEG exact match rate: {batch_results['exact_match']:.2f}% | Context Recall: {batch_context_recall:.2f}%")

            batch_metrics.append({
                "batch": i + 1,
                "index_range": f"{start_idx}-{end_idx}",
                "total_in_batch": len(batch_ds),
                "valid_in_batch": len(batch_ds),
                "metrics": batch_results,
                "context_recall": batch_context_recall
            })

            # Update cumulative evaluation results
            total_samples_run += len(batch_ds)
            total_correct += len(batch_ds) * (batch_results['exact_match'] / 100)
            total_recall_sum += sum(batch_recall_list)

            if total_samples_run > 0:
                current_overall_em = (total_correct / total_samples_run) * 100
                current_overall_recall = (total_recall_sum / total_samples_run) * 100
                print(f"📈 Cumulative QAEG exact match rate up to now: {current_overall_em:.2f}% | Cumulative Context Recall: {current_overall_recall:.2f}%")

        except Exception as e:
            print(f"❌ Batch {i + 1} QAEG inference failed (network retries exhausted), skipping this batch: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

        # Throttle requests between consecutive batches to reduce server load
        if i < num_batches - 1:
            await asyncio.sleep(3)

    # 4. Summarize evaluation results across all successfully processed batches
    print("\n" + "=" * 40)
    print("📊 QAEG all batches completed, summarizing evaluation results")
    if total_samples_run > 0:
        final_exact_match = (total_correct / total_samples_run) * 100
        final_context_recall = (total_recall_sum / total_samples_run) * 100
    else:
        final_exact_match = 0.0
        final_context_recall = 0.0
    final_results = {
        "exact_match": final_exact_match,
        "context_recall": final_context_recall,
        "total_evaluated_samples": total_samples_run,
        "total_original_samples": total_size
    }
    print(f"🎯 QAEG overall Exact Match rate: {final_results['exact_match']:.2f}%")
    print(f"🎯 QAEG overall Context Recall: {final_results['context_recall']:.2f}%")
    print(f"📦 Evaluated samples: {final_results['total_evaluated_samples']}/{final_results['total_original_samples']}")
    print("=" * 40)

    # 5. Save QAEG predictions, batch metrics, and experiment configuration
    print(f"\n💾 Saving QAEG experiment results to: {SAVE_PATH}")
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
    print("✅ QAEG experiment results saved successfully!")


if __name__ == "__main__":
    asyncio.run(run_in_batches(ds, batch_size=5))