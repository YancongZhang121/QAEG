"""
QAEG audit interface causal intervention evaluation script.

This script verifies, around the role-constrained evidence flow described in the paper,
that retrieved context is the only admissible evidence for supporting answers,
and that the parametric view is used only for auditing coverage and disagreement.
It fixes the query, context evidence view, and necessary audit states,
and sequentially executes the original path, Frozen-s identity intervention,
answer-object replacement, full parametric view permutation, and direct-exposure positive control.

The final generator always uses the context-bounded interface; except for direct-exposure
positive control, explicit parametric triples never enter the answer evidence field.
The script also records diagnostic information such as empty-response retries,
intervention validity, audit state changes, answer flips, and decoy adoption,
to test whether answer identity passes through the compressed audit channel.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import os
import re
import string
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import httpcore
import httpx
from datasets import Dataset, load_dataset
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from QAEG import qaeg


# =============================================================================
# 0. User-defined configuration (modify according to your setup)
# =============================================================================

logging.basicConfig(level=logging.WARNING)
logging.getLogger("qaeg").setLevel(logging.WARNING)
logging.getLogger("logger").setLevel(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"

# QAEG custom inference server URL
SERVER_URL = "http://your-server-address:port/chat"   # Replace with your LLM service address

# QAEG result save directory and file name
SAVE_DIR = "./results"                                 # Replace with your save directory
SAVE_FILE_NAME = "causal_interface_results.json"       # Replace with your desired file name
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_FILE_NAME)

# Evaluation dataset path (supports local or Hugging Face path)
DATA_FILE = "./datasets/your_dataset.json"             # Replace with your dataset path

# Embedding model path (for similarity computation, required by QAEG)
SIMILARITY_MODEL_PATH = "/path/to/embedding/model"     # Replace with your embedding model directory

# Batch size kept at 5 to control server load and align with existing experiment settings.
BATCH_SIZE = 5
START_BATCH_IDX = 0
BATCH_SLEEP_SECONDS = 3
SAVE_AFTER_EACH_BATCH = True

# MuSiQue includes multi-hop questions, so expand candidate sentence and chunk ranges;
# these parameters only affect evidence selection within fixed source texts.
RETRIEVAL_SENT_TOPK = 8
RETRIEVAL_CHUNK_TOPK = 12
RETRIEVAL_CHUNK_SIZE = 45
MIN_SELECTED_CONTEXT_CHARS = 240
MAX_CONTEXT_CHARS = 14000

# Generation uses deterministic settings; given possible drift on remote services,
# the Frozen‑s primary metric directly reuses the original output from the same decoding input.
TEMPERATURE = 0.0
TOP_P = 1.0
PARAMETRIC_MAX_TOKENS = 500
DECOY_MAX_TOKENS = 120
COVERAGE_MAX_TOKENS = 120
GENERATION_MAX_TOKENS = 360
SERVER_EMPTY_RETRIES = 3
SERVER_RETRY_SLEEP_SECONDS = 1.0

# Audit state consists of context coverage and parametric disagreement:
# coverage controls whether answering is allowed, disagreement serves only as audit metadata and cannot provide answers.
COVERAGE_THRESHOLD = 0.55
DISAGREEMENT_REPORT_PENALTY = 0.10
DEFAULT_COVERAGE_WITH_STRONG_EVIDENCE = 0.82
DEFAULT_COVERAGE_WITH_WEAK_EVIDENCE = 0.68
DEFAULT_COVERAGE_WITHOUT_EVIDENCE = 0.20

# Frozen-s is an identity intervention; the final decoding input remains unchanged.
# The optional remote re‑decoding is only for diagnosing service drift and does not contribute to primary causal metrics.
RUN_FROZEN_REDECODE_DIAGNOSTIC = False

# Direct exposure is a positive control: only under this condition are the intervened target parametric triples
# temporarily made answer‑admissible; this is not part of the formal QAEG answer path.
RUN_DIRECT_EXPOSURE_CONTROL = True


# =============================================================================
# 1. Dataset loading and QAEG instance initialization
# =============================================================================


ds = load_dataset("json", data_files=DATA_FILE, split="train")

rag = qaeg(
    backend_type="custom_server",
    model_name="custom_model",
    server_url=SERVER_URL,
    similarity_model=SIMILARITY_MODEL_PATH,
    enable_dual_evidence_graph=True,
    enable_sufficiency_estimation=True,
)


# =============================================================================
# 2. Output normalization and evaluation helper functions
# =============================================================================


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("_", " ")
    punctuation = set(string.punctuation + "‘’´`")
    text = "".join(" " if ch in punctuation else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def as_answer_list(answer: Any) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, (list, tuple, set)):
        return [str(x).strip() for x in answer if str(x).strip()]
    value = str(answer).strip()
    return [value] if value else []


def normalize_choices(choices: Any) -> List[str]:
    """Compatible with common multiple-choice field representations in HuggingFace datasets."""
    if choices is None:
        return []

    if isinstance(choices, Mapping):
        for key in ("text", "choices", "options"):
            value = choices.get(key)
            if isinstance(value, (list, tuple)):
                return [str(x).strip() for x in value if str(x).strip()]
        return [str(v).strip() for v in choices.values() if str(v).strip()]

    if isinstance(choices, (list, tuple)):
        result: List[str] = []
        for choice in choices:
            if isinstance(choice, Mapping):
                value = (
                    choice.get("text")
                    or choice.get("option")
                    or choice.get("answer")
                    or choice.get("label")
                )
                if value is not None and str(value).strip():
                    result.append(str(value).strip())
            elif str(choice).strip():
                result.append(str(choice).strip())
        return result

    if isinstance(choices, str):
        stripped = choices.strip()
        if not stripped:
            return []
        try:
            return normalize_choices(json.loads(stripped))
        except Exception:
            return [
                line.strip(" -\t")
                for line in stripped.splitlines()
                if line.strip(" -\t")
            ]

    return []


def extract_json_object(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = cleaned.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        ch = cleaned[index]
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
    return None


def extract_answer_from_output(raw_output: Any) -> str:
    if raw_output is None:
        return ""
    text = str(raw_output).strip()
    if not text:
        return ""

    parsed = extract_json_object(text)
    if parsed:
        for key in ("Answer", "answer", "final_answer", "prediction", "result", "ans"):
            if key in parsed and parsed[key] is not None:
                value = parsed[key]
                if isinstance(value, (str, int, float)):
                    return str(value).strip()
        for key, value in parsed.items():
            if "answer" in str(key).lower() and value is not None:
                return str(value).strip()

    patterns = [
        r'(?is)["\']?answer["\']?\s*:\s*["\']([^"\'\n\r}]+)',
        r"(?is)\bfinal\s+answer\s*:\s*([^\n\r}]+)",
        r"(?is)\banswer\s*:\s*([^\n\r}]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().strip('"\'` ')

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0].strip('"\'` ')
    return lines[-1].strip('"\'` ')


def canonicalize_prediction(prediction: Any, choices: Sequence[str]) -> str:
    answer = str(prediction or "").strip().strip('"\'` ')
    if not answer or not choices:
        return answer

    answer_n = normalize_text(answer)
    for choice in choices:
        if answer_n == normalize_text(choice):
            return choice

    # Map label forms like A/B/C/D or "option B" to the corresponding option text.
    label_match = re.fullmatch(
        r"(?:option|choice)?\s*([a-z])(?:[\).:]\s*)?",
        answer_n,
        flags=re.IGNORECASE,
    )
    if label_match:
        index = ord(label_match.group(1).lower()) - ord("a")
        if 0 <= index < len(choices):
            return choices[index]

    number_match = re.fullmatch(r"(?:option|choice)?\s*(\d+)", answer_n)
    if number_match:
        index = int(number_match.group(1)) - 1
        if 0 <= index < len(choices):
            return choices[index]

    # When there is a unique match, normalize verbose output to the exact option text.
    matches = [
        choice
        for choice in choices
        if normalize_text(choice)
        and (
            normalize_text(choice) in answer_n
            or answer_n in normalize_text(choice)
        )
    ]
    return matches[0] if len(matches) == 1 else answer


def answer_equals(left: Any, right: Any) -> bool:
    left_n = normalize_text(left)
    right_n = normalize_text(right)
    return bool(left_n and right_n and left_n == right_n)


def answer_matches_target(prediction: Any, target: Any) -> bool:
    pred_n = normalize_text(prediction)
    target_n = normalize_text(target)
    if not pred_n or not target_n:
        return False
    return pred_n == target_n or pred_n in target_n or target_n in pred_n


def repository_style_correct(prediction: Any, ground_truth: Any) -> bool:
    """Perform normalized prediction‑in‑ground‑truth containment as per the unified evaluation protocol."""
    pred_n = normalize_text(prediction)
    if not pred_n:
        return False
    return any(
        bool(gt_n) and pred_n in gt_n
        for gt_n in (normalize_text(x) for x in as_answer_list(ground_truth))
    )


def relaxed_correct(prediction: Any, ground_truth: Any) -> bool:
    pred_n = normalize_text(prediction)
    if not pred_n:
        return False
    for answer in as_answer_list(ground_truth):
        gt_n = normalize_text(answer)
        if gt_n and (pred_n == gt_n or pred_n in gt_n or gt_n in pred_n):
            return True
    return False


def triples_to_text(triples: Sequence[Sequence[Any]]) -> str:
    if not triples:
        return "None"
    lines = []
    for triple in triples:
        if len(triple) == 3:
            lines.append(f"({triple[0]}, {triple[1]}, {triple[2]})")
    return "\n".join(lines) if lines else "None"


def make_id_map(items: Sequence[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {item["id"]: item for item in items}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def wrap_llama3_prompt(system_prompt: str, user_content: str) -> str:
    return (
        "<|begin_of_solution|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


def is_nonempty_output(value: Any) -> bool:
    return bool(str(value or "").strip())


async def generate_with_empty_retry(
    backend: Any,
    ids: Sequence[Any],
    prompts: Sequence[str],
    system_prompt: str,
    *,
    max_tokens: int,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    empty_retries: int = SERVER_EMPTY_RETRIES,
) -> Tuple[Dict[Any, str], Dict[Any, int]]:
    """Perform targeted retries only for samples where the server returned empty or missing."""
    if len(ids) != len(prompts):
        raise ValueError("ids and prompts must have the same length")

    output_by_id: Dict[Any, str] = {sample_id: "" for sample_id in ids}
    attempts_by_id: Dict[Any, int] = {sample_id: 0 for sample_id in ids}
    pending = list(range(len(ids)))

    for attempt in range(1, empty_retries + 2):
        if not pending:
            break

        pending_prompts = [prompts[index] for index in pending]
        raw_results = await backend.generate(
            prompts=pending_prompts,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
        )
        raw_results = list(raw_results or [])

        next_pending: List[int] = []
        for local_position, original_index in enumerate(pending):
            sample_id = ids[original_index]
            attempts_by_id[sample_id] = attempt
            result = raw_results[local_position] if local_position < len(raw_results) else ""
            output_by_id[sample_id] = str(result or "")
            if not is_nonempty_output(result):
                next_pending.append(original_index)

        pending = next_pending
        if pending and attempt <= empty_retries:
            await asyncio.sleep(SERVER_RETRY_SLEEP_SECONDS * attempt)

    return output_by_id, attempts_by_id


# =============================================================================
# 3. Fixed answer‑admissible context evidence
# =============================================================================


def serialize_context(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        preferred: List[str] = []
        for key in ("title", "paragraph_text", "paragraph", "text", "content"):
            if key in value and value[key] is not None:
                text = serialize_context(value[key])
                if text:
                    preferred.append(text)
        if preferred:
            return "\n".join(preferred)
        return "\n".join(
            text
            for text in (serialize_context(v) for v in value.values())
            if text
        )
    if isinstance(value, (list, tuple)):
        return "\n\n".join(
            text
            for text in (serialize_context(v) for v in value)
            if text
        )
    return str(value).strip()


@dataclass
class FixedEvidence:
    sample_id: Any
    context: str
    retrieved_triples: List[Tuple[str, str, str]]
    selected_chunks: List[str]
    context_source: str


def build_fixed_evidence(
    dataset: Dataset,
    retrieved_facts: Sequence[Dict[str, Any]],
    ranked_chunks: Sequence[Dict[str, Any]],
) -> Dict[Any, FixedEvidence]:
    fact_map = make_id_map(retrieved_facts)
    chunk_map = make_id_map(ranked_chunks)
    evidence_by_id: Dict[Any, FixedEvidence] = {}

    for item in dataset:
        sample_id = item["id"]
        selected_chunks: List[str] = []
        seen = set()
        for chunk_item in chunk_map.get(sample_id, {}).get("topk_chunks", []):
            text = str(chunk_item.get("chunk", "")).strip()
            key = normalize_text(text)
            if text and key and key not in seen:
                selected_chunks.append(text)
                seen.add(key)

        selected_context = "\n\n".join(selected_chunks).strip()
        raw_context = serialize_context(item.get("context", ""))

        if len(selected_context) >= MIN_SELECTED_CONTEXT_CHARS:
            fixed_context = selected_context
            source = "retrieved_topk_chunks"
        elif raw_context:
            # If triple extraction yields nothing, do not let the final decoder get an empty context;
            # this fallback source is recorded per sample.
            fixed_context = raw_context
            source = "raw_context_fallback"
        else:
            fixed_context = selected_context
            source = "empty"

        fixed_context = fixed_context[:MAX_CONTEXT_CHARS]
        triples = []
        for triple in fact_map.get(sample_id, {}).get("facts", []):
            if len(triple) == 3:
                triples.append(tuple(str(x).strip() for x in triple))

        evidence_by_id[sample_id] = FixedEvidence(
            sample_id=sample_id,
            context=fixed_context,
            retrieved_triples=triples,
            selected_chunks=selected_chunks,
            context_source=source,
        )

    return evidence_by_id


# =============================================================================
# 4. Audit‑only parametric view with explicit answer objects
# =============================================================================


PARAMETRIC_SYSTEM_PROMPT = """You independently elicit parametric-memory knowledge.
Do not use retrieved context, hidden documents, or tools. For each question,
produce a compact factual view and identify exactly one answer-bearing triple.
The object of that target triple must be your own best final answer. Output only
valid JSON and never include explanations outside JSON."""


def parse_triple(value: Any) -> Optional[Tuple[str, str, str]]:
    if isinstance(value, Mapping):
        subject = value.get("subject") or value.get("s")
        predicate = value.get("predicate") or value.get("relation") or value.get("p")
        obj = value.get("object") or value.get("answer") or value.get("o")
        if subject is not None and predicate is not None and obj is not None:
            return (str(subject).strip(), str(predicate).strip(), str(obj).strip())
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(str(x).strip() for x in value)  # type: ignore[return-value]
    if isinstance(value, str):
        match = re.fullmatch(r"\s*\((.*?),\s*(.*?),\s*(.*?)\)\s*", value, flags=re.DOTALL)
        if match:
            return tuple(part.strip() for part in match.groups())  # type: ignore[return-value]
    return None


def parse_parametric_view(raw: Any) -> Tuple[List[Tuple[str, str, str]], Optional[int], str]:
    text = str(raw or "").strip()
    parsed = extract_json_object(text)
    triples: List[Tuple[str, str, str]] = []
    target_index: Optional[int] = None
    answer_value = ""

    if parsed:
        raw_triples = parsed.get("triples") or parsed.get("parametric_triples") or []
        if isinstance(raw_triples, Mapping):
            raw_triples = [raw_triples]
        if isinstance(raw_triples, (list, tuple)):
            for value in raw_triples:
                triple = parse_triple(value)
                if triple and all(triple):
                    triples.append(triple)

        raw_index = parsed.get("answer_triple_index")
        if raw_index is None:
            raw_index = parsed.get("target_index")
        try:
            if raw_index is not None:
                target_index = int(raw_index)
        except (TypeError, ValueError):
            target_index = None

        answer_value = str(parsed.get("answer") or parsed.get("final_answer") or "").strip()

    if not triples:
        for subject, predicate, obj in re.findall(
            r"\((.*?),\s*(.*?),\s*(.*?)\)", text, flags=re.DOTALL
        ):
            triple = (subject.strip(), predicate.strip(), obj.strip())
            if all(triple):
                triples.append(triple)

    if target_index is not None and not (0 <= target_index < len(triples)):
        # Some models return 1‑based triple indices.
        one_based = target_index - 1
        target_index = one_based if 0 <= one_based < len(triples) else None

    if target_index is None and answer_value:
        for index, triple in enumerate(triples):
            if answer_matches_target(triple[2], answer_value):
                target_index = index
                break

    if not triples and answer_value:
        triples = [("the question", "has answer", answer_value)]
        target_index = 0

    if triples and target_index is None:
        # The prompt asks for an explicit answer‑triple marker; if we cannot parse an index,
        # fall back to marking the last triple as the target.
        target_index = len(triples) - 1
        return triples, target_index, "elicited_last_triple_fallback"

    method = "elicited_answer_bearing_triple" if target_index is not None else "parse_failed"
    return triples, target_index, method


async def elicit_answer_bearing_parametric_views(
    rag_model: qaeg,
    dataset: Dataset,
    fallback_parametric_facts: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    ids: List[Any] = []
    prompts: List[str] = []

    for item in dataset:
        choices = normalize_choices(item.get("choices"))
        options_text = "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
        ) or "None"
        user_content = f"""Question:
{item.get('question', '')}

Options (if any):
{options_text}

Return exactly this JSON schema:
{{
  "triples": [
    {{"subject": "...", "predicate": "...", "object": "..."}}
  ],
  "answer_triple_index": 0,
  "answer": "the exact object of the answer-bearing triple"
}}

Requirements:
- Use only your parametric memory, not retrieved context.
- Include at most five triples.
- The indexed triple must directly encode your final answer in its object.
- When options exist, the answer object must be exact option text.
"""
        ids.append(item["id"])
        prompts.append(wrap_llama3_prompt(PARAMETRIC_SYSTEM_PROMPT, user_content))

    raw_by_id, attempts_by_id = await generate_with_empty_retry(
        rag_model.reasoning_engine.llm_backend,
        ids,
        prompts,
        PARAMETRIC_SYSTEM_PROMPT,
        max_tokens=PARAMETRIC_MAX_TOKENS,
    )

    fallback_map = make_id_map(fallback_parametric_facts)
    views: List[Dict[str, Any]] = []
    metadata: Dict[Any, Dict[str, Any]] = {}

    for item in dataset:
        sample_id = item["id"]
        raw = raw_by_id.get(sample_id, "")
        triples, target_index, method = parse_parametric_view(raw)

        if not triples:
            fallback_triples = []
            for triple in fallback_map.get(sample_id, {}).get("parametric_triples", []):
                if len(triple) == 3:
                    fallback_triples.append(tuple(str(x).strip() for x in triple))
            triples = fallback_triples
            target_index = len(triples) - 1 if triples else None
            method = "legacy_parametric_fallback" if triples else "no_parametric_view"

        views.append({"id": sample_id, "parametric_triples": triples})
        metadata[sample_id] = {
            "target_triple_index": target_index,
            "target_selection_method": method,
            "strict_answer_object_match": (
                target_index is not None and method.startswith("elicited_")
            ),
            "parametric_raw": raw,
            "parametric_generation_attempts": attempts_by_id.get(sample_id, 0),
        }

    return views, metadata


# =============================================================================
# 5. Type‑compatible answer‑object intervention and parametric‑view permutation
# =============================================================================


def infer_answer_type(value: Any) -> str:
    text = str(value or "").strip()
    normalized = normalize_text(text)
    if not normalized:
        return "empty"
    if normalized in {"yes", "no", "true", "false"}:
        return "boolean"
    if re.fullmatch(r"(?:1[0-9]{3}|20[0-9]{2})", normalized):
        return "year"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*[%a-z]+)?", normalized):
        return "number"
    if re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        normalized,
    ):
        return "date"
    words = text.split()
    if 1 <= len(words) <= 6 and sum(word[:1].isupper() for word in words) >= max(1, len(words) // 2):
        return "named_entity"
    return "free_text"


def choose_pool_decoy(
    original_object: str,
    ground_truth: Any,
    global_answer_pool: Sequence[str],
    offset: int,
) -> Optional[str]:
    original_type = infer_answer_type(original_object)
    forbidden = {normalize_text(original_object)} | {
        normalize_text(answer) for answer in as_answer_list(ground_truth)
    }
    candidates = [
        value
        for value in global_answer_pool
        if normalize_text(value)
        and normalize_text(value) not in forbidden
        and infer_answer_type(value) == original_type
    ]
    if not candidates:
        candidates = [
            value
            for value in global_answer_pool
            if normalize_text(value) and normalize_text(value) not in forbidden
        ]
    return candidates[offset % len(candidates)] if candidates else None


DECOY_SYSTEM_PROMPT = """Generate a controlled counterfactual answer object.
Return only JSON. The decoy must be plausible for the question, have the same
semantic type as the original object, and be different from the original object.
Do not provide explanations."""


async def generate_open_ended_decoys(
    rag_model: qaeg,
    dataset: Dataset,
    parametric_views: Sequence[Dict[str, Any]],
    target_metadata: Mapping[Any, Dict[str, Any]],
) -> Tuple[Dict[Any, str], Dict[Any, int]]:
    view_map = make_id_map(parametric_views)
    ids: List[Any] = []
    prompts: List[str] = []

    for item in dataset:
        if normalize_choices(item.get("choices")):
            continue
        sample_id = item["id"]
        target_index = target_metadata.get(sample_id, {}).get("target_triple_index")
        triples = view_map.get(sample_id, {}).get("parametric_triples", [])
        if target_index is None or not (0 <= target_index < len(triples)):
            continue
        original_object = str(triples[target_index][2]).strip()
        user_content = f"""Question:
{item.get('question', '')}

Original answer object:
{original_object}

Return exactly:
{{"decoy": "a plausible but incorrect alternative of the same semantic type"}}
"""
        ids.append(sample_id)
        prompts.append(wrap_llama3_prompt(DECOY_SYSTEM_PROMPT, user_content))

    if not ids:
        return {}, {}

    raw_by_id, attempts_by_id = await generate_with_empty_retry(
        rag_model.reasoning_engine.llm_backend,
        ids,
        prompts,
        DECOY_SYSTEM_PROMPT,
        max_tokens=DECOY_MAX_TOKENS,
    )

    decoys: Dict[Any, str] = {}
    for sample_id in ids:
        raw = raw_by_id.get(sample_id, "")
        parsed = extract_json_object(raw)
        decoy = ""
        if parsed:
            decoy = str(parsed.get("decoy") or parsed.get("answer") or "").strip()
        if not decoy:
            decoy = extract_answer_from_output(raw)
        if decoy:
            decoys[sample_id] = decoy
    return decoys, attempts_by_id


async def build_object_swap_parametric_view(
    rag_model: qaeg,
    dataset: Dataset,
    parametric_views: Sequence[Dict[str, Any]],
    target_metadata: Mapping[Any, Dict[str, Any]],
    global_answer_pool: Sequence[str],
    batch_start_index: int,
) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    generated_decoys, decoy_attempts = await generate_open_ended_decoys(
        rag_model,
        dataset,
        parametric_views,
        target_metadata,
    )
    view_map = make_id_map(parametric_views)
    corrupted_views: List[Dict[str, Any]] = []
    metadata: Dict[Any, Dict[str, Any]] = {}

    for local_index, item in enumerate(dataset):
        sample_id = item["id"]
        triples = copy.deepcopy(view_map.get(sample_id, {}).get("parametric_triples", []))
        target_index = target_metadata.get(sample_id, {}).get("target_triple_index")
        strict_target = bool(target_metadata.get(sample_id, {}).get("strict_answer_object_match"))

        original_target_triple = None
        original_object = ""
        if target_index is not None and 0 <= target_index < len(triples):
            original_target_triple = tuple(triples[target_index])
            original_object = str(triples[target_index][2]).strip()

        choices = normalize_choices(item.get("choices"))
        gold_norms = {normalize_text(x) for x in as_answer_list(item.get("answer"))}
        forbidden = gold_norms | {normalize_text(original_object)}

        decoy: Optional[str] = None
        decoy_source = "unavailable"
        wrong_options = [
            choice
            for choice in choices
            if normalize_text(choice)
            and normalize_text(choice) not in forbidden
            and "don't know" not in normalize_text(choice)
            and "do not know" not in normalize_text(choice)
        ]
        if wrong_options:
            decoy = wrong_options[(batch_start_index + local_index) % len(wrong_options)]
            decoy_source = "wrong_option"
        elif generated_decoys.get(sample_id):
            candidate = generated_decoys[sample_id]
            if normalize_text(candidate) not in forbidden:
                decoy = candidate
                decoy_source = "generated_same_type"

        if decoy is None and original_object:
            decoy = choose_pool_decoy(
                original_object,
                item.get("answer"),
                global_answer_pool,
                batch_start_index + local_index,
            )
            if decoy is not None:
                decoy_source = "same_type_answer_pool"

        intervention_valid = bool(
            strict_target
            and decoy
            and target_index is not None
            and 0 <= target_index < len(triples)
            and normalize_text(decoy) not in forbidden
        )

        corrupted_target_triple = None
        if intervention_valid:
            subject, predicate, _ = triples[target_index]
            triples[target_index] = (subject, predicate, str(decoy))
            corrupted_target_triple = tuple(triples[target_index])

        corrupted_views.append({"id": sample_id, "parametric_triples": triples})
        metadata[sample_id] = {
            **dict(target_metadata.get(sample_id, {})),
            "decoy": decoy,
            "decoy_source": decoy_source,
            "decoy_generation_attempts": decoy_attempts.get(sample_id, 0),
            "intervention_valid": intervention_valid,
            "original_target_triple": original_target_triple,
            "corrupted_target_triple": corrupted_target_triple,
        }

    return corrupted_views, metadata


def build_shuffled_parametric_view(
    dataset: Dataset,
    parametric_views: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    view_map = make_id_map(parametric_views)
    ids = [item["id"] for item in dataset]
    nonempty_ids = [
        sample_id
        for sample_id in ids
        if view_map.get(sample_id, {}).get("parametric_triples")
    ]

    shuffled: List[Dict[str, Any]] = []
    metadata: Dict[Any, Dict[str, Any]] = {}
    for position, target_id in enumerate(ids):
        donor_id = None
        if len(nonempty_ids) >= 2:
            if target_id in nonempty_ids:
                source_position = nonempty_ids.index(target_id)
                donor_id = nonempty_ids[(source_position + 1) % len(nonempty_ids)]
            else:
                donor_id = nonempty_ids[position % len(nonempty_ids)]
        elif len(nonempty_ids) == 1 and nonempty_ids[0] != target_id:
            donor_id = nonempty_ids[0]

        valid = donor_id is not None and donor_id != target_id
        donor_triples = copy.deepcopy(
            view_map.get(donor_id, {}).get("parametric_triples", [])
            if valid
            else view_map.get(target_id, {}).get("parametric_triples", [])
        )
        shuffled.append({"id": target_id, "parametric_triples": donor_triples})
        metadata[target_id] = {
            "shuffle_valid": valid,
            "shuffle_donor_id": donor_id,
        }

    return shuffled, metadata


# =============================================================================
# 6. QAEG compressed audit state construction
# =============================================================================


COVERAGE_SYSTEM_PROMPT = """You judge whether retrieved evidence alone is
sufficient to answer a question. Parametric memory is not available. Do not
answer the question and do not reveal an answer candidate. Output only JSON with
one numeric field named coverage between 0 and 1. A high score requires an
explicit, complete evidence path; multi-hop composition is allowed when every
step is present."""


@dataclass
class AuditState:
    coverage: float
    disagreement: float
    score: float
    admissible: bool
    coverage_raw: str = ""
    coverage_source: str = ""


def structural_coverage_fallback(evidence: FixedEvidence) -> float:
    context_length = len(evidence.context.strip())
    triple_count = len(evidence.retrieved_triples)
    if context_length >= MIN_SELECTED_CONTEXT_CHARS and triple_count >= 2:
        return DEFAULT_COVERAGE_WITH_STRONG_EVIDENCE
    if context_length >= 80 or triple_count >= 1:
        return DEFAULT_COVERAGE_WITH_WEAK_EVIDENCE
    return DEFAULT_COVERAGE_WITHOUT_EVIDENCE


def parse_coverage(raw: Any) -> Optional[float]:
    parsed = extract_json_object(raw)
    value: Any = None
    if parsed:
        for key in ("coverage", "coverage_score", "score"):
            if key in parsed:
                value = parsed[key]
                break
    if value is None:
        match = re.search(r"(?<![\d.])(-?\d+(?:\.\d+)?)(?![\d.])", str(raw or ""))
        value = match.group(1) if match else None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < score <= 100.0:
        score /= 100.0
    return max(0.0, min(1.0, score))


async def estimate_retrieval_coverage_once(
    rag_model: qaeg,
    dataset: Dataset,
    evidence_by_id: Mapping[Any, FixedEvidence],
) -> Tuple[Dict[Any, float], Dict[Any, Dict[str, Any]]]:
    ids: List[Any] = []
    prompts: List[str] = []

    for item in dataset:
        sample_id = item["id"]
        evidence = evidence_by_id[sample_id]
        user_content = f"""Question:
{item.get('question', '')}

Retrieved evidence chunks:
{evidence.context}

Retrieved evidence triples:
{triples_to_text(evidence.retrieved_triples)}

Judge only evidence coverage. Do not state the answer.
Return exactly: {{"coverage": 0.00}}
"""
        ids.append(sample_id)
        prompts.append(wrap_llama3_prompt(COVERAGE_SYSTEM_PROMPT, user_content))

    raw_by_id, attempts_by_id = await generate_with_empty_retry(
        rag_model.sufficiency_estimator.llm_backend,
        ids,
        prompts,
        COVERAGE_SYSTEM_PROMPT,
        max_tokens=COVERAGE_MAX_TOKENS,
    )

    coverage_by_id: Dict[Any, float] = {}
    diagnostics: Dict[Any, Dict[str, Any]] = {}
    for item in dataset:
        sample_id = item["id"]
        evidence = evidence_by_id[sample_id]
        fallback = structural_coverage_fallback(evidence)
        raw = raw_by_id.get(sample_id, "")
        parsed = parse_coverage(raw)

        if parsed is None:
            coverage = fallback
            source = "structural_fallback_empty_or_unparseable"
        else:
            # The already selected fixed retrieval evidence itself provides a structural coverage signal;
            # taking the upper bound avoids unnecessary refusals due to occasional underestimation by the auditor.
            coverage = max(parsed, fallback)
            source = "llm_plus_structural_lower_bound"

        coverage_by_id[sample_id] = coverage
        diagnostics[sample_id] = {
            "coverage_raw": raw,
            "coverage_parsed": parsed,
            "coverage_fallback": fallback,
            "coverage_final": coverage,
            "coverage_source": source,
            "coverage_generation_attempts": attempts_by_id.get(sample_id, 0),
        }

    return coverage_by_id, diagnostics


def object_supported_by_retrieval(obj: Any, evidence: FixedEvidence) -> bool:
    obj_n = normalize_text(obj)
    if not obj_n:
        return False
    context_n = normalize_text(evidence.context)
    if obj_n in context_n:
        return True
    for triple in evidence.retrieved_triples:
        if obj_n == normalize_text(triple[2]) or obj_n in normalize_text(triple[2]):
            return True
    return False


def compute_disagreement(
    parametric_triples: Sequence[Sequence[Any]],
    evidence: FixedEvidence,
) -> float:
    if not parametric_triples:
        return 0.5

    retrieved_pair_to_objects: Dict[Tuple[str, str], set[str]] = {}
    for triple in evidence.retrieved_triples:
        if len(triple) != 3:
            continue
        key = (normalize_text(triple[0]), normalize_text(triple[1]))
        retrieved_pair_to_objects.setdefault(key, set()).add(normalize_text(triple[2]))

    weighted_total = 0.0
    weighted_conflict = 0.0
    for triple in parametric_triples:
        if len(triple) != 3:
            continue
        subject, predicate, obj = triple
        key = (normalize_text(subject), normalize_text(predicate))
        obj_n = normalize_text(obj)
        weight = 1.0
        weighted_total += weight

        if key in retrieved_pair_to_objects:
            if obj_n and obj_n not in retrieved_pair_to_objects[key]:
                weighted_conflict += 1.0
        elif not object_supported_by_retrieval(obj, evidence):
            # Lack of support is weaker than explicit object conflict under the same relation,
            # so assign a lower conflict weight.
            weighted_conflict += 0.45

    if weighted_total <= 0:
        return 0.5
    return max(0.0, min(1.0, weighted_conflict / weighted_total))


def build_audit_states(
    dataset: Dataset,
    evidence_by_id: Mapping[Any, FixedEvidence],
    coverage_by_id: Mapping[Any, float],
    parametric_views: Sequence[Dict[str, Any]],
    coverage_diagnostics: Mapping[Any, Dict[str, Any]],
) -> Dict[Any, AuditState]:
    view_map = make_id_map(parametric_views)
    states: Dict[Any, AuditState] = {}
    for item in dataset:
        sample_id = item["id"]
        coverage = float(coverage_by_id.get(sample_id, DEFAULT_COVERAGE_WITHOUT_EVIDENCE))
        triples = view_map.get(sample_id, {}).get("parametric_triples", [])
        disagreement = compute_disagreement(triples, evidence_by_id[sample_id])
        score = max(
            0.0,
            min(1.0, coverage - DISAGREEMENT_REPORT_PENALTY * disagreement),
        )
        diagnostic = coverage_diagnostics.get(sample_id, {})
        states[sample_id] = AuditState(
            coverage=coverage,
            disagreement=disagreement,
            score=score,
            admissible=coverage >= COVERAGE_THRESHOLD,
            coverage_raw=str(diagnostic.get("coverage_raw", "")),
            coverage_source=str(diagnostic.get("coverage_source", "")),
        )
    return states


# =============================================================================
# 7. Context‑bounded generation and direct‑exposure positive control
# =============================================================================


CONTEXT_DECODER_SYSTEM_PROMPT = """You are a context-authoritative retrieval QA
decoder. Retrieved evidence is the sole answer-admissible source. Parametric
memory, world knowledge, option frequency and audit metadata cannot support an
answer. Audit metadata may only indicate whether to be cautious or abstain.
Multi-hop reasoning is allowed only when every link is explicit in retrieved
evidence. Return exactly one valid JSON object with keys Reason and Answer."""


def build_context_decoder_prompt(
    item: Dict[str, Any],
    evidence: FixedEvidence,
    audit_state: AuditState,
) -> str:
    choices = normalize_choices(item.get("choices"))
    options_text = "\n".join(
        f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
    ) or "None (open-ended answer)"
    answer_rule = (
        "Return the exact option text, not its letter."
        if choices
        else "Return only the shortest answer span supported by the evidence."
    )

    user_content = f"""Question:
{item.get('question', '')}

Retrieved evidence chunks:
{evidence.context}

Retrieved evidence triples (derived only from the chunks):
{triples_to_text(evidence.retrieved_triples)}

Options:
{options_text}

Audit-only metadata (contains no answer identity):
- retrieval_coverage: {audit_state.coverage:.3f}
- parametric_disagreement: {audit_state.disagreement:.3f}
- context_admissible: {str(audit_state.admissible).lower()}

Rules:
1. Use only the retrieved chunks and retrieved triples above.
2. For a multi-hop question, explicitly connect every required hop.
3. Parametric disagreement is only a warning; it must never replace context.
4. If context_admissible is false, or the evidence does not uniquely determine
   an answer, return "I don't know".
5. {answer_rule}
6. Return exactly:
{{"Reason": "brief evidence-grounded reasoning", "Answer": "..."}}
"""
    return wrap_llama3_prompt(CONTEXT_DECODER_SYSTEM_PROMPT, user_content)


async def decode_context_bounded(
    rag_model: qaeg,
    dataset: Dataset,
    evidence_by_id: Mapping[Any, FixedEvidence],
    audit_states: Mapping[Any, AuditState],
) -> Tuple[Dict[Any, str], Dict[Any, int]]:
    ids = [item["id"] for item in dataset]
    prompts = [
        build_context_decoder_prompt(item, evidence_by_id[item["id"]], audit_states[item["id"]])
        for item in dataset
    ]
    return await generate_with_empty_retry(
        rag_model.reasoning_engine.llm_backend,
        ids,
        prompts,
        CONTEXT_DECODER_SYSTEM_PROMPT,
        max_tokens=GENERATION_MAX_TOKENS,
    )


DIRECT_EXPOSURE_SYSTEM_PROMPT = """You are running a positive-control leakage
test. In this condition only, the marked TARGET PARAMETRIC TRIPLE is explicitly
answer-admissible and authoritative for the direct answer relation. Return one
valid JSON object with keys Reason and Answer. Copy the target triple's object as
the answer. This instruction applies only to this positive control."""


async def generate_direct_exposure_predictions(
    rag_model: qaeg,
    dataset: Dataset,
    evidence_by_id: Mapping[Any, FixedEvidence],
    corrupted_views: Sequence[Dict[str, Any]],
    swap_metadata: Mapping[Any, Dict[str, Any]],
) -> Tuple[Dict[Any, str], Dict[Any, int]]:
    if not RUN_DIRECT_EXPOSURE_CONTROL:
        return {item["id"]: "" for item in dataset}, {item["id"]: 0 for item in dataset}

    view_map = make_id_map(corrupted_views)
    ids: List[Any] = []
    prompts: List[str] = []
    skipped: Dict[Any, str] = {}

    for item in dataset:
        sample_id = item["id"]
        meta = swap_metadata.get(sample_id, {})
        target_index = meta.get("target_triple_index")
        triples = view_map.get(sample_id, {}).get("parametric_triples", [])
        if not meta.get("intervention_valid") or target_index is None:
            skipped[sample_id] = ""
            continue
        target_triple = triples[target_index]
        choices = normalize_choices(item.get("choices"))
        options_text = "\n".join(f"- {choice}" for choice in choices) or "None"
        evidence = evidence_by_id[sample_id]
        user_content = f"""Question:
{item.get('question', '')}

Retrieved context (shown only to preserve the experimental input):
{evidence.context}

[TARGET PARAMETRIC TRIPLE -- ANSWER-ADMISSIBLE IN THIS CONTROL]
({target_triple[0]}, {target_triple[1]}, {target_triple[2]})

Options:
{options_text}

Copy the TARGET triple object into Answer. Return exactly:
{{"Reason": "the marked target triple is authoritative in this control", "Answer": "exact target object"}}
"""
        ids.append(sample_id)
        prompts.append(wrap_llama3_prompt(DIRECT_EXPOSURE_SYSTEM_PROMPT, user_content))

    output_by_id: Dict[Any, str] = dict(skipped)
    attempts_by_id: Dict[Any, int] = {sample_id: 0 for sample_id in skipped}
    if ids:
        generated, attempts = await generate_with_empty_retry(
            rag_model.reasoning_engine.llm_backend,
            ids,
            prompts,
            DIRECT_EXPOSURE_SYSTEM_PROMPT,
            max_tokens=GENERATION_MAX_TOKENS,
        )
        output_by_id.update(generated)
        attempts_by_id.update(attempts)
    return output_by_id, attempts_by_id


# =============================================================================
# 8. QAEG causal intervention batch execution
# =============================================================================


RETRYABLE_EXCEPTIONS = (
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpcore.RemoteProtocolError,
    ConnectionError,
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=20),
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    reraise=True,
)
async def process_causal_batch_with_retry(
    rag_model: qaeg,
    batch_ds: Dataset,
    global_answer_pool: Sequence[str],
    batch_start_index: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print("    🧭 Building QAEG role evidence states and fixing query contract A and context view T_c ...")
    facts_dict = await rag_model.get_self_facts(batch_ds)
    retrieved_facts = facts_dict.get("retrieved_facts", [])
    legacy_parametric_facts = facts_dict.get("parametric_facts") or []

    print("    📚 Fixing the actual answer‑admissible context evidence received by the final generator ...")
    ranked_chunks = rag_model.get_topk_chunks(
        batch_ds,
        retrieved_facts,
        sent_topk=RETRIEVAL_SENT_TOPK,
        chunk_topk=RETRIEVAL_CHUNK_TOPK,
        chunk_size=RETRIEVAL_CHUNK_SIZE,
    )
    chunk_map = make_id_map(ranked_chunks)
    for retrieved_item in retrieved_facts:
        retrieved_item.update(chunk_map.get(retrieved_item["id"], {}))
    evidence_by_id = build_fixed_evidence(batch_ds, retrieved_facts, ranked_chunks)

    print("    🔎 Independently building audit‑only parametric view T_m and marking its answer object ...")
    original_parametric_views, target_metadata = await elicit_answer_bearing_parametric_views(
        rag_model,
        batch_ds,
        legacy_parametric_facts,
    )

    print("    🛡️ Estimating context evidence coverage and compressing into original audit state s ...")
    coverage_by_id, coverage_diagnostics = await estimate_retrieval_coverage_once(
        rag_model,
        batch_ds,
        evidence_by_id,
    )
    original_states = build_audit_states(
        batch_ds,
        evidence_by_id,
        coverage_by_id,
        original_parametric_views,
        coverage_diagnostics,
    )

    print("    1/5 Original QAEG path: performing context‑bounded generation ...")
    original_predictions, original_attempts = await decode_context_bounded(
        rag_model,
        batch_ds,
        evidence_by_id,
        original_states,
    )

    print("    2/5 Frozen‑s identity intervention: keeping audit state and decoding input unchanged ...")
    frozen_predictions = copy.deepcopy(original_predictions)
    frozen_diagnostic_predictions: Dict[Any, str] = {}
    frozen_diagnostic_attempts: Dict[Any, int] = {}
    if RUN_FROZEN_REDECODE_DIAGNOSTIC:
        frozen_diagnostic_predictions, frozen_diagnostic_attempts = await decode_context_bounded(
            rag_model,
            batch_ds,
            evidence_by_id,
            original_states,
        )

    print("    🎯 Constructing type‑compatible parametric answer‑object intervention ...")
    corrupted_views, swap_metadata = await build_object_swap_parametric_view(
        rag_model,
        batch_ds,
        original_parametric_views,
        target_metadata,
        global_answer_pool,
        batch_start_index,
    )

    print("    3/5 Object‑swap intervention: recomputing compressed audit state and performing context‑bounded generation ...")
    swap_states = build_audit_states(
        batch_ds,
        evidence_by_id,
        coverage_by_id,
        corrupted_views,
        coverage_diagnostics,
    )
    swap_predictions, swap_attempts = await decode_context_bounded(
        rag_model,
        batch_ds,
        evidence_by_id,
        swap_states,
    )

    print("    4/5 T_m‑shuffle intervention: permuting the full audit view and reconstructing audit state ...")
    shuffled_views, shuffle_metadata = build_shuffled_parametric_view(
        batch_ds,
        original_parametric_views,
    )
    shuffle_states = build_audit_states(
        batch_ds,
        evidence_by_id,
        coverage_by_id,
        shuffled_views,
        coverage_diagnostics,
    )
    shuffle_predictions, shuffle_attempts = await decode_context_bounded(
        rag_model,
        batch_ds,
        evidence_by_id,
        shuffle_states,
    )

    print("    5/5 Direct‑exposure positive control: temporarily allowing intervened parametric object into answer evidence ...")
    direct_predictions, direct_attempts = await generate_direct_exposure_predictions(
        rag_model,
        batch_ds,
        evidence_by_id,
        corrupted_views,
        swap_metadata,
    )

    original_view_map = make_id_map(original_parametric_views)
    corrupted_view_map = make_id_map(corrupted_views)
    shuffled_view_map = make_id_map(shuffled_views)

    records: List[Dict[str, Any]] = []
    for item in batch_ds:
        sample_id = item["id"]
        choices = normalize_choices(item.get("choices"))

        original_raw = original_predictions.get(sample_id, "")
        frozen_raw = frozen_predictions.get(sample_id, "")
        swap_raw = swap_predictions.get(sample_id, "")
        shuffle_raw = shuffle_predictions.get(sample_id, "")
        direct_raw = direct_predictions.get(sample_id, "")

        original_answer = canonicalize_prediction(extract_answer_from_output(original_raw), choices)
        frozen_answer = canonicalize_prediction(extract_answer_from_output(frozen_raw), choices)
        swap_answer = canonicalize_prediction(extract_answer_from_output(swap_raw), choices)
        shuffle_answer = canonicalize_prediction(extract_answer_from_output(shuffle_raw), choices)
        direct_answer = canonicalize_prediction(extract_answer_from_output(direct_raw), choices)

        diagnostic_raw = frozen_diagnostic_predictions.get(sample_id, "")
        diagnostic_answer = canonicalize_prediction(
            extract_answer_from_output(diagnostic_raw),
            choices,
        )

        swap_meta = swap_metadata.get(sample_id, {})
        shuffle_meta = shuffle_metadata.get(sample_id, {})
        decoy = swap_meta.get("decoy")
        original_state = original_states[sample_id]
        swap_state = swap_states[sample_id]
        shuffle_state = shuffle_states[sample_id]
        evidence = evidence_by_id[sample_id]

        record = {
            "id": sample_id,
            "question": item.get("question", ""),
            "gold_answer": item.get("answer"),
            "choices": choices,
            "fixed_evidence": asdict(evidence),
            "original_parametric_triples": original_view_map.get(sample_id, {}).get(
                "parametric_triples", []
            ),
            "corrupted_parametric_triples": corrupted_view_map.get(sample_id, {}).get(
                "parametric_triples", []
            ),
            "shuffled_parametric_triples": shuffled_view_map.get(sample_id, {}).get(
                "parametric_triples", []
            ),
            **swap_meta,
            **shuffle_meta,
            "audit": {
                "original": asdict(original_state),
                "swap": asdict(swap_state),
                "shuffle": asdict(shuffle_state),
                "swap_abs_score_delta": abs(swap_state.score - original_state.score),
                "shuffle_abs_score_delta": abs(shuffle_state.score - original_state.score),
                "coverage_invariant_swap": math.isclose(
                    original_state.coverage, swap_state.coverage, abs_tol=1e-12
                ),
                "coverage_invariant_shuffle": math.isclose(
                    original_state.coverage, shuffle_state.coverage, abs_tol=1e-12
                ),
                "swap_admissibility_crossed": original_state.admissible != swap_state.admissible,
                "shuffle_admissibility_crossed": original_state.admissible != shuffle_state.admissible,
                "coverage_diagnostic": coverage_diagnostics.get(sample_id, {}),
            },
            "outputs": {
                "original_raw": original_raw,
                "original_answer": original_answer,
                "frozen_raw": frozen_raw,
                "frozen_answer": frozen_answer,
                "frozen_redecode_raw": diagnostic_raw,
                "frozen_redecode_answer": diagnostic_answer,
                "swap_raw": swap_raw,
                "swap_answer": swap_answer,
                "shuffle_raw": shuffle_raw,
                "shuffle_answer": shuffle_answer,
                "direct_raw": direct_raw,
                "direct_answer": direct_answer,
            },
            "server_attempts": {
                "original": original_attempts.get(sample_id, 0),
                "frozen_redecode": frozen_diagnostic_attempts.get(sample_id, 0),
                "swap": swap_attempts.get(sample_id, 0),
                "shuffle": shuffle_attempts.get(sample_id, 0),
                "direct": direct_attempts.get(sample_id, 0),
            },
            "indicators": {
                "original_correct": repository_style_correct(original_answer, item.get("answer")),
                "original_relaxed_correct": relaxed_correct(original_answer, item.get("answer")),
                # Identity intervention keeps decoding input unchanged, so primary metric reuses the same cached output.
                "frozen_answer_consistent": answer_equals(original_answer, frozen_answer),
                "frozen_raw_consistent": str(original_raw).strip() == str(frozen_raw).strip(),
                "frozen_redecode_consistent": (
                    answer_equals(original_answer, diagnostic_answer)
                    if RUN_FROZEN_REDECODE_DIAGNOSTIC
                    else None
                ),
                "swap_correct": repository_style_correct(swap_answer, item.get("answer")),
                "swap_relaxed_correct": relaxed_correct(swap_answer, item.get("answer")),
                "swap_answer_flip": not answer_equals(original_answer, swap_answer),
                "swap_decoy_adopted": bool(decoy) and answer_matches_target(swap_answer, decoy),
                "shuffle_correct": repository_style_correct(shuffle_answer, item.get("answer")),
                "shuffle_relaxed_correct": relaxed_correct(shuffle_answer, item.get("answer")),
                "shuffle_answer_flip": not answer_equals(original_answer, shuffle_answer),
                "direct_correct": repository_style_correct(direct_answer, item.get("answer")),
                "direct_decoy_adopted": bool(decoy) and answer_matches_target(direct_answer, decoy),
                "original_parse_success": bool(normalize_text(original_answer)),
                "frozen_parse_success": bool(normalize_text(frozen_answer)),
                "swap_parse_success": bool(normalize_text(swap_answer)),
                "shuffle_parse_success": bool(normalize_text(shuffle_answer)),
                "direct_parse_success": bool(normalize_text(direct_answer)),
            },
        }
        records.append(json_safe(record))

    diagnostics = {
        "num_samples": len(batch_ds),
        "num_valid_object_swaps": sum(
            1 for meta in swap_metadata.values() if meta.get("intervention_valid")
        ),
        "num_strict_object_targets": sum(
            1 for meta in swap_metadata.values() if meta.get("strict_answer_object_match")
        ),
        "num_valid_shuffles": sum(
            1 for meta in shuffle_metadata.values() if meta.get("shuffle_valid")
        ),
        "num_raw_context_fallbacks": sum(
            1 for evidence in evidence_by_id.values()
            if evidence.context_source == "raw_context_fallback"
        ),
    }
    return records, diagnostics


# =============================================================================
# 9. Causal metric aggregation and result recording
# =============================================================================


def percentage(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def mean_or_none(values: Iterable[float]) -> Optional[float]:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else None


def aggregate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    valid_swaps = [record for record in records if record.get("intervention_valid")]
    valid_shuffles = [record for record in records if record.get("shuffle_valid")]

    def count_indicator(subset: Sequence[Dict[str, Any]], key: str) -> int:
        return sum(
            1
            for record in subset
            if record.get("indicators", {}).get(key) is True
        )

    metrics = {
        "original_accuracy": percentage(count_indicator(records, "original_correct"), total),
        "original_relaxed_accuracy": percentage(
            count_indicator(records, "original_relaxed_correct"), total
        ),
        "frozen_s_answer_consistency": percentage(
            count_indicator(records, "frozen_answer_consistent"), total
        ),
        "frozen_s_raw_consistency": percentage(
            count_indicator(records, "frozen_raw_consistent"), total
        ),
        "frozen_redecode_consistency": (
            percentage(count_indicator(records, "frozen_redecode_consistent"), total)
            if RUN_FROZEN_REDECODE_DIAGNOSTIC
            else None
        ),
        # Object‑swap primary metrics are computed only over samples where the explicit answer object was successfully located and replaced.
        "swap_accuracy": percentage(count_indicator(valid_swaps, "swap_correct"), len(valid_swaps)),
        "swap_relaxed_accuracy": percentage(
            count_indicator(valid_swaps, "swap_relaxed_correct"), len(valid_swaps)
        ),
        "swap_answer_flip_valid": percentage(
            count_indicator(valid_swaps, "swap_answer_flip"), len(valid_swaps)
        ),
        "swap_decoy_adoption_valid": percentage(
            count_indicator(valid_swaps, "swap_decoy_adopted"), len(valid_swaps)
        ),
        "shuffle_accuracy": percentage(
            count_indicator(valid_shuffles, "shuffle_correct"), len(valid_shuffles)
        ),
        "shuffle_relaxed_accuracy": percentage(
            count_indicator(valid_shuffles, "shuffle_relaxed_correct"), len(valid_shuffles)
        ),
        "shuffle_answer_flip_valid": percentage(
            count_indicator(valid_shuffles, "shuffle_answer_flip"), len(valid_shuffles)
        ),
        "direct_decoy_adoption_valid": percentage(
            count_indicator(valid_swaps, "direct_decoy_adopted"), len(valid_swaps)
        ),
        "direct_accuracy": percentage(count_indicator(valid_swaps, "direct_correct"), len(valid_swaps)),
        "mean_swap_abs_audit_delta": mean_or_none(
            float(record["audit"]["swap_abs_score_delta"]) for record in valid_swaps
        ),
        "mean_shuffle_abs_audit_delta": mean_or_none(
            float(record["audit"]["shuffle_abs_score_delta"]) for record in valid_shuffles
        ),
        "coverage_invariance_swap": percentage(
            sum(1 for record in valid_swaps if record["audit"]["coverage_invariant_swap"]),
            len(valid_swaps),
        ),
        "coverage_invariance_shuffle": percentage(
            sum(1 for record in valid_shuffles if record["audit"]["coverage_invariant_shuffle"]),
            len(valid_shuffles),
        ),
        "parse_success": {
            condition: percentage(
                count_indicator(records, f"{condition}_parse_success"), total
            )
            for condition in ("original", "frozen", "swap", "shuffle", "direct")
        },
    }

    return {
        "num_samples": total,
        "num_valid_object_swaps": len(valid_swaps),
        "num_strict_object_targets": sum(
            1 for record in records if record.get("strict_answer_object_match")
        ),
        "num_valid_shuffles": len(valid_shuffles),
        "metrics_percent": metrics,
    }


def format_metric(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def print_primary_table(summary: Dict[str, Any], title: str) -> None:
    metrics = summary["metrics_percent"]
    print("\n" + "=" * 109)
    print(title)
    print("=" * 109)
    print(
        f"{'Orig. Acc.':>11} | "
        f"{'Frozen Cons.':>12} | "
        f"{'Swap Acc.':>10} | "
        f"{'Flip':>8} | "
        f"{'Decoy':>8} | "
        f"{'Shuffle Acc.':>12} | "
        f"{'Direct Decoy':>12}"
    )
    print("-" * 109)
    print(
        f"{format_metric(metrics['original_accuracy']):>11} | "
        f"{format_metric(metrics['frozen_s_answer_consistency']):>12} | "
        f"{format_metric(metrics['swap_accuracy']):>10} | "
        f"{format_metric(metrics['swap_answer_flip_valid']):>8} | "
        f"{format_metric(metrics['swap_decoy_adoption_valid']):>8} | "
        f"{format_metric(metrics['shuffle_accuracy']):>12} | "
        f"{format_metric(metrics['direct_decoy_adoption_valid']):>12}"
    )
    print("=" * 109)
    print(
        f"Valid object swaps: {summary['num_valid_object_swaps']}/{summary['num_samples']}; "
        f"Strict target objects located: {summary['num_strict_object_targets']}/{summary['num_samples']}; "
        f"Valid view permutations: {summary['num_valid_shuffles']}/{summary['num_samples']}"
    )
    print(
        "QAEG interface diagnostics: "
        f"Frozen raw={format_metric(metrics['frozen_s_raw_consistency'])}, "
        f"remote re-decode={format_metric(metrics['frozen_redecode_consistency'])}, "
        f"relaxed original={format_metric(metrics['original_relaxed_accuracy'])}, "
        f"mean |Δs| swap={metrics['mean_swap_abs_audit_delta']}, "
        f"mean |Δs| shuffle={metrics['mean_shuffle_abs_audit_delta']}, "
        f"coverage invariant(swap/shuffle)="
        f"{format_metric(metrics['coverage_invariance_swap'])}/"
        f"{format_metric(metrics['coverage_invariance_shuffle'])}"
    )


def build_save_payload(
    total_original_samples: int,
    total_batches: int,
    completed_batches: Sequence[Dict[str, Any]],
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "experiment": "QAEG audit-interface causal intervention evaluation",
        "description": {
            "original": "Fixed query, context evidence, and original audit state, then context-bounded generation.",
            "frozen_s": "Identity intervention: q, T_c, and s remain unchanged, and original output is reused.",
            "object_swap": "Replace the object of the explicit answer parametric triple, recompute disagreement, but do not expose T_m to the final generator.",
            "tm_shuffle": "Permute the full parametric view, recompute disagreement, but do not expose T_m to the final generator.",
            "direct_exposure": "Positive control: temporarily make the intervened target parametric triple answer‑admissible.",
        },
        "config": {
            "data_file": DATA_FILE,
            "server_url": SERVER_URL,
            "batch_size": BATCH_SIZE,
            "start_batch_idx": START_BATCH_IDX,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "retrieval_sent_topk": RETRIEVAL_SENT_TOPK,
            "retrieval_chunk_topk": RETRIEVAL_CHUNK_TOPK,
            "retrieval_chunk_size": RETRIEVAL_CHUNK_SIZE,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "disagreement_report_penalty": DISAGREEMENT_REPORT_PENALTY,
            "empty_response_retries": SERVER_EMPTY_RETRIES,
            "run_frozen_redecode_diagnostic": RUN_FROZEN_REDECODE_DIAGNOSTIC,
            "run_direct_exposure_control": RUN_DIRECT_EXPOSURE_CONTROL,
            "frozen_primary_semantics": "cached identity output",
            "answer_metric": "repository normalized prediction-in-ground-truth containment",
        },
        "total_original_samples": total_original_samples,
        "total_batches": total_batches,
        "completed_batches": list(completed_batches),
        "aggregate": aggregate_records(records),
        "records": list(records),
    }


def save_checkpoint(payload: Dict[str, Any]) -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    temporary_path = SAVE_PATH + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(json_safe(payload), file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, SAVE_PATH)


# =============================================================================
# 10. QAEG causal interface evaluation main loop
# =============================================================================


async def run_causal_interface_stress_test(
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
) -> None:
    total_size = len(dataset)
    total_batches = (total_size + batch_size - 1) // batch_size
    all_records: List[Dict[str, Any]] = []
    completed_batches: List[Dict[str, Any]] = []

    global_answer_pool: List[str] = []
    for item in dataset:
        global_answer_pool.extend(as_answer_list(item.get("answer")))

    for batch_index in range(START_BATCH_IDX, total_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, total_size)
        batch_ds = dataset.select(range(start_index, end_index))

        print(
            f"\n--- QAEG audit interface causal intervention evaluation: batch {batch_index + 1}/{total_batches} "
            f"(samples {start_index}-{end_index - 1}) ---"
        )

        try:
            batch_records, diagnostics = await process_causal_batch_with_retry(
                rag,
                batch_ds,
                global_answer_pool,
                start_index,
            )
            all_records.extend(batch_records)
            batch_summary = aggregate_records(batch_records)
            completed_batches.append(
                {
                    "batch": batch_index + 1,
                    "index_range": [start_index, end_index],
                    "status": "completed",
                    "diagnostics": diagnostics,
                    "summary": batch_summary,
                }
            )

            print_primary_table(
                batch_summary,
                title=f"Batch {batch_index + 1} QAEG causal intervention results",
            )
            print_primary_table(
                aggregate_records(all_records),
                title="Cumulative QAEG causal intervention results up to now",
            )

            if SAVE_AFTER_EACH_BATCH:
                payload = build_save_payload(
                    total_size,
                    total_batches,
                    completed_batches,
                    all_records,
                )
                save_checkpoint(payload)
                print(f"    💾 Saved QAEG causal evaluation checkpoint to: {SAVE_PATH}")

        except Exception as error:
            print(
                f"❌ Batch {batch_index + 1} QAEG causal intervention failed after retries exhausted, skipping: "
                f"{type(error).__name__}: {error}"
            )
            traceback.print_exc()
            completed_batches.append(
                {
                    "batch": batch_index + 1,
                    "index_range": [start_index, end_index],
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

        if batch_index < total_batches - 1:
            await asyncio.sleep(BATCH_SLEEP_SECONDS)

    final_payload = build_save_payload(
        total_size,
        total_batches,
        completed_batches,
        all_records,
    )
    save_checkpoint(final_payload)

    print_primary_table(
        final_payload["aggregate"],
        title="QAEG audit interface causal intervention final results",
    )
    print(f"\n✅ QAEG causal interface evaluation results fully saved to: {SAVE_PATH}")


if __name__ == "__main__":
    asyncio.run(run_causal_interface_stress_test(ds, batch_size=BATCH_SIZE))