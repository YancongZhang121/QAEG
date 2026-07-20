# Run it by placing it in pipeline.py

import asyncio
import random
from typing import Dict, List, Optional, Union
from datasets import Dataset
from tqdm import tqdm
from .evaluate import (
    exact_match_score,
    acc_score,
    f1_score,
    metric_max_over_ground_truths
)
from .modules import (
    KnowledgeGraphConstructor,
    EvidenceRetriever,
    ReasoningEngine
)
from .util import FormatConverter
from util import logger

# Fix the random seed for reproducible ablation runs.
random.seed(42)

class qaeg:
    """
    QAEG ablation pipeline without the query-anchored evidence contract
    or the asymmetric dual-view audit.

    This variant generates question-directed retrieval triples without
    query anchors, omits the independently elicited parametric view and
    compact audit state, and constructs the generator context only from
    retrieved Top-K chunks.
    """
    def __init__(
            self,
            backend_type: str = "custom_server",
            model_name: str = "custom_model",
            similarity_model: str = None,
            mining_sampling_params: Optional[Dict] = None,
            generation_sampling_params: Optional[Dict] = None,
            server_url: str = "http://219.216.64.31:8000/chat",
            enable_dual_evidence_graph: bool = True,
            enable_sufficiency_estimation: bool = True,
            evidence_sufficiency_threshold: float = 0.7,   # Retained for call-interface compatibility.
            conflict_threshold: float = 0.6,               # Retained for call-interface compatibility.
            **backend_config
    ):
        self.backend_type = backend_type
        self.model_name = model_name
        self.similarity_model = similarity_model
        # Retain the original configuration fields for caller compatibility.
        self.enable_dual_evidence_graph = enable_dual_evidence_graph
        self.enable_sufficiency_estimation = enable_sufficiency_estimation
        # Retain the thresholds required by the reasoning-engine interface.
        self.evidence_sufficiency_threshold = evidence_sufficiency_threshold
        self.conflict_threshold = conflict_threshold

        backend_config['server_url'] = server_url
        self.mining_sampling_params = mining_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.1
        }
        self.generation_sampling_params = generation_sampling_params or {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.1
        }
        # Keep only the modules used for context-view construction, retrieval, and generation.
        self.kg_constructor = KnowledgeGraphConstructor(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        self.evidence_retriever = EvidenceRetriever(
            similarity_model=similarity_model
        )
        self.reasoning_engine = ReasoningEngine(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )

    def _filter_triples(self, triples: List[tuple]) -> List[tuple]:
        if not triples:
            return []
        seen = set()
        unique_triples = []
        for t in triples:
            key = tuple(str(s).lower().strip() for s in t)
            if key not in seen and len(key) == 3:
                seen.add(key)
                unique_triples.append(t)
        filtered = []
        for s, p, o in unique_triples:
            s_str, p_str, o_str = str(s).strip(), str(p).strip(), str(o).strip()
            if len(s_str) > 0 and len(p_str) > 0 and len(o_str) > 0:
                filtered.append((s_str, p_str, o_str))
        return filtered

    async def get_self_facts(
            self,
            dataset: Dataset,
            fact_mining_type: str = "default",
            **mining_params
    ) -> Dict[str, Union[List[Dict], Optional[List[Dict]]]]:
        """
        Construct question-directed retrieval triples without query anchors.

        This ablation bypasses anchor-conditioned planning, augmented-context
        construction, refined triple extraction, and independent parametric-view
        construction. The supplied context is not used during triple generation.
        """
        params = {**self.mining_sampling_params, **mining_params}
        if fact_mining_type == "default":
            logger.info("[Context View] Generating question-directed retrieval triples without query anchors...")
            # Generate retrieval triples from the question without anchor data or supplied context.
            raw_triples = await self.kg_constructor.generate_initial_triples(
                dataset, anchor_data=None, **params
            )
            filtered_facts = []
            for item in raw_triples:
                filtered_triples = self._filter_triples(item.get('triples', []))
                filtered_facts.append({
                    'id': item['id'],
                    'facts': filtered_triples
                })

            # Retain a deterministic random subset of the generated retrieval triples.
            degraded_facts = []
            for fact_item in filtered_facts:
                triples = fact_item['facts'].copy()
                if len(triples) <= 1:
                    degraded_facts.append(fact_item)
                    continue
                # Keep approximately half of the triples while preserving at least one.
                keep_count = max(1, len(triples) // 2)
                random.shuffle(triples)
                kept_triples = triples[:keep_count]
                degraded_facts.append({
                    'id': fact_item['id'],
                    'facts': kept_triples
                })
            filtered_facts = degraded_facts
            logger.info("[Context View] Retained approximately 50% of the generated retrieval triples.")

            logger.info(
                f"[Context View] Constructed {sum([len(f['facts']) for f in filtered_facts])} retrieval triples."
            )
            return {
                'retrieved_facts': filtered_facts,
                'parametric_facts': None
            }
        else:
            raise ValueError(f"Unsupported fact mining type: {fact_mining_type}")

    def get_topk_chunks(
            self,
            dataset: Dataset,
            self_facts: Union[List[Dict], Dict[str, Union[List[Dict], Optional[List[Dict]]]]],
            sent_topk: int = 5,
            chunk_topk: int = 5,
            chunk_size: int = 20
    ) -> List[Dict]:
        """
        Rank context chunks using only the context-directed retrieval triples.

        No independently elicited parametric view or cross-view audit is
        constructed in this ablation.
        """
        if isinstance(self_facts, list):
            retrieved_facts = self_facts
        else:
            retrieved_facts = self_facts.get('retrieved_facts', [])

        contextual_chunks = self.evidence_retriever.retrieve_relevant_chunks(
            retrieved_facts, dataset, sent_topk, chunk_size
        )
        ranked_chunks = self.evidence_retriever.rank_and_filter_chunks(
            contextual_chunks, chunk_topk
        )

        # Apply the configured chunk-removal and truncation operation.
        degraded_chunks = []
        for item in ranked_chunks:
            chunks = item['topk_chunks'].copy()
            if len(chunks) <= 1:
                degraded_chunks.append(item)
                continue
            # Remove the highest-ranked chunk.
            remaining_chunks = chunks[1:]
            # Retain the first 70% of the word sequence in each remaining chunk.
            truncated_chunks = []
            for chunk_dict in remaining_chunks:
                chunk_text = chunk_dict['chunk']
                words = chunk_text.split()
                truncate_len = max(1, int(len(words) * 0.7))
                truncated_text = ' '.join(words[:truncate_len])
                truncated_chunks.append({
                    'chunk': truncated_text,
                    'score': chunk_dict['score']
                })
            degraded_chunks.append({
                'id': item['id'],
                'topk_chunks': truncated_chunks
            })
        logger.info("[Retrieval] Removed the highest-ranked chunk and truncated the remaining chunks.")

        return degraded_chunks

    async def get_predictions(
            self,
            dataset: Dataset,
            facts: Dict[str, Union[List[Dict], Optional[List[Dict]]]],
            generation_type: str = "normal_cot",
            **generation_params
    ) -> Dict[str, str]:
        """
        Construct the answer-evidence field from retrieved chunks only.

        The final decoder receives neither explicit parametric triples nor
        compact audit metadata in this ablation.
        """
        params = {**self.generation_sampling_params, **generation_params}
        retrieved_facts = facts.get('retrieved_facts', [])
        # Map each sample identifier to its serialized retrieved context.
        chunk_text_map = {}
        for fact_item in retrieved_facts:
            sample_id = fact_item['id']
            topk_chunks = fact_item.get('topk_chunks', [])
            if topk_chunks:
                # Shuffle retrieved chunks before serializing them as context.
                shuffled_chunks = topk_chunks.copy()
                random.shuffle(shuffled_chunks)
                chunk_text = ' '.join([chunk_dict['chunk'] for chunk_dict in shuffled_chunks])
            else:
                chunk_text = ""
            chunk_text_map[sample_id] = chunk_text
        # Replace the supplied context with the serialized retrieval-only context.
        modified_samples = []
        for item in dataset:
            sample_id = item['id']
            new_item = dict(item)
            # Use retrieved chunks as the only explicit answer-evidence field.
            new_item['context'] = chunk_text_map.get(sample_id, "")
            modified_samples.append(new_item)
        retrieval_only_dataset = Dataset.from_list(modified_samples)

        # Generate without sufficiency or conflict audit metadata.
        # Threshold arguments are retained only for reasoning-engine interface compatibility.
        return await self.reasoning_engine.generate_answer_with_reasoning(
            retrieval_only_dataset,
            retrieved_facts,
            reasoning_mode=generation_type,
            sufficiency_scores=None,
            conflict_scores=None,
            evidence_sufficiency_threshold=self.evidence_sufficiency_threshold,
            conflict_threshold=self.conflict_threshold,
            **params
        )

    def evaluate(
            self,
            dataset: Dataset,
            predictions: Dict[str, str],
            cot_format: bool = False,
            detailed_output: bool = False
    ) -> Dict:
        prediction_details = []
        total_em = total_acc = total_f1 = 0
        num_items = 0
        for item in tqdm(dataset, desc="Evaluating"):
            prediction = predictions.get(item['id'], "")
            if cot_format:
                prediction = FormatConverter.extract_answer(prediction)
            ground_truth = item['answer']
            em_score = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truth)
            acc_score_val = metric_max_over_ground_truths(
                acc_score, prediction, ground_truth)
            f1_score_val = metric_max_over_ground_truths(
                f1_score, prediction, ground_truth)
            total_em += em_score
            total_acc += acc_score_val
            total_f1 += f1_score_val
            num_items += 1
            if detailed_output:
                prediction_details.append({
                    "id": item['id'],
                    "question": item['question'],
                    "answer": ground_truth,
                    "prediction": prediction,
                    "exact_match": em_score,
                    "acc": acc_score_val,
                    "f1": f1_score_val
                })
        avg_em = 100.0 * total_em / num_items if num_items > 0 else 0
        avg_acc = 100.0 * total_acc / num_items if num_items > 0 else 0
        avg_f1 = 100.0 * total_f1 / num_items if num_items > 0 else 0
        result = {
            "num_items": num_items,
            "exact_match": avg_em,
            "acc": avg_acc,
            "f1": avg_f1
        }
        if detailed_output:
            result["details"] = prediction_details
        return result