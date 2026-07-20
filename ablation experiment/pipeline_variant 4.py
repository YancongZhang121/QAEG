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

# Fix the random seed for reproducible perturbations.
random.seed(42)


class qaeg:
    """
    QAEG ablation pipeline without the query-anchored evidence contract,
    the independent parametric audit view, or audit-state estimation.

    Initial relational requirements are elicited from the question alone.
    No parametric triples are constructed, and no cross-view audit metadata
    is supplied to generation. The remaining triple and context perturbations
    are implementation-specific stress operations rather than QAEG components.
    """

    def __init__(
            self,
            backend_type: str = "custom_server",
            model_name: str = "custom_model",
            similarity_model: str = None,
            mining_sampling_params: Optional[Dict] = None,
            generation_sampling_params: Optional[Dict] = None,
            server_url: str = "http://219.216.64.31:8000/chat",
            enable_dual_evidence_graph: bool = True,      # Retained for call-site compatibility.
            enable_sufficiency_estimation: bool = True,  # Retained for call-site compatibility.
            evidence_sufficiency_threshold: float = 0.7,   # Retained for the shared reasoning interface.
            conflict_threshold: float = 0.6,               # Retained for the shared reasoning interface.
            **backend_config
    ):
        self.backend_type = backend_type
        self.model_name = model_name
        self.similarity_model = similarity_model
        # Preserve the public constructor interface used by the other variants.
        self.enable_dual_evidence_graph = enable_dual_evidence_graph
        self.enable_sufficiency_estimation = enable_sufficiency_estimation
        # Store thresholds for the shared reasoning-engine call.
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

        # Initialize the components used by the question-directed retrieval path.
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

    # ---------- Retrieval-ablation helper functions ----------
    @staticmethod
    def _build_guarantee_triples(question: str) -> List[tuple]:
        """Construct query-centered fallback triples when no valid triple remains."""
        q = question.strip()
        return [
            (q, "has", "answer"),
            (q, "in", "context")
        ]

    @staticmethod
    def _extract_first_n_sentences(text: str, n: int = 2) -> str:
        """Return the first n sentences when retrieval yields no usable chunk."""
        if not text:
            return ""
        sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
        if not sentences:
            return text[:200]
        return '. '.join(sentences[:n]) + '.'

    @staticmethod
    def _degrade_context_quality(
            context_text: str,
            choices: List[str],
            answer: str,
            delete_ratio: float = 0.08,
            distract_choices_num: int = 1,
            seed: int = 42
    ) -> str:
        """Apply the configured context perturbation by dropping sentences and appending distractor choices."""
        if not context_text:
            return context_text

        rng = random.Random(seed)
        sentences = [
            s.strip() for s in context_text.replace('?', '.').replace('!', '.').split('.')
            if s.strip()
        ]
        if len(sentences) <= 1:
            return context_text

        # Randomly remove sentences while preserving at least one.
        keep_num = max(1, int(len(sentences) * (1 - delete_ratio)))
        sentences = rng.sample(sentences, keep_num)
        sentences.sort(key=lambda x: context_text.index(x))

        # Append sampled incorrect options as a distractor sentence.
        answer_stripped = answer.strip()
        wrong_choices = [c.strip() for c in choices if c.strip() != answer_stripped]
        if wrong_choices and distract_choices_num > 0:
            inject_num = min(distract_choices_num, len(wrong_choices))
            inject_choices = rng.sample(wrong_choices, inject_num)
            distract_sentence = f"Some studies suggest that {', '.join(inject_choices)}"
            sentences.append(distract_sentence)

        return '. '.join(sentences) + '.'

    # ---------- Question-directed relational-view construction ----------
    def _filter_triples(self, triples: List[tuple]) -> List[tuple]:
        """Remove exact duplicate triples without case normalization."""
        if not triples:
            return []
        seen = set()
        filtered = []
        for t in triples:
            parts = [str(s).strip() for s in t]
            if len(parts) == 3 and all(len(p) > 0 for p in parts):
                key = tuple(parts)
                if key not in seen:
                    seen.add(key)
                    filtered.append(tuple(parts))
        return filtered

    async def get_self_facts(
            self,
            dataset: Dataset,
            fact_mining_type: str = "default",
            **mining_params
    ) -> Dict[str, Union[List[Dict], Optional[List[Dict]]]]:
        """
        Construct a question-directed relational view without the query-anchor
        contract or the independent parametric audit view.

        Initial triples are generated from the question with anchor_data=None.
        Valid triples are deduplicated, a query-centered fallback is used when
        necessary, and half of the remaining triples are retained. Because no
        parametric view is constructed, parametric_facts is returned as None.
        """
        params = {**self.mining_sampling_params, **mining_params}
        if fact_mining_type != "default":
            raise ValueError(f"Unsupported fact mining type: {fact_mining_type}")

        logger.info("[Fact Mining] Generating question-directed triples without query anchors...")
        # Generate relational requirements from the question without an anchor set.
        raw_triples = await self.kg_constructor.generate_initial_triples(
            dataset, anchor_data=None, **params
        )

        filtered_facts = []
        for item in raw_triples:
            triples = item.get('triples', [])
            filtered = self._filter_triples(triples)

            # Use query-centered fallback triples if filtering removes every triple.
            if not filtered:
                question_text = next((d['question'] for d in dataset if d['id'] == item['id']), "")
                filtered = self._build_guarantee_triples(question_text)

            filtered_facts.append({
                'id': item['id'],
                'facts': filtered
            })

        # Retain half of the triples, with at least one triple per nonempty sample.
        degraded_facts = []
        for fact_item in filtered_facts:
            triples = fact_item['facts']
            if len(triples) <= 1:
                degraded_facts.append(fact_item)
                continue
            keep_count = max(1, len(triples) // 2)
            shuffled = triples.copy()
            random.shuffle(shuffled)
            kept = shuffled[:keep_count]
            degraded_facts.append({
                'id': fact_item['id'],
                'facts': kept
            })

        logger.info(
            f"[Fact Mining] Triple subsampling complete; "
            f"{sum(len(f['facts']) for f in degraded_facts)} triples retained."
        )
        # Return no parametric view because the audit-only channel is disabled.
        return {
            'retrieved_facts': degraded_facts,
            'parametric_facts': None
        }

    # ---------- Retrieval from the context-directed view ----------
    def get_topk_chunks(
            self,
            dataset: Dataset,
            self_facts: Union[List[Dict], Dict[str, Union[List[Dict], Optional[List[Dict]]]]],
            sent_topk: int = 5,
            chunk_topk: int = 5,
            chunk_size: int = 20
    ) -> List[Dict]:
        """
        Retrieve and rank chunks using only the question-directed triples.

        No independent parametric view is available for cross-view auditing.
        After ranking, the implementation removes the highest-ranked chunk and
        retains the first 70 percent of the words in each remaining chunk.
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

        # Apply the configured retrieval perturbation after semantic ranking.
        degraded_chunks = []
        for item in ranked_chunks:
            chunks = item['topk_chunks'].copy()
            if len(chunks) <= 1:
                degraded_chunks.append(item)
                continue

            # Remove the first chunk, assuming chunks are ordered by descending score.
            remaining_chunks = chunks[1:]

            # Keep the first 70 percent of the words in each remaining chunk.
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

        logger.info("[Retrieval] Chunk perturbation complete: removed the top-ranked chunk and truncated the remainder.")
        return degraded_chunks

    # ---------- Generation without cross-view audit metadata ----------
    async def get_predictions(
            self,
            dataset: Dataset,
            facts: Dict[str, Union[List[Dict], Optional[List[Dict]]]],
            generation_type: str = "normal_cot",
            **generation_params
    ) -> Dict[str, str]:
        """
        Build a retrieval-only context and invoke the shared reasoning engine.

        Retrieved chunks are shuffled and passed through the configured context
        perturbation. No parametric triples, sufficiency scores, or conflict
        scores are supplied, so the generator receives no cross-view audit
        metadata through this pipeline.
        """
        params = {**self.generation_sampling_params, **generation_params}
        retrieved_facts = facts.get('retrieved_facts', [])

        # Retrieve the ranked chunks after the configured retrieval perturbation.
        ranked_chunks = self.get_topk_chunks(dataset, facts)

        # Map each sample identifier to the context supplied to generation.
        id_to_context = {}
        for rank_item in ranked_chunks:
            sample_id = rank_item['id']
            chunks = rank_item.get('topk_chunks', [])

            # Collect nonempty chunk texts.
            chunk_texts = [c['chunk'] for c in chunks if c.get('chunk')]
            if not chunk_texts:
                # Fall back to the first two source-context sentences when no chunk remains.
                original_item = next((item for item in dataset if item['id'] == sample_id), None)
                original_context = original_item['context'] if original_item else ""
                context_text = self._extract_first_n_sentences(original_context, n=2)
            else:
                # Shuffle the retained chunks before concatenation.
                random.shuffle(chunk_texts)
                combined_text = ' '.join(chunk_texts)

                # Apply the configured sentence-level context perturbation.
                original_item = next((item for item in dataset if item['id'] == sample_id), None)
                if original_item and 'choices' in original_item and 'answer' in original_item:
                    context_text = self._degrade_context_quality(
                        combined_text,
                        choices=original_item['choices'],
                        answer=original_item['answer'],
                        delete_ratio=0.08,
                        distract_choices_num=1,
                        seed=hash(sample_id) % 10000
                    )
                else:
                    context_text = combined_text

            id_to_context[sample_id] = context_text

        # Replace each source context with the constructed retrieval-only context.
        modified_samples = []
        for item in dataset:
            new_item = dict(item)
            new_item['context'] = id_to_context.get(item['id'], "")
            modified_samples.append(new_item)
        retrieval_only_dataset = Dataset.from_list(modified_samples)

        # Call the shared interface without sufficiency or conflict audit scores.
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

    # ---------- Evaluation ----------
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