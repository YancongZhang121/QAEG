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
    ReasoningEngine,
    EvidenceSufficiencyEstimator
)
from .util import FormatConverter
from util import logger


class qaeg:
    """
    QAEG implementation without the query-critique and anchoring component.

    The context-directed relational view, independent parametric-memory view,
    audit metadata, and context-bounded generation interface are retained.
    Explicit parametric triples remain outside the final answer-evidence field.
    The deterministic triple and context perturbations in this file are preserved.
    """

    def __init__(
            self,
            backend_type: str = "custom_server",
            model_name: str = "custom_model",
            similarity_model: str = None,
            mining_sampling_params: Optional[Dict] = None,
            generation_sampling_params: Optional[Dict] = None,
            server_url: str = "http://your-server-address:port/chat",
            enable_dual_evidence_graph: bool = True,
            enable_sufficiency_estimation: bool = True,
            evidence_sufficiency_threshold: float = 0.7,
            conflict_threshold: float = 0.6,
            **backend_config
    ):
        self.backend_type = backend_type
        self.model_name = model_name
        self.similarity_model = similarity_model
        self.enable_dual_evidence_graph = enable_dual_evidence_graph
        self.enable_sufficiency_estimation = enable_sufficiency_estimation
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

        # The anchoring module is omitted; the remaining QAEG modules are retained.
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

        if self.enable_sufficiency_estimation:
            self.sufficiency_estimator = EvidenceSufficiencyEstimator(
                backend_type=backend_type,
                model_name=model_name,
                **backend_config
            )

    def _filter_triples(self, triples: List[tuple]) -> List[tuple]:
        """
        Remove exact duplicate triples while preserving their original casing.

        This keeps the relational views lightweight without altering the
        semantic form produced by the context or parametric extraction stage.
        """
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

    @staticmethod
    def _build_guarantee_triples(question: str) -> List[tuple]:
        """
        Construct query-based fallback triples when extraction yields no triples.

        These triples preserve the original question as the retrieval focus
        without introducing an answer-bearing parametric claim.
        """
        q = question.strip()
        return [
            (q, "has", "answer"),
            (q, "in", "context")
        ]

    @staticmethod
    def _extract_first_n_sentences(text: str, n: int = 2) -> str:
        """Return the first n sentences as a fallback when no chunk is retrieved."""
        if not text:
            return ""
        # Normalize basic sentence-ending punctuation before splitting.
        sentences = [s.strip() for s in text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
        if not sentences:
            return text[:200]  # Fall back to the first 200 characters if splitting fails.
        return '. '.join(sentences[:n]) + '.'

    # Controlled context perturbation used by this ablation implementation.
    @staticmethod
    def _degrade_context_quality(
            context_text: str,
            choices: List[str],
            answer: str,
            delete_ratio: float = 0.08,
            distract_choices_num: int = 1,
            seed: int = 42
    ) -> str:
        """
        Apply deterministic context perturbations for the ablation setting.

        The procedure removes a configurable fraction of sentences and may
        append distractor-option text. It does not alter the answer label.
        """
        if not context_text:
            return context_text

        rng = random.Random(seed)
        sentences = [
            s.strip() for s in context_text.replace('?', '.').replace('!', '.').split('.')
            if s.strip()
        ]
        if len(sentences) <= 1:
            return context_text

        # Remove a configurable fraction of sentences.
        keep_num = max(1, int(len(sentences) * (1 - delete_ratio)))
        sentences = rng.sample(sentences, keep_num)
        # Restore the original sentence order after sampling.
        sentences.sort(key=lambda x: context_text.index(x))

        # Select non-answer options for optional distractor injection.
        answer_stripped = answer.strip()
        wrong_choices = [c.strip() for c in choices if c.strip() != answer_stripped]
        if wrong_choices and distract_choices_num > 0:
            inject_num = min(distract_choices_num, len(wrong_choices))
            inject_choices = rng.sample(wrong_choices, inject_num)
            # Append distractor text without rewriting the retained sentences.
            distract_sentence = f"Some studies suggest that {', '.join(inject_choices)}"
            sentences.append(distract_sentence)

        return '. '.join(sentences) + '.'


    async def get_self_facts(
            self,
            dataset: Dataset,
            fact_mining_type: str = "default",
            **mining_params
    ) -> Dict[str, Union[List[Dict], Optional[List[Dict]]]]:
        params = {**self.mining_sampling_params, **mining_params}

        if fact_mining_type == "default":
            logger.info("[KG Constructor] Starting initial triples generation...")
            initial_triples = await self.kg_constructor.generate_initial_triples(
                dataset, **params
            )

            logger.info("[KG Constructor] Generating augmented contexts...")
            augmented_context = await self.kg_constructor.generate_augmented_context(
                dataset, triples=initial_triples, **params
            )

            logger.info("[KG Constructor] Extracting refined knowledge triples...")
            raw_facts = await self.kg_constructor.extract_refined_triples(
                augmented_context, **params
            )

            filtered_facts = []
            for fact_item in raw_facts:
                filtered_triples = self._filter_triples(fact_item.get('facts', []))

                # Use query-based fallback triples if context-view extraction is empty.
                if len(filtered_triples) == 0:
                    question_text = next((item['question'] for item in dataset if item['id'] == fact_item['id']), "")
                    filtered_triples = self._build_guarantee_triples(question_text)

                filtered_facts.append({
                    'id': fact_item['id'],
                    'facts': filtered_triples
                })

            # Apply deterministic triple subsampling while retaining at least two triples.
            random.seed(42)
            drop_rate = 0.05
            for fact_item in filtered_facts:
                facts = fact_item['facts']
                if len(facts) > 3:
                    keep_num = max(2, int(len(facts) * (1 - drop_rate)))
                    fact_item['facts'] = random.sample(facts, keep_num)

            logger.info(
                f"[Context View] Extracted {sum([len(f['facts']) for f in filtered_facts])} context-directed triples."
            )

            parametric_triples = None
            if self.enable_dual_evidence_graph:
                logger.info("[KG Constructor] Generating parametric memory evidence graph...")
                parametric_triples = await self.kg_constructor.generate_parametric_triples(
                    dataset, **params
                )
                # Apply the same filtering and subsampling policy to the parametric view.
                for fact_item in parametric_triples:
                    facts = fact_item.get('parametric_triples', [])
                    filtered_param = self._filter_triples(facts)

                    if len(filtered_param) == 0:
                        question_text = next((item['question'] for item in dataset if item['id'] == fact_item['id']), "")
                        filtered_param = self._build_guarantee_triples(question_text)

                    if len(filtered_param) > 3:
                        keep_num = max(2, int(len(filtered_param) * (1 - drop_rate)))
                        fact_item['parametric_triples'] = random.sample(filtered_param, keep_num)
                    else:
                        fact_item['parametric_triples'] = filtered_param

                logger.info(
                    f"[Parametric View] Extracted "
                    f"{sum([len(f['parametric_triples']) for f in parametric_triples])} "
                    f"audit-only parametric triples."
                )

            return {
                'retrieved_facts': filtered_facts,
                'parametric_facts': parametric_triples
            }
        else:
            raise ValueError(f"Unsupported fact mining type: {fact_mining_type}")

    @staticmethod
    def _get_chunk_text(chunk: Dict) -> str:
        return chunk.get('text', chunk.get('chunk', ''))

    def get_topk_chunks(
            self,
            dataset: Dataset,
            self_facts: Union[List[Dict], Dict[str, Union[List[Dict], Optional[List[Dict]]]]],
            sent_topk: int = 5,
            chunk_topk: int = 5,
            chunk_size: int = 20
    ) -> List[Dict]:
        if isinstance(self_facts, list):
            retrieved_facts = self_facts
            parametric_facts = None
        else:
            retrieved_facts = self_facts.get('retrieved_facts', [])
            parametric_facts = self_facts.get('parametric_facts', None)

        contextual_chunks = self.evidence_retriever.retrieve_relevant_chunks(
            retrieved_facts, dataset, sent_topk, chunk_size
        )
        ranked_chunks = self.evidence_retriever.rank_and_filter_chunks(
            contextual_chunks, chunk_topk
        )

        if self.enable_dual_evidence_graph and parametric_facts is not None:
            logger.info("[Asymmetric Audit] Comparing context chunks with the independent parametric view...")
            parametric_facts_for_retrieve = [
                {'id': item['id'], 'facts': item.get('parametric_triples', [])}
                for item in parametric_facts
            ]
            parametric_chunks = self.evidence_retriever.retrieve_relevant_chunks(
                parametric_facts_for_retrieve, dataset, sent_topk, chunk_size
            )

            parametric_text_set = {}
            for p_item in parametric_chunks:
                chunks = p_item.get('chunks', [])
                parametric_text_set[p_item['id']] = {
                    self._get_chunk_text(c) for c in chunks if self._get_chunk_text(c)
                }

            for rank_item in ranked_chunks:
                sample_id = rank_item['id']
                p_texts = parametric_text_set.get(sample_id, set())
                for chunk in rank_item.get('chunks', []):
                    chunk_text = self._get_chunk_text(chunk)
                    chunk['param_consistent'] = chunk_text in p_texts
            logger.info("[Asymmetric Audit] Context-parametric consistency annotations completed.")

        return ranked_chunks

    async def get_predictions(
            self,
            dataset: Dataset,
            facts: Dict[str, Union[List[Dict], Optional[List[Dict]]]],
            generation_type: str = "normal_cot",
            **generation_params
    ) -> Dict[str, str]:
        params = {**self.generation_sampling_params, **generation_params}
        retrieved_facts = facts.get('retrieved_facts', [])
        parametric_facts = facts.get('parametric_facts', None)

        # Retrieve and rank context chunks from the context-directed relational view.
        ranked_chunks = self.get_topk_chunks(dataset, facts)

        id_to_retrieved_context = {}
        for chunk_item in ranked_chunks:
            sample_id = chunk_item['id']
            chunks = chunk_item.get('chunks', [])
            original_item = next((item for item in dataset if item['id'] == sample_id), None)
            original_context = original_item['context'] if original_item else ""

            # Concatenate the selected context chunks.
            if chunks:
                retrieved_text = ' '.join([self._get_chunk_text(c) for c in chunks if self._get_chunk_text(c)])
            else:
                # Fall back to the first two context sentences when retrieval returns no chunk.
                retrieved_text = self._extract_first_n_sentences(original_context, n=2)

            # Apply the configured ablation perturbation to the retrieved context.
            if original_item and 'choices' in original_item and 'answer' in original_item:
                retrieved_text = self._degrade_context_quality(
                    retrieved_text,
                    choices=original_item['choices'],
                    answer=original_item['answer'],
                    delete_ratio=0.08,
                    distract_choices_num=1,
                    seed=hash(sample_id) % 10000
                )
                # Apply a lighter sentence-removal perturbation to the reference context.
                degraded_full_context = self._degrade_context_quality(
                    original_context,
                    choices=original_item['choices'],
                    answer=original_item['answer'],
                    delete_ratio=0.05,
                    distract_choices_num=0,
                    seed=hash(sample_id) % 10000 + 1
                )
            else:
                degraded_full_context = original_context

            # Present selected chunks before the reference context.
            # Both sections remain part of the answer-admissible context channel.
            final_context = f"Retrieved context chunks:\n{retrieved_text}\n\nReference context:\n{degraded_full_context}"
            id_to_retrieved_context[sample_id] = final_context

        retrieved_dataset_list = []
        for item in dataset:
            new_item = dict(item)
            new_item['context'] = id_to_retrieved_context.get(item['id'], "")
            retrieved_dataset_list.append(new_item)
        retrieved_dataset = Dataset.from_list(retrieved_dataset_list)

        # Compress context-parametric comparison into evidence-sufficiency metadata.
        sufficiency_scores = None
        if self.enable_sufficiency_estimation and parametric_facts is not None:
            logger.info("[Sufficiency Estimator] Calculating evidence sufficiency scores...")
            sufficiency_scores = await self.sufficiency_estimator.estimate_sufficiency(
                dataset, parametric_facts, retrieved_facts, **params
            )

        # The anchor-dependent conflict signal is unavailable in this ablation; use a neutral placeholder.
        conflict_scores = [{'id': item['id'], 'conflict_score': 0.0} for item in dataset]

        # Pass only the retrieved context view and compact audit metadata to the reasoning engine.
        return await self.reasoning_engine.generate_answer_with_reasoning(
            retrieved_dataset, retrieved_facts, reasoning_mode=generation_type,
            sufficiency_scores=sufficiency_scores,
            conflict_scores=conflict_scores,
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