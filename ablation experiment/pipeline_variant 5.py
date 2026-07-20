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
    # QuerySelfCriticAndAnchorGenerator and EvidenceSufficiencyEstimator are intentionally omitted.
)
from .util import FormatConverter
from util import logger


class qaeg:
    """
    QAEG ablation pipeline without query anchoring or cross-view audit signals.

    The query self-critic and evidence-sufficiency estimator are not instantiated.
    Context-directed triples are generated without an anchor contract. An
    independently elicited parametric view may still be returned for interface
    compatibility, but it is not used for consistency annotation, audit-state
    construction, or answer generation. The shared reasoning interface receives
    no sufficiency or conflict metadata.
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

        # Initialize only the modules retained by this ablation.
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
        """
        Remove exact duplicate triples after trimming their fields.

        Letter case is preserved so that only identical normalized field strings
        are treated as duplicates.
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
        Construct query-derived fallback triples when extraction returns no
        usable relations.
        """
        q = question.strip()
        return [
            (q, "has", "answer"),
            (q, "in", "context")
        ]

    @staticmethod
    def _extract_first_n_sentences(text: str, n: int = 2) -> str:
        """Return the first n sentence-like segments as a retrieval fallback."""
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
        """
        Apply controlled context perturbations by removing a subset of sentences
        and appending text derived from distractor answer choices.
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

        # Randomly retain a subset of sentences.
        keep_num = max(1, int(len(sentences) * (1 - delete_ratio)))
        sentences = rng.sample(sentences, keep_num)
        sentences.sort(key=lambda x: context_text.index(x))

        # Append text derived from sampled distractor choices.
        answer_stripped = answer.strip()
        wrong_choices = [c.strip() for c in choices if c.strip() != answer_stripped]
        if wrong_choices and distract_choices_num > 0:
            inject_num = min(distract_choices_num, len(wrong_choices))
            inject_choices = rng.sample(wrong_choices, inject_num)
            distract_sentence = f"Some studies suggest that {', '.join(inject_choices)}"
            sentences.append(distract_sentence)

        return '. '.join(sentences) + '.'

    async def get_self_facts(
            self,
            dataset: Dataset,
            fact_mining_type: str = "default",
            **mining_params
    ) -> Dict[str, Union[List[Dict], Optional[List[Dict]]]]:
        """
        Construct the context-directed relational view without a query-anchor
        contract. A parametric-memory view may also be elicited, but it is not
        consumed by later audit or generation stages in this ablation.
        """
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

                # Use query-derived fallback triples when no valid relation remains.
                if len(filtered_triples) == 0:
                    question_text = next((item['question'] for item in dataset if item['id'] == fact_item['id']), "")
                    filtered_triples = self._build_guarantee_triples(question_text)

                filtered_facts.append({
                    'id': fact_item['id'],
                    'facts': filtered_triples
                })

            # Apply deterministic subsampling with a 5% target drop rate and retain at least two triples.
            random.seed(42)
            drop_rate = 0.05
            for fact_item in filtered_facts:
                facts = fact_item['facts']
                if len(facts) > 3:
                    keep_num = max(2, int(len(facts) * (1 - drop_rate)))
                    fact_item['facts'] = random.sample(facts, keep_num)

            logger.info(f"[Fact Mining] Completed with {sum([len(f['facts']) for f in filtered_facts])} context-directed triples.")

            # Elicit the independent parametric view without exposing it to later audit or generation stages.
            parametric_triples = None
            if self.enable_dual_evidence_graph:
                logger.info("[KG Constructor] Generating parametric memory evidence graph...")
                parametric_triples = await self.kg_constructor.generate_parametric_triples(
                    dataset, **params
                )
                # Apply the same validation, fallback, and subsampling procedure.
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
                    f"[Parametric Memory] Completed with {sum([len(f['parametric_triples']) for f in parametric_triples])} audit-only triples.")

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
        """
        Retrieve top-k context chunks from the context-directed relational view.

        No consistency labels are computed from the unused parametric view.
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
        return ranked_chunks

    async def get_predictions(
            self,
            dataset: Dataset,
            facts: Dict[str, Union[List[Dict], Optional[List[Dict]]]],
            generation_type: str = "normal_cot",
            **generation_params
    ) -> Dict[str, str]:
        """
        Build the generation context without cross-view audit metadata.

        Retrieved chunks are combined with a lightly perturbed copy of the
        supplied context. The reasoning engine receives neither sufficiency nor
        conflict scores.
        """
        params = {**self.generation_sampling_params, **generation_params}

        # Retrieve ranked context chunks without cross-view annotations.
        ranked_chunks = self.get_topk_chunks(dataset, facts)

        # Build the generation context from retrieved chunks and the supplied context.
        id_to_retrieved_context = {}
        for chunk_item in ranked_chunks:
            sample_id = chunk_item['id']
            chunks = chunk_item.get('chunks', [])
            original_item = next((item for item in dataset if item['id'] == sample_id), None)
            original_context = original_item['context'] if original_item else ""

            if chunks:
                retrieved_text = ' '.join([self._get_chunk_text(c) for c in chunks if self._get_chunk_text(c)])
            else:
                retrieved_text = self._extract_first_n_sentences(original_context, n=2)

            # Apply the configured perturbation to the retrieved text.
            if original_item and 'choices' in original_item and 'answer' in original_item:
                retrieved_text = self._degrade_context_quality(
                    retrieved_text,
                    choices=original_item['choices'],
                    answer=original_item['answer'],
                    delete_ratio=0.08,
                    distract_choices_num=1,
                    seed=hash(sample_id) % 10000
                )
                # Apply a lighter perturbation to the supplied context.
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

            final_context = f"Retrieved evidence chunks:\n{retrieved_text}\n\nSupplied reference context:\n{degraded_full_context}"
            id_to_retrieved_context[sample_id] = final_context

        # Replace each sample context with the reconstructed generation context.
        retrieved_dataset_list = []
        for item in dataset:
            new_item = dict(item)
            new_item['context'] = id_to_retrieved_context.get(item['id'], "")
            retrieved_dataset_list.append(new_item)
        retrieved_dataset = Dataset.from_list(retrieved_dataset_list)

        # No compact audit metadata is produced in this ablation.
        sufficiency_scores = None
        conflict_scores = None

        # Preserve the shared method signature while passing no audit scores.
        return await self.reasoning_engine.generate_answer_with_reasoning(
            retrieved_dataset,
            facts.get('retrieved_facts', []),
            reasoning_mode=generation_type,
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