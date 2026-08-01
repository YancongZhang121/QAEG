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
    QuerySelfCriticAndAnchorGenerator,
    EvidenceSufficiencyEstimator
)
from .util import FormatConverter
from util import logger


class qaeg:
    """
    QAEG: Query-Anchored Asymmetric Evidence Governance for context-faithful RAG.

    The pipeline implements three coupled stages:
    1. Query critique and anchoring define a persistent evidence contract.
    2. Context-directed and parametric relational views are constructed
       independently. Retrieved context is answer-admissible, whereas the
       parametric view is audit-only.
    3. Compact audit metadata is passed to context-bounded generation, while
       explicit parametric triples remain outside the answer-evidence field.
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

        # Register the server endpoint used by the LLM-backed modules
        backend_config['server_url'] = server_url

        # Default sampling parameters for evidence construction and answer generation
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

        # Initialize query anchoring, relational-view construction, retrieval, and generation modules
        self.query_self_critic = QuerySelfCriticAndAnchorGenerator(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
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

        # Initialize the audit-state estimator when enabled
        if self.enable_sufficiency_estimation:
            self.sufficiency_estimator = EvidenceSufficiencyEstimator(
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
        Construct the query-anchored context and parametric relational views.

        The context-directed view is organized by the query anchors. The
        parametric-memory view is elicited independently and retained only for
        audit-state construction, not as answer evidence.
        """
        params = {**self.mining_sampling_params, **mining_params}
        if fact_mining_type == "default":
            logger.info("[QSAG Module] Starting query self-criticism and anchor generation...")
            anchor_data = await self.query_self_critic.generate_critique_and_anchors(
                dataset, **params
            )
            logger.info("[KG Constructor] Starting initial triples generation...")
            initial_triples = await self.kg_constructor.generate_initial_triples(
                dataset, anchor_data=anchor_data, **params
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
                filtered_facts.append({
                    'id': fact_item['id'],
                    'facts': filtered_triples
                })
            logger.info(f"[Fact Mining] Completed, extracted {sum([len(f['facts']) for f in filtered_facts])} retrieved triples")

            # Independently elicit the parametric-memory view for audit-only use
            parametric_triples = None
            if self.enable_dual_evidence_graph:
                logger.info("[KG Constructor] Generating parametric memory evidence graph...")
                parametric_triples = await self.kg_constructor.generate_parametric_triples(
                    dataset, **params
                )
                logger.info(
                    f"[Parametric Memory] Completed, extracted {sum([len(f['parametric_triples']) for f in parametric_triples])} parametric triples")

            return {
                'retrieved_facts': filtered_facts,
                'parametric_facts': parametric_triples
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
        Rank and deduplicate retrieved-context chunks using the context-directed view.

        Retrieved-context chunks remain the only answer-admissible evidence. Any
        optional marker derived from the parametric view is diagnostic audit
        metadata and does not alter the context ranking or authorize parametric
        content as answer evidence.
        """
        # Accept either the legacy fact list or the role-separated view dictionary
        if isinstance(self_facts, list):
            retrieved_facts = self_facts
            parametric_facts = None
        else:
            retrieved_facts = self_facts.get('retrieved_facts', [])
            parametric_facts = self_facts.get('parametric_facts', None)

        # Rank and deduplicate context chunks using the context-directed triples
        contextual_chunks = self.evidence_retriever.retrieve_relevant_chunks(
            retrieved_facts, dataset, sent_topk, chunk_size
        )
        ranked_chunks = self.evidence_retriever.rank_and_filter_chunks(
            contextual_chunks, chunk_topk
        )

        # Attach optional audit-only consistency metadata without changing the selected context chunks
        if self.enable_dual_evidence_graph and parametric_facts is not None:
            logger.info("[Dual Evidence Graph] Performing consistency validation between retrieved segments and parametric evidence...")
            parametric_chunks = self.evidence_retriever.retrieve_relevant_chunks(
                parametric_facts, dataset, sent_topk, chunk_size
            )
            parametric_text_set = {}
            for p_item in parametric_chunks:
                parametric_text_set[p_item['id']] = {c['text'] for c in p_item['chunks']}

            for rank_item in ranked_chunks:
                sample_id = rank_item['id']
                p_texts = parametric_text_set.get(sample_id, set())
                for chunk in rank_item['chunks']:
                    # Record a diagnostic marker only; it does not make parametric content answer-admissible
                    chunk['param_consistent'] = chunk['text'] in p_texts
            logger.info("[Dual Evidence Graph] Consistency validation marking completed")

        return ranked_chunks

    async def get_predictions(
            self,
            dataset: Dataset,
            facts: Dict[str, Union[List[Dict], Optional[List[Dict]]]],
            generation_type: str = "normal_cot",
            **generation_params
    ) -> Dict[str, str]:
        """
        Build compact audit metadata and invoke context-bounded generation.

        The final decoder uses retrieved evidence as the answer-supporting source.
        Parametric triples remain outside the answer-evidence field and may affect
        only the audit state. When the context does not directly determine an
        answer, the generation policy requires abstention.
        """
        params = {**self.generation_sampling_params, **generation_params}

        retrieved_facts = facts.get('retrieved_facts', [])
        parametric_facts = facts.get('parametric_facts', None)

        # Compress the context/parametric comparison into per-sample audit metadata
        sufficiency_scores = None
        if self.enable_sufficiency_estimation and parametric_facts is not None:
            logger.info("[Sufficiency Estimator] Calculating evidence sufficiency scores...")
            sufficiency_scores = await self.sufficiency_estimator.estimate_sufficiency(
                dataset, parametric_facts, retrieved_facts, **params
            )

        # Supply a neutral disagreement field when no separate disagreement estimator is available.
        conflict_scores = [{'id': item['id'], 'conflict_score': 0.0} for item in dataset]

        # Invoke context-bounded generation with retrieved evidence and compact audit metadata
        # Explicit parametric triples are not passed to the decoder as answer evidence
        return await self.reasoning_engine.generate_answer_with_reasoning(
            dataset, retrieved_facts, reasoning_mode=generation_type,
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