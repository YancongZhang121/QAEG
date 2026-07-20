# Run it by placing it in pipeline.py

import asyncio
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
    # EvidenceSufficiencyEstimator is intentionally omitted in the bounded-generation ablation.
)
from .util import FormatConverter
from util import logger


class qaeg:
    """
    QAEG bounded-generation ablation ("w/o Bound").

    The query-anchored evidence contract and the independently constructed
    context and parametric relational views are retained. This variant omits
    the explicit evidence-sufficiency estimator, so no computed sufficiency
    audit state is supplied to the final reasoning interface. Threshold
    attributes and placeholder fields are preserved for compatibility with
    the shared pipeline API.
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
        # Preserve the shared threshold attributes for interface compatibility.
        self.evidence_sufficiency_threshold = evidence_sufficiency_threshold
        self.conflict_threshold = conflict_threshold

        # Forward the server endpoint to all retained model-backed modules.
        backend_config['server_url'] = server_url

        # Default sampling parameters for evidence construction and answer generation.
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

        # Initialize the modules retained by this ablation with a shared backend configuration.
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
        # No sufficiency estimator is instantiated in this bounded-generation ablation.

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
        Construct the two role-separated relational views retained by this ablation.

        Query anchors guide the context-directed view, while the parametric
        view is elicited independently from model memory. The views remain
        separate and are returned through distinct fields.
        """
        params = {**self.mining_sampling_params, **mining_params}
        if fact_mining_type == "default":
            logger.info("[Query Contract] Generating query critique and evidence anchors...")
            anchor_data = await self.query_self_critic.generate_critique_and_anchors(
                dataset, **params
            )
            logger.info("[Context View] Generating query-conditioned requirement triples...")
            initial_triples = await self.kg_constructor.generate_initial_triples(
                dataset, anchor_data=anchor_data, **params
            )
            logger.info("[Context View] Building an intermediate evidence sketch...")
            augmented_context = await self.kg_constructor.generate_augmented_context(
                dataset, triples=initial_triples, **params
            )
            logger.info("[Context View] Extracting refined context-directed triples...")
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
            logger.info(
                f"[Context View] Completed: extracted "
                f"{sum([len(f['facts']) for f in filtered_facts])} context-directed triples."
            )

            # Elicit the independent parametric view retained for the asymmetric evidence state.
            parametric_triples = None
            if self.enable_dual_evidence_graph:
                logger.info("[Parametric View] Generating independent parametric-memory triples...")
                parametric_triples = await self.kg_constructor.generate_parametric_triples(
                    dataset, **params
                )
                logger.info(
                    f"[Parametric View] Completed: extracted "
                    f"{sum([len(f['parametric_triples']) for f in parametric_triples])} "
                    f"parametric-memory triples."
                )
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
        Retrieve and rank context chunks from the context-directed relational view.

        The parametric view is not used as answer-retrieval evidence, and this
        method does not attach cross-view consistency annotations.
        """
        # Accept either the legacy fact list or the role-separated fact dictionary.
        if isinstance(self_facts, list):
            retrieved_facts = self_facts
        else:
            retrieved_facts = self_facts.get('retrieved_facts', [])

        # Rank context chunks using the context-directed triples only.
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
        Generate answers without a computed evidence-sufficiency audit state.

        The explicit sufficiency operator used by the full QAEG decision
        interface is absent here. A zero-valued conflict field is passed only
        to satisfy the shared reasoning-engine signature; it is not an
        evaluated audit signal.
        """
        params = {**self.generation_sampling_params, **generation_params}
        retrieved_facts = facts.get('retrieved_facts', [])

        # No audit operator is evaluated here; retain a neutral compatibility field.
        conflict_scores = [{'id': item['id'], 'conflict_score': 0.0} for item in dataset]

        # Invoke the shared answer stage without a computed sufficiency signal.
        return await self.reasoning_engine.generate_answer_with_reasoning(
            dataset,
            retrieved_facts,
            reasoning_mode=generation_type,
            sufficiency_scores=None,                               # No sufficiency audit state in this variant.
            conflict_scores=conflict_scores,                       # Compatibility placeholder, not an audit judgment.
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