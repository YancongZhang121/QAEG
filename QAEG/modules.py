from typing import List, Tuple, Dict, Union, Optional
import re
from sentence_transformers import SentenceTransformer, util
import nltk
from datasets import Dataset
import json
from .prompts import PromptGenerator
from .util import FormatConverter
from .llm import LLMBackend
from util import logger


# ===================================================================
# Structured-output normalization for QAEG modules
# ===================================================================
def clean_and_repair_json(text: str) -> str:
    """
    Normalize malformed or truncated JSON returned by QAEG's LLM modules.
    """
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
    text = text.strip()
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    start_idx = text.find('{')
    if start_idx == -1:
        return "{}"
    partial_json = text[start_idx:]
    open_braces = partial_json.count('{')
    close_braces = partial_json.count('}')
    open_brackets = partial_json.count('[')
    close_brackets = partial_json.count(']')
    repaired = partial_json
    if open_brackets > close_brackets:
        temp_repaired = repaired.rstrip()
        if temp_repaired and temp_repaired[-1] not in ['"', ',', '[', '{']:
            last_quote_idx = temp_repaired.rfind('"')
            if last_quote_idx != -1:
                repaired = temp_repaired[:last_quote_idx + 1]
        while repaired.count('[') > repaired.count(']'):
            if repaired.endswith(','):
                repaired = repaired[:-1]
            repaired += ']'
    while repaired.count('{') > repaired.count('}'):
        repaired += '}'
    text = repaired
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(?<!")(\b\w+\b)(?=\s*:)', r'"\1"', text)
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


# -------------------------------------------------------------------
# Query-Anchored Evidence Contract
# -------------------------------------------------------------------
class QuerySelfCriticAndAnchorGenerator:
    def __init__(
            self,
            backend_type: str,
            model_name: str,
            **backend_config
    ):
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        self.prompt_generator = PromptGenerator(
            llm_type=backend_type,
            task="self_critic"
        )
        self.default_sampling_params = {
            'max_tokens': 1500,
            'top_p': 1.0,
            'temperature': 0.1
        }

    async def generate_critique_and_anchors(
            self,
            dataset: Dataset,
            **sampling_kwargs
    ) -> List[Dict]:
        prompts = [
            self.prompt_generator.generate_self_critic_prompt(
                user_query=item['question']
            )
            for item in dataset
        ]
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator.system_prompt,
            **merged_params
        )
        results = []
        for item, raw_result in zip(dataset, raw_results):
            critique = ""
            anchor_questions = []
            try:
                cleaned_json_str = clean_and_repair_json(raw_result)
                parsed = json.loads(cleaned_json_str)
                for key in ["Critical_Analysis", "critical_analysis", "Analysis", "analysis", "critique", "Critique"]:
                    if key in parsed:
                        critique = str(parsed[key]).strip()
                        if critique: break
                anchors_data = None
                for key in ["Anchor_Questions", "anchor_questions", "Anchors", "anchors", "questions", "Questions"]:
                    if key in parsed:
                        anchors_data = parsed[key]
                        break
                if anchors_data:
                    if isinstance(anchors_data, list):
                        anchor_questions = [str(q).strip() for q in anchors_data if str(q).strip()]
                    elif isinstance(anchors_data, str):
                        potential_questions = [q.strip() for q in anchors_data.split('\n') if q.strip()]
                        anchor_questions = [q for q in potential_questions if '?' in q or len(q) > 10]
            except Exception as e:
                logger.debug(f"[QSAG Plan B] SampleID:{item['id']} JSON parsing failed, using regex extraction.")
                critique_match = re.search(
                    r'(?:Critical_Analysis|critical_analysis|Analysis)["\s]*:["\s]*(.*?)(?="?\s*,\s*"Anchor|$)',
                    raw_result,
                    re.DOTALL | re.IGNORECASE
                )
                if critique_match:
                    critique = critique_match.group(1).strip()
                    critique = re.sub(r'^["\s]+|["\s]+$', '', critique)
                if not critique:
                    critique = "Proceeding with regex-extracted anchors."
                bracket_match = re.search(
                    r'(?:Anchor_Questions|anchor_questions|Anchors)["\s]*:[\s]*\[(.*?)\]',
                    raw_result,
                    re.DOTALL
                )
                if bracket_match:
                    array_content = bracket_match.group(1)
                    potential_q = re.findall(r'"([^"]+)"', array_content)
                    anchor_questions = [q.strip() for q in potential_q if q.strip() and '?' in q]
                if not anchor_questions:
                    candidates = re.findall(r'([^\.\n]{10,}\?)', raw_result)
                    anchor_questions = [q.strip() for q in candidates if q.strip()]
                anchor_questions = list(dict.fromkeys(anchor_questions))[:3]
            results.append({
                'id': item['id'],
                'original_question': item['question'],
                'critique': critique,
                'anchor_questions': anchor_questions
            })
        return results


# -------------------------------------------------------------------
# Asymmetric Dual-View Evidence Construction
# -------------------------------------------------------------------
class KnowledgeGraphConstructor:
    def __init__(
            self,
            backend_type: str,
            model_name: str,
            **backend_config
    ):
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        self.prompt_generator = PromptGenerator(
            llm_type=backend_type,
            task="normal"
        )
        self.prompt_generator_extract = PromptGenerator(
            llm_type=backend_type,
            task="extract"
        )
        self.prompt_generator_parametric = PromptGenerator(
            llm_type=backend_type,
            task="parametric_memory"
        )
        self.default_sampling_params = {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.1
        }
        self.triple_pattern = r'\((.*?),\s*(.*?),\s*(.*?)\)'

    async def generate_initial_triples(
            self,
            dataset: Dataset,
            anchor_data: Optional[List[Dict]] = None,
            **sampling_kwargs
    ) -> List[Dict]:
        prompts = []
        for item in dataset:
            anchors = []
            if anchor_data:
                anchor_item = next((a for a in anchor_data if a['id'] == item['id']), None)
                if anchor_item:
                    anchors = anchor_item.get('anchor_questions', [])
            base_prompt = self.prompt_generator_extract.generate_factual_knowledge(
                user_query=item['question']
            )
            if anchors:
                anchor_str = "\n\n【Supplementary Perspective】To extract knowledge more comprehensively, please also consider the following related questions:\n" + "\n".join(
                    [f"- {q}" for q in anchors])
                if "Now, please analyze the following question:" in base_prompt:
                    parts = base_prompt.split("Now, please analyze the following question:")
                    prompts.append(parts[0] + anchor_str + "\n\nNow, please analyze the following question:" + parts[1])
                else:
                    prompts.append(base_prompt + anchor_str)
            else:
                prompts.append(base_prompt)
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator.system_prompt,
            **merged_params
        )
        for i, raw_result in enumerate(raw_results):
            logger.debug(f"[KG Constructor] SampleID:{dataset[i]['id']} initial triple raw output: {raw_result[:600]}")
        return [
            {
                'id': item['id'],
                'triples': re.findall(self.triple_pattern, result)
            }
            for item, result in zip(dataset, raw_results)
        ]

    async def generate_parametric_triples(
            self,
            dataset: Dataset,
            **sampling_kwargs
    ) -> List[Dict]:
        prompts = [
            self.prompt_generator_parametric.generate_parametric_memory_prompt(
                user_query=item['question']
            )
            for item in dataset
        ]
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_parametric.system_prompt,
            **merged_params
        )
        return [
            {
                'id': item['id'],
                'parametric_triples': re.findall(self.triple_pattern, result)
            }
            for item, result in zip(dataset, raw_results)
        ]

    async def generate_augmented_context(
            self,
            dataset: Dataset,
            triples: Optional[Union[Dict, List[Dict]]] = None,
            **sampling_kwargs
    ) -> List[Dict]:
        prompts = []
        for item in dataset:
            if triples is None:
                prompt = self.prompt_generator.generate_context_directly_prompt(
                    user_query=item['question']
                )
            else:
                triple_data = next(
                    (k for k in triples if k['id'] == item['id']),
                    None
                )
                if triple_data is None:
                    logger.warning(f"No triples found for item {item['id']}")
                    prompt = self.prompt_generator.generate_context_directly_prompt(
                        user_query=item['question']
                    )
                else:
                    triples_str = '\n'.join([f"({s}, {p}, {o})" for s, p, o in triple_data['triples']])
                    prompt = self.prompt_generator.generate_context_by_factual_knowledge(
                        user_query=item['question'],
                        factual_knowledge=triples_str
                    )
            prompts.append(prompt)
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Generating augmented contexts...")
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator.system_prompt,
            **merged_params
        )
        return [
            {'id': item['id'], 'context': result}
            for item, result in zip(dataset, raw_results)
        ]

    async def extract_refined_triples(
            self,
            contexts: Union[List[Dict], Dict],
            **sampling_kwargs
    ) -> List[Dict]:
        if isinstance(contexts, dict):
            contexts = [contexts]
        prompts = [
            self.prompt_generator_extract.generate_context_extract(
                user_context=ctx['context']
            )
            for ctx in contexts
        ]
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Extracting refined knowledge triples...")
        raw_results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=self.prompt_generator_extract.system_prompt,
            **merged_params
        )
        for i, raw_result in enumerate(raw_results):
            logger.debug(f"[KG Constructor] SampleID:{contexts[i]['id']} refined triple raw output: {raw_result[:600]}")
        return [
            {
                'id': ctx['id'],
                'facts': re.findall(self.triple_pattern, result)
            }
            for ctx, result in zip(contexts, raw_results)
        ]

    def merge_triples(self, retrieved_triples: List[tuple], parametric_triples: List[tuple]) -> List[tuple]:
        merged = {}
        for s, p, o in retrieved_triples:
            key = (s.lower().strip(), p.lower().strip())
            merged[key] = (s, p, o)
        for s, p, o in parametric_triples:
            key = (s.lower().strip(), p.lower().strip())
            if key not in merged:
                merged[key] = (s, p, o)
        return list(merged.values())

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


# -------------------------------------------------------------------
# Asymmetric Audit-State Construction
# -------------------------------------------------------------------
class EvidenceSufficiencyEstimator:
    def __init__(
            self,
            backend_type: str,
            model_name: str,
            **backend_config
    ):
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        self.prompt_generator = PromptGenerator(
            llm_type=backend_type,
            task="sufficiency_estimation"
        )
        self.default_sampling_params = {
            'max_tokens': 50,
            'top_p': 1.0,
            'temperature': 0.0
        }

    async def estimate_sufficiency(
            self,
            dataset: Dataset,
            parametric_triples: List[Dict],
            retrieved_triples: List[Dict],
            **sampling_kwargs
    ) -> List[Dict]:
        results = []
        prompts = []
        prompt_items = []

        for item in dataset:
            param_data = next((p for p in parametric_triples if p['id'] == item['id']), None)
            retrieved_data = next((r for r in retrieved_triples if r['id'] == item['id']), None)

            question_lower = item['question'].lower()
            has_direct_answer = False

            answer_predicates = {
                'died', 'died in', 'died in year', 'born', 'born in', 'born in year',
                'located in', 'located at', 'wrote', 'authored', 'founded', 'founder of',
                'departed', 'left', 'returned', 'married', 'spouse of'
            }

            if retrieved_data and retrieved_data.get('facts', []):
                for (s, p, o) in retrieved_data['facts']:
                    p_lower = p.lower().strip()
                    if any(pred in p_lower for pred in answer_predicates):
                        s_lower = s.lower().strip()
                        o_lower = o.lower().strip()
                        if s_lower in question_lower or o_lower in question_lower:
                            has_direct_answer = True
                            break

            if has_direct_answer:
                results.append({
                    'id': item['id'],
                    'sufficiency_score': 0.8
                })
                continue

            param_str = "\n".join(
                [f"({s}, {p}, {o})" for s, p, o in param_data.get('parametric_triples', [])]) if param_data else "None"
            retrieved_str = "\n".join(
                [f"({s}, {p}, {o})" for s, p, o in retrieved_data.get('facts', [])]) if retrieved_data else "None"

            prompts.append(
                self.prompt_generator.generate_sufficiency_prompt(
                    question=item['question'],
                    parametric_triples=param_str,
                    retrieved_triples=retrieved_str
                )
            )
            prompt_items.append(item)

        if prompts:
            merged_params = {**self.default_sampling_params, **sampling_kwargs}
            raw_results = await self.llm_backend.generate(
                prompts=prompts,
                system_prompt=self.prompt_generator.system_prompt,
                **merged_params
            )

            for item, raw_result in zip(prompt_items, raw_results):
                try:
                    score_match = re.search(r'(\d+\.?\d*)', raw_result.strip())
                    if score_match:
                        score = float(score_match.group(1))
                        score = 0.9 * max(0.0, min(1.0, score)) + 0.1 * 0.5
                    else:
                        score = 0.5
                except Exception as e:
                    logger.debug(f"[Sufficiency Estimator] SampleID:{item['id']} score parsing failed, using default 0.5")
                    score = 0.5

                score = max(0.8, score)

                results.append({
                    'id': item['id'],
                    'sufficiency_score': score
                })

        id_to_idx = {item['id']: idx for idx, item in enumerate(dataset)}
        results.sort(key=lambda x: id_to_idx[x['id']])
        return results


# -------------------------------------------------------------------
# Context-Directed Chunk Retrieval
# -------------------------------------------------------------------
class EvidenceRetriever:
    def __init__(self,
                 similarity_model: str):
        self.similarity_model = SentenceTransformer(similarity_model)

    def chunk_text(self, paragraph: str, chunk_size: int = 20, overlap_size: int = 5) -> List[str]:
        sentences = nltk.sent_tokenize(paragraph)
        chunks = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                backtrack = min(overlap_size, len(current_chunk))
                current_chunk = current_chunk[-backtrack:] if backtrack > 0 else []
                current_length = sum(len(s.split()) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_length
            i += 1
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _triple_to_text(self, triple):
        return f"{triple[0]} {triple[1]} {triple[2]}"

    def compute_similarity_scores(
            self,
            paragraph: str,
            triples: List[Tuple],
            top_k: int = 5,
            chunk_size: int = 50
    ):
        chunks = self.chunk_text(paragraph, chunk_size=chunk_size)
        chunk_embeddings = self.similarity_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
        triple_texts = [self._triple_to_text(t) for t in triples]
        triple_embeddings = self.similarity_model.encode(triple_texts, convert_to_tensor=True, show_progress_bar=False)
        results = []
        for i, triple_text in enumerate(triple_texts):
            cosine_scores = util.cos_sim(triple_embeddings[i], chunk_embeddings)
            top_results = sorted(
                enumerate(cosine_scores[0].tolist()), key=lambda x: x[1], reverse=True
            )[:top_k]
            top_chunks = [(chunks[idx], score) for idx, score in top_results]
            results.append((triple_text, top_chunks))
        return results

    def retrieve_relevant_chunks(self, facts: List[Dict], dataset: Dataset, sent_topk=5, chunck_size=20):
        all_chunks = []
        for item, fact in zip(dataset, facts):
            if len(fact['facts']) == 0:
                logger.warning(f'SampleID:{fact["id"]} no knowledge triples extracted')
                all_chunks.append({'id': fact['id'], 'chunks': []})
                continue
            paragraph = FormatConverter.remove_brackets_and_content(item['context'])
            results = self.compute_similarity_scores(paragraph, fact['facts'], top_k=sent_topk, chunk_size=chunck_size)
            chunks = []
            for _, match in results:
                for chunk, score in match:
                    chunks.append({'chunk': chunk, 'score': score})
            all_chunks.append({'id': fact['id'], 'chunks': chunks})
        return all_chunks

    def rank_and_filter_chunks(self, all_chunks: List[Dict], chunk_topk=5):
        all_topk_chunks = []
        for chunk in all_chunks:
            topk_chunks = []
            sorted_chunks = sorted(chunk['chunks'], key=lambda x: x['score'], reverse=True)
            seen_chunks = set()
            for sub_chunk in sorted_chunks:
                if sub_chunk['chunk'] not in seen_chunks:
                    topk_chunks.append(sub_chunk)
                    seen_chunks.add(sub_chunk['chunk'])
                if len(topk_chunks) == chunk_topk:
                    break
            all_topk_chunks.append({'id': chunk['id'], 'topk_chunks': topk_chunks})
        return all_topk_chunks


# -------------------------------------------------------------------
# Context-Bounded Generation
# -------------------------------------------------------------------
class ReasoningEngine:
    def __init__(
            self,
            backend_type: str,
            model_name: str,
            **backend_config
    ):
        self.llm_backend = LLMBackend(
            backend_type=backend_type,
            model_name=model_name,
            **backend_config
        )
        self.prompt_generator_qa_cot = PromptGenerator(
            llm_type=backend_type,
            task="qa-cot"
        )
        self.prompt_generator_qa = PromptGenerator(
            llm_type=backend_type,
            task="qa"
        )
        self.default_sampling_params = {
            'max_tokens': 1000,
            'top_p': 1.0,
            'temperature': 0.1
        }

    async def generate_answer_with_reasoning(
            self,
            dataset: Dataset,
            facts: List[Dict],
            reasoning_mode: str = "normal_cot",
            sufficiency_scores: Optional[List[Dict]] = None,
            conflict_scores: Optional[List[Dict]] = None,
            evidence_sufficiency_threshold: float = 0.0,
            conflict_threshold: float = 1.0,
            **sampling_kwargs
    ) -> Dict[str, str]:
        if reasoning_mode == "normal_cot":
            return await self._normal_cot_reasoning(
                dataset, facts, sufficiency_scores, conflict_scores,
                evidence_sufficiency_threshold, conflict_threshold, **sampling_kwargs
            )
        elif reasoning_mode == "scheduled_cot":
            return await self._scheduled_cot_reasoning(
                dataset, facts, sufficiency_scores, conflict_scores,
                evidence_sufficiency_threshold, conflict_threshold, **sampling_kwargs
            )
        elif reasoning_mode == "wo_cot":
            return await self._direct_answer(dataset, facts, **sampling_kwargs)
        else:
            raise ValueError(f"Unsupported reasoning mode: {reasoning_mode}")

    def _calculate_option_counts(self, context: str, options: List[str]) -> tuple:
        """
        Count explicit answer-option mentions in the supplied context.
        This diagnostic cannot authorize an answer unsupported by the context.
        Returns formatted counts, an abstention flag, and the raw count mapping.
        """
        if not options:
            return "No options provided.", False, {}

        context_lower = context.lower()
        count_dict = {}
        idk_option = None

        for opt in options:
            if "don't know" in opt.lower() or "dont know" in opt.lower() or "i do not know" in opt.lower():
                idk_option = opt
                break

        for opt in options:
            opt_clean = opt.strip()
            if not opt_clean:
                continue
            opt_escaped = re.escape(opt_clean.lower())
            count = len(re.findall(r'\b' + opt_escaped + r'\b', context_lower))
            count_dict[opt_clean] = count

        force_idk = False
        if idk_option:
            total_non_idk_count = sum(cnt for opt, cnt in count_dict.items() if opt != idk_option)
            if total_non_idk_count == 0:
                force_idk = True
                logger.debug(f"[Reasoning Engine] Forcing I don't know: no option mentioned in context and no evidence")

        sorted_counts = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        count_str_lines = []
        for opt, cnt in sorted_counts:
            count_str_lines.append(f"- '{opt}': {cnt} explicit mentions")

        return "\n".join(count_str_lines), force_idk, count_dict

    async def _normal_cot_reasoning(
            self,
            dataset: Dataset,
            facts: List[Dict],
            sufficiency_scores: Optional[List[Dict]] = None,
            conflict_scores: Optional[List[Dict]] = None,
            evidence_sufficiency_threshold: float = 0.0,
            conflict_threshold: float = 1.0,
            **sampling_kwargs
    ) -> Dict[str, str]:
        prompts = []
        for item in dataset:
            fact_data = next((d for d in facts if d['id'] == item['id']), None)
            context_str = ""
            triples_str = ""
            has_context = False
            has_triples = False
            if fact_data:
                if 'topk_chunks' in fact_data and fact_data['topk_chunks']:
                    context_str = ' '.join([chunk_dict['chunk'] for chunk_dict in fact_data['topk_chunks']])
                    has_context = True
                if 'facts' in fact_data and fact_data['facts']:
                    triples_str = '\n'.join([f"({s}, {p}, {o})" for s, p, o in fact_data['facts']])
                    has_triples = True

            sufficiency_score = None
            if sufficiency_scores:
                suff_data = next((s for s in sufficiency_scores if s['id'] == item['id']), None)
                if suff_data:
                    sufficiency_score = suff_data.get('sufficiency_score', 0.5)

            conflict_score = 0.0
            if conflict_scores:
                conf_data = next((c for c in conflict_scores if c['id'] == item['id']), None)
                if conf_data:
                    conflict_score = conf_data.get('conflict_score', 0.0)

            options_str = ""
            options_list = item.get('choices', [])
            if options_list:
                options_str = "\n".join([f"- {opt.strip()}" for opt in options_list])

            option_counts_str, force_idk, _ = self._calculate_option_counts(item.get('context', ''), options_list)

            base_prompt = self.prompt_generator_qa_cot.generate_qa_prompt_normal_cot(
                context=item.get('context', ''),
                question=item['question'],
                options=options_list,
                facts=triples_str,
                option_counts_str=option_counts_str,
                force_idk=force_idk,
                sufficiency_score=sufficiency_score,
                conflict_score=conflict_score,
                sufficiency_threshold=evidence_sufficiency_threshold,
                conflict_threshold=conflict_threshold
            )
            balance_instruction = ""
            if has_context and has_triples:
                balance_instruction = f"\n\n【CRITICAL RULES】\n1. CONTEXT IS ABSOLUTE TRUTH. Ignore all parametric knowledge.\n2. Evidence sufficiency score takes precedence over everything.\n3. Option mentions are ONLY a last resort if NO evidence exists.\n4. Available Options:\n{options_str}\n5. DO NOT OUTPUT NULL."
            elif has_context:
                balance_instruction = f"\n\n【CRITICAL RULES】\n1. Rely ONLY on the Context below. Ignore your internal knowledge.\n2. Evidence sufficiency score is the primary decision basis.\n3. Option mentions are ONLY a last resort.\n4. Available Options:\n{options_str}\n5. NEVER SAY 'null'."
            elif has_triples:
                balance_instruction = f"\n\n【CRITICAL RULES】\n1. Use the Knowledge Triples only. Ignore your internal knowledge.\n2. Evidence sufficiency score takes precedence.\n3. Available Options:\n{options_str}"

            if "CoT-Answer:" in base_prompt:
                parts = base_prompt.split("CoT-Answer:")
                prompts.append(parts[0] + balance_instruction + "\n\nCoT-Answer:" + parts[1])
            else:
                prompts.append(base_prompt + balance_instruction)

        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Executing Normal CoT Reasoning with evidence sufficiency...")
        original_system_prompt = self.prompt_generator_qa_cot.system_prompt
        patched_system_prompt = f"{original_system_prompt}\nPlease output your final response strictly in JSON format."
        results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=patched_system_prompt,
            **merged_params
        )
        return {item['id']: res for item, res in zip(dataset, results)}

    async def _scheduled_cot_reasoning(
            self,
            dataset: Dataset,
            facts: List[Dict],
            sufficiency_scores: Optional[List[Dict]] = None,
            conflict_scores: Optional[List[Dict]] = None,
            evidence_sufficiency_threshold: float = 0.0,
            conflict_threshold: float = 1.0,
            **sampling_kwargs
    ) -> Dict[str, str]:
        prompts = []
        for item in dataset:
            fact_data = next((d for d in facts if d['id'] == item['id']), None)
            triples_str = ""
            if fact_data and 'facts' in fact_data and fact_data['facts']:
                triples_str = '\n'.join([f"({s}, {p}, {o})" for s, p, o in fact_data['facts']])

            sufficiency_score = None
            if sufficiency_scores:
                suff_data = next((s for s in sufficiency_scores if s['id'] == item['id']), None)
                if suff_data:
                    sufficiency_score = suff_data.get('sufficiency_score', 0.5)

            conflict_score = 0.0
            if conflict_scores:
                conf_data = next((c for c in conflict_scores if c['id'] == item['id']), None)
                if conf_data:
                    conflict_score = conf_data.get('conflict_score', 0.0)

            options_str = ""
            options_list = item.get('choices', [])
            if options_list:
                options_str = "\n".join([f"- {opt.strip()}" for opt in options_list])

            option_counts_str, force_idk, _ = self._calculate_option_counts(item.get('context', ''), options_list)

            base_prompt = self.prompt_generator_qa_cot.generate_qa_prompt_schedule_cot(
                context=item.get('context', ''),
                question=item['question'],
                facts=triples_str,
                options=options_list,
                option_counts_str=option_counts_str,
                force_idk=force_idk,
                sufficiency_score=sufficiency_score,
                conflict_score=conflict_score,
                sufficiency_threshold=evidence_sufficiency_threshold,
                conflict_threshold=conflict_threshold
            )
            balance_note = f"\n\n【REMINDER】Context is absolute truth. Evidence sufficiency takes precedence over option mentions. Available Options: \n{options_str}"
            if "CoT-Answer:" in base_prompt:
                parts = base_prompt.split("CoT-Answer:")
                prompts.append(parts[0] + balance_note + "\n\nCoT-Answer:" + parts[1])
            else:
                prompts.append(base_prompt + balance_note)

        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Executing Scheduled CoT Reasoning with evidence sufficiency...")
        original_system_prompt = self.prompt_generator_qa_cot.system_prompt
        patched_system_prompt = f"{original_system_prompt}\nPlease output your final response strictly in JSON format."
        results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=patched_system_prompt,
            **merged_params
        )
        return {item['id']: res for item, res in zip(dataset, results)}

    async def _direct_answer(
            self,
            dataset: Dataset,
            facts: List[Dict],
            **sampling_kwargs
    ) -> Dict[str, str]:
        prompts = []
        for item in dataset:
            fact_data = next((d for d in facts if d['id'] == item['id']), None)
            triples_str = ""
            if fact_data and 'facts' in fact_data and fact_data['facts']:
                triples_str = '\n'.join([f"({s}, {p}, {o})" for s, p, o in fact_data['facts']])
            prompts.append(
                self.prompt_generator_qa.generate_qa_prompt(
                    context=item.get('context', ''),
                    question=item['question'],
                    options=item.get('choices'),
                    facts=triples_str
                )
            )
        merged_params = {**self.default_sampling_params, **sampling_kwargs}
        logger.info(f"Generating Direct Answer (w/o CoT)...")
        original_system_prompt = self.prompt_generator_qa.system_prompt
        patched_system_prompt = f"{original_system_prompt}\nPlease output your final response strictly in JSON format."
        results = await self.llm_backend.generate(
            prompts=prompts,
            system_prompt=patched_system_prompt,
            **merged_params
        )
        return {item['id']: res for item, res in zip(dataset, results)}