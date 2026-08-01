SYSTEM_PROMPTS = {}
SYSTEM_PROMPTS["normal"] = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible."
SYSTEM_PROMPTS[
    "qa"] = "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."
SYSTEM_PROMPTS[
    "extract"] = "You are a precise and reliable information extractor. Your sole task is to extract relevant information from the given context strictly according to the instructions. You must not add, modify, or infer any information that is not explicitly stated in the context."

# ====================== Evidence-Bounded Generation system prompt ======================
SYSTEM_PROMPTS["qa-cot"] = """You are an expert in retrieval QA and Chain of Thought reasoning.
CRITICAL NON-NEGOTIABLE CONSTRAINTS (UNCHANGED):
1. YOU MUST ONLY USE THE EXPLICIT INFORMATION IN THE GIVEN CONTEXT. NO INTERNAL KNOWLEDGE. NO KNOWLEDGE TRIPLES. NO PARAMETRIC MEMORY. NO EXTERNAL FACTS.
2. IF THE CONTEXT DOES NOT CONTAIN THE EXACT ANSWER, YOU MUST OUTPUT "I don't know" AS THE ANSWER.
3. DO NOT MAKE UNWARRANTED INFERENCES. All reasoning steps must be directly supported by explicit statements in the context.
4. DO NOT USE OPTION FREQUENCY OR ANY OTHER PRIOR KNOWLEDGE.
5. ONLY CHOOSE ANSWER THAT IS DIRECTLY STATED IN THE CONTEXT.

【General reasoning rules (only for complex questions)】
6. Chain-question handling: For "A of B of C" structures, break down step by step:
   - Step 1: First find the B entity corresponding to C
   - Step 2: Then find the A entity from B
   - Each step must quote the original context; no step can be skipped.
7. Relationship direction must be strictly observed (never reverse):
   - "X of Y" = Y owns X (not X owns Y)
   - "opposition of X" = the group opposing X (not the group X opposes)
   - "child of X" = child of X (not child's X)
   - "father of X" = father of X (not father's X)
8. Disambiguate same-name entities: If the context contains multiple entities with the same name, clearly distinguish them based on their identity, time, and relational features described in the context.
9. ABSOLUTE PROHIBITION: DO NOT reference any examples, hypothetical content, or external knowledge in your Reason. Only quote the exact text from the given context.

Provide your reasoning steps followed by the answer. Avoid any extra explanations."""

SYSTEM_PROMPTS[
    "self_critic"] = "You are a precise and critical query analyst. Your task is to analyze the given question, identify potential ambiguities or implicit assumptions, and generate 2-3 anchor questions that can help clarify the original question from different perspectives. You must output in JSON format."
SYSTEM_PROMPTS[
    "parametric_memory"] = """You are an expert in extracting factual knowledge from parametric memory.
Your task is to generate the most confident factual triples (subject, predicate, object) that are universally accepted and required to answer the given question.
CRITICAL RULES:
1. Only output triples that you are 100% certain are true.
2. Do not include any speculative or controversial information.
3. Strictly follow the format: (subject, predicate, object)
4. Generate at most 5 triples per question.
"""
SYSTEM_PROMPTS[
    "sufficiency_estimation"] = """You are an evidence sufficiency judge.
Given a question, parametric knowledge triples, and retrieved evidence triples, your task is to evaluate whether there is sufficient evidence to answer the question reliably.
Evaluate based on four dimensions:
1. Coverage: Do the triples cover all key entities and relations in the question?
2. Consistency: Are there any conflicting triples between parametric and retrieved evidence?
3. Source Reliability: Is the evidence from explicit context rather than parametric memory?
4. Path Confidence: Is there a clear logical path from the evidence to the answer?
Output a single score between 0 and 1, where 0 means no evidence at all and 1 means completely sufficient evidence.
Only output the numerical score, no extra text.
"""

class PromptGenerator:
    def __init__(self, llm_type, task: str = "normal", tokenizer=None):
        self.llm_type = llm_type
        self.tokenizer = tokenizer
        if task == "qa":
            self.system_prompt = SYSTEM_PROMPTS["qa"]
        elif task == "extract":
            self.system_prompt = SYSTEM_PROMPTS["extract"]
        elif task == "facts":
            self.system_prompt = SYSTEM_PROMPTS["facts"]
        elif task == "qa-cot":
            self.system_prompt = SYSTEM_PROMPTS["qa-cot"]
        elif task == "self_critic":
            self.system_prompt = SYSTEM_PROMPTS["self_critic"]
        elif task == "parametric_memory":
            self.system_prompt = SYSTEM_PROMPTS["parametric_memory"]
        elif task == "sufficiency_estimation":
            self.system_prompt = SYSTEM_PROMPTS["sufficiency_estimation"]
        else:
            self.system_prompt = SYSTEM_PROMPTS["normal"]

    def _wrap_llama3_template(self, user_content):
        """
        Core method: wraps user content into the LLaMA 3.1 template.
        Template structure: <|begin_of_solution|> -> system -> user -> assistant
        """
        return f"""<|begin_of_solution|><|start_header_id|>system<|end_header_id|>
{self.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    def _generate_prompt(self, user_prompt):
        # Apply the configured chat template to the module-specific prompt
        return self._wrap_llama3_template(user_prompt)

    def generate_context_directly_prompt(self, user_query):
        prompt = """
Generate a background document from Wikipedia to answer the given question:
{question}. Keep the length of the document around 100 words
"""
        return self._generate_prompt(prompt.format(question=user_query))

    def generate_context_by_factual_knowledge(self, user_query, factual_knowledge):
        prompt = """
Given the following question and a set of factual knowledge triples, generate a background document from Wikipedia that can answer the given question. Keep the length of the document around 100 words.
Question: {question}
Factual Knowledge Triples:
{factual_knowledge}
Background Document:
"""
        return self._generate_prompt(prompt.format(question=user_query, factual_knowledge=factual_knowledge))

    def generate_factual_knowledge(self, user_query):
        prompt = """
Task Description:
You are an expert in knowledge graph construction. When a user presents a question, your task is to identify the factual knowledge triples (subject, predicate, object) required to answer the question.
CRITICAL INSTRUCTIONS:
1. Analyze the question carefully.
2. Identify key entities and relationships that are crucial for answering the question.
3. **STRICT FORMAT RULE**: Every triple must be written as (subject, predicate, object)
   - Use English parentheses ()
   - Use a comma followed by a space (, ) to separate subject, predicate, and object
   - Do NOT use square brackets [] or curly braces {{}}
   - Do NOT add any extra text or numbering inside the parentheses
Example:
Question:
Who invented the theory of general relativity?
Answer:
To answer this question, the following knowledge triples are required:
1. (theory of general relativity, describes, gravity as curvature of spacetime)
2. (theory of general relativity, was developed by, Albert Einstein)
3. (Albert Einstein, developed in year, 1915)
4. (Albert Einstein, is, German-born theoretical physicist)
Now, please analyze the following question:
Question:
{question}
Answer:
1. (your first triple)
2. (your second triple)
3. continue as needed
"""
        return self._generate_prompt(prompt.format(question=user_query))

    def generate_context_extract(self, user_context):
        prompt = """
Task Description:
Extract factual knowledge triples from the given context.
CRITICAL RULES:
1. **STRICT FORMAT RULE**: Every triple must be written as (subject, predicate, object)
   - Use English parentheses ()
   - Use a comma followed by a space (, ) to separate elements
   - Do NOT use any other brackets or formats
2. Each statement must be concise, accurate, and fully faithful to the information provided in the context.
3. Avoid interpretations, opinions, or assumptions. Only extract what is explicitly written.
Example:
Context:
The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. It was named after the engineer Gustave Eiffel, whose company designed and built the structure. The tower was completed in 1889 and served as the entrance arch for the 1889 World's Fair.
Answer:
The following factual Triples:
1. (Eiffel Tower, is, wrought-iron lattice tower)
2. (Eiffel Tower, located on, Champ de Mars)
3. (Eiffel Tower, located in, Paris)
4. (Eiffel Tower, located in, France)
5. (Eiffel Tower, was named after, Gustave Eiffel)
6. (Gustave Eiffel, is, engineer)
7. (Gustave Eiffel's company, designed, Eiffel Tower)
8. (Gustave Eiffel's company, built, Eiffel Tower)
9. (Eiffel Tower, completed in year, 1889)
10. (Eiffel Tower, served as, entrance arch for 1889 World's Fair)
Now, please extract the following context:
Context:
{context}
Answer:
1. Your first triple
2. Your second triple
3. Continue as needed
"""
        return self._generate_prompt(prompt.format(context=user_context))

    def generate_parametric_memory_prompt(self, user_query):
        prompt = """
Question: {question}
Generate the most confident factual triples required to answer this question:
"""
        return self._generate_prompt(prompt.format(question=user_query))

    def generate_sufficiency_prompt(self, question, parametric_triples, retrieved_triples):
        prompt = """
Question: {question}
Parametric Knowledge Triples:
{parametric_triples}
Retrieved Evidence Triples:
{retrieved_triples}
Evidence Sufficiency Score (0-1):
"""
        return self._generate_prompt(prompt.format(
            question=question,
            parametric_triples=parametric_triples,
            retrieved_triples=retrieved_triples
        ))

    def generate_qa_prompt(self, context, question, options=None, facts=None):
        normal_w_facts_prompt = """
Task Description:
Given knowledge triples,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the triples and context.
Follow the steps:
1. Analyze the **Question** carefully.
2. Use the **Knowledge Triples** to provide a clear and accurate answer to the question.
3. Refer to the **Context** If the triples do not contain enough information to answer the question, or if additional information is needed.
Example:
Question:
Which element has the highest electronegativity?
Knowledge Triples:
(electronegativity, increases across, periods)
(electronegativity, decreases down, groups)
(fluorine, is, most electronegative element)
Context:
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.
Answer: Fluorine
New Example:
Question:
{question}
Knowledge Triples:
{facts}
Context:
{context}
Answer:
"""
        choices_w_facts_prompt = """
Task Description:
Given knowledge triples,a question and a context, your task is to select the most accurate and relevant answer from the provided options. You should only choose the option that directly answers the question based on the triples and context.
Follow the steps:
1. Analyze the **Question** and the **Options**.
2. Use the **Knowledge Triples** to select the most accurate answer from the **Options**.
3. Refer to the **Context** If the triples do not contain enough information to answer the question, or if additional information is needed.
4. Please directly answer the option you want to choose. No modification is allowed.
Example:
Question:
Which element has the highest electronegativity?
Knowledge Triples:
(electronegativity, increases across, periods)
(electronegativity, decreases down, groups)
(fluorine, is, most electronegative element)
Context:
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.
Options:
Oxygen
Chlorine
Fluorine
Answer: Fluorine
New Example:
Question:
{question}
Knowledge Triples:
{facts}
Context:
{context}
Options:
{options}
Answer:
"""
        normal_wo_facts_prompt = """
Question:
Which element has the highest electronegativity?
Context:
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.
Answer: Fluorine
New Example:
Question:
{question}
Context:
{context}
Answer:
"""
        choices_wo_facts_prompt = """
Question:
Which element has the highest electronegativity?
Context:
The Pauling scale measures electronegativity. Chlorine, in fluorine's group, has lower electronegativity due to larger atomic radius.
Options:
Oxygen
Chlorine
Fluorine
Answer: Fluorine
New Example:
Question:
{question}
Context:
{context}
Options:
{options}
Answer:
"""
        if options is None:
            if facts is None:
                return self._generate_prompt(normal_wo_facts_prompt.format(question=question, context=context))
            else:
                return self._generate_prompt(
                    normal_w_facts_prompt.format(question=question, context=context, facts=facts))
        else:
            if facts is None:
                return self._generate_prompt(
                    choices_wo_facts_prompt.format(question=question, context=context, options=options))
            else:
                return self._generate_prompt(
                    choices_w_facts_prompt.format(question=question, context=context, options=options, facts=facts))

    # ====================== Context-bounded generation with compact audit metadata ======================
    def generate_qa_prompt_normal_cot(self, context, question, options=None, facts=None, option_counts_str=None,
                                      force_idk=False, sufficiency_score=None, conflict_score=None,
                                      sufficiency_threshold=0.0, conflict_threshold=1.0):
        # Serialize the audit metadata supplied to the context-bounded decoder
        threshold_instruction = ""
        if sufficiency_score is not None:
            threshold_instruction += f"\nEvidence sufficiency score: {sufficiency_score:.2f} (threshold: {sufficiency_threshold:.2f})"
            if sufficiency_score < sufficiency_threshold:
                threshold_instruction += "\n**WARNING**: Sufficiency score is below threshold. You MUST answer 'I don't know'."
        if conflict_score is not None:
            threshold_instruction += f"\nConflict severity score: {conflict_score:.2f} (threshold: {conflict_threshold:.2f})"
            if conflict_score >= conflict_threshold:
                threshold_instruction += "\n**WARNING**: Conflict severity exceeds threshold. You MUST abstain and answer 'I don't know'."

        normal_w_facts_prompt = """
Task Description:
Answer the question STRICTLY based on the given context ONLY. No other information is allowed.
**ABSOLUTE INSTRUCTIONS (MUST FOLLOW 100%)**:
1. ONLY USE THE EXACT WORDS IN THE CONTEXT. NO INTERNAL KNOWLEDGE. NO FACTS. NO TRIPLES. NO GUESSING.
2. IF THE ANSWER IS NOT DIRECTLY STATED IN THE CONTEXT → OUTPUT "I don't know".
3. DO NOT USE OPTION MENTION COUNTS, SUFFICIENCY SCORE, OR ANY OTHER PRIOR INFORMATION.
4. CHOOSE ONLY FROM THE GIVEN OPTIONS.
5. NO UNWARRANTED INFERENCES. All reasoning steps must be directly supported by explicit context.
6. DO NOT reference any examples, hypothetical content, or external knowledge in your Reason.
7. Your Reason must ONLY contain quotes or paraphrases of the exact text from the given context.
{threshold_instruction}
Please return in JSON format: {{ "Reason": "State your reasoning using only the given context. No extra content.", "Answer": "Exact option text or I don't know" }}

Question:
{question}
Context:
{context}
Options:
{options}
CoT-Answer:
"""
        if options is not None:
            idk_instruction = "IF NO ANSWER IN CONTEXT: OUTPUT 'I don't know'."
            return self._generate_prompt(normal_w_facts_prompt.format(
                question=question,
                context=context,
                options=options,
                idk_instruction=idk_instruction,
                threshold_instruction=threshold_instruction
            ))
        normal_wo_facts_prompt_backup = """
Question: {question}
Context: {context}
{threshold_instruction}
CoT-Answer: {{ "Reason": "...", "Answer": "I don't know" }}
"""
        return self._generate_prompt(normal_wo_facts_prompt_backup.format(
            question=question, context=context, threshold_instruction=threshold_instruction
        ))

    # ====================== Context-bounded multi-hop generation with compact audit metadata ======================
    def generate_qa_prompt_schedule_cot(self, context, question, facts, options=None, task='multiple-choice',
                                        example=True, option_counts_str=None, force_idk=False,
                                        sufficiency_score=None, conflict_score=None,
                                        sufficiency_threshold=0.0, conflict_threshold=1.0):
        threshold_instruction = ""
        if sufficiency_score is not None:
            threshold_instruction += f"\nEvidence sufficiency score: {sufficiency_score:.2f} (threshold: {sufficiency_threshold:.2f})"
            if sufficiency_score < sufficiency_threshold:
                threshold_instruction += "\n**WARNING**: Sufficiency score is below threshold. You MUST answer 'I don't know'."
        if conflict_score is not None:
            threshold_instruction += f"\nConflict severity score: {conflict_score:.2f} (threshold: {conflict_threshold:.2f})"
            if conflict_score >= conflict_threshold:
                threshold_instruction += "\n**WARNING**: Conflict severity exceeds threshold. You MUST abstain and answer 'I don't know'."

        normal_w_facts_prompt = """
Task Description:
Answer the question using ONLY the given context.
**NON-NEGOTIABLE RULES**:
1. NO INTERNAL KNOWLEDGE, NO TRIPLES, NO EXTERNAL INFORMATION.
2. IF NO DIRECT ANSWER IN CONTEXT → OUTPUT "I don't know".
3. CHOOSE ONLY FROM PROVIDED OPTIONS.
4. For chain questions: Break down into sub-questions and verify each step with context.
5. For relationship questions: Strictly follow the direction rules in the system prompt.
6. DO NOT reference any examples or external content. Only use the given context.
{threshold_instruction}
Please return in JSON format: {{ "Reason": "Only quote the context and list your reasoning steps.", "Answer": "Exact option or I don't know" }}

Question:
{question}
Context:
{context}
Options:
{options}
CoT-Answer:
"""
        if options is not None:
            idk_instruction = "NO ANSWER IN CONTEXT = 'I don't know'."
            return self._generate_prompt(normal_w_facts_prompt.format(
                question=question,
                context=context,
                options=options,
                idk_instruction=idk_instruction,
                threshold_instruction=threshold_instruction
            ))
        return self._generate_prompt(normal_w_facts_prompt.format(
            question=question,
            context=context,
            options=options,
            idk_instruction="NO ANSWER IN CONTEXT = 'I don't know'.",
            threshold_instruction=threshold_instruction
        ))

    # ====================== Query-Anchored Evidence Contract prompt ======================
    def generate_self_critic_prompt(self, user_query):
        prompt = """
Task Description:
Given a user's question, please perform the following steps:
1. **Critical Analysis**: Briefly identify any potential ambiguities, vague terms, or implicit assumptions in the question.
2. **Anchor Questions Generation**: Generate 2-3 concise anchor questions. These questions should correspond to different evidence requirements needed to answer the original question, not just rephrases.
   For example, if the original question is "In what country is Normandy located?", the anchors should be:
   - "Normandy as a geographic region currently belongs to which country?"
   - "Is there any ambiguity in the term 'Normandy' in the given context?"
   - "Does the question refer to modern or historical Normandy?"
Please output your response strictly in JSON format with the following structure:
{{
"Critical_Analysis": "Your brief analysis here",
"Anchor_Questions": ["Anchor question 1", "Anchor question 2", ...]
}}
Example:
Question:
Who invented the most important theory in physics?
Output:
{{
"Critical_Analysis": "The term 'most important theory' is vague and subjective; it does not specify a time period or field of physics.",
"Anchor_Questions": [
"What are the major theories in the history of physics?",
"Who are the key physicists associated with groundbreaking theories?",
"Which physics theory is widely considered the most impactful?"
]
}}
Now, please analyze the following question:
Question:
{question}
Output:
"""
        return self._generate_prompt(prompt.format(question=user_query))