import json
from typing import List, Dict
from pydantic import BaseModel
from flotorch.sdk.utils.llm_utils import convert_pydantic_to_custom_json_schema

from utils.models import validate_environment, setup_model


PROMPT_GENERATION_SYSTEM = """You are a senior prompt engineer specializing in optimizing prompts for Retrieval-Augmented Generation (RAG) systems.
Your task is to create improved system + user prompt pairs that will guide the model to produce higher-quality answers
for tasks like the examples provided.

The goal is to design prompts that score better on RAG metrics:
- Context Precision - Relevance of retrieved context.
- Faithfulness - Factual consistency with the context.
- Answer Relevancy - How well the answer addresses the question.
- Context Relevancy - Evaluates the relevance of the context retrieved.
- Context Recall - Measures how well the retrieved context aligns with expected answer.
- Hallucination - Checks whether generated answer is factually correct.
- Maliciousness - Checks for harmful or unsafe content.

You will receive:
- One or more existing system and user prompts (used as reference)
- Optionally, several example question-answer pairs (representative of the task)

Your job:
1. Analyze the provided prompts to understand what kind of reasoning and answer structure the task requires.
2. Identify potential weaknesses or limitations in the existing prompts.
3. Generate a new prompt pair (system_prompt and user_prompt) that differs from all the existing ones and could yield better RAG metric performance.
4. The prompts does not require any kinds of tags including the ones for the question. They will be added later. Both the system and user prompt should be text only.

The new pair must:
- Differ in strategy or tone from all existing prompts (e.g., evidence citation, reasoning depth, conciseness, explicit context handling, uncertainty handling)
- Be structured to improve model grounding, contextual precision, and clarity.

Output strict JSON in the following schema:
- system_prompt: complete text of the new system prompt
- user_prompt: complete text of the new user prompt
- strategy_used: short label for the design idea (e.g., "evidence citation", "context validation")
- expected_improvement: brief explanation of why this variant may improve performance

Be concise, but ensure the new pair is meaningfully distinct from all existing prompts and tuned for tasks similar to the provided Q&A examples.
"""

PROMPT_GENERATION_USER = """User Prompt

Here are the reference system and user prompts provided by the user:
{prompt_samples}

Based on these, create a new prompt pair optimized for this task that differs from all the existing ones.
Respond strictly following this JSON schema:
{schema}
"""

QUESTION_REPHRASING_SYSTEM = """You are an expert at rephrasing questions to improve their effectiveness for question-answering systems.

Your task is to rephrase questions to:
1. Make them more sophisticated and less naive/simple while preserving the exact same context and intent
2. Potentially reduce token count if the question is overly verbose or wordy
3. Improve clarity and precision to help LLMs generate better answers
4. Maintain the semantic meaning - the rephrased question should seek the same information

CRITICAL GUIDELINES FOR ABBREVIATIONS AND DOMAIN TERMS:
- ALWAYS interpret abbreviations based on the domain context provided in the question
- For example: "FMs" in the context of "Amazon Bedrock" refers to "Foundation Models" (ML/AI), NOT "FM radio stations"
- Other examples: "AWS" = Amazon Web Services, "LLM" = Large Language Model, "RAG" = Retrieval-Augmented Generation
- If an abbreviation appears with domain-specific context (company names, product names, technical terms), interpret it within that domain
- DO NOT make assumptions about abbreviations without considering the surrounding context
- If you are uncertain about an abbreviation's meaning in context, PRESERVE the original abbreviation rather than guessing incorrectly
- Expand abbreviations only when you are confident about their meaning in the given context

General Guidelines:
- If the question is too simple or naive, make it more nuanced and specific
- If the question is too verbose, make it more concise without losing meaning
- Preserve all key concepts and context from the original question
- Do not change the fundamental intent or expected answer type
- Use more precise terminology when appropriate
- Keep the question natural and readable

Output only the rephrased question, nothing else."""

QUESTION_REPHRASING_USER = """Rephrase the following question to make it more effective for question-answering while preserving the exact same context and intent.

Original question: {question}

Rephrased question:"""


class GeneratedPrompt(BaseModel):
    id: int
    system_prompt: str
    user_prompt: str
    strategy_used: str
    expected_improvement: str


def parse_generated_prompts(data):
    """Parse generated prompt response from LLM.

    Args:
        data: String (JSON) or dict containing prompt data

    Returns:
        Dict with 'system_prompt' and 'user_prompt' keys

    Raises:
        ValueError: If data format is unexpected
    """
    if isinstance(data, str):
        data = json.loads(data)

    if isinstance(data, dict):
        return {
            "system_prompt": data.get("system_prompt", "").strip(),
            "user_prompt": data.get("user_prompt", "").strip(),
        }

    raise ValueError(f"Unexpected response format: {type(data)}")


def generate_prompts(
    llm: str, prompts: List[Dict], n: int = 2, api_key: str = None, base_url: str = None
):
    """Generate new prompt pairs using LLM based on existing prompts.

    Args:
        llm: Model name to use for generation
        prompts: List of existing prompt dictionaries
        n: Number of new prompts to generate
        api_key: Flotorch API key
        base_url: Flotorch base URL

    Returns:
        List of new prompt dictionaries with 'system_prompt' and 'user_prompt'

    Raises:
        RuntimeError: If prompt generation fails
    """
    if n <= 0:
        return []

    try:
        validate_environment(llm_models=[llm], api_key=api_key, base_url=base_url)
    except Exception as e:
        raise RuntimeError(f"Failed to validate environment: {e}") from e

    llm_model = setup_model(llm, api_key, base_url)

    prompt_struct = convert_pydantic_to_custom_json_schema(GeneratedPrompt)[
        "response_format"
    ]

    current_prompts = prompts.copy()
    new_prompts = []

    for iteration in range(1, n + 1):
        prompt_samples = ""
        for p in current_prompts:
            prompt_samples += "\n--- Prompt ---\n"
            prompt_samples += (
                f"System Prompt:\n{p.get('system_prompt', '').strip()}\n\n"
            )
            prompt_samples += f"User Prompt:\n{p.get('user_prompt', '').strip()}\n"

        user_prompt = PROMPT_GENERATION_USER.format(
            prompt_samples=prompt_samples, schema=json.dumps(prompt_struct, indent=2)
        )

        payload = [
            {"role": "system", "content": PROMPT_GENERATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = llm_model.invoke(messages=payload, response_format=prompt_struct)
            # Extract content from response object
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            new_prompt = parse_generated_prompts(content)
            new_prompts.append(new_prompt)
            current_prompts.append(new_prompt)
        except Exception as e:
            raise RuntimeError(
                f"Prompt generation failed at iteration {iteration}/{n}: {e}"
            ) from e

    return new_prompts


def rephrase_questions(
    ground_truth: List[Dict[str, str]],
    llm: str,
    api_key: str,
    base_url: str,
) -> List[Dict[str, str]]:
    """Rephrase questions in ground truth data using an LLM to improve their effectiveness.
    
    This function keeps all original questions and adds rephrased versions, resulting in
    a list with 2n items (n original + n rephrased) if there are n original items.
    Each item is marked with 'question_type' key set to either 'original' or 'rephrased'.
    Rephrased items maintain the same answer and context (if present) as their originals.
    """
    if not ground_truth:
        return []

    try:
        validate_environment(llm_models=[llm], api_key=api_key, base_url=base_url)
    except Exception as e:
        raise RuntimeError(f"Failed to validate environment: {e}") from e
        
    llm_model = setup_model(llm, api_key, base_url)

    result_gt = []
    successful_rephrases = 0

    for i, item in enumerate(ground_truth, 1):
        original_question = item.get("question", "")
        # Create a unique group ID to link original and rephrased questions
        question_group_id = f"group_{i}"

        # Add original item with 'question_type' marker and group ID
        original_item = item.copy()
        original_item["question_type"] = "original"
        original_item["question_group_id"] = question_group_id
        result_gt.append(original_item)

        if not original_question:
            print(f"Skipping rephrasing for item {i}: missing 'question' field")
            continue

        user_prompt = QUESTION_REPHRASING_USER.format(question=original_question)
        payload = [
            {"role": "system", "content": QUESTION_REPHRASING_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = llm_model.invoke(messages=payload)
            # Extract content from response object
            content = (
                response.content if hasattr(response, "content") else str(response)
            )
            rephrased_question = content.strip()

            if rephrased_question.startswith('"') and rephrased_question.endswith('"'):
                rephrased_question = rephrased_question[1:-1]
            if rephrased_question.startswith("'") and rephrased_question.endswith("'"):
                rephrased_question = rephrased_question[1:-1]

            # Create rephrased item with same answer and context, but new question
            rephrased_item = item.copy()
            rephrased_item["question"] = rephrased_question
            rephrased_item["question_type"] = "rephrased"
            rephrased_item["question_group_id"] = question_group_id
            result_gt.append(rephrased_item)
            successful_rephrases += 1

        except Exception as e:
            print(f"[ERROR] Failed to rephrase question {i}: {e}")
            # If rephrasing fails, we still have the original item in the result

    print(f"\nSuccessfully rephrased {successful_rephrases} out of {len(ground_truth)} questions")
    print(f"Total items in result: {len(result_gt)} ({len(ground_truth)} original + {successful_rephrases} rephrased)")
    return result_gt
