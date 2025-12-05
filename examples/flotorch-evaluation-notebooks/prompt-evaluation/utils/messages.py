from typing import List, Optional


def create_messages(
    system_prompt: str,
    user_prompt: str,
    question: str,
    context: Optional[List[str]] = None,
    examples: Optional[List[dict]] = None,
    assembly_rule: Optional[dict] = None,
):
    """Assemble LLM messages using a user-defined assembly rule.

    Args:
        system_prompt: System prompt text
        user_prompt: User prompt text
        question: Question to include
        context: Optional list of context chunks
        examples: Optional list of example dicts with 'example' key
        assembly_rule: Optional dict defining how to combine components

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    DEFAULT_RULE = {
        "separator": "",
        "system_prompt": ["system", "context", "examples"],
        "user_prompt": ["user", "question"],
    }

    rule = assembly_rule if assembly_rule is not None else DEFAULT_RULE
    separator = rule.get("separator", "")

    context_text = ""
    if context:
        if isinstance(context, list):
            context_text = "\n\nContext:\n" + "\n\n---\n\n".join(context)
        else:
            context_text = "\n\nContext:\n" + str(context)

    examples_text = ""
    if examples:
        examples_text = "\nExamples:"
        for ex in examples:
            examples_text += "\n\nExample:" + ex["example"]

    components = {
        "system": system_prompt,
        "user": user_prompt,
        "context": context_text,
        "question": question,
        "examples": examples_text,
    }

    def assemble(parts):
        out = [components[p] for p in parts if components.get(p)]
        return separator.join(out)

    final_system = assemble(rule.get("system_prompt", []))
    final_user = assemble(rule.get("user_prompt", []))

    return [
        {"role": "system", "content": final_system},
        {"role": "user", "content": final_user},
    ]
