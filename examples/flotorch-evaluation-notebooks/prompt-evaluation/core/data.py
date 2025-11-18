from typing import Any, Dict, List


def convert_ground_truth_format(
    ground_truth: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert ground truth context to standardized list format.

    Handles various context formats:
    - None -> []
    - str -> [str]
    - list of dicts with 'chunk_text' -> list of strings
    - list of strings -> list of strings
    """
    converted = []
    for item in ground_truth:
        converted_item = item.copy()

        if "context" in converted_item:
            ctx = converted_item["context"]

            if ctx is None:
                converted_item["context"] = []
            elif isinstance(ctx, str):
                converted_item["context"] = [ctx]
            elif isinstance(ctx, list) and len(ctx) > 0:
                if isinstance(ctx[0], dict) and "chunk_text" in ctx[0]:
                    converted_item["context"] = [
                        chunk.get("chunk_text", "")
                        for chunk in ctx
                        if isinstance(chunk, dict) and chunk.get("chunk_text")
                    ]
                elif isinstance(ctx[0], str):
                    converted_item["context"] = ctx
                else:
                    converted_item["context"] = [str(chunk) for chunk in ctx if chunk]
            else:
                converted_item["context"] = []

        converted.append(converted_item)

    return converted
