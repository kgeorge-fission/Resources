from typing import Any, Dict, List, Union
from collections import defaultdict
import json
from pathlib import Path
from flotorch_eval.llm_eval import EvaluationItem
from utils.schema import PromptEvaluationResult, EvaluationDatasetType
from utils.display import get_best_prompt_pair


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


def save_evaluation_results(
    results: PromptEvaluationResult,
    filepath: str,
    indent: int = 2
) -> str:
    """Save evaluation results to a JSON file.

    Args:
        results: The PromptEvaluationResult to save
        filepath: Path to the output JSON file
        indent: JSON indentation level (default: 2)

    Returns:
        The absolute path to the saved file

    Example:
        >>> results = run_evaluation(evaluation_data)
        >>> save_evaluation_results(results, "results.json")
        '/path/to/results.json'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=indent, ensure_ascii=False)

    return str(filepath.resolve())


def load_evaluation_results(filepath: str) -> PromptEvaluationResult:
    """Load evaluation results from a JSON file.

    Args:
        filepath: Path to the JSON file containing saved results

    Returns:
        The loaded PromptEvaluationResult

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON

    Example:
        >>> results = load_evaluation_results("results.json")
        >>> display_prompt_results(results)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results


def _evaluation_item_to_dict(item: EvaluationItem) -> Dict[str, Any]:
    """Convert an EvaluationItem to a dictionary for JSON serialization."""
    return {
        "question": item.question,
        "generated_answer": item.generated_answer,
        "expected_answer": item.expected_answer,
        "context": item.context,
        "metadata": item.metadata if hasattr(item, 'metadata') else {}
    }


def _dict_to_evaluation_item(data: Dict[str, Any]) -> EvaluationItem:
    """Convert a dictionary back to an EvaluationItem."""
    return EvaluationItem(
        question=data.get("question", ""),
        generated_answer=data.get("generated_answer", ""),
        expected_answer=data.get("expected_answer", ""),
        context=data.get("context", []),
        metadata=data.get("metadata", {})
    )


def save_evaluation_data(
    evaluation_data: Union[Dict[str, Any], EvaluationDatasetType],
    filepath: str,
    indent: int = 2
) -> str:
    """Save evaluation data from ExperimentRunner to a JSON file.

    This saves the expensive experiment data (containing EvaluationItem objects)
    so it can be loaded later to run evaluation without re-running experiments.

    Args:
        evaluation_data: The evaluation data from ExperimentRunner.run_async()
            Can be either a dict with 'runs' key or just the runs list
        filepath: Path to the output JSON file
        indent: JSON indentation level (default: 2)

    Returns:
        The absolute path to the saved file

    Example:
        >>> evaluation_data = await runner.run_async(concurrency=10)
        >>> save_evaluation_data(evaluation_data, "experiment_data.json")
        '/path/to/experiment_data.json'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Handle both dict format (with 'runs') and list format
    if isinstance(evaluation_data, dict):
        # Full format with units_count, warnings, runs
        serializable_data = {
            "units_count": evaluation_data.get("units_count", 0),
            "warnings": evaluation_data.get("warnings", []),
            "runs": []
        }
        runs = evaluation_data.get("runs", [])
    else:
        # Just the runs list
        serializable_data = {"runs": []}
        runs = evaluation_data

    # Convert EvaluationItem objects to dicts
    for run in runs:
        serializable_run = {
            "model": run.get("model", ""),
            "system_prompt": run.get("system_prompt", ""),
            "user_prompt": run.get("user_prompt", ""),
            "context_size": run.get("context_size"),
            "experiments": []
        }
        
        experiments = run.get("experiments", [])
        for item in experiments:
            if isinstance(item, EvaluationItem):
                serializable_run["experiments"].append(_evaluation_item_to_dict(item))
            else:
                # Already a dict
                serializable_run["experiments"].append(item)
        
        serializable_data["runs"].append(serializable_run)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=indent, ensure_ascii=False)

    return str(filepath.resolve())


def load_evaluation_data(filepath: str) -> Dict[str, Any]:
    """Load evaluation data from a JSON file and reconstruct EvaluationItem objects.

    This loads previously saved experiment data so you can run evaluation
    without re-running the expensive experiments.

    Args:
        filepath: Path to the JSON file containing saved evaluation data

    Returns:
        The loaded evaluation data in the same format as ExperimentRunner.run_async()
        Returns a dict with 'units_count', 'warnings', and 'runs' keys

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON

    Example:
        >>> evaluation_data = load_evaluation_data("experiment_data.json")
        >>> results = run_evaluation(evaluation_data)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Evaluation data file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Reconstruct EvaluationItem objects
    runs = []
    for run in data.get("runs", []):
        reconstructed_run = {
            "model": run.get("model", ""),
            "system_prompt": run.get("system_prompt", ""),
            "user_prompt": run.get("user_prompt", ""),
            "context_size": run.get("context_size"),
            "experiments": []
        }
        
        for item_dict in run.get("experiments", []):
            reconstructed_run["experiments"].append(_dict_to_evaluation_item(item_dict))
        
        runs.append(reconstructed_run)

    # Return in the same format as ExperimentRunner.run_async()
    return {
        "units_count": data.get("units_count", len(runs)),
        "warnings": data.get("warnings", []),
        "runs": runs
    }


def save_best_prompt_pair(
    results: PromptEvaluationResult,
    filename: str = "best_prompt_pair.json"
) -> str:
    """Save the best performing prompt pair to disk based on average_score.
    
    Args:
        results: List of evaluation result dicts
        filename: Name of the file to save (default: "best_prompt_pair.json")
    
    Returns:
        The absolute path to the saved file
    
    Example:
        >>> results = run_evaluation(evaluation_data)
        >>> save_best_prompt_pair(results)
        '/path/to/best_prompt_pair.json'
    """
    if not results:
        raise ValueError("No results provided")
    
    # Find the best result (uses weighted_final_score if available, else average_score)
    best_result = get_best_prompt_pair(results)
    
    # Format as the original prompt input format
    prompt_pair = [{
        "system_prompt": best_result.get("system_prompt", ""),
        "user_prompt": best_result.get("user_prompt", "")
    }]
    
    filepath = Path(filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(prompt_pair, f, indent=2, ensure_ascii=False)
    
    metrics = best_result.get('evaluation_metrics', {})
    weighted_available = "weighted_final_score" in metrics
    key_metric = "weighted_final_score" if weighted_available else "average_score"
    
    print(f"Saved best prompt pair to {filepath.resolve()}")
    print(f"  {key_metric.replace('_', ' ').title()}: {metrics.get(key_metric, 'N/A')}")
    if weighted_available:
        print(f"  Average Score: {metrics.get('average_score', 'N/A')}")
    print(f"  Model: {best_result.get('inference_model', 'N/A')}")
    print(f"  Context Size: {best_result.get('context_size', 'N/A')}")
    
    return str(filepath.resolve())


def save_best_ground_truth(
    evaluation_data: Union[Dict[str, Any], EvaluationDatasetType],
    results: PromptEvaluationResult,
    filename: str = "best_ground_truth.json"
) -> str:
    """Save ground truth items, keeping only the better performing version
    between original and rephrased questions based on per-question metrics.
    
    Uses question_level_results from evaluation results to compute actual per-question
    scores. For questions with question_type "original" or "rephrased", compares their
    performance across all prompt pairs and saves only the better one.
    All saved items include a question_type parameter indicating "original" or "rephrased".
    
    Args:
        evaluation_data: Evaluation data dict with 'runs' key containing experiments
        results: List of evaluation result dicts with question_level_results
        filename: Name of the file to save (default: "best_ground_truth.json")
    
    Returns:
        The absolute path to the saved file
    
    Example:
        >>> evaluation_data = await runner.run_async()
        >>> results = run_evaluation(evaluation_data)
        >>> save_best_ground_truth(evaluation_data, results)
        '/path/to/best_ground_truth.json'
    """
    # Extract runs from evaluation_data
    if isinstance(evaluation_data, dict):
        data_runs = evaluation_data.get("runs", [])
    elif isinstance(evaluation_data, list):
        data_runs = evaluation_data
    else:
        raise ValueError("Invalid evaluation_data format")
    
    # Build question to ground truth item mapping
    question_to_gt = {}
    question_groups = defaultdict(list)  # group_id -> list of (question, gt_item)
    
    for run in data_runs:
        experiments = run.get("experiments", [])
        for item in experiments:
            if hasattr(item, "question"):
                question = item.question
                expected_answer = item.expected_answer
                context = item.context
                metadata = item.metadata if hasattr(item, "metadata") else {}
            elif isinstance(item, dict):
                question = item.get("question", "")
                expected_answer = item.get("expected_answer", "")
                context = item.get("context", [])
                metadata = item.get("metadata", {})
            else:
                continue
            
            question_type = metadata.get("question_type") if isinstance(metadata, dict) else None
            question_group_id = metadata.get("question_group_id") if isinstance(metadata, dict) else None
            
            # Create ground truth item
            gt_item = {
                "question": question,
                "answer": expected_answer,
                "context": context
            }
            
            # Always include question_type (default to "original" if not specified)
            if question_type:
                gt_item["question_type"] = question_type
            else:
                gt_item["question_type"] = "original"
            
            if question_group_id:
                gt_item["question_group_id"] = question_group_id
            
            question_to_gt[question] = gt_item
            
            if question_group_id:
                question_groups[question_group_id].append((question, gt_item))
    
    # Compute average score per question using question_level_results
    question_scores = defaultdict(list)
    
    for result in results:
        question_level_results = result.get("question_level_results", [])
        
        if not question_level_results:
            # Fallback: if question_level_results not available, skip this result
            continue
        
        for q_result in question_level_results:
            question = q_result.get("question", "")
            if not question:
                continue
            
            metrics = q_result.get("metrics", {})
            if not metrics:
                continue
            
            # Compute average score from all available metrics
            metric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
            if metric_values:
                avg_score = sum(metric_values) / len(metric_values)
                question_scores[question].append(avg_score)
    
    # Compute average score per question across all prompt pairs
    question_avg_scores = {
        q: sum(scores) / len(scores) if scores else 0
        for q, scores in question_scores.items()
    }
    
    # Build best ground truth items
    best_gt_items = []
    processed_groups = set()
    
    for question, gt_item in question_to_gt.items():
        question_group_id = gt_item.get("question_group_id")
        
        if question_group_id and question_group_id not in processed_groups:
            # This is a group with original/rephrased pairs
            group_items = question_groups[question_group_id]
            if len(group_items) >= 2:
                # Compare scores and keep the better one
                best_item = None
                best_score = -1
                best_question_type = None
                
                for q, item in group_items:
                    score = question_avg_scores.get(q, 0)
                    if score > best_score:
                        best_score = score
                        best_item = item
                        best_question_type = item.get("question_type", "original")
                
                if best_item:
                    # Remove question_group_id but keep question_type
                    clean_item = {
                        "question": best_item["question"],
                        "answer": best_item["answer"],
                        "context": best_item["context"],
                        "question_type": best_question_type
                    }
                    best_gt_items.append(clean_item)
                
                processed_groups.add(question_group_id)
            elif len(group_items) == 1:
                # Only one item in group (original only), keep it
                item = group_items[0][1]
                clean_item = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "context": item["context"],
                    "question_type": item.get("question_type", "original")
                }
                best_gt_items.append(clean_item)
                processed_groups.add(question_group_id)
        elif not question_group_id:
            # No group ID, keep as-is (single question, no rephrased version)
            clean_item = {
                "question": gt_item["question"],
                "answer": gt_item["answer"],
                "context": gt_item["context"],
                "question_type": gt_item.get("question_type", "original")
            }
            best_gt_items.append(clean_item)
    
    # Save to file
    filepath = Path(filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(best_gt_items, f, indent=2, ensure_ascii=False)
    
    print(f"Saved best ground truth items to {filepath.resolve()}")
    print(f"  Total items: {len(best_gt_items)}")
    
    return str(filepath.resolve())
