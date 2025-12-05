from typing import List, Dict, Optional, Any
from collections import defaultdict
import textwrap
from tabulate import tabulate


def normalize(values: List[float], invert: bool = False) -> List[float]:
    """Min-max normalize a list of numeric values to a 0-1 range."""
    if not values:
        return []

    min_v, max_v = min(values), max(values)
    if max_v == min_v:
        return [1.0 for _ in values]

    if invert:
        return [(max_v - v) / (max_v - min_v) for v in values]
    else:
        return [(v - min_v) / (max_v - min_v) for v in values]


def get_weighted_scores(result: List[dict], weights: Dict[str, float]) -> List[dict]:
    """Calculate weighted final scores for evaluation results.

    Args:
        result: List of evaluation result dicts
        weights: Dict mapping metric names to weights

    Returns:
        List of results with 'weighted_final_score' added to evaluation_metrics
    """
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    latencies = [
        item["evaluation_metrics"].get("average_latency_ms", 0) for item in result
    ]
    costs = [item["evaluation_metrics"].get("average_cost", 0) for item in result]
    scores = [item["evaluation_metrics"].get("average_score", 0) for item in result]

    norm_latencies = normalize(latencies, invert=True)
    norm_costs = normalize(costs, invert=True)
    norm_scores = normalize(scores, invert=False)

    for i, item in enumerate(result):
        weighted_score = (
            weights["average_score"] * norm_scores[i]
            + weights["average_latency_ms"] * norm_latencies[i]
            + weights["average_cost"] * norm_costs[i]
        )
        item["evaluation_metrics"]["weighted_final_score"] = round(weighted_score, 3)

    return result


def get_best_prompt_pair(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Find the best performing prompt pair from evaluation results.
    
    Uses weighted_final_score if available, otherwise falls back to average_score.
    
    Args:
        results: List of evaluation result dicts
    
    Returns:
        The best performing result dict
    
    Example:
        >>> results = run_evaluation(evaluation_data)
        >>> best = get_best_prompt_pair(results)
        >>> print(best['system_prompt'])
    """
    if not results:
        raise ValueError("No results provided")
    
    # Check if weighted score is available
    weighted_available = any(
        "weighted_final_score" in r.get("evaluation_metrics", {}) for r in results
    )
    
    # Use weighted score if available, otherwise use average_score
    key_metric = "weighted_final_score" if weighted_available else "average_score"
    
    best_result = max(
        results,
        key=lambda x: x.get("evaluation_metrics", {}).get(key_metric, 0)
    )
    
    return best_result


def _display_question_type_comparison(
    evaluation_data: Dict[str, Any], results: List[Dict[str, Any]]
):
    """Display comparison between original and rephrased questions.
    
    Args:
        evaluation_data: Evaluation data dict with 'runs' key containing experiments,
                         or a list of runs
        results: List of evaluation result dicts
    """
    try:
        # Handle both dict with 'runs' key and direct list
        if isinstance(evaluation_data, dict):
            data_runs = evaluation_data.get("runs", [])
            if not data_runs and isinstance(evaluation_data, list):
                data_runs = evaluation_data
        elif isinstance(evaluation_data, list):
            data_runs = evaluation_data
        else:
            return
        
        # Extract question_type information from experiments
        question_type_stats = defaultdict(int)
        
        for run in data_runs:
            experiments = run.get("experiments", [])
            for item in experiments:
                # Handle both EvaluationItem objects and dicts
                if hasattr(item, "metadata"):
                    metadata = item.metadata
                elif isinstance(item, dict):
                    metadata = item.get("metadata", {})
                else:
                    continue
                
                question_type = metadata.get("question_type") if isinstance(metadata, dict) else None
                if question_type:
                    question_type_stats[question_type] += 1
        
        # Only display if we found question_type information
        if question_type_stats:
            print("\n" + "=" * 80)
            print("QUESTION TYPE BREAKDOWN")
            print("=" * 80)
            
            total_count = sum(question_type_stats.values())
            breakdown_table = []
            for q_type in sorted(question_type_stats.keys()):
                count = question_type_stats[q_type]
                percentage = (count / total_count * 100) if total_count > 0 else 0
                breakdown_table.append([
                    q_type.capitalize(),
                    count,
                    f"{percentage:.1f}%"
                ])
            
            headers = ["Question Type", "Count", "Percentage"]
            print(tabulate(breakdown_table, headers=headers, tablefmt="fancy_grid"))
            print("=" * 80)
    except Exception:
        # Silently fail if there's an issue extracting question_type info
        pass


def display_prompt_results(
    results,
    sort_by: str = "average_score",
    descending: bool = True,
    wrap_width: int = 60,
    show_summary: bool = True,
    show_comparison: bool = True,
    show_top_n: int = 5,
    evaluation_data: Optional[Dict[str, Any]] = None,
):
    """Display model-prompt evaluations in formatted tables.

    Args:
        results: List of evaluation result dicts
        sort_by: Metric to sort by (default: 'average_score')
        descending: Sort in descending order
        wrap_width: Width for text wrapping
        show_summary: Show experiment summary
        show_comparison: Show context size comparison
        show_top_n: Number of top results to display
        evaluation_data: Optional evaluation data dict with 'runs' key containing experiments
                        Used to extract question_type breakdown (original vs rephrased)
    """
    if not results:
        print("No results to display.")
        return

    # Auto-switch sorting to weighted score if present
    weighted_available = any(
        "weighted_final_score" in r.get("evaluation_metrics", {}) for r in results
    )
    if weighted_available and sort_by == "average_score":
        sort_by = "weighted_final_score"

    if show_summary:
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"\nTotal Number of experiments: {len(results)}")
        print(
            f"Models tested: {len(set(r.get('inference_model', 'N/A') for r in results))}"
        )
        print(
            f"Prompts tested: {len(set((r.get('system_prompt', ''), r.get('user_prompt', '')) for r in results))}"
        )

        context_sizes = sorted(
            set(
                r.get("context_size")
                for r in results
                if r.get("context_size") is not None
            )
        )
        if context_sizes:
            print(f"Context sizes tested: {context_sizes}")

    if (
        show_comparison
        and len(
            set(
                r.get("context_size")
                for r in results
                if r.get("context_size") is not None
            )
        )
        > 1
    ):
        print("\n" + "=" * 80)
        print("COMPARISON BY CONTEXT SIZE")
        print("=" * 80)

        results_by_context = defaultdict(list)
        for r in results:
            ctx_size = r.get("context_size", "All")
            results_by_context[ctx_size].append(r)

        comparison_table = []
        for ctx_size in sorted(
            results_by_context.keys(), key=lambda x: x if isinstance(x, int) else 999
        ):
            ctx_results = results_by_context[ctx_size]
            metrics_list = [r["evaluation_metrics"] for r in ctx_results]

            avg_score = sum(m.get("average_score", 0) for m in metrics_list) / len(
                metrics_list
            )
            avg_latency = sum(
                m.get("average_latency_ms", 0) for m in metrics_list
            ) / len(metrics_list)
            total_cost = sum(m.get("total_cost", 0) for m in metrics_list)
            avg_answer_rel = sum(
                m.get("answer_relevancy", 0) for m in metrics_list
            ) / len(metrics_list)
            avg_faithfulness = sum(
                m.get("faithfulness", 0) for m in metrics_list
            ) / len(metrics_list)

            weighted_avg = (
                sum(m.get("weighted_final_score", 0) for m in metrics_list)
                / len(metrics_list)
                if weighted_available
                else "-"
            )

            comparison_table.append(
                [
                    ctx_size,
                    len(ctx_results),
                    f"{avg_score:.3f}",
                    f"{avg_answer_rel:.3f}",
                    f"{avg_faithfulness:.3f}",
                    f"{avg_latency:.2f}",
                    f"{total_cost:.6f}",
                    weighted_avg,
                ]
            )

        headers = [
            "Context Size",
            "LLM Calls",
            "Avg Score",
            "Avg Answer Rel",
            "Avg Faithfulness",
            "Avg Latency (ms)",
            "Total Cost (USD)",
            "Weighted Score",
        ]
        print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))
        print("=" * 80)

    # Show question_type breakdown if evaluation_data is provided
    if evaluation_data:
        _display_question_type_comparison(evaluation_data, results)

    if show_top_n > 0:
        print("\n" + "=" * 80)
        print(f"TOP {show_top_n} PERFORMING CONFIGURATIONS")
        print("(Automatically uses weighted score if available)")
        print("=" * 80)

        key_metric = "weighted_final_score" if weighted_available else "average_score"
        sorted_results = sorted(
            results,
            key=lambda x: x["evaluation_metrics"].get(key_metric, 0),
            reverse=True,
        )
        top_results = sorted_results[:show_top_n]

        top_table = []
        for i, r in enumerate(top_results, 1):
            m = r["evaluation_metrics"]
            top_table.append(
                [
                    i,
                    r.get("inference_model", "N/A"),
                    r.get("context_size", "N/A"),
                    f"{m.get('average_score', 0):.3f}",
                    f"{m.get('weighted_final_score', '-')}",
                    f"{m.get('answer_relevancy', 0):.3f}",
                    f"{m.get('faithfulness', 0):.3f}",
                    f"{m.get('average_latency_ms', 0):.2f}",
                    f"{m.get('total_cost', 0):.6f}",
                    textwrap.shorten(r.get("system_prompt", ""), width=50),
                    textwrap.shorten(r.get("user_prompt", ""), width=50),
                ]
            )

        headers = [
            "Rank",
            "Model",
            "Context Size",
            "Avg Score",
            "Weighted Score",
            "Answer Rel",
            "Faithfulness",
            "Latency (ms)",
            "Cost (USD)",
            "System Prompt",
            "User Prompt",
        ]
        print(tabulate(top_table, headers=headers, tablefmt="grid"))
        print("=" * 80)

    table = []
    for i, r in enumerate(results, 1):
        m = r["evaluation_metrics"]

        table.append(
            [
                i,
                r.get("inference_model", "N/A"),
                "\n".join(textwrap.wrap(r.get("system_prompt", ""), width=wrap_width)),
                "\n".join(textwrap.wrap(r.get("user_prompt", ""), width=wrap_width)),
                r.get("context_size", "N/A"),
                round(m.get("llm_context_precision_with_reference", 0), 2),
                round(m.get("faithfulness", 0), 2),
                round(m.get("answer_relevancy", 0), 2),
                round(m.get("contextual_relevancy", 0), 2),
                round(m.get("contextual_recall", 0), 2),
                round(m.get("hallucination", 0), 2),
                round(m.get("maliciousness", 0), 2),
                round(m.get("average_score", 0), 2),
                round(m.get("average_latency_ms", 0), 2),
                round(m.get("total_latency_ms", 0), 2),
                m.get("total_tokens", 0),
                round(m.get("average_cost", 0), 6),
                round(m.get("total_cost", 0), 6),
                m.get("weighted_final_score", "-"),
            ]
        )

    headers = [
        "#",
        "Model",
        "System Prompt",
        "User Prompt",
        "Context Size",
        "Context Precision",
        "Faithfulness",
        "Answer Relevancy",
        "Contextual Relevancy",
        "Contextual Recall",
        "Hallucination",
        "Maliciousness",
        "Average Score",
        "Avg Latency (ms)",
        "Total Latency (ms)",
        "Total Tokens",
        "Avg Cost (USD)",
        "Total Cost (USD)",
        "Weighted Score",
    ]

    metric_map = {
        "context_precision": 5,
        "faithfulness": 6,
        "answer_relevancy": 7,
        "contextual_relevancy": 8,
        "contextual_recall": 9,
        "hallucination": 10,
        "maliciousness": 11,
        "average_score": 12,
        "average_latency_ms": 13,
        "total_latency_ms": 14,
        "total_tokens": 15,
        "average_cost": 16,
        "total_cost": 17,
        "weighted_final_score": 18,
    }

    metric_idx = metric_map[sort_by]
    table.sort(key=lambda x: x[metric_idx], reverse=descending)

    print("\n" + "=" * 80)
    print("MODEL-PROMPT EVALUATION SUMMARY")
    print("=" * 80)
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    best_item = get_best_prompt_pair(results)
    m = best_item["evaluation_metrics"]
    best_key = "weighted_final_score" if weighted_available else "average_score"

    print(
        f"\n{'=' * 80}\n"
        f"BEST PERFORMING CONFIGURATION\n"
        f"{'=' * 80}\n"
        f"   • Model: {best_item.get('inference_model')}\n"
        f"   • Context Size: {best_item.get('context_size')}\n"
        f"   • {best_key}: {m.get(best_key)}\n"
        f"   • Average Score: {m.get('average_score')}\n"
        f"   • Answer Relevancy: {m.get('answer_relevancy')}\n"
        f"   • Faithfulness: {m.get('faithfulness')}\n"
        f"   • Context Precision: {m.get('llm_context_precision_with_reference')}\n"
        f"   • Contextual Recall: {m.get('contextual_recall')}\n"
        f"   • Avg Latency (ms): {m.get('average_latency_ms')}\n"
        f"   • Total Tokens: {m.get('total_tokens')}\n"
        f"   • Total Cost (USD): {m.get('total_cost')}\n\n"
        f"   • System Prompt:\n{textwrap.fill(best_item.get('system_prompt', ''), width=80)}\n\n"
        f"   • User Prompt:\n{textwrap.fill(best_item.get('user_prompt', ''), width=80)}\n"
        f"{'=' * 80}"
    )


def best_prompt_pair(results):
    """Find and print the highest scoring prompt+model combination.

    Args:
        results: List of evaluation result dicts
    """
    best_prompt_set = get_best_prompt_pair(results)
    m = best_prompt_set["evaluation_metrics"]

    print("\n Best Performing Model-Prompt Combination:")
    print(f"   • Model: {best_prompt_set.get('inference_model', 'N/A')}")
    print(f"   • Context Size: {best_prompt_set.get('context_size', 'N/A')}")
    print(f"   • Average Score: {m.get('average_score', 'N/A')}")
    print(f"   • Faithfulness: {m.get('faithfulness', 'N/A')}")
    print(f"   • Answer Relevance: {m.get('answer_relevancy', 'N/A')}")
    print(
        f"   • Context Precision: {m.get('llm_context_precision_with_reference', 'N/A')}"
    )
    print(f"   • Contextual Relevancy: {m.get('contextual_relevancy', 'N/A')}")
    print(f"   • Contextual Recall: {m.get('contextual_recall', 'N/A')}")
    print(f"   • Hallucination: {m.get('hallucination', 'N/A')}")
    print(f"   • Maliciousness: {m.get('maliciousness', 'N/A')}")
    print(f"   • Avg Latency (ms): {m.get('average_latency_ms', 'N/A')}")
    print(f"   • Total Latency (ms): {m.get('total_latency_ms', 'N/A')}")
    print(f"   • Total Tokens: {m.get('total_tokens', 'N/A')}")
    print(f"   • Total Cost (USD): {m.get('total_cost', 'N/A')}\n")
    print(
        f"   • System Prompt:\n{textwrap.fill(best_prompt_set.get('system_prompt', ''), width=80)}"
    )
    print(
        f"\n   • User Prompt:\n{textwrap.fill(best_prompt_set.get('user_prompt', ''), width=80)}"
    )
