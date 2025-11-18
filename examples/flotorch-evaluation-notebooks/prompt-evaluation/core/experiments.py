import asyncio
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from flotorch_eval.llm_eval import EvaluationItem
from flotorch.sdk.memory import FlotorchAsyncVectorStore
from flotorch.sdk.llm import FlotorchLLM

from core.memory import async_search_vectorstore
from core.messages import create_messages


@dataclass
class ExperimentUnit:
    """Single experiment configuration."""

    model_name: str
    model_obj: FlotorchLLM
    system_prompt: str
    user_prompt: str
    question: str
    expected_answer: str
    examples: Optional[List[Dict[str, str]]]
    context_size: Optional[int]
    context_chunks: List[str]
    assembly_rule: Optional[Dict] = None


class ContextProvider:
    """Retrieves and selects context chunks for questions."""

    def __init__(
        self,
        ground_truth: List[Dict[str, Any]],
        knowledge_base: Optional[FlotorchAsyncVectorStore] = None,
        strategy: str = "top",
        random_seed: Optional[int] = None,
        max_results: Optional[int] = None,
    ):
        self.ground_truth_map = {
            item.get("question", ""): item for item in ground_truth
        }
        self.knowledge_base = knowledge_base
        self.strategy = strategy
        self.max_results = max_results
        if random_seed is not None:
            random.seed(random_seed)

    async def get_context(self, question: str, size: Optional[int] = None) -> List[str]:
        """Get context for a question. Returns up to `size` chunks (or all if size is None).

        Priority order:
        1. Context from ground_truth (if present) - takes precedence
        2. Context from knowledge_base (if available)
        3. Empty list (if neither available)
        """
        gt_item = self.ground_truth_map.get(question)

        # Check if context is explicitly provided in ground_truth (takes precedence)
        if gt_item is not None and "context" in gt_item:
            ctx = gt_item.get("context")
            # Handle different context formats
            if ctx is None:
                chunks = []
            elif isinstance(ctx, str):
                chunks = [ctx] if ctx else []
            elif isinstance(ctx, list):
                chunks = list(ctx)
            else:
                chunks = [str(ctx)]
        # Fall back to knowledge base only if ground_truth doesn't have context
        elif self.knowledge_base:
            try:
                chunks = (
                    await async_search_vectorstore(
                        self.knowledge_base,
                        question,
                        max_number_of_result=self.max_results,
                    )
                    or []
                )
            except Exception as e:
                print(f"[WARN] KB search failed for '{question}': {e}")
                chunks = []
        else:
            chunks = []

        if not chunks or size is None:
            return chunks

        if size <= 0 or size >= len(chunks):
            return chunks

        if self.strategy == "random":
            return random.sample(chunks, size)
        else:
            return chunks[:size]


class ExperimentRunner:
    """Runs experiments across models, prompts, and questions with context experimentation."""

    def __init__(
        self,
        models: List[str],
        prompts: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        api_key: str,
        base_url: str,
        knowledge_base: Optional[str] = None,
        context_sizes: Optional[List[int]] = None,
        context_strategy: str = "top",
        random_seed: Optional[int] = None,
        assembly_rule: Optional[Dict] = None,
        n: Optional[int] = None,
    ):
        self.models = models
        self.prompts = prompts
        self.ground_truth = ground_truth
        self.api_key = api_key
        self.base_url = base_url
        self.context_sizes = context_sizes
        self.assembly_rule = assembly_rule
        self.n = n

        kb_obj = None
        if knowledge_base:
            kb_obj = FlotorchAsyncVectorStore(
                api_key=api_key, base_url=base_url, vectorstore_id=knowledge_base
            )

        # Calculate max_results from context_sizes to ensure we retrieve enough chunks
        max_results = None
        if context_sizes:
            max_results = max(context_sizes)

        self.context_provider = ContextProvider(
            ground_truth=ground_truth,
            knowledge_base=kb_obj,
            strategy=context_strategy,
            random_seed=random_seed,
            max_results=max_results,
        )

    async def _build_units(self) -> Tuple[List[ExperimentUnit], List[str]]:
        """Build all experiment units. Returns (units, warnings)."""
        units = []
        warnings = []
        model_objects = {
            name: FlotorchLLM(name, self.api_key, self.base_url) for name in self.models
        }

        for model_name, model_obj in model_objects.items():
            for prompt in self.prompts:
                for qa in self.ground_truth:
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")

                    sizes_to_test = self.context_sizes if self.context_sizes else [None]
                    full_context = await self.context_provider.get_context(
                        question, None
                    )

                    for size in sizes_to_test:
                        if size is not None and len(full_context) < size:
                            warning = f"Skipped context size {size} for question '{question}': only {len(full_context)} context chunks available."
                            warnings.append(warning)
                            continue

                        if size is None:
                            context_chunks = full_context
                        elif self.context_provider.strategy == "top":
                            context_chunks = full_context[:size]
                        else:
                            context_chunks = random.sample(full_context, size)

                        # Handle n-shot examples
                        prompt_examples = prompt.get("examples", None)
                        selected_examples = None

                        if prompt_examples and isinstance(prompt_examples, list):
                            if self.n is not None and self.n > 0:
                                available_count = len(prompt_examples)
                                if available_count >= self.n:
                                    # Randomly sample n examples
                                    selected_examples = random.sample(
                                        prompt_examples, self.n
                                    )
                                else:
                                    # Use all available examples and warn
                                    selected_examples = prompt_examples
                                    warning = f"Question '{question}': Requested {self.n} examples but only {available_count} available. Using all {available_count} examples."
                                    warnings.append(warning)
                            else:
                                # If n is not specified, use all examples
                                selected_examples = prompt_examples

                        unit = ExperimentUnit(
                            model_name=model_name,
                            model_obj=model_obj,
                            system_prompt=prompt.get("system_prompt", ""),
                            user_prompt=prompt.get("user_prompt", ""),
                            question=question,
                            expected_answer=answer,
                            examples=selected_examples,
                            context_size=size,
                            context_chunks=context_chunks,
                            assembly_rule=self.assembly_rule,
                        )
                        units.append(unit)

        return units, warnings

    async def _run_unit(
        self, unit: ExperimentUnit, semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Execute a single experiment unit."""
        async with semaphore:
            # Examples are already filtered in _build_units
            message_input = {
                "system_prompt": unit.system_prompt,
                "user_prompt": unit.user_prompt,
                "question": unit.question,
                "assembly_rule": unit.assembly_rule,
            }
            if unit.examples:
                message_input["examples"] = unit.examples
            if unit.context_chunks:
                message_input["context"] = unit.context_chunks

            messages = create_messages(**message_input)

            try:
                response, headers = await unit.model_obj.ainvoke(
                    messages=messages,
                    return_headers=True,
                    assembly_rule=unit.assembly_rule,
                )
                result = EvaluationItem(
                    question=unit.question,
                    generated_answer=response.content,
                    expected_answer=unit.expected_answer,
                    context=unit.context_chunks,
                    metadata=headers,
                )
            except Exception as e:
                result = EvaluationItem(
                    question=unit.question,
                    generated_answer="",
                    expected_answer=unit.expected_answer,
                    context=unit.context_chunks,
                    metadata={"error": str(e)},
                )

            return {"unit": unit, "result": result}

    async def run_async(self, concurrency: int = 10) -> Dict[str, Any]:
        """Run all experiments asynchronously.

        Args:
            concurrency: Maximum number of concurrent experiments

        Returns:
            Dict with 'units_count', 'warnings', and 'runs' keys
        """
        units, warnings = await self._build_units()
        semaphore = asyncio.Semaphore(concurrency)
        tasks = [self._run_unit(u, semaphore) for u in units]
        runs = await asyncio.gather(*tasks, return_exceptions=False)

        return {"units_count": len(units), "warnings": warnings, "runs": runs}

    def run(self, concurrency: int = 10) -> Dict[str, Any]:
        """Synchronous wrapper for run_async.

        Args:
            concurrency: Maximum number of concurrent experiments

        Returns:
            Dict with 'units_count', 'warnings', and 'runs' keys
        """
        return asyncio.run(self.run_async(concurrency=concurrency))
