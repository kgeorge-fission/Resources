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
        """Builds one ExperimentUnit per (model x prompt pair x context size),
        and stores context for ALL questions inside that unit.
        """
        units = []
        warnings = []

        # Build all model objects once
        model_objects = {
            name: FlotorchLLM(name, self.api_key, self.base_url)
            for name in self.models
        }

        # MAIN LOOPS: model × prompt × context-size
        for model_name, model_obj in model_objects.items():
            for prompt in self.prompts:

                # Handle n-shot example subset
                prompt_examples = prompt.get("examples", None)
                selected_examples = None

                if prompt_examples and isinstance(prompt_examples, list):
                    if self.n is not None and self.n > 0:
                        available_count = len(prompt_examples)
                        if available_count >= self.n:
                            selected_examples = random.sample(prompt_examples, self.n)
                        else:
                            selected_examples = prompt_examples
                            warnings.append(
                                f"Requested {self.n} examples but only {available_count} available; using all."
                            )
                    else:
                        selected_examples = prompt_examples

                # Iterate over all context sizes (or one None)
                for size in (self.context_sizes or [None]):

                    context_map = {}   # {question: [context_chunks]}
                    skip_unit = False

                    # Get context for ALL QUESTIONS
                    for qa in self.ground_truth:
                        question = qa.get("question", "")

                        full_context = await self.context_provider.get_context(question, None)

                        # If insufficient context, skip this whole unit
                        if size is not None and len(full_context) < size:
                            warnings.append(
                                f"Skipped context size {size} for model '{model_name}', "
                                f"prompt '{prompt.get('user_prompt','')[:30]}...' "
                                f"because question '{question}' has only {len(full_context)} chunks."
                            )
                            skip_unit = True
                            break

                        # Choose chunks based on strategy
                        if size is None:
                            chosen = full_context
                        elif self.context_provider.strategy == "top":
                            chosen = full_context[:size]
                        else:
                            chosen = random.sample(full_context, size)

                        context_map[question] = chosen

                    if skip_unit:
                        continue

                    unit = ExperimentUnit(
                        model_name=model_name,
                        model_obj=model_obj,
                        system_prompt=prompt.get("system_prompt", ""),
                        user_prompt=prompt.get("user_prompt", ""),
                        question=None,
                        expected_answer=None,
                        examples=selected_examples,
                        context_size=size,
                        context_chunks=context_map,
                        assembly_rule=self.assembly_rule,
                    )

                    units.append(unit)

        return units, warnings

    async def _run_unit(
        self,
        unit: ExperimentUnit,
        semaphore: asyncio.Semaphore
    ) -> Dict[str, Any]:
        """Executes one experiment unit, running ALL questions and returning
        the exact output format needed by the evaluator.
        """
        async with semaphore:
            experiments = []

            for qa in self.ground_truth:
                question = qa.get("question", "")
                expected = qa.get("answer", "")
                context_chunks = unit.context_chunks.get(question, [])

                message_input = {
                    "system_prompt": unit.system_prompt,
                    "user_prompt": unit.user_prompt,
                    "question": question,
                    "assembly_rule": unit.assembly_rule,
                }

                if unit.examples:
                    message_input["examples"] = unit.examples

                if context_chunks:
                    message_input["context"] = context_chunks

                messages = create_messages(**message_input)

                try:
                    response, headers = await unit.model_obj.ainvoke(
                        messages=messages,
                        return_headers=True,
                        assembly_rule=unit.assembly_rule,
                    )
                    item = EvaluationItem(
                        question=question,
                        generated_answer=response.content,
                        expected_answer=expected,
                        context=context_chunks,
                        metadata=headers,
                    )

                except Exception as e:
                    item = EvaluationItem(
                        question=question,
                        generated_answer="",
                        expected_answer=expected,
                        context=context_chunks,
                        metadata={"error": str(e)},
                    )

                experiments.append(item)

            return {
                "model": unit.model_name,
                "system_prompt": unit.system_prompt,
                "user_prompt": unit.user_prompt,
                "context_size": unit.context_size,
                "experiments": experiments,
            }

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
