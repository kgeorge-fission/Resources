from typing import List, Optional
from flotorch.sdk.memory import FlotorchAsyncVectorStore
from flotorch.sdk.utils import memory_utils


async def async_search_vectorstore(
    knowledge_base: FlotorchAsyncVectorStore,
    question: str,
    max_number_of_result: Optional[int] = None,
) -> List[str]:
    """Asynchronously searches the Knowledge base and returns extracted text chunks.

    Args:
        knowledge_base: The FlotorchAsyncVectorStore instance to search
        question: The query string to search for
        max_number_of_result: Maximum number of results to retrieve. If None, uses default (5).

    Returns:
        List of text chunks extracted from the search results.
    """
    try:
        search_kwargs = {"query": question}
        if max_number_of_result is not None:
            search_kwargs["max_number_of_result"] = max_number_of_result

        results = await knowledge_base.search(**search_kwargs)
        chunks = memory_utils.extract_vectorstore_texts(results)

        # extract_vectorstore_texts returns list[str], ensure it's a list
        if not isinstance(chunks, list):
            chunks = [str(chunks)] if chunks else []

        return chunks
    except Exception as e:
        print(f"[WARN] Knowledge base search failed for query '{question}': {e}")
        return []
