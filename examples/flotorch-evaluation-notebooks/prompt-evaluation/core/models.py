from typing import List, Optional
from flotorch.sdk.llm import FlotorchLLM
from flotorch.sdk.memory import FlotorchVectorStore
import httpx


def setup_model(model_name: str, api_key: str, base_url: str) -> FlotorchLLM:
    """Initialize a FlotorchLLM model."""
    return FlotorchLLM(api_key=api_key, base_url=base_url, model_id=model_name)


def create_embeddings(model_name: str, text_input, api_key: str, base_url: str):
    """Create embeddings for text input using Flotorch embedding API.

    Args:
        model_name: Name of the embedding model
        text_input: Single string or list of strings to embed
        api_key: Flotorch API key
        base_url: Flotorch base URL

    Returns:
        Dictionary containing embedding data from API response

    Raises:
        RuntimeError: If API request fails or response is invalid
    """
    if isinstance(text_input, str):
        text_input = [text_input]

    url = f"{base_url}/openai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {"model": model_name, "input": text_input}

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not isinstance(data["data"], list):
            raise RuntimeError("Invalid embedding response: missing 'data' field.")

        print(
            f"[OK] Received {len(data['data'])} embeddings from model '{model_name}'."
        )
        return data

    except httpx.RequestError as e:
        raise RuntimeError(f"Network error while requesting embeddings: {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Embedding API returned {e.response.status_code}: {e.response.text}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during embedding request: {e}") from e


def validate_environment(
    llm_models: List,
    api_key: str,
    base_url: str,
    embedding_models: Optional[List] = None,
    knowledge_base_id: str = "",
):
    """Validate that models and vectorstore (if provided) are reachable and working.

    Args:
        llm_models: List of LLM model names to validate
        api_key: Flotorch API key
        base_url: Flotorch base URL
        embedding_models: Optional list of embedding model names to validate
        knowledge_base_id: Optional knowledge base ID to validate

    Raises:
        RuntimeError: If validation fails for any component
    """
    print("Validating LLM model responses")
    try:
        for model_name in llm_models:
            model = setup_model(model_name, api_key, base_url)
            test_msg = [{"role": "user", "content": "ping"}]
            response = model.invoke(messages=test_msg)

            if not response or not hasattr(response, "content"):
                raise RuntimeError("Model response is empty or invalid structure.")

            print(f"[OK] Model '{model_name}' responded successfully.")
    except Exception as e:
        raise RuntimeError(f"Model validation failed: {e}") from e

    if embedding_models is None:
        embedding_models = []

    if embedding_models:
        print("\n[STEP] Validating embedding models...")
        try:
            for emb_model in embedding_models:
                resp = create_embeddings(
                    emb_model, "Test sentence for embedding", api_key, base_url
                )

                if "data" not in resp or not isinstance(resp["data"], list):
                    raise RuntimeError(
                        f"Embedding model '{emb_model}' returned invalid format (missing 'data')."
                    )

                if not resp["data"] or "embedding" not in resp["data"][0]:
                    raise RuntimeError(
                        f"Embedding model '{emb_model}' returned no embeddings."
                    )

                embedding_vector = resp["data"][0]["embedding"]
                if not isinstance(embedding_vector, list) or len(embedding_vector) == 0:
                    raise RuntimeError(
                        f"Embedding model '{emb_model}' returned empty embedding vector."
                    )

                print(
                    f"[OK] Embedding model '{emb_model}' produced vector of length {len(embedding_vector)}."
                )
        except Exception as e:
            raise RuntimeError(f"Embedding model validation failed: {e}") from e

    if knowledge_base_id:
        print("Validating vectorstore connectivity")
        try:
            kb = FlotorchVectorStore(
                api_key=api_key, base_url=base_url, vectorstore_id=knowledge_base_id
            )

            test_query = "test query"
            results = kb.search(query=test_query)

            if not results:
                raise RuntimeError("Vectorstore returned empty or invalid response.")

            print(f"[OK] Vectorstore '{knowledge_base_id}' responded successfully.")
        except Exception as e:
            raise RuntimeError(f"Vectorstore validation failed: {e}") from e

    print("[SUCCESS] All system checks passed.\n")
