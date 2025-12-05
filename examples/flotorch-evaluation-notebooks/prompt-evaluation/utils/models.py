from typing import List, Optional, Union, Any
from flotorch.sdk.llm import FlotorchLLM
from flotorch.sdk.memory import FlotorchVectorStore
import httpx
import json


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

def _inspect_response_error(response_data: Union[dict, Any], component_name: str) -> None:
    """
    Parses response data for specific API Gateway error patterns.
    """
    if not response_data:
        return

    data = response_data
    if hasattr(response_data, "json") and callable(response_data.json):
        try:
            data = response_data.json()
        except:
            pass 
    elif hasattr(response_data, "__dict__"):
        data = response_data.__dict__

    if not isinstance(data, dict):
        return

    if "statusCode" in data and data.get("error") is True:
        code = data.get("statusCode")
        msg = data.get("message", "Unknown Error")
        
        if code == 401:
            raise ValueError(f"Authentication Failed for {component_name}: {msg}")
        if code == 404:
            raise ValueError(f"Route Not Found for {component_name}: {msg}")
        raise ValueError(f"Gateway Error ({code}) for {component_name}: {msg}")

    if "error" in data and isinstance(data["error"], dict):
        err_body = data["error"]
        err_code = err_body.get("code", "")
        err_msg = err_body.get("message", "")

        if "PROVIDER_NOT_FOUND" in str(err_code) or "NO CREDITS" in str(err_code):
            raise ValueError(f"Provider Config Error for {component_name}: {err_msg}")
        
        if "MODEL_NOT_FOUND" in str(err_code):
            raise ValueError(f"Model Not Found for {component_name}: {err_msg}")
            
        if "VECTOR_STORE_NOT_FOUND" in str(err_code):
            raise ValueError(f"Vector Store Error for {component_name}: {err_msg}")

        raise ValueError(f"API Logic Error for {component_name}: {err_msg}")


def _handle_api_exception(e: Exception, component_name: str):
    if isinstance(e, ValueError):
        raise e

    if hasattr(e, "response") and e.response is not None:
        data = None
        try:
            if hasattr(e.response, "json"):
                data = e.response.json()
            elif hasattr(e.response, "text"):
                data = json.loads(e.response.text)
        except:
            pass
        
        if data:
            _inspect_response_error(data, component_name)

    e_str = str(e)
    if "{" in e_str:
        json_data = None
        
        try:
            json_start = e_str.find("{")
            potential_json = e_str[json_start:]
            
            json_data = json.loads(potential_json)
        except json.JSONDecodeError:
            try:
                json_end = e_str.rfind("}") + 1
                clean_json = e_str[json_start:json_end]
                json_data = json.loads(clean_json)
            except:
                pass

        if json_data:
            _inspect_response_error(json_data, component_name)

    if isinstance(e, httpx.ConnectError):
        raise RuntimeError(f"Connection Failed for {component_name}: Could not reach server.")
    
    if isinstance(e, httpx.TimeoutException):
        raise RuntimeError(f"Timeout for {component_name}: Server took too long to respond.")

    raise RuntimeError(f"Call failed for {component_name}: {str(e)}")


def validate_environment(
    api_key: str,
    base_url: str,
    llm_models: Optional[List[str]] = None,
    embedding_models: Optional[List[str]] = None,
    knowledge_base_id: Optional[str] = None,
):
    
    target_llms = llm_models if llm_models else []
    target_embeddings = embedding_models if embedding_models else []
    target_kb = knowledge_base_id if knowledge_base_id else None
    
    components_found = False
    print(f"--- Starting Environment Validation ---")

    # --- 1. Validate LLMs ---
    if target_llms:
        components_found = True
        print(f"\n[Check] Validating {len(target_llms)} LLM(s)...")
        
        for model_name in target_llms:
            try:
                model = setup_model(model_name, api_key, base_url)
                test_msg = [{"role": "user", "content": "<TEST>ping</TEST>. Respond only with: READY"}]
                
                response = model.invoke(messages=test_msg)
                
                # Valid response check
                _inspect_response_error(response, f"LLM '{model_name}'")

                # Content check
                content = response.content
                if not content and isinstance(response, dict):
                    content = response.get("content")
                
                if not content:
                    raise RuntimeError("Response empty (no content).")
                print(f"  [OK] LLM: {model_name}")
                
            except Exception as e:
                _handle_api_exception(e, f"LLM '{model_name}'")

    # --- 2. Validate Embeddings ---
    if target_embeddings:
        components_found = True
        print(f"\n[Check] Validating {len(target_embeddings)} Embedding Model(s)...")
        
        for emb_model in target_embeddings:
            try:
                resp = create_embeddings(emb_model, "Test", api_key, base_url)

                _inspect_response_error(resp, f"Embedding '{emb_model}'")

                if "data" not in resp or not isinstance(resp["data"], list):
                    raise RuntimeError("Invalid response format.")
                print(f"  [OK] Embedding: {emb_model}")

            except Exception as e:
                _handle_api_exception(e, f"Embedding '{emb_model}'")

    # --- 3. Validate Knowledge Base ---
    if target_kb:
        components_found = True
        print(f"\n[Check] Validating Knowledge Base '{target_kb}'...")
        
        try:
            kb = FlotorchVectorStore(
                api_key=api_key, base_url=base_url, vectorstore_id=target_kb
            )
            results = kb.search(query="test")

            _inspect_response_error(results, f"VectorStore '{target_kb}'")
            
            if results is None:
                 raise RuntimeError("VectorStore returned None.")
            print(f"  [OK] Knowledge Base connected.")

        except Exception as e:
            _handle_api_exception(e, f"VectorStore '{target_kb}'")

    if not components_found:
        print("\n[Warning] No components provided to validate.")
    else:
        print("\n[SUCCESS] All provided components are valid.")
