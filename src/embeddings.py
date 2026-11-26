import requests

from config import OLLAMA_BASE_URL, EMBEDDING_MODEL, REQUEST_TIMEOUT


def create_embedding(text: str) -> list[float]:
    # validate input
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty.")

    # Ollama API request
    url = f"{OLLAMA_BASE_URL}/api/embeddings"

    payload = {"model": EMBEDDING_MODEL, "prompt": text}

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)

        # Check HTTP errors
        response.raise_for_status()

        # Extract embedding
        result = response.json()

        embedding = result.get("embedding")

        if not embedding:
            raise ValueError("No embedding returned by Ollama.")

        return embedding

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama API request timed out.")

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Ollama. Make sure Ollama is running.")

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")


# emb = create_embedding("RAG combines information retrieval with language generation.")

# print("Embedding type:", type(emb))
# print("Embedding dimensions:", len(emb))
# print("First five values:", emb[:5])
