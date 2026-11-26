import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDINGS_FILE
from embeddings import create_embedding

# Load Saved Embeddings


def load_embeddings() -> pd.DataFrame:
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(f"Embeddings file not found: {EMBEDDINGS_FILE}")

    df = joblib.load(EMBEDDINGS_FILE)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Saved embeddings must be a Pandas DataFrame.")

    if df.empty:
        raise ValueError("Embeddings DataFrame is empty.")

    required_columns = {
        "chunk_id",
        "title",
        "number",
        "start",
        "end",
        "text",
        "embedding",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in embeddings DataFrame: {missing_columns}")

    return df


# df = load_embeddings().head()
# print(df)


# Semantic Search
def retrieve_relevant_chunks(
    query: str, df: pd.DataFrame, top_k: int = 3
) -> pd.DataFrame:
    # Validate query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    # Validate top_k
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if df.empty:
        raise ValueError("No chunks available for retrieval.")

    # Generate query embedding
    question_embedding = create_embedding(query)

    # Convert stored embeddings into a 2D NumPy array
    chunk_embeddings = np.vstack(df["embedding"].to_numpy())

    # Check embedding dimensions
    if len(question_embedding) != chunk_embeddings.shape[1]:
        raise ValueError("Query embedding dimensions do not match stored embeddings.")

    # Calculate cosine similarity
    similarities = cosine_similarity(chunk_embeddings, [question_embedding]).flatten()

    # Retrieve top K matching chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = df.iloc[top_indices].copy()

    # Attach similarity scores
    results["similarity_score"] = similarities[top_indices]

    return results


# query = "Where is tailwind css taught"

# results = retrieve_relevant_chunks(query, df=load_embeddings(), top_k=3)
# print(results)
