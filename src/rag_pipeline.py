import requests

from src.config import OLLAMA_BASE_URL, LLM_MODEL, REQUEST_TIMEOUT
from src.retriever import load_embeddings, retrieve_relevant_chunks


# RAG Prompt
def build_prompt(query, retrieved_chunks):

    # Converts the DataFrame into JSON records
    context = retrieved_chunks[["title", "number", "start", "end", "text"]].to_json(
        orient="records"
    )

    prompt = f"""
            You are an AI teaching assistant for the Sigma Web Development course.

            Your task is to answer questions using ONLY the video chunks provided below.

            COURSE CONTEXT:
            {context}

            USER QUESTION:
            {query}

            INSTRUCTIONS:

            1. Answer only questions related to the course.
            2. Use the provided context to answer the question.
            3. Mention the video title and video number wherever relevant.
            4. Mention the timestamp (start and end time) where the topic is taught.
            5. Never invent video titles, timestamps, or course content.
            6. If the provided context does not contain enough information,
            clearly say that you could not find the answer in the available course material.
            7. If the question is unrelated to the course, politely explain
            that you can only answer questions related to the course.

            Provide a clear and beginner-friendly answer.
            """

    return prompt


# Generate Response Using Ollama
def generate_response(prompt):
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)

        response.raise_for_status()

        result = response.json()

        answer = result.get("response")

        if not answer:
            raise ValueError("Ollama returned an empty response.")

        return answer

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama API request timed out.")

    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to Ollama. Make sure Ollama is running.")

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}")


# Complete RAG Pipeline
def answer_question(query, df, top_k=3):

    # Retrieve relevent chunks
    retrieved_chunks = retrieve_relevant_chunks(query=query, df=df, top_k=top_k)

    # Construct prompt
    prompt = build_prompt(query=query, retrieved_chunks=retrieved_chunks)

    # Generate answer
    answer = generate_response(prompt)

    return {"answer": answer, "sources": retrieved_chunks}
