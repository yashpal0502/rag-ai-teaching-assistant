from src.retriever import load_embeddings
from src.rag_pipeline import answer_question

# Load saved embeddings
df = load_embeddings()

print("Total chunks:", len(df))


# Ask a question
query = "How can I create a navigation bar using Flexbox?"


# Run complete RAG pipeline
result = answer_question(query=query, df=df, top_k=3)


# Display answer
print("\nAI Answer:")
print(result["answer"])


# Display retrieved sources
print("\nRetrieved Sources:")

print(result["sources"][["title", "number", "start", "end", "similarity_score"]])
