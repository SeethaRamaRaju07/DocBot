import ollama
from app.vector_store import search_query

def generate_rag_answer(query, text_chunks):
    retrieved_docs = search_query(query)
    context = "\n".join(retrieved_docs)
    prompt = f"""You are an intelligent document assistant. Use only the provided document context to answer the query.

    Document Context:
    ------------------
    {context}
    
    Query: {query}
    Answer:
    """
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return response["message"]