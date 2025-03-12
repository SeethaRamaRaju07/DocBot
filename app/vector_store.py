import faiss
import numpy as np
import os
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Load embedding model
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Global variables
bm25 = None
text_chunks = []


def store_embeddings(chunks):
    """
    Store text embeddings in FAISS and create a BM25 index.
    """
    global bm25, text_chunks  # Ensure we update global variables

    if not chunks:
        print("⚠️ No text chunks found! Check PDF processing.")
        return
    
    text_chunks = chunks  # Store text chunks globally
    print(f"✅ Total text chunks: {len(text_chunks)}")

    # Compute embeddings
    embeddings = embedding_model.encode(text_chunks)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # Initialize FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings))

    # Save FAISS index
    os.makedirs("index", exist_ok=True)
    faiss.write_index(faiss_index, "index/faiss_index.bin")
    print("✅ FAISS index stored successfully!")

    # Initialize BM25
    bm25 = BM25Okapi([chunk.split() for chunk in text_chunks])
    print("✅ BM25 index created successfully!")


def search_query(query):
    """
    Perform hybrid search using FAISS (vector search) and BM25 (keyword search).
    """
    global bm25, text_chunks

    if not text_chunks:
        print("⚠️ No text chunks found! Ensure PDF was processed.")
        return []

    if not os.path.exists("index/faiss_index.bin"):
        print("⚠️ FAISS index file not found!")
        return []
    
    print("🔍 Loading FAISS index...")
    faiss_index = faiss.read_index("index/faiss_index.bin")

    # Perform FAISS search
    query_embedding = embedding_model.encode([query])
    _, faiss_indices = faiss_index.search(np.array(query_embedding), k=3)

    # Extract FAISS results
    faiss_results = [text_chunks[i] for i in faiss_indices[0] if i < len(text_chunks)]

    if bm25 is None:
        print("⚠️ BM25 index is not initialized! Re-run store_embeddings().")
        return faiss_results

    # Perform BM25 search
    bm25_results = bm25.get_top_n(query.split(), text_chunks, n=3)

    return list(set(faiss_results + bm25_results))
