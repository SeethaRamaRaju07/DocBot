import os
os.makedirs("data", exist_ok=True)
os.makedirs("index", exist_ok=True)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
FAISS_INDEX_PATH = "index/faiss_index.bin"