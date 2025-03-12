from fastapi import FastAPI, File, UploadFile
import shutil
from app.pdf_processor import extract_text_from_pdf, chunk_text
from app.vector_store import store_embeddings, search_query
from app.query_handler import generate_rag_answer

app = FastAPI()
text_chunks = []
bm25 = None

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global text_chunks, bm25
    with open(f"data/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = extract_text_from_pdf(f"data/{file.filename}")
    text_chunks = chunk_text(text)
    bm25 = store_embeddings(text_chunks)
    return {"message": "File processed successfully!"}

@app.get("/query/")
async def query_api(q: str):
    response = generate_rag_answer(q, text_chunks)
    return {"answer": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)