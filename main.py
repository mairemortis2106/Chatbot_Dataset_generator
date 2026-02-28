from fastapi import FastAPI, HTTPException, UploadFile, File
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import fitz 
import io
import os

from chunker import chunk_text
from embedding import embed_chunks

client = QdrantClient(
    url="https://qdrantdb.mhas.my.id",
    api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=os.environ.get("QDRANT_TIMEOUT"),
    https=True,
    port=443,
)

COLLECTION = "documents"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION for c in collections)
        if not exists:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            print(f"Collection '{COLLECTION}' created")
        else:
            print(f"Collection '{COLLECTION}' already exists")
    except Exception as e:
        print("Qdrant init error:", e)
    
    yield
    
    print("App shutting down")

app = FastAPI(lifespan=lifespan)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        pdf_doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        pdf_doc.close()
    elif filename.endswith(".txt"):
        text = content.decode("utf-8")
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Tidak ada teks yang bisa diekstrak dari file.")

    chunks = chunk_text(text)
    vectors = embed_chunks(chunks)
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"text": chunk})
        for chunk, vector in zip(chunks, vectors)
    ]

    # ✅ Upsert dalam batch kecil
    BATCH_SIZE = os.environ.get("BATCH_SIZE")
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=COLLECTION, points=batch)

    return {"status": "ok", "chunks": len(points)}

@app.get("/test-qdrant")
def test_qdrant():
    return client.get_collections().model_dump()