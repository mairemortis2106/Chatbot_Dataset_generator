from fastapi import FastAPI, UploadFile, File
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid

from chunker import chunk_text
from embedding import embed_chunks

app = FastAPI()

client = QdrantClient(
    url="http://qdrant:6333",   # internal docker hostname
    api_key="vT2CddA2GODG7NkchvH23P9t3GYUu9Rv"
)

COLLECTION = "documents"

@app.on_event("startup")
def init():
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    text = (await file.read()).decode()

    chunks = chunk_text(text)

    vectors = embed_chunks(chunks)

    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk}
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    return {
        "status": "ok",
        "chunks": len(points)
    }