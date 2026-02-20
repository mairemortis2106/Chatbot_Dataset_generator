from fastapi import FastAPI, UploadFile, File
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import os

from chunker import chunk_text
from embedding import embed_chunks

app = FastAPI()

client = QdrantClient(
    host="qdrantdb.mhas.my.id",
    https=True,
    api_key="vT2CddA2GODG7NkchvH23P9t3GYUu9Rv",
    prefer_grpc=False,
    timeout=30,
)

COLLECTION = "documents"


# startup safe init
@app.on_event("startup")
def init():
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION for c in collections)

        if not exists:
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            print(f"Collection '{COLLECTION}' created")
        else:
            print(f"Collection '{COLLECTION}' already exists")

    except Exception as e:
        print("Qdrant init error:", e)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    text = (await file.read()).decode()

    chunks = chunk_text(text)

    vectors = embed_chunks(chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": chunk}
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    client.upsert(
        collection_name=COLLECTION,
        points=points
    )

    return {
        "status": "ok",
        "chunks": len(points)
    }


@app.get("/test-qdrant")
def test_qdrant():
    return client.get_collections().model_dump()