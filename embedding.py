from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/multilingual-e5-base")

def embed_chunks(chunks):
    return model.encode(chunks).tolist()