from sentence_transformers import SentenceTransformer
import torch

# 1. Cache model agar tidak re-download tiap run
model = SentenceTransformer(
    "intfloat/multilingual-e5-base",
    cache_folder="./model_cache"  # simpan lokal
)

# 2. Gunakan GPU kalau tersedia
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("intfloat/multilingual-e5-base", device=device)

# 3. Batch processing + normalisasi (penting untuk e5 model!)
def embed_chunks(chunks: list[str], batch_size: int = 32) -> list[list[float]]:
    # multilingual-e5 butuh prefix "query: " atau "passage: "
    passages = [f"passage: {chunk}" for chunk in chunks]
    
    embeddings = model.encode(
        passages,
        batch_size=batch_size,
        normalize_embeddings=True,  # untuk cosine similarity
        show_progress_bar=len(chunks) > 100
    )
    return embeddings.tolist()