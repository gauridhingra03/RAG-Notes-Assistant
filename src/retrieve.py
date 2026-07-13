# src/retrieve.py
from sentence_transformers import SentenceTransformer


def retrieve_top_k(query: str, index, metadata: list[dict], model: SentenceTransformer, k: int = 4):
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vector, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx]
        results.append({
            "text": chunk["text"],
            "page": chunk.get("page", "?"),
            "score": float(scores[0][rank])   # similarity score (higher = better)
        })
    return results