# src/embed.py
import faiss
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME


def build_faiss_index(chunks: list[dict], model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    # BGE models cosine similarity ke liye normalized embeddings recommend karte hain
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)   # normalized vectors + inner product = cosine similarity
    index.add(embeddings.astype("float32"))
    return index