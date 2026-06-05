"""Build rag_db_v3 from tagged chunks using local BGE-small embeddings.

DEVIATION FROM GUIDE: the guide stores vectors in ChromaDB. ChromaDB's native
dependency (chroma-hnswlib) has no installable wheel for this Python/OS combo
(0.7.5 has a cp312 wheel but every chromadb that pins a *buildable* version
pins 0.7.6, which doesn't). At ~2.8k chunks a brute-force NumPy cosine matrix
is faster than HNSW and has zero native deps, so rag_db_v3 is just:
    rag_db_v3/embeddings.npy   float32 (N, dim), L2-normalized
    rag_db_v3/chunks.jsonl     one {"id","text","metadata"} per line
"""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "rag_db_v3"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build(jsonl_path: str = "data/kb_chunks_tagged.jsonl",
          db_path: str = DB_PATH):
    Path(db_path).mkdir(parents=True, exist_ok=True)
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    embedder = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, show_progress_bar=True,
                          normalize_embeddings=True, batch_size=64)
    emb = np.asarray(emb, dtype=np.float32)

    np.save(Path(db_path) / "embeddings.npy", emb)
    with open(Path(db_path) / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Indexed {len(chunks)} chunks -> {db_path} (emb shape {emb.shape})")


if __name__ == "__main__":
    build()
