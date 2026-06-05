"""Hybrid (dense + BM25) retrieval with posture metadata filtering.

NumPy vector store (see build_index.py for why not ChromaDB). Dense cosine over
an L2-normalized matrix + BM25, fused with Reciprocal Rank Fusion, then a
Python-side posture/anatomy metadata filter.
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DB_PATH = "rag_db_v3"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if len(w) > 2]


class PostureRetriever:
    def __init__(self, db_path: str = DB_PATH):
        self.embedder = SentenceTransformer(MODEL_NAME)
        self.emb = np.load(Path(db_path) / "embeddings.npy")  # (N, d), normalized
        self.chunks: List[Dict] = []
        with open(Path(db_path) / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line))
        self.ids = [c["id"] for c in self.chunks]
        self.docs = [c["text"] for c in self.chunks]
        self.metadatas = [c["metadata"] for c in self.chunks]
        self.bm25 = BM25Okapi([_tokenize(d) for d in self.docs])
        self.id_to_idx = {i: idx for idx, i in enumerate(self.ids)}
        # Warm the embedder so the first real query isn't slow
        self.embedder.encode(["warm up"], normalize_embeddings=True)

    def retrieve(self,
                 query: str,
                 current_posture: Optional[str] = None,
                 anatomical_filter: Optional[List[str]] = None,
                 top_k: int = 5,
                 candidates_per_stream: int = 20) -> List[Dict]:
        """Hybrid retrieval. Filters by posture class metadata."""
        # 1. Dense (cosine == dot product on normalized vectors)
        q = self.embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims = self.emb @ q
        dense_rank = np.argsort(-sims)[:candidates_per_stream]
        dense_ids = [self.ids[i] for i in dense_rank]

        # 2. BM25
        scores = self.bm25.get_scores(_tokenize(query))
        bm25_rank = sorted(range(len(scores)), key=lambda i: -scores[i])[:candidates_per_stream]
        bm25_ids = [self.ids[i] for i in bm25_rank]

        # 3. RRF fusion
        rrf_k = 60
        rrf_scores: Dict[str, float] = {}
        for rank, did in enumerate(dense_ids):
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
        for rank, did in enumerate(bm25_ids):
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (rrf_k + rank)
        fused = sorted(rrf_scores.items(), key=lambda kv: -kv[1])

        # 4. Posture filter (Python-side)
        results = []
        for did, score in fused:
            idx = self.id_to_idx[did]
            md = self.metadatas[idx]
            postures = md.get("applicable_postures", [])
            if current_posture is not None and current_posture not in postures:
                continue
            if anatomical_filter and md.get("anatomical_region") not in anatomical_filter:
                continue
            results.append({"id": did, "text": self.docs[idx],
                            "metadata": md, "rrf_score": score})
            if len(results) >= top_k:
                break

        # Fall back to top-k unfiltered if the filter removed everything
        if not results:
            for did, score in fused[:top_k]:
                idx = self.id_to_idx[did]
                results.append({"id": did, "text": self.docs[idx],
                                "metadata": self.metadatas[idx], "rrf_score": score})
        return results
