"""Optional hosted reranker. Returns top-k of input candidates by relevance.

Default is NoopReranker (no hosted-reranking budget assumed). Wire CohereReranker
into the retriever if a COHERE_API_KEY is available.
"""
import os
from typing import List, Dict, Optional


class CohereReranker:
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "rerank-english-v3.0"):
        import cohere
        self.client = cohere.Client(api_key or os.environ["COHERE_API_KEY"])
        self.model = model

    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        docs = [c["text"] for c in candidates]
        result = self.client.rerank(query=query, documents=docs,
                                    model=self.model, top_n=top_k)
        out = []
        for r in result.results:
            cand = candidates[r.index]
            cand["rerank_score"] = float(r.relevance_score)
            out.append(cand)
        return out


class NoopReranker:
    """Drop-in replacement if no reranker is configured."""
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5):
        return candidates[:top_k]
