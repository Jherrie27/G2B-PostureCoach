"""Phase 6 RAG-stack smoke test.

ChromaDB was dropped on this platform (chroma-hnswlib has no installable wheel);
the vector store is a NumPy cosine matrix instead. This verifies the RAG deps
that ARE used.
"""
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import groq

print("RAG IMPORTS OK")
