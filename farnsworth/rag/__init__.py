"""
Farnsworth RAG Module

Self-evolving retrieval-augmented generation with:
- Hybrid semantic + keyword search with RRF fusion
- RL-optimized retrieval strategy learning
- Adaptive chunking and reranking
- Query success feedback loops
"""

from farnsworth.rag.embeddings import EmbeddingManager
from farnsworth.rag.hybrid_retriever import HybridRetriever
from farnsworth.rag.document_processor import DocumentProcessor
from farnsworth.rag.self_refining_rag import SelfRefiningRAG

__all__ = [
    "EmbeddingManager",
    "HybridRetriever",
    "DocumentProcessor",
    "SelfRefiningRAG",
]
