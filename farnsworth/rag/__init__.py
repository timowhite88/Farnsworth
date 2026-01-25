"""
Farnsworth RAG Module

Self-evolving retrieval-augmented generation with:
- Hybrid semantic + keyword search with RRF fusion
- RL-optimized retrieval strategy learning
- Adaptive chunking and reranking
- Query success feedback loops

Q1 2025 (v0.2.0) Features:
- Hybrid Search v2 with intent classification
- Multi-hop retrieval for complex questions
- Context compression for token efficiency
"""

from farnsworth.rag.embeddings import EmbeddingManager
from farnsworth.rag.hybrid_retriever import HybridRetriever
from farnsworth.rag.document_processor import DocumentProcessor
from farnsworth.rag.self_refining_rag import SelfRefiningRAG

# Q1 2025 Features
from farnsworth.rag.hybrid_search_v2 import (
    HybridSearchV2,
    QueryIntent,
    QueryAnalysis,
    SearchHop,
    AttributedResult,
    HybridSearchResult,
)
from farnsworth.rag.context_compression import (
    ContextCompressor,
    CompressionLevel,
    ContentPriority,
    ContextBlock,
    CompressionResult,
    ContextBudget,
)

__all__ = [
    # Core RAG components
    "EmbeddingManager",
    "HybridRetriever",
    "DocumentProcessor",
    "SelfRefiningRAG",
    # Q1 2025: Hybrid Search v2
    "HybridSearchV2",
    "QueryIntent",
    "QueryAnalysis",
    "SearchHop",
    "AttributedResult",
    "HybridSearchResult",
    # Q1 2025: Context Compression
    "ContextCompressor",
    "CompressionLevel",
    "ContentPriority",
    "ContextBlock",
    "CompressionResult",
    "ContextBudget",
]
