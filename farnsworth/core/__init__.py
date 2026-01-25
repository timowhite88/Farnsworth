"""
Farnsworth Core Module

Provides multi-backend LLM inference with:
- Automatic hardware detection and optimization
- Speculative decoding for 2x throughput
- Dynamic model switching based on task complexity
- Graceful fallback chains
"""

from farnsworth.core.llm_backend import (
    LLMBackend,
    OllamaBackend,
    LlamaCppBackend,
    BitNetBackend,
)
from farnsworth.core.model_manager import ModelManager
from farnsworth.core.inference_engine import InferenceEngine

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "BitNetBackend",
    "ModelManager",
    "InferenceEngine",
]
