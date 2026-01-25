"""
Farnsworth Core Module

Provides multi-backend LLM inference with:
- Automatic hardware detection and optimization
- Speculative decoding for 2x throughput
- Dynamic model switching based on task complexity
- Graceful fallback chains
- Model Swarm collaborative inference (PSO-based)
"""

from farnsworth.core.llm_backend import (
    LLMBackend,
    OllamaBackend,
    LlamaCppBackend,
    BitNetBackend,
    CascadeBackend,
    GenerationConfig,
    GenerationResult,
    StreamChunk,
)
from farnsworth.core.model_manager import ModelManager
from farnsworth.core.inference_engine import InferenceEngine
from farnsworth.core.model_swarm import (
    ModelSwarm,
    ModelParticle,
    SwarmStrategy,
    SwarmResponse,
    ModelRole,
    QueryAnalyzer,
    QueryAnalysis,
)

__all__ = [
    # Backends
    "LLMBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "BitNetBackend",
    "CascadeBackend",
    # Generation
    "GenerationConfig",
    "GenerationResult",
    "StreamChunk",
    # Model Management
    "ModelManager",
    "InferenceEngine",
    # Model Swarm
    "ModelSwarm",
    "ModelParticle",
    "SwarmStrategy",
    "SwarmResponse",
    "ModelRole",
    "QueryAnalyzer",
    "QueryAnalysis",
]
