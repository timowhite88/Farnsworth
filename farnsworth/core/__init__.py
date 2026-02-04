"""
Farnsworth Core Module

Provides multi-backend LLM inference with:
- Automatic hardware detection and optimization
- Speculative decoding for 2x throughput
- Dynamic model switching based on task complexity
- Graceful fallback chains
- Model Swarm collaborative inference (PSO-based)
- Embedded prompting system for agent coordination
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

# Embedded prompting system
try:
    from farnsworth.core.embedded_prompts import (
        prompt_manager,
        EmbeddedPromptManager,
        PromptTemplate,
        PromptCategory,
        ModelTier,
        get_agent_init_prompt,
        get_swarm_prompt,
        get_memory_prompt,
        get_handoff_prompt,
    )
    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False

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
    # Embedded Prompts (conditionally available)
    "prompt_manager",
    "EmbeddedPromptManager",
    "PromptTemplate",
    "PromptCategory",
    "ModelTier",
    "get_agent_init_prompt",
    "get_swarm_prompt",
    "get_memory_prompt",
    "get_handoff_prompt",
]
