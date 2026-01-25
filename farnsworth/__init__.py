"""
Farnsworth: Self-Evolving Companion AI

A modular, self-evolving companion AI that runs entirely locally with zero-cost operation.
Features MemGPT-style memory paging, LangGraph agent swarms, and genetic evolution.
"""

__version__ = "0.5.0"
__author__ = "Farnsworth Team"

from farnsworth.core import LLMBackend, ModelManager, InferenceEngine
from farnsworth.memory import MemorySystem
from farnsworth.agents import SwarmOrchestrator

__all__ = [
    "LLMBackend",
    "ModelManager",
    "InferenceEngine",
    "MemorySystem",
    "SwarmOrchestrator",
    "__version__",
]
