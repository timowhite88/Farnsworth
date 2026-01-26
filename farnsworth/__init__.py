"""
Farnsworth: Self-Evolving Companion AI

A modular, self-evolving companion AI that runs entirely locally with zero-cost operation.
Features MemGPT-style memory paging, LangGraph agent swarms, and genetic evolution.
"""

__version__ = "2.1.0-alpha"
__author__ = "Farnsworth Team"

from farnsworth.core.nexus import nexus, Signal, SignalType
from farnsworth.core.fcp import FCPEngine
from farnsworth.core.neuromorphic.engine import neuro_engine
from farnsworth.os_integration.bridge import os_bridge
from farnsworth.core.cognition.theory_of_mind import tom_engine

__all__ = [
    "nexus", "Signal", "SignalType",
    "FCPEngine",
    "neuro_engine",
    "os_bridge",
    "tom_engine",
    "__version__",
]
