"""
Farnsworth Memory Module

MemGPT-style hierarchical memory system with:
- Virtual context window paging
- Attention-weighted importance scoring
- Graph-augmented semantic retrieval
- Background "dreaming" consolidation
"""

from farnsworth.memory.virtual_context import VirtualContext, ContextWindow, PageManager
from farnsworth.memory.working_memory import WorkingMemory
from farnsworth.memory.archival_memory import ArchivalMemory
from farnsworth.memory.recall_memory import RecallMemory
from farnsworth.memory.knowledge_graph import KnowledgeGraph
from farnsworth.memory.memory_dreaming import MemoryDreamer
from farnsworth.memory.memory_system import MemorySystem

__all__ = [
    "VirtualContext",
    "ContextWindow",
    "PageManager",
    "WorkingMemory",
    "ArchivalMemory",
    "RecallMemory",
    "KnowledgeGraph",
    "MemoryDreamer",
    "MemorySystem",
]
