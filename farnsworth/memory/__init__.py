"""
Farnsworth Memory Module

MemGPT-style hierarchical memory system with:
- Virtual context window paging
- Attention-weighted importance scoring
- Graph-augmented semantic retrieval
- Background "dreaming" consolidation

Q1 2025 (v0.2.0) Features:
- Episodic memory timeline
- Semantic memory layers with concept hierarchies
- Memory sharing (export/import)
- Enhanced knowledge graph with temporal edges
"""

from farnsworth.memory.virtual_context import VirtualContext, ContextWindow, PageManager
from farnsworth.memory.working_memory import WorkingMemory
from farnsworth.memory.archival_memory import ArchivalMemory
from farnsworth.memory.recall_memory import RecallMemory
from farnsworth.memory.knowledge_graph import KnowledgeGraph
from farnsworth.memory.memory_dreaming import MemoryDreamer
from farnsworth.memory.memory_system import MemorySystem

# Q1 2025 Features
from farnsworth.memory.episodic_memory import (
    EpisodicMemory,
    Episode,
    Session,
    EventType,
    TimelineQuery,
    OnThisDayResult,
)
from farnsworth.memory.semantic_layers import (
    SemanticLayerSystem,
    SemanticConcept,
    AbstractionLevel,
    DomainCluster,
    CrossDomainConnection,
)
from farnsworth.memory.memory_sharing import (
    MemorySharing,
    ExportFormat,
    MergeStrategy,
    ExportManifest,
    ImportResult,
    BackupInfo,
)
from farnsworth.memory.knowledge_graph_v2 import (
    KnowledgeGraphV2,
    TemporalEdge,
    EntityCluster,
    EntityResolutionCandidate,
)
from farnsworth.memory.conversation_export import (
    ConversationExporter,
    ConversationExportFormat,
    ExportOptions,
    ExportResult,
)

__all__ = [
    # Core memory components
    "VirtualContext",
    "ContextWindow",
    "PageManager",
    "WorkingMemory",
    "ArchivalMemory",
    "RecallMemory",
    "KnowledgeGraph",
    "MemoryDreamer",
    "MemorySystem",
    # Q1 2025: Episodic Memory
    "EpisodicMemory",
    "Episode",
    "Session",
    "EventType",
    "TimelineQuery",
    "OnThisDayResult",
    # Q1 2025: Semantic Layers
    "SemanticLayerSystem",
    "SemanticConcept",
    "AbstractionLevel",
    "DomainCluster",
    "CrossDomainConnection",
    # Q1 2025: Memory Sharing
    "MemorySharing",
    "ExportFormat",
    "MergeStrategy",
    "ExportManifest",
    "ImportResult",
    "BackupInfo",
    # Q1 2025: Enhanced Knowledge Graph
    "KnowledgeGraphV2",
    "TemporalEdge",
    "EntityCluster",
    "EntityResolutionCandidate",
    # Conversation Export
    "ConversationExporter",
    "ConversationExportFormat",
    "ExportOptions",
    "ExportResult",
]
