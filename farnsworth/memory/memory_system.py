"""
Farnsworth Memory System - Unified Memory Interface

Integrates all memory components:
- Virtual Context (working memory paging)
- Working Memory (scratchpad)
- Archival Memory (long-term storage)
- Recall Memory (conversation history)
- Knowledge Graph (entity relationships)
- Memory Dreaming (consolidation)
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable

from loguru import logger

from farnsworth.memory.virtual_context import VirtualContext, MemoryBlock
from farnsworth.memory.working_memory import WorkingMemory, SlotType
from farnsworth.memory.archival_memory import ArchivalMemory, SearchResult
from farnsworth.memory.recall_memory import RecallMemory, ConversationTurn
from farnsworth.memory.knowledge_graph import KnowledgeGraph, Entity
from farnsworth.memory.memory_dreaming import MemoryDreamer


@dataclass
class MemorySearchResult:
    """Unified search result across all memory systems."""
    content: str
    source: str  # "archival", "recall", "graph", "working"
    score: float
    metadata: dict


class MemorySystem:
    """
    Unified memory system integrating all memory components.

    Provides a single interface for:
    - Storing and retrieving memories
    - Managing conversation history
    - Building and querying knowledge graphs
    - Background memory consolidation
    """

    def __init__(
        self,
        data_dir: str = "./data",
        context_window_size: int = 4096,
        embedding_dim: int = 384,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.virtual_context = VirtualContext(
            context_window_size=context_window_size,
            data_dir=str(self.data_dir / "context"),
        )

        self.working_memory = WorkingMemory(
            max_slots=50,
            default_ttl_minutes=30,
        )

        self.archival_memory = ArchivalMemory(
            data_dir=str(self.data_dir / "archival"),
            embedding_dim=embedding_dim,
        )

        self.recall_memory = RecallMemory(
            data_dir=str(self.data_dir / "conversations"),
        )

        self.knowledge_graph = KnowledgeGraph(
            data_dir=str(self.data_dir / "graph"),
        )

        self.dreamer = MemoryDreamer(
            idle_threshold_minutes=5,
            consolidation_interval_hours=1.0,
        )

        # Embedding function (set by user)
        self._embed_fn: Optional[Callable] = None

        self._initialized = False

    async def initialize(self):
        """Initialize all memory components."""
        if self._initialized:
            return

        await self.archival_memory.initialize()
        await self.knowledge_graph.initialize()

        # Set up dreamer callbacks
        self.dreamer.set_callbacks(
            get_memories=self._get_memories_for_dreaming,
            get_embedding=self._get_embedding,
            store_memory=self.remember,
            delete_memory=self.forget,
        )

        # Start background dreaming
        await self.dreamer.start_background_dreaming()

        self._initialized = True
        logger.info("Memory system initialized")

    async def shutdown(self):
        """Shutdown memory system."""
        await self.dreamer.stop_background_dreaming()
        await self.knowledge_graph.save()

    def set_embedding_function(self, embed_fn: Callable):
        """Set the embedding function for all components."""
        self._embed_fn = embed_fn
        self.archival_memory.embed_fn = embed_fn
        self.knowledge_graph.embed_fn = embed_fn

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text."""
        if self._embed_fn is None:
            return None
        try:
            if asyncio.iscoroutinefunction(self._embed_fn):
                return await self._embed_fn(text)
            return self._embed_fn(text)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    async def remember(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        extract_entities: bool = True,
    ) -> str:
        """
        Store a memory.

        - Adds to archival memory for long-term storage
        - Optionally extracts entities for knowledge graph
        - May add to context window if important

        Returns the memory ID.
        """
        self.dreamer.record_activity()

        # Get embedding
        embedding = await self._get_embedding(content)

        # Store in archival memory
        memory_id = await self.archival_memory.store(
            content=content,
            metadata=metadata,
            tags=tags,
            embedding=embedding,
        )

        # Extract entities for knowledge graph
        if extract_entities:
            entities = await self.knowledge_graph.extract_entities_from_text(content)
            if len(entities) > 1:
                await self.knowledge_graph.extract_relationships_from_text(
                    content, entities
                )

        # Add to context if important
        if importance > 0.7:
            block = MemoryBlock(
                id=memory_id,
                content=content,
                importance_score=importance,
                tags=tags or [],
            )
            self.virtual_context.context_window.add_block(block)

        logger.debug(f"Remembered: {content[:50]}... (id={memory_id})")
        return memory_id

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        search_archival: bool = True,
        search_conversation: bool = True,
        search_graph: bool = True,
        min_score: float = 0.3,
    ) -> list[MemorySearchResult]:
        """
        Recall memories relevant to a query.

        Searches across all memory systems and combines results.
        """
        self.dreamer.record_activity()

        results = []

        # Search archival memory
        if search_archival:
            archival_results = await self.archival_memory.search(
                query, top_k=top_k, min_score=min_score
            )
            for r in archival_results:
                results.append(MemorySearchResult(
                    content=r.entry.content,
                    source="archival",
                    score=r.score,
                    metadata={
                        "id": r.entry.id,
                        "tags": r.entry.tags,
                        "search_type": r.search_type,
                    },
                ))

        # Search conversation history
        if search_conversation:
            conv_results = await self.recall_memory.search(
                query, top_k=top_k
            )
            for r in conv_results:
                results.append(MemorySearchResult(
                    content=r.turn.content,
                    source="recall",
                    score=r.score,
                    metadata={
                        "role": r.turn.role,
                        "timestamp": r.turn.timestamp.isoformat(),
                        "context": [t.content for t in r.context_turns[:2]],
                    },
                ))

        # Search knowledge graph
        if search_graph:
            graph_results = await self.knowledge_graph.query(query)
            for entity in graph_results.entities[:top_k]:
                results.append(MemorySearchResult(
                    content=f"{entity.name} ({entity.entity_type})",
                    source="graph",
                    score=graph_results.score,
                    metadata={
                        "entity_id": entity.id,
                        "properties": entity.properties,
                        "mention_count": entity.mention_count,
                    },
                ))

        # Sort by score and deduplicate
        results.sort(key=lambda x: x.score, reverse=True)

        # Deduplicate similar content
        seen_content = set()
        unique_results = []
        for r in results:
            content_key = r.content[:100].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(r)

        return unique_results[:top_k]

    async def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        return await self.archival_memory.delete(memory_id)

    async def add_conversation_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ConversationTurn:
        """Add a turn to conversation history."""
        self.dreamer.record_activity()
        return await self.recall_memory.add_turn(role, content, metadata)

    async def get_conversation_context(self, max_turns: int = 10) -> str:
        """Get recent conversation as context string."""
        return self.recall_memory.to_context_string(max_turns=max_turns)

    async def set_working_memory(
        self,
        name: str,
        value: Any,
        slot_type: SlotType = SlotType.SCRATCH,
    ):
        """Set a value in working memory."""
        self.dreamer.record_activity()
        await self.working_memory.set(name, value, slot_type)

    async def get_working_memory(self, name: str, default: Any = None) -> Any:
        """Get a value from working memory."""
        return await self.working_memory.get(name, default)

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[dict] = None,
    ) -> Entity:
        """Add an entity to the knowledge graph."""
        self.dreamer.record_activity()
        return await self.knowledge_graph.add_entity(name, entity_type, properties)

    async def link_entities(
        self,
        source: str,
        target: str,
        relation_type: str,
    ):
        """Create a relationship between entities."""
        return await self.knowledge_graph.add_relationship(
            source, target, relation_type
        )

    def get_context(self) -> str:
        """Get the full context for LLM input."""
        parts = []

        # System prompt
        context = self.virtual_context.get_context()
        if context:
            parts.append(context)

        # Working memory
        working = self.working_memory.to_context_string(max_length=500)
        if len(working) > 50:
            parts.append(working)

        # Recent conversation
        conv = self.recall_memory.to_context_string(max_turns=5, max_length=1000)
        if len(conv) > 50:
            parts.append(conv)

        return "\n\n".join(parts)

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self.virtual_context.set_system_prompt(prompt)

    async def _get_memories_for_dreaming(self, limit: int = 200) -> list[dict]:
        """Get memories for dreaming consolidation."""
        results = []

        for entry_id, entry in list(self.archival_memory.entries.items())[:limit]:
            results.append({
                "id": entry.id,
                "content": entry.content,
                "embedding": entry.embedding,
                "created_at": entry.created_at.isoformat(),
                "access_count": entry.retrieval_count,
                "importance_score": 0.5,  # Default
            })

        return results

    async def trigger_dream(self):
        """Manually trigger a dreaming session."""
        return await self.dreamer.dream()

    def get_stats(self) -> dict:
        """Get comprehensive memory system statistics."""
        return {
            "virtual_context": self.virtual_context.get_status(),
            "working_memory": self.working_memory.get_status(),
            "archival_memory": self.archival_memory.get_stats(),
            "recall_memory": self.recall_memory.get_stats(),
            "knowledge_graph": self.knowledge_graph.get_stats(),
            "dreamer": self.dreamer.get_stats(),
        }

    async def get_memory_summary(self) -> str:
        """Get a human-readable summary of memory state."""
        stats = self.get_stats()

        return f"""Memory System Status:
- Archival: {stats['archival_memory']['total_entries']} entries
- Conversation: {stats['recall_memory']['total_turns']} turns
- Knowledge Graph: {stats['knowledge_graph']['total_entities']} entities
- Working Memory: {stats['working_memory']['slot_count']} active slots
- Dreamer: {'Idle' if stats['dreamer']['is_idle'] else 'Active'}, {stats['dreamer']['total_dreams']} sessions
"""
