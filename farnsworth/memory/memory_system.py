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
import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable

from loguru import logger


class QueryCache:
    """
    Simple LRU cache for memory queries with TTL support.

    Features:
    - LRU eviction when max size reached
    - TTL-based expiration
    - Cache key normalization
    """

    def __init__(self, max_size: int = 100, ttl_seconds: float = 60.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()  # Thread-safe for multi-user scenarios

    def _make_key(self, query: str, **kwargs) -> str:
        """Create normalized cache key."""
        key_parts = [query.lower().strip()]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[Any]:
        """Get cached result if valid (thread-safe)."""
        key = self._make_key(query, **kwargs)
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    # Expired
                    del self._cache[key]
            self._misses += 1
            return None

    def set(self, query: str, value: Any, **kwargs):
        """Cache a result (thread-safe)."""
        key = self._make_key(query, **kwargs)
        with self._lock:
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)

            # Evict oldest if over size
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def invalidate(self, query: Optional[str] = None):
        """Invalidate cache entries (thread-safe)."""
        with self._lock:
            if query is None:
                self._cache.clear()
            else:
                key = self._make_key(query)
                self._cache.pop(key, None)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }

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
        data_dir: str = None,
        context_window_size: int = 4096,
        embedding_dim: int = 384,
    ):
        # Use persistent storage on server, local storage otherwise
        if data_dir is None:
            import os
            if os.path.exists("/workspace/farnsworth_memory"):
                data_dir = "/workspace/farnsworth_memory"
            else:
                data_dir = "./data"

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

        # Query cache for fast repeated lookups
        self._query_cache = QueryCache(max_size=100, ttl_seconds=60.0)

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

        # Invalidate query cache since memory has changed
        self._query_cache.invalidate()

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
        Recall memories relevant to a query with parallel execution.

        Optimized for <100ms latency on hot queries.
        """
        self.dreamer.record_activity()

        # Check cache for hot queries
        cache_key_params = {
            "top_k": top_k,
            "archival": search_archival,
            "conv": search_conversation,
            "graph": search_graph,
            "min_score": min_score,
        }
        cached = self._query_cache.get(query, **cache_key_params)
        if cached is not None:
            logger.debug(f"Cache hit for query: {query[:30]}...")
            return cached

        tasks = []

        # Helper coroutine for empty results
        async def _empty_result():
            return []

        # 1. Archival Search Task
        if search_archival:
            tasks.append(self._search_archival_wrapped(query, top_k, min_score))
        else:
            tasks.append(_empty_result())

        # 2. Conversation Search Task
        if search_conversation:
            tasks.append(self._search_conversation_wrapped(query, top_k))
        else:
            tasks.append(_empty_result())

        # 3. Graph Search Task
        if search_graph:
            tasks.append(self._search_graph_wrapped(query, top_k))
        else:
            tasks.append(_empty_result())
            
        # Execute in parallel
        results_archival, results_conv, results_graph = await asyncio.gather(*tasks)
        
        # Combine results
        all_results = results_archival + results_conv + results_graph
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Deduplicate
        seen_content = set()
        unique_results = []
        for r in all_results:
            content_key = r.content[:100].lower()
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(r)

        final_results = unique_results[:top_k]

        # Cache results for fast repeated queries
        self._query_cache.set(query, final_results, **cache_key_params)

        return final_results

    async def _search_archival_wrapped(self, query: str, top_k: int, min_score: float) -> list[MemorySearchResult]:
        try:
            archival_results = await self.archival_memory.search(query, top_k=top_k, min_score=min_score)
            return [
                MemorySearchResult(
                    content=r.entry.content,
                    source="archival",
                    score=r.score,
                    metadata={
                        "id": r.entry.id,
                        "tags": r.entry.tags,
                        "search_type": r.search_type,
                    }
                ) for r in archival_results
            ]
        except Exception as e:
            logger.error(f"Archival search error: {e}")
            return []

    async def _search_conversation_wrapped(self, query: str, top_k: int) -> list[MemorySearchResult]:
        try:
            conv_results = await self.recall_memory.search(query, top_k=top_k)
            return [
                MemorySearchResult(
                    content=r.turn.content,
                    source="recall",
                    score=r.score,
                    metadata={
                        "role": r.turn.role,
                        "timestamp": r.turn.timestamp.isoformat(),
                    }
                ) for r in conv_results
            ]
        except Exception as e:
            logger.error(f"Conversation search error: {e}")
            return []

    async def _search_graph_wrapped(self, query: str, top_k: int) -> list[MemorySearchResult]:
        try:
            graph_results = await self.knowledge_graph.query(query)
            return [
                MemorySearchResult(
                    content=f"{entity.name} ({entity.entity_type})",
                    source="graph",
                    score=graph_results.score,
                    metadata={
                        "entity_id": entity.id,
                        "properties": entity.properties,
                    }
                ) for entity in graph_results.entities[:top_k]
            ]
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []

    async def forget(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        self._query_cache.invalidate()  # Invalidate cache on mutation
        return await self.archival_memory.delete(memory_id)

    async def add_conversation_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> ConversationTurn:
        """Add a turn to conversation history."""
        self._query_cache.invalidate()  # Invalidate cache on mutation
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
        self._query_cache.invalidate()  # Invalidate cache on mutation
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
        self._query_cache.invalidate()  # Invalidate cache on mutation
        self.dreamer.record_activity()
        return await self.knowledge_graph.add_entity(name, entity_type, properties)

    async def link_entities(
        self,
        source: str,
        target: str,
        relation_type: str,
    ):
        """Create a relationship between entities."""
        self._query_cache.invalidate()  # Invalidate cache on mutation
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
            "query_cache": self._query_cache.get_stats(),
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


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_memory_system: Optional["MemorySystem"] = None


def get_memory_system() -> "MemorySystem":
    """Get or create the global memory system instance."""
    global _memory_system
    if _memory_system is None:
        _memory_system = MemorySystem()
    return _memory_system


async def initialize_memory_system() -> "MemorySystem":
    """Initialize and return the global memory system."""
    system = get_memory_system()
    await system.initialize()
    return system
