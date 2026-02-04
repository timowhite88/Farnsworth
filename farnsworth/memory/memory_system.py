"""
Farnsworth Memory System - Unified Memory Interface

Integrates all memory components:
- Virtual Context (working memory paging)
- Working Memory (scratchpad)
- Archival Memory (long-term storage)
- Recall Memory (conversation history)
- Knowledge Graph (entity relationships)
- Memory Dreaming (consolidation)

v1.4 Enhancements:
- Optional at-rest encryption (Fernet)
- Nexus signal integration for consolidation events
- Hysteresis-aware activity tracking
- Affective valence bias in retrieval
- Async throttling with semaphore
- Snapshot backup before pruning
- Surprise signaling for novel memories
"""

import asyncio
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any, Callable

import numpy as np
from loguru import logger


# =============================================================================
# AGI MEMORY CONFIGURATION
# =============================================================================

@dataclass
class MemoryAGIConfig:
    """
    Configuration for AGI memory upgrades.

    All features are enabled by default but can be toggled via environment
    variables with the FARNSWORTH_ prefix.
    """
    # Feature toggles
    sync_enabled: bool = True
    hybrid_enabled: bool = True
    proactive_context: bool = True
    cost_aware: bool = True
    drift_detection: bool = True

    # Sync settings (Federated memory sharing)
    sync_epsilon: float = 1.0  # Differential privacy budget
    sync_max_batch: int = 100

    # Retrieval settings (Hybrid recall)
    hybrid_oversample: int = 3
    hybrid_use_attention: bool = True

    # Context settings (Proactive compaction)
    proactive_threshold: float = 0.7  # Trigger at 70% capacity
    preserve_ratio: float = 0.3

    # Cost settings (Budget-aware operations)
    cost_daily_limit: float = 1.0  # USD
    prefer_local: bool = True

    # Schema settings (Adaptive drift detection)
    drift_threshold: float = 0.3
    decay_halflife: float = 24.0  # hours

    @classmethod
    def from_env(cls) -> "MemoryAGIConfig":
        """Load configuration from environment variables with FARNSWORTH_ prefix."""
        def get_bool(key: str, default: bool) -> bool:
            return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

        def get_float(key: str, default: float) -> float:
            return float(os.getenv(key, str(default)))

        def get_int(key: str, default: int) -> int:
            return int(os.getenv(key, str(default)))

        return cls(
            sync_enabled=get_bool("FARNSWORTH_SYNC_ENABLED", True),
            hybrid_enabled=get_bool("FARNSWORTH_HYBRID_ENABLED", True),
            proactive_context=get_bool("FARNSWORTH_CONTEXT_PROACTIVE", True),
            cost_aware=get_bool("FARNSWORTH_COST_AWARE", True),
            drift_detection=get_bool("FARNSWORTH_DRIFT_DETECTION", True),
            sync_epsilon=get_float("FARNSWORTH_SYNC_EPSILON", 1.0),
            sync_max_batch=get_int("FARNSWORTH_SYNC_MAX_BATCH", 100),
            hybrid_oversample=get_int("FARNSWORTH_HYBRID_OVERSAMPLE", 3),
            hybrid_use_attention=get_bool("FARNSWORTH_HYBRID_ATTENTION", True),
            proactive_threshold=get_float("FARNSWORTH_CONTEXT_THRESHOLD", 0.7),
            preserve_ratio=get_float("FARNSWORTH_CONTEXT_PRESERVE_RATIO", 0.3),
            cost_daily_limit=get_float("FARNSWORTH_COST_DAILY_LIMIT", 1.0),
            prefer_local=get_bool("FARNSWORTH_PREFER_LOCAL", True),
            drift_threshold=get_float("FARNSWORTH_DRIFT_THRESHOLD", 0.3),
            decay_halflife=get_float("FARNSWORTH_DECAY_HALFLIFE", 24.0),
        )

# Optional encryption
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    Fernet = None

# Nexus integration
try:
    from farnsworth.core.nexus import nexus, SignalType
    NEXUS_AVAILABLE = True
except ImportError:
    NEXUS_AVAILABLE = False
    nexus = None


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


# =============================================================================
# FEDERATED SYNC DATACLASSES
# =============================================================================

@dataclass
class SyncResult:
    """Result of memory synchronization with a peer."""
    pushed_count: int
    pulled_count: int
    merged_count: int
    conflicts_resolved: int
    privacy_budget_used: float
    sync_timestamp: datetime
    peer_id: str

    def summary(self) -> str:
        """Human-readable summary of sync result."""
        return (
            f"Sync with {self.peer_id}: "
            f"pushed={self.pushed_count}, pulled={self.pulled_count}, "
            f"merged={self.merged_count}, conflicts={self.conflicts_resolved}, "
            f"privacy_budget={self.privacy_budget_used:.3f}"
        )


class MemorySystem:
    """
    Unified memory system integrating all memory components.

    Provides a single interface for:
    - Storing and retrieving memories
    - Managing conversation history
    - Building and querying knowledge graphs
    - Background memory consolidation

    v1.4 Enhancements:
    - Optional at-rest encryption
    - Nexus signal integration
    - Hysteresis activity tracking
    - Affective bias in retrieval
    - Async throttling
    """

    def __init__(
        self,
        data_dir: str = None,
        context_window_size: int = 4096,
        embedding_dim: int = 384,
        encrypt_at_rest: bool = False,
        max_concurrent_ops: int = 10,
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

        # v1.4: Encryption setup
        self.encrypt_at_rest = encrypt_at_rest and ENCRYPTION_AVAILABLE
        self._cipher = None
        if self.encrypt_at_rest:
            key_path = self.data_dir / ".encryption_key"
            if key_path.exists():
                self._cipher = Fernet(key_path.read_bytes())
            else:
                key = Fernet.generate_key()
                key_path.write_bytes(key)
                self._cipher = Fernet(key)
            logger.info("Memory encryption enabled")

        # v1.4: Async throttling semaphore
        self._op_semaphore = asyncio.Semaphore(max_concurrent_ops)

        # v1.4: Hysteresis activity tracking
        self._last_activity = datetime.now()
        self._activity_count = 0

        # v1.4: Backup directory
        self._backup_dir = self.data_dir / "backups"
        self._backup_dir.mkdir(parents=True, exist_ok=True)

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

        # v1.4: Cluster centroids for surprise detection
        self._cluster_centroids: list = []

        self._initialized = False

    def notify_activity(self):
        """
        v1.4: Notify system of user/message activity.
        Used for hysteresis-based consolidation triggering.
        """
        self._last_activity = datetime.now()
        self._activity_count += 1
        self.dreamer.record_activity()

        # Emit activity signal if Nexus available
        if NEXUS_AVAILABLE and nexus:
            asyncio.create_task(self._emit_activity_signal())

    async def _emit_activity_signal(self):
        """Emit activity signal via Nexus."""
        try:
            await nexus.emit(
                type=SignalType.USER_MESSAGE,
                payload={"activity_count": self._activity_count},
                source="memory_system",
            )
        except Exception:
            pass

    def get_idle_seconds(self) -> float:
        """Get seconds since last activity."""
        return (datetime.now() - self._last_activity).total_seconds()

    def _encrypt(self, content: str) -> str:
        """Encrypt content if encryption enabled."""
        if self._cipher:
            return self._cipher.encrypt(content.encode()).decode()
        return content

    def _decrypt(self, content: str) -> str:
        """Decrypt content if encryption enabled."""
        if self._cipher:
            try:
                return self._cipher.decrypt(content.encode()).decode()
            except Exception:
                return content
        return content

    def snapshot_backup(self, reason: str = "manual"):
        """
        v1.4: Create backup snapshot before destructive operations.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self._backup_dir / f"memory_backup_{timestamp}_{reason}.json"

        try:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "archival_count": len(self.archival_memory.entries) if hasattr(self.archival_memory, 'entries') else 0,
                "stats": self.get_stats(),
            }
            backup_file.write_text(json.dumps(snapshot, indent=2))
            logger.info(f"Created backup snapshot: {backup_file.name}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")

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
        emotional_valence: float = 0.0,
    ) -> str:
        """
        Store a memory.

        - Adds to archival memory for long-term storage
        - Optionally extracts entities for knowledge graph
        - May add to context window if important

        v1.4: Added emotional_valence for affective bias, surprise detection

        Returns the memory ID.
        """
        async with self._op_semaphore:  # v1.4: Throttle concurrent ops
            self.notify_activity()

            # v1.4: Boost importance based on emotional valence
            adjusted_importance = importance + abs(emotional_valence) * 0.2
            adjusted_importance = min(1.0, adjusted_importance)

            # Get embedding
            embedding = await self._get_embedding(content)

            # v1.4: Detect surprise (novel memory)
            surprise_score = 0.0
            if embedding and self._cluster_centroids:
                surprise_score = await self._calculate_surprise(embedding)
                if surprise_score > 0.7:
                    adjusted_importance = min(1.0, adjusted_importance + 0.1)
                    if NEXUS_AVAILABLE and nexus:
                        asyncio.create_task(nexus.emit(
                            type=SignalType.ANOMALY_DETECTED,
                            payload={
                                "event": "surprise_memory",
                                "surprise_score": surprise_score,
                                "content_preview": content[:100],
                            },
                            source="memory_system",
                        ))

            # v1.4: Encrypt if enabled
            store_content = self._encrypt(content) if self.encrypt_at_rest else content

            # Prepare enhanced metadata
            enhanced_metadata = metadata or {}
            enhanced_metadata.update({
                "emotional_valence": emotional_valence,
                "surprise_score": surprise_score,
                "encrypted": self.encrypt_at_rest,
            })

            # Store in archival memory
            memory_id = await self.archival_memory.store(
                content=store_content,
                metadata=enhanced_metadata,
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
            if adjusted_importance > 0.7:
                block = MemoryBlock(
                    id=memory_id,
                    content=content,
                    importance_score=adjusted_importance,
                    tags=tags or [],
                )
                self.virtual_context.context_window.add_block(block)

            # Invalidate query cache since memory has changed
            self._query_cache.invalidate()

            # v1.4: Emit storage signal
            if NEXUS_AVAILABLE and nexus:
                asyncio.create_task(nexus.emit(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "memory_stored",
                        "memory_id": memory_id,
                        "importance": adjusted_importance,
                        "surprise": surprise_score,
                    },
                    source="memory_system",
                ))

            logger.debug(f"Remembered: {content[:50]}... (id={memory_id}, importance={adjusted_importance:.2f})")
            return memory_id

    async def _calculate_surprise(self, embedding: list) -> float:
        """
        v1.4: Calculate how surprising/novel a memory is.
        Higher surprise = more different from existing cluster centroids.
        """
        if not self._cluster_centroids:
            return 0.5  # Neutral if no centroids yet

        try:
            import numpy as np
            emb_np = np.array(embedding)
            min_distance = float('inf')

            for centroid in self._cluster_centroids:
                centroid_np = np.array(centroid)
                distance = np.linalg.norm(emb_np - centroid_np)
                min_distance = min(min_distance, distance)

            # Normalize to 0-1 using sigmoid
            import math
            surprise = 1.0 / (1.0 + math.exp(-min_distance + 1))
            return surprise
        except Exception:
            return 0.5

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        search_archival: bool = True,
        search_conversation: bool = True,
        search_graph: bool = True,
        min_score: float = 0.3,
        emotion_filter: Optional[str] = None,
        valence_bias: float = 0.0,
    ) -> list[MemorySearchResult]:
        """
        Recall memories relevant to a query with parallel execution.

        Optimized for <100ms latency on hot queries.

        v1.4: Added emotion_filter and valence_bias for affective retrieval
        - emotion_filter: "positive", "negative", or None
        - valence_bias: -1.0 to 1.0, boosts memories with matching valence
        """
        self.notify_activity()

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

        # v1.4: Apply affective bias filtering
        if emotion_filter or valence_bias != 0.0:
            unique_results = self._apply_affective_bias(
                unique_results, emotion_filter, valence_bias
            )

        final_results = unique_results[:top_k]

        # Cache results for fast repeated queries
        self._query_cache.set(query, final_results, **cache_key_params)

        # v1.4: Emit retrieval signal
        if NEXUS_AVAILABLE and nexus:
            asyncio.create_task(nexus.emit(
                type=SignalType.EXTERNAL_EVENT,
                payload={
                    "event": "memory_retrieved",
                    "count": len(final_results),
                    "query_preview": query[:50],
                },
                source="memory_system",
            ))

        return final_results

    def _apply_affective_bias(
        self,
        results: list,
        emotion_filter: Optional[str],
        valence_bias: float,
    ) -> list:
        """
        v1.4: Apply affective bias to search results.
        Boosts/filters memories based on emotional valence.
        """
        filtered = []

        for r in results:
            valence = r.metadata.get("emotional_valence", 0.0)

            # Filter by emotion category
            if emotion_filter == "positive" and valence < 0.0:
                continue
            elif emotion_filter == "negative" and valence > 0.0:
                continue

            # Apply valence bias to score
            if valence_bias != 0.0:
                # Boost score if valence matches bias direction
                valence_match = valence * valence_bias
                if valence_match > 0:
                    r.score *= (1.0 + abs(valence_match) * 0.3)
                else:
                    r.score *= (1.0 - abs(valence_match) * 0.1)

            filtered.append(r)

        # Re-sort after bias adjustment
        filtered.sort(key=lambda x: x.score, reverse=True)
        return filtered

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

    # =========================================================================
    # FEDERATED SYNC WITH DIFFERENTIAL PRIVACY (AGI Upgrade 1)
    # =========================================================================

    async def sync_memories(
        self,
        peer_id: str,
        direction: str = "bidirectional",  # "push", "pull", "bidirectional"
        epsilon: float = 1.0,  # Differential privacy budget
        max_memories: int = 100,
        importance_threshold: float = 0.7,
        swarm_fabric=None,  # P2P SwarmFabric instance
    ) -> SyncResult:
        """
        Federated memory synchronization with differential privacy.

        Enables privacy-preserving memory sharing across Farnsworth instances
        using Laplacian noise for differential privacy guarantees.

        Args:
            peer_id: ID of the peer to sync with
            direction: "push", "pull", or "bidirectional"
            epsilon: Differential privacy budget (higher = less privacy, more utility)
            max_memories: Maximum memories to sync per direction
            importance_threshold: Only sync memories above this importance
            swarm_fabric: P2P networking instance (SwarmFabric)

        Returns:
            SyncResult with sync statistics
        """
        async with self._op_semaphore:
            pushed_count = 0
            pulled_count = 0
            merged_count = 0
            conflicts_resolved = 0
            privacy_budget_used = 0.0

            # Step 1: Push memories if direction allows
            if direction in ("push", "bidirectional"):
                push_result = await self._push_memories(
                    peer_id=peer_id,
                    epsilon=epsilon,
                    max_memories=max_memories,
                    importance_threshold=importance_threshold,
                    swarm_fabric=swarm_fabric,
                )
                pushed_count = push_result.get("pushed", 0)
                privacy_budget_used += push_result.get("budget_used", 0)

            # Step 2: Pull memories if direction allows
            if direction in ("pull", "bidirectional"):
                pull_result = await self._pull_memories(
                    peer_id=peer_id,
                    max_memories=max_memories,
                    swarm_fabric=swarm_fabric,
                )
                pulled_count = pull_result.get("pulled", 0)
                merged_count = pull_result.get("merged", 0)
                conflicts_resolved = pull_result.get("conflicts", 0)

            result = SyncResult(
                pushed_count=pushed_count,
                pulled_count=pulled_count,
                merged_count=merged_count,
                conflicts_resolved=conflicts_resolved,
                privacy_budget_used=privacy_budget_used,
                sync_timestamp=datetime.now(),
                peer_id=peer_id,
            )

            # Emit sync event via Nexus
            if NEXUS_AVAILABLE and nexus:
                asyncio.create_task(nexus.emit(
                    type=SignalType.EXTERNAL_EVENT,
                    payload={
                        "event": "memory_sync_completed",
                        "peer_id": peer_id,
                        "direction": direction,
                        "pushed": pushed_count,
                        "pulled": pulled_count,
                        "merged": merged_count,
                    },
                    source="memory_system",
                ))

            logger.info(f"Memory sync completed: {result.summary()}")
            return result

    async def _push_memories(
        self,
        peer_id: str,
        epsilon: float,
        max_memories: int,
        importance_threshold: float,
        swarm_fabric=None,
    ) -> dict:
        """Push important memories to peer with differential privacy."""
        pushed = 0
        budget_used = 0.0

        # Select high-importance memories for sharing
        candidates = []
        for entry_id, entry in list(self.archival_memory.entries.items()):
            # Skip encrypted or private entries
            if entry.metadata.get("encrypted"):
                continue
            if entry.metadata.get("private"):
                continue

            # Calculate importance (using emotional valence and access patterns)
            base_importance = 0.5
            emotional_boost = abs(entry.metadata.get("emotional_valence", 0)) * 0.2
            access_boost = min(0.3, entry.retrieval_count * 0.05)
            importance = base_importance + emotional_boost + access_boost

            if importance >= importance_threshold:
                candidates.append((entry_id, entry, importance))

        # Sort by importance and take top max_memories
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = candidates[:max_memories]

        for entry_id, entry, importance in candidates:
            # Apply differential privacy to embedding
            if entry.embedding:
                noisy_embedding = self._add_differential_noise(
                    entry.embedding,
                    epsilon=epsilon,
                    sensitivity=1.0,
                )
                budget_used += 1.0 / epsilon

                # Prepare sync message
                sync_data = {
                    "type": "GOSSIP_MEMORY",
                    "memory_id": entry_id,
                    "content_hash": hashlib.sha256(entry.content.encode()).hexdigest()[:16],
                    "embedding": noisy_embedding,
                    "tags": entry.tags,
                    "importance": importance,
                    "created_at": entry.created_at.isoformat(),
                    "from_node": self.node_id if hasattr(self, 'node_id') else "local",
                }

                # Send via P2P if available
                if swarm_fabric:
                    try:
                        await swarm_fabric.broadcast_message(sync_data)
                        pushed += 1
                    except Exception as e:
                        logger.debug(f"Failed to push memory {entry_id}: {e}")

        return {"pushed": pushed, "budget_used": budget_used}

    async def _pull_memories(
        self,
        peer_id: str,
        max_memories: int,
        swarm_fabric=None,
    ) -> dict:
        """Pull and integrate memories from peer."""
        pulled = 0
        merged = 0
        conflicts = 0

        # In a real implementation, this would request memories from the peer
        # For now, we process any incoming memory sync events
        # The actual pull would be triggered by receiving GOSSIP_MEMORY events

        return {"pulled": pulled, "merged": merged, "conflicts": conflicts}

    def _add_differential_noise(
        self,
        embedding: list[float],
        epsilon: float,
        sensitivity: float = 1.0,
    ) -> list[float]:
        """
        Add Laplacian noise for differential privacy.

        The Laplace mechanism adds noise calibrated to sensitivity/epsilon
        to provide epsilon-differential privacy.

        Args:
            embedding: Original embedding vector
            epsilon: Privacy budget (higher = less noise, less privacy)
            sensitivity: L1 sensitivity of the query (default 1.0 for normalized embeddings)

        Returns:
            Noisy embedding preserving approximate utility
        """
        # Scale parameter for Laplace distribution
        scale = sensitivity / epsilon

        # Generate Laplacian noise
        noise = np.random.laplace(0, scale, len(embedding))

        # Add noise to embedding
        noisy = np.array(embedding) + noise

        # Re-normalize to unit sphere (embeddings are typically normalized)
        norm = np.linalg.norm(noisy)
        if norm > 0:
            noisy = noisy / norm

        return noisy.tolist()

    async def _semantic_merge(
        self,
        local: MemorySearchResult,
        remote: dict,
        threshold: float = 0.95,
    ) -> Optional[str]:
        """
        Merge semantically similar memories, keeping higher importance.

        Args:
            local: Local memory result
            remote: Remote memory data
            threshold: Similarity threshold for merging (0.95 = 95% similar)

        Returns:
            Merged memory ID if merge occurred, None otherwise
        """
        # Get embeddings
        local_embedding = await self._get_embedding(local.content)
        remote_embedding = remote.get("embedding")

        if not local_embedding or not remote_embedding:
            return None

        # Calculate cosine similarity
        local_vec = np.array(local_embedding)
        remote_vec = np.array(remote_embedding)

        similarity = np.dot(local_vec, remote_vec) / (
            np.linalg.norm(local_vec) * np.linalg.norm(remote_vec) + 1e-8
        )

        if similarity < threshold:
            return None  # Not similar enough to merge

        # Determine which to keep based on importance
        local_importance = local.metadata.get("importance", 0.5)
        remote_importance = remote.get("importance", 0.5)

        if remote_importance > local_importance:
            # Remote is more important, update local
            local_id = local.metadata.get("id")
            if local_id:
                entry = await self.archival_memory.get_entry(local_id)
                if entry:
                    # Update with remote tags (union)
                    entry.tags = list(set(entry.tags + remote.get("tags", [])))
                    # Boost importance
                    entry.metadata["merged_importance"] = max(local_importance, remote_importance)
                    await self.archival_memory._save_entry(entry)
                    return local_id

        return None


# =============================================================================
# ADAPTIVE SCHEMA MANAGER (AGI Upgrade 5)
# =============================================================================

@dataclass
class DriftResult:
    """Result of concept drift detection."""
    concept_name: str
    drift_magnitude: float
    drift_direction: list[float]  # Unit vector
    samples_analyzed: int
    is_significant: bool
    recommended_action: str  # "update_centroid", "create_branch", "ignore"


@dataclass
class SchemaEvolution:
    """Record of schema evolution event."""
    timestamp: datetime
    concept_name: str
    old_centroid: list[float]
    new_centroid: list[float]
    drift_magnitude: float
    action_taken: str


class AdaptiveSchemaManager:
    """
    Manages adaptive memory schemas for long-horizon tasks.

    Features:
    - Concept drift detection using EMA centroid tracking
    - Importance decay with configurable halflife
    - Schema evolution with branching for significant drift

    This enables the memory system to adapt to evolving concepts
    over extended time periods (days to weeks).
    """

    def __init__(
        self,
        drift_threshold: float = 0.3,
        decay_halflife_hours: float = 24.0,
        min_samples_for_drift: int = 10,
        ema_alpha: float = 0.1,
    ):
        """
        Initialize the schema manager.

        Args:
            drift_threshold: Magnitude threshold for significant drift
            decay_halflife_hours: Half-life for importance decay
            min_samples_for_drift: Minimum samples before drift detection
            ema_alpha: Exponential moving average smoothing factor
        """
        self.drift_threshold = drift_threshold
        self.decay_halflife_hours = decay_halflife_hours
        self.min_samples_for_drift = min_samples_for_drift
        self.ema_alpha = ema_alpha

        # Concept tracking
        self._concept_centroids: dict[str, np.ndarray] = {}
        self._concept_sample_counts: dict[str, int] = {}
        self._concept_timestamps: dict[str, list[datetime]] = {}

        # Evolution history
        self._drift_history: list[SchemaEvolution] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def detect_concept_drift(
        self,
        concept_name: str,
        new_embedding: list[float],
    ) -> DriftResult:
        """
        Detect if a concept has drifted from its historical centroid.

        Uses exponential moving average (EMA) for smooth centroid updates
        and cosine distance for drift magnitude.

        Args:
            concept_name: Name of the concept to track
            new_embedding: New embedding vector for this concept

        Returns:
            DriftResult with drift analysis
        """
        async with self._lock:
            new_vec = np.array(new_embedding)
            new_vec_normalized = new_vec / (np.linalg.norm(new_vec) + 1e-8)

            # Initialize if new concept
            if concept_name not in self._concept_centroids:
                self._concept_centroids[concept_name] = new_vec_normalized.copy()
                self._concept_sample_counts[concept_name] = 1
                self._concept_timestamps[concept_name] = [datetime.now()]

                return DriftResult(
                    concept_name=concept_name,
                    drift_magnitude=0.0,
                    drift_direction=new_vec_normalized.tolist(),
                    samples_analyzed=1,
                    is_significant=False,
                    recommended_action="ignore",
                )

            # Get current centroid
            current_centroid = self._concept_centroids[concept_name]
            sample_count = self._concept_sample_counts[concept_name]

            # Calculate drift (cosine distance)
            dot_product = np.dot(current_centroid, new_vec_normalized)
            drift_magnitude = 1.0 - max(-1.0, min(1.0, dot_product))

            # Calculate drift direction (unit vector from centroid to new)
            drift_vec = new_vec_normalized - current_centroid
            drift_norm = np.linalg.norm(drift_vec)
            drift_direction = (drift_vec / (drift_norm + 1e-8)).tolist()

            # Update sample count and timestamps
            self._concept_sample_counts[concept_name] = sample_count + 1
            self._concept_timestamps[concept_name].append(datetime.now())

            # Keep bounded history
            if len(self._concept_timestamps[concept_name]) > 1000:
                self._concept_timestamps[concept_name] = \
                    self._concept_timestamps[concept_name][-500:]

            # Determine if drift is significant
            is_significant = (
                drift_magnitude > self.drift_threshold and
                sample_count >= self.min_samples_for_drift
            )

            # Determine recommended action
            if drift_magnitude > 0.5:
                recommended_action = "create_branch"
            elif drift_magnitude > self.drift_threshold:
                recommended_action = "update_centroid"
            else:
                recommended_action = "ignore"

            return DriftResult(
                concept_name=concept_name,
                drift_magnitude=drift_magnitude,
                drift_direction=drift_direction,
                samples_analyzed=sample_count + 1,
                is_significant=is_significant,
                recommended_action=recommended_action,
            )

    def apply_importance_decay(
        self,
        memories: list,
        current_context: str = "",
    ) -> list:
        """
        Apply time-based importance decay to memories.

        Formula: importance' = importance * (0.5 ^ (age_hours / halflife)) * freq_weight
        Where freq_weight = log(1 + access_count) / log(10)

        Args:
            memories: List of MemorySearchResult objects
            current_context: Current context for relevance boosting

        Returns:
            Memories with adjusted scores
        """
        import math

        context_lower = current_context.lower() if current_context else ""

        for memory in memories:
            # Get age in hours
            created_at = memory.metadata.get("created_at")
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    created_at = datetime.now()
            elif not isinstance(created_at, datetime):
                created_at = datetime.now()

            age_hours = (datetime.now() - created_at).total_seconds() / 3600

            # Time decay factor (half-life decay)
            time_decay = math.pow(0.5, age_hours / self.decay_halflife_hours)

            # Frequency weight (logarithmic access count boost)
            access_count = memory.metadata.get("access_count", 1)
            freq_weight = math.log(1 + access_count) / math.log(10)
            freq_weight = max(0.5, min(2.0, freq_weight))  # Clamp to [0.5, 2.0]

            # Context relevance boost
            context_boost = 1.0
            if context_lower and hasattr(memory, 'content'):
                # Boost if content is mentioned in current context
                content_words = memory.content.lower().split()[:5]
                matches = sum(1 for word in content_words if word in context_lower)
                context_boost = 1.0 + matches * 0.1

            # Apply decay
            original_score = memory.score
            memory.score = original_score * time_decay * freq_weight * context_boost

            # Store decay metadata
            if not hasattr(memory, 'metadata') or memory.metadata is None:
                memory.metadata = {}
            memory.metadata["decay_applied"] = {
                "time_decay": time_decay,
                "freq_weight": freq_weight,
                "context_boost": context_boost,
                "original_score": original_score,
            }

        # Re-sort by adjusted scores
        memories.sort(key=lambda m: m.score, reverse=True)
        return memories

    async def evolve_schema(
        self,
        concept_name: str,
        detected_drift: DriftResult,
    ) -> SchemaEvolution:
        """
        Evolve schema based on detected drift.

        Actions based on drift magnitude:
        - High (>0.5): Create new concept branch
        - Moderate (0.3-0.5): Update centroid with EMA
        - Low (<0.3): Log and ignore

        Args:
            concept_name: Name of the concept
            detected_drift: Result from detect_concept_drift

        Returns:
            SchemaEvolution record of action taken
        """
        async with self._lock:
            old_centroid = self._concept_centroids.get(
                concept_name, np.zeros(len(detected_drift.drift_direction))
            ).tolist()

            action_taken = "none"

            if detected_drift.drift_magnitude > 0.5:
                # Create new branch (new concept variant)
                branch_name = f"{concept_name}_v{len(self._drift_history) + 1}"
                new_centroid = np.array(detected_drift.drift_direction)
                self._concept_centroids[branch_name] = new_centroid
                self._concept_sample_counts[branch_name] = 1
                self._concept_timestamps[branch_name] = [datetime.now()]
                action_taken = f"create_branch:{branch_name}"

                logger.info(
                    f"Schema evolution: Created branch '{branch_name}' from '{concept_name}' "
                    f"(drift={detected_drift.drift_magnitude:.3f})"
                )

            elif detected_drift.drift_magnitude > self.drift_threshold:
                # Update centroid using EMA
                current = self._concept_centroids[concept_name]
                new_point = np.array(detected_drift.drift_direction)
                updated = (1 - self.ema_alpha) * current + self.ema_alpha * new_point
                updated = updated / (np.linalg.norm(updated) + 1e-8)  # Normalize
                self._concept_centroids[concept_name] = updated
                action_taken = "update_centroid"

                logger.debug(
                    f"Schema evolution: Updated centroid for '{concept_name}' "
                    f"(drift={detected_drift.drift_magnitude:.3f})"
                )

            else:
                action_taken = "ignore"

            # Record evolution
            evolution = SchemaEvolution(
                timestamp=datetime.now(),
                concept_name=concept_name,
                old_centroid=old_centroid,
                new_centroid=self._concept_centroids.get(
                    concept_name, np.zeros(len(old_centroid))
                ).tolist(),
                drift_magnitude=detected_drift.drift_magnitude,
                action_taken=action_taken,
            )

            self._drift_history.append(evolution)

            # Keep bounded history
            if len(self._drift_history) > 1000:
                self._drift_history = self._drift_history[-500:]

            # Emit drift signal if significant and Nexus available
            if detected_drift.is_significant and NEXUS_AVAILABLE and nexus:
                asyncio.create_task(nexus.emit(
                    type=SignalType.ANOMALY_DETECTED,
                    payload={
                        "event": "concept_drift_detected",
                        "concept": concept_name,
                        "drift_magnitude": detected_drift.drift_magnitude,
                        "action": action_taken,
                    },
                    source="adaptive_schema_manager",
                ))

            return evolution

    def get_stats(self) -> dict:
        """Get schema manager statistics."""
        return {
            "tracked_concepts": len(self._concept_centroids),
            "total_evolutions": len(self._drift_history),
            "recent_evolutions": [
                {
                    "concept": e.concept_name,
                    "magnitude": e.drift_magnitude,
                    "action": e.action_taken,
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in self._drift_history[-5:]
            ],
            "drift_threshold": self.drift_threshold,
            "decay_halflife_hours": self.decay_halflife_hours,
        }


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
