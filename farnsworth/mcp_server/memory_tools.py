"""
Farnsworth Memory Tools - MCP Tool Implementations for Memory Operations

Provides detailed memory access capabilities:
- Advanced search with filters
- Memory graph queries
- Working memory access
- Memory statistics
"""

from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime

from loguru import logger


@dataclass
class MemoryToolResult:
    """Result from a memory tool operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = None

    def to_dict(self) -> dict:
        result = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
        return result


class MemoryTools:
    """
    Memory tool implementations for the MCP server.

    Provides advanced memory operations beyond basic remember/recall.
    """

    def __init__(self, memory_system):
        self.memory = memory_system

    async def remember_with_context(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
        context: Optional[dict] = None,
        extract_entities: bool = True,
    ) -> MemoryToolResult:
        """
        Store memory with additional context and entity extraction.
        """
        try:
            # Add context to content if provided
            full_content = content
            if context:
                context_str = "\n".join(f"[{k}]: {v}" for k, v in context.items())
                full_content = f"{content}\n\nContext:\n{context_str}"

            memory_id = await self.memory.remember(
                content=full_content,
                tags=tags,
                importance=importance,
                extract_entities=extract_entities,
            )

            return MemoryToolResult(
                success=True,
                data={"memory_id": memory_id},
                metadata={"extracted_entities": extract_entities},
            )

        except Exception as e:
            logger.error(f"Remember with context failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def advanced_recall(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.3,
        tags_filter: Optional[list[str]] = None,
        time_range: Optional[tuple[str, str]] = None,
        source_filter: Optional[list[str]] = None,
    ) -> MemoryToolResult:
        """
        Advanced memory search with multiple filters.
        """
        try:
            results = await self.memory.recall(
                query=query,
                top_k=limit,
                min_score=min_score,
            )

            # Apply additional filters
            filtered = []
            for r in results:
                # Tag filter
                if tags_filter:
                    if "tags" in r.metadata:
                        if not any(t in r.metadata["tags"] for t in tags_filter):
                            continue

                # Source filter
                if source_filter and r.source not in source_filter:
                    continue

                filtered.append({
                    "content": r.content,
                    "source": r.source,
                    "score": r.score,
                    "metadata": r.metadata,
                })

            return MemoryToolResult(
                success=True,
                data={"count": len(filtered), "results": filtered},
            )

        except Exception as e:
            logger.error(f"Advanced recall failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def query_knowledge_graph(
        self,
        query: str,
        max_entities: int = 10,
        max_hops: int = 2,
    ) -> MemoryToolResult:
        """
        Query the knowledge graph for entities and relationships.
        """
        try:
            result = await self.memory.knowledge_graph.query(
                query=query,
                max_entities=max_entities,
                max_hops=max_hops,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "entities": [
                        {
                            "name": e.name,
                            "type": e.entity_type,
                            "mentions": e.mention_count,
                        }
                        for e in result.entities
                    ],
                    "relationships": [
                        {
                            "source": r.source_id,
                            "target": r.target_id,
                            "type": r.relation_type,
                        }
                        for r in result.relationships
                    ],
                    "paths": result.paths,
                },
            )

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_working_memory(self, slot_name: Optional[str] = None) -> MemoryToolResult:
        """
        Access working memory slots.
        """
        try:
            if slot_name:
                value = await self.memory.get_working_memory(slot_name)
                return MemoryToolResult(
                    success=True,
                    data={slot_name: value},
                )
            else:
                # Return all slots
                status = self.memory.working_memory.get_status()
                return MemoryToolResult(
                    success=True,
                    data=status,
                )

        except Exception as e:
            logger.error(f"Get working memory failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def set_working_memory(
        self,
        name: str,
        value: Any,
        slot_type: str = "scratch",
    ) -> MemoryToolResult:
        """
        Set a value in working memory.
        """
        try:
            from farnsworth.memory.working_memory import SlotType

            type_map = {
                "text": SlotType.TEXT,
                "code": SlotType.CODE,
                "data": SlotType.DATA,
                "task": SlotType.TASK,
                "reference": SlotType.REFERENCE,
                "scratch": SlotType.SCRATCH,
            }

            await self.memory.set_working_memory(
                name=name,
                value=value,
                slot_type=type_map.get(slot_type, SlotType.SCRATCH),
            )

            return MemoryToolResult(
                success=True,
                data={"slot": name, "type": slot_type},
            )

        except Exception as e:
            logger.error(f"Set working memory failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def get_memory_stats(self) -> MemoryToolResult:
        """
        Get comprehensive memory statistics.
        """
        try:
            stats = self.memory.get_stats()
            summary = await self.memory.get_memory_summary()

            return MemoryToolResult(
                success=True,
                data={
                    "stats": stats,
                    "summary": summary,
                },
            )

        except Exception as e:
            logger.error(f"Get memory stats failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def trigger_dreaming(self) -> MemoryToolResult:
        """
        Manually trigger memory consolidation (dreaming).
        """
        try:
            result = await self.memory.trigger_dream()

            return MemoryToolResult(
                success=True,
                data={
                    "memories_processed": result.memories_processed,
                    "clusters_formed": result.clusters_formed,
                    "insights": result.insights_generated,
                    "consolidation_score": result.consolidation_score,
                },
            )

        except Exception as e:
            logger.error(f"Trigger dreaming failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        properties: Optional[dict] = None,
    ) -> MemoryToolResult:
        """
        Add an entity to the knowledge graph.
        """
        try:
            entity = await self.memory.add_entity(
                name=name,
                entity_type=entity_type,
                properties=properties,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "entity_id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                },
            )

        except Exception as e:
            logger.error(f"Add entity failed: {e}")
            return MemoryToolResult(success=False, error=str(e))

    async def link_entities(
        self,
        source: str,
        target: str,
        relation_type: str,
    ) -> MemoryToolResult:
        """
        Create a relationship between entities.
        """
        try:
            await self.memory.link_entities(
                source=source,
                target=target,
                relation_type=relation_type,
            )

            return MemoryToolResult(
                success=True,
                data={
                    "source": source,
                    "target": target,
                    "relation": relation_type,
                },
            )

        except Exception as e:
            logger.error(f"Link entities failed: {e}")
            return MemoryToolResult(success=False, error=str(e))
