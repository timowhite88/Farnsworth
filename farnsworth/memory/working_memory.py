"""
Farnsworth Working Memory - In-Context Scratchpad

Novel Approaches:
1. Structured Scratchpad - Type-safe slots for different data types
2. Temporal Decay - Automatic relevance decay over time
3. Cross-Reference Tracking - Automatic linking between related items
4. Snapshot/Restore - Point-in-time memory states
"""

import asyncio
import copy
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic
from collections import deque

from loguru import logger


class SlotType(Enum):
    """Types of working memory slots."""
    TEXT = "text"
    CODE = "code"
    DATA = "data"
    TASK = "task"
    REFERENCE = "reference"
    SCRATCH = "scratch"


@dataclass
class WorkingMemorySlot:
    """A typed slot in working memory."""
    name: str
    slot_type: SlotType
    content: Any
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    importance: float = 0.5
    references: list[str] = field(default_factory=list)  # Other slot names
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if slot has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def get_age_seconds(self) -> float:
        """Get age of slot in seconds."""
        return (datetime.now() - self.created_at).total_seconds()

    def to_dict(self) -> dict:
        """Serialize slot."""
        return {
            "name": self.name,
            "slot_type": self.slot_type.value,
            "content": self.content if isinstance(self.content, (str, int, float, bool, list, dict)) else str(self.content),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "importance": self.importance,
            "references": self.references,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkingMemorySlot":
        """Deserialize slot."""
        return cls(
            name=data["name"],
            slot_type=SlotType(data["slot_type"]),
            content=data["content"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            importance=data.get("importance", 0.5),
            references=data.get("references", []),
            access_count=data.get("access_count", 0),
        )


@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of working memory."""
    timestamp: datetime
    slots: dict[str, dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "slots": self.slots,
            "metadata": self.metadata,
        }


class WorkingMemory:
    """
    In-context scratchpad for active task processing.

    Provides structured storage for:
    - Current task context
    - Intermediate results
    - Active references
    - Scratch calculations

    Features:
    - Typed slots with validation
    - Automatic decay and cleanup
    - Cross-reference tracking
    - Snapshot/restore for backtracking
    """

    def __init__(
        self,
        max_slots: int = 50,
        default_ttl_minutes: int = 30,
        decay_rate: float = 0.1,
    ):
        self.max_slots = max_slots
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.decay_rate = decay_rate

        self.slots: dict[str, WorkingMemorySlot] = {}
        self.snapshots: deque[MemorySnapshot] = deque(maxlen=10)
        self.history: list[dict] = []  # Operation history

        self._lock = asyncio.Lock()

    async def set(
        self,
        name: str,
        content: Any,
        slot_type: SlotType = SlotType.SCRATCH,
        importance: float = 0.5,
        ttl_minutes: Optional[int] = None,
        references: Optional[list[str]] = None,
    ) -> WorkingMemorySlot:
        """
        Set a value in working memory.

        Args:
            name: Slot name (unique identifier)
            content: Content to store
            slot_type: Type of content
            importance: Importance score (0-1)
            ttl_minutes: Time-to-live in minutes (None for default)
            references: Names of related slots

        Returns:
            The created/updated slot
        """
        async with self._lock:
            # Check capacity
            if name not in self.slots and len(self.slots) >= self.max_slots:
                await self._evict_least_important()

            expires_at = None
            if ttl_minutes is not None:
                expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
            elif self.default_ttl:
                expires_at = datetime.now() + self.default_ttl

            if name in self.slots:
                # Update existing slot
                slot = self.slots[name]
                slot.content = content
                slot.updated_at = datetime.now()
                slot.expires_at = expires_at
                slot.importance = importance
                if references:
                    slot.references = list(set(slot.references + references))
            else:
                # Create new slot
                slot = WorkingMemorySlot(
                    name=name,
                    slot_type=slot_type,
                    content=content,
                    expires_at=expires_at,
                    importance=importance,
                    references=references or [],
                )
                self.slots[name] = slot

            # Update cross-references
            self._update_references(name, references or [])

            # Record history
            self._record_operation("set", name, slot_type.value)

            return slot

    async def get(self, name: str, default: Any = None) -> Any:
        """
        Get a value from working memory.

        Returns the content (not the slot) or default if not found/expired.
        """
        async with self._lock:
            slot = self.slots.get(name)

            if slot is None:
                return default

            if slot.is_expired():
                del self.slots[name]
                return default

            # Update access tracking
            slot.access_count += 1
            slot.updated_at = datetime.now()

            # Apply decay to other slots
            self._apply_decay(exclude=name)

            return slot.content

    async def get_slot(self, name: str) -> Optional[WorkingMemorySlot]:
        """Get the full slot object (not just content)."""
        async with self._lock:
            slot = self.slots.get(name)
            if slot and not slot.is_expired():
                return slot
            return None

    async def delete(self, name: str) -> bool:
        """Delete a slot from working memory."""
        async with self._lock:
            if name in self.slots:
                # Remove from other slots' references
                for other_slot in self.slots.values():
                    if name in other_slot.references:
                        other_slot.references.remove(name)

                del self.slots[name]
                self._record_operation("delete", name)
                return True
            return False

    async def clear(self, slot_type: Optional[SlotType] = None):
        """Clear working memory, optionally by type."""
        async with self._lock:
            if slot_type is None:
                self.slots.clear()
            else:
                self.slots = {
                    k: v for k, v in self.slots.items()
                    if v.slot_type != slot_type
                }
            self._record_operation("clear", str(slot_type) if slot_type else "all")

    async def _evict_least_important(self):
        """Evict the least important slot."""
        if not self.slots:
            return

        # Score: lower = more evictable
        def eviction_score(slot: WorkingMemorySlot) -> float:
            age_factor = 1.0 / (1.0 + slot.get_age_seconds() / 600)  # Decay over 10 min
            access_factor = min(1.0, slot.access_count / 5)
            ref_factor = min(1.0, len(slot.references) / 3)

            return (
                slot.importance * 0.4 +
                age_factor * 0.3 +
                access_factor * 0.2 +
                ref_factor * 0.1
            )

        min_slot = min(self.slots.values(), key=eviction_score)
        del self.slots[min_slot.name]
        logger.debug(f"Evicted working memory slot: {min_slot.name}")

    def _update_references(self, source_name: str, target_names: list[str]):
        """Update bidirectional references."""
        for target_name in target_names:
            if target_name in self.slots:
                if source_name not in self.slots[target_name].references:
                    self.slots[target_name].references.append(source_name)

    def _apply_decay(self, exclude: Optional[str] = None):
        """Apply temporal decay to importance scores."""
        for name, slot in self.slots.items():
            if name != exclude:
                slot.importance *= (1.0 - self.decay_rate * 0.01)

    def _record_operation(self, op_type: str, slot_name: str, extra: str = ""):
        """Record operation for history."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": op_type,
            "slot": slot_name,
            "extra": extra,
        })

        # Keep bounded history
        if len(self.history) > 100:
            self.history = self.history[-50:]

    async def snapshot(self, metadata: Optional[dict] = None) -> MemorySnapshot:
        """Create a snapshot of current state."""
        async with self._lock:
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                slots={name: slot.to_dict() for name, slot in self.slots.items()},
                metadata=metadata or {},
            )
            self.snapshots.append(snapshot)
            return snapshot

    async def restore(self, snapshot: MemorySnapshot):
        """Restore working memory from a snapshot."""
        async with self._lock:
            self.slots = {
                name: WorkingMemorySlot.from_dict(data)
                for name, data in snapshot.slots.items()
            }
            self._record_operation("restore", f"snapshot_{snapshot.timestamp.isoformat()}")

    async def get_related(self, name: str, depth: int = 1) -> list[WorkingMemorySlot]:
        """Get slots related to the given slot."""
        async with self._lock:
            if name not in self.slots:
                return []

            related = set()
            to_explore = [name]
            explored = set()

            for _ in range(depth):
                next_explore = []
                for current in to_explore:
                    if current in explored:
                        continue
                    explored.add(current)

                    if current in self.slots:
                        for ref in self.slots[current].references:
                            if ref not in explored:
                                related.add(ref)
                                next_explore.append(ref)
                to_explore = next_explore

            return [self.slots[r] for r in related if r in self.slots]

    async def cleanup_expired(self) -> int:
        """Remove all expired slots."""
        async with self._lock:
            expired = [name for name, slot in self.slots.items() if slot.is_expired()]
            for name in expired:
                del self.slots[name]
            return len(expired)

    def get_by_type(self, slot_type: SlotType) -> list[WorkingMemorySlot]:
        """Get all slots of a specific type."""
        return [slot for slot in self.slots.values() if slot.slot_type == slot_type]

    def to_context_string(self, max_length: int = 2000) -> str:
        """
        Convert working memory to a string for LLM context.

        Prioritizes important and recently accessed slots.
        """
        # Sort by importance * recency
        sorted_slots = sorted(
            self.slots.values(),
            key=lambda s: s.importance * (1.0 / (1.0 + s.get_age_seconds() / 600)),
            reverse=True,
        )

        parts = ["[Working Memory]"]
        current_length = len(parts[0])

        for slot in sorted_slots:
            content_str = str(slot.content)[:500]  # Truncate long content
            slot_str = f"\n[{slot.slot_type.value}:{slot.name}] {content_str}"

            if current_length + len(slot_str) > max_length:
                break

            parts.append(slot_str)
            current_length += len(slot_str)

        return "".join(parts)

    def get_status(self) -> dict:
        """Get working memory status for UI."""
        return {
            "slot_count": len(self.slots),
            "max_slots": self.max_slots,
            "slots_by_type": {
                st.value: len([s for s in self.slots.values() if s.slot_type == st])
                for st in SlotType
            },
            "total_importance": sum(s.importance for s in self.slots.values()),
            "snapshot_count": len(self.snapshots),
            "recent_operations": self.history[-5:] if self.history else [],
        }
