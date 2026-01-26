"""
Farnsworth Nexus: The Neural Event Bus.

The Nexus is the central nervous system of Farnsworth v1.3.
It replaces traditional "function calls" with a high-speed, asynchronous event bus
that allows the Agent Swarm to coordinate in real-time.

Unlike simple message queues, the Nexus uses "Neural Routing" to determine
which agents should react to a given signal based on their specialization.
"""

import asyncio
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional, Awaitable
from loguru import logger
import uuid

class SignalType(Enum):
    # Core Lifecycle
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    # Cognitive Signals
    THOUGHT_EMITTED = "cognitive.thought"
    DECISION_REACHED = "cognitive.decision"
    ANOMALY_DETECTED = "cognitive.anomaly"
    CONFUSIOM_DETECTED = "cognitive.confusion"
    
    # Task Signals
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_BLOCKED = "task.blocked"
    
    # External I/O (The "Connected" part)
    USER_MESSAGE = "io.user.message"
    USER_INTERRUPTION = "io.user.interruption"
    EXTERNAL_ALERT = "io.external.alert"  # e.g. from GitHub, CI/CD

@dataclass
class Signal:
    """A quantified event propagating through the Nexus."""
    type: SignalType
    payload: Dict[str, Any]
    source_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    urgency: float = 0.5  # 0.0 to 1.0
    
    # Neural context (optional embedding vector or semantic tags)
    context_vector: Optional[List[float]] = None
    semantic_tags: List[str] = field(default_factory=list)

class Nexus:
    """
    The central event bus.
    """
    _instance = None
    
    def __init__(self):
        self._subscribers: Dict[SignalType, List[Callable[[Signal], Awaitable[None]]]] = {}
        self._history: List[Signal] = []  # Short-term memory of signals
        self._interceptors: List[Callable[[Signal], bool]] = []
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, signal_type: SignalType, handler: Callable[[Signal], Awaitable[None]]):
        """Connect a synapse (handler) to a specific signal type."""
        if signal_type not in self._subscribers:
            self._subscribers[signal_type] = []
        self._subscribers[signal_type].append(handler)
        logger.debug(f"Nexus: Synapse connected for {signal_type.value}")

    async def broadcast(self, signal: Signal):
        """
        Propagate a signal through the Nexus.
        """
        # 1. Store in short-term history
        self._history.append(signal)
        if len(self._history) > 1000:
            self._history.pop(0)
            
        # 2. Run interceptors (e.g., for safety or filtering)
        for interceptor in self._interceptors:
            if not interceptor(signal):
                logger.warning(f"Nexus: Signal {signal.id} intercepted/blocked")
                return

        # 3. Neural Routing (Find handlers)
        handlers = self._subscribers.get(signal.type, [])
        
        # 4. Asynchronous Propagation
        if handlers:
            logger.debug(f"Nexus: Propagating {signal.type.value} to {len(handlers)} synapses")
            await asyncio.gather(*[h(signal) for h in handlers])
        else:
            logger.trace(f"Nexus: No synapses active for {signal.type.value}")

    async def emit(self, type: SignalType, payload: Dict[str, Any], source: str, urgency: float = 0.5):
        """Helper to create and broadcast a signal."""
        signal = Signal(
            type=type,
            payload=payload,
            source_id=source,
            urgency=urgency
        )
        await self.broadcast(signal)

# Global accessor
nexus = Nexus.get_instance()
