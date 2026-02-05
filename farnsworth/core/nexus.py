"""
Farnsworth Nexus: The Neural Event Bus.

The Nexus is the central nervous system of Farnsworth v1.3.
It replaces traditional "function calls" with a high-speed, asynchronous event bus
that allows the Agent Swarm to coordinate in real-time.

UPDATES:
- Added Middleware pipeline support
- Added Priority Queues (via urgency sort)
- Added 'Signal Black Box' for debugging

AGI UPGRADES (v1.4):
- Priority queue with urgency-based ordering
- Semantic/vector-based subscription (neural routing)
- Self-evolving middleware (dynamic subscriber modification)
- Spontaneous thought generator (idle creativity)
- Signal persistence and collective memory recall
- Backpressure handling and rate limiting
"""

import asyncio
import inspect
import random
import math
import uuid
import json
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any, Optional, Awaitable, Tuple, Set, Union
from loguru import logger


# =============================================================================
# AGI v1.8: SAFE HANDLER INVOCATION
# =============================================================================

async def _safe_invoke_handler(handler: Callable, signal: Any) -> Any:
    """
    Safely invoke a signal handler, handling both sync and async handlers.

    AGI v1.8: Prevents 'asyncio.Future, a coroutine or an awaitable is required' errors
    by properly wrapping sync handlers and handling non-awaitable returns.

    Args:
        handler: The handler function (sync or async)
        signal: The signal to pass to the handler

    Returns:
        The handler result, or None on error
    """
    try:
        result = handler(signal)
        # If result is a coroutine or awaitable, await it
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            return await result
        # If it's an awaitable (has __await__), await it
        if hasattr(result, '__await__'):
            return await result
        # Otherwise return the sync result
        return result
    except Exception as e:
        logger.error(f"Nexus: Handler {getattr(handler, '__name__', 'unknown')} failed: {e}")
        return None


# =============================================================================
# SIGNAL TYPES (must be defined before dataclasses that reference it)
# =============================================================================

class SignalType(Enum):
    # Core Lifecycle
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"

    # Cognitive Signals
    THOUGHT_EMITTED = "cognitive.thought"
    DECISION_REACHED = "cognitive.decision"
    ANOMALY_DETECTED = "cognitive.anomaly"
    CONFUSION_DETECTED = "cognitive.confusion"
    MEMORY_CONSOLIDATION = "cognitive.memory_consolidation"

    # Task Signals
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_BLOCKED = "task.blocked"

    # External I/O
    USER_MESSAGE = "io.user.message"
    USER_INTERRUPTION = "io.user.interruption"
    EXTERNAL_ALERT = "io.external.alert"

    # P2P / Network Signals
    EXTERNAL_EVENT = "p2p.external_event"
    TASK_RECEIVED = "p2p.task_received"
    PEER_CONNECTED = "p2p.peer_connected"
    PEER_DISCONNECTED = "p2p.peer_disconnected"
    SKILL_RECEIVED = "p2p.skill_received"

    # Dialogue / Deliberation Signals
    DIALOGUE_STARTED = "dialogue.started"
    DIALOGUE_PROPOSE = "dialogue.propose"
    DIALOGUE_CRITIQUE = "dialogue.critique"
    DIALOGUE_REFINE = "dialogue.refine"
    DIALOGUE_VOTE = "dialogue.vote"
    DIALOGUE_CONSENSUS = "dialogue.consensus"
    DIALOGUE_COMPLETED = "dialogue.completed"
    DIALOGUE_TOOL_DECISION = "dialogue.tool_decision"

    # Collective Resonance Signals (Inter-Collective Communication)
    COLLECTIVE_THOUGHT = "resonance.collective_thought"      # Visible deliberation thought
    RESONANT_THOUGHT = "resonance.resonant_thought"          # Thought shared via P2P
    RESONANCE_RECEIVED = "resonance.received"                # Incoming thought from another collective
    RESONANCE_BROADCAST = "resonance.broadcast"              # Outgoing thought to P2P network

    # Handler Benchmark Signals (AGI v1.7 - Dynamic Selection)
    HANDLER_BENCHMARK_START = "benchmark.start"              # Tournament starting
    HANDLER_BENCHMARK_RESULT = "benchmark.result"            # Single handler result
    HANDLER_EVALUATION = "benchmark.evaluation"              # Collaborative evaluation request
    BEST_HANDLER_SELECTED = "benchmark.selected"             # Winner announced
    HANDLER_PERFORMANCE_UPDATE = "benchmark.performance"     # Fitness update

    # Sub-Swarm Signals (AGI v1.7 - API-Triggered Swarms)
    SUBSWARM_SPAWN = "subswarm.spawn"                        # Spin up sub-swarm
    SUBSWARM_COMPLETE = "subswarm.complete"                  # Sub-swarm finished
    SUBSWARM_MERGE = "subswarm.merge"                        # Merge results back

    # Provider Session Signals (AGI v1.7 - Persistent Sessions)
    SESSION_CREATED = "session.created"                      # tmux/persistent session created
    SESSION_COMMAND = "session.command"                      # Command sent to session
    SESSION_OUTPUT = "session.output"                        # Output captured
    SESSION_DESTROYED = "session.destroyed"                  # Session ended

    # =========================================================================
    # AGI v1.8 SIGNALS
    # =========================================================================

    # LangGraph Workflow Signals
    WORKFLOW_STARTED = "workflow.started"                    # Workflow execution started
    WORKFLOW_NODE_ENTERED = "workflow.node_entered"          # Entered a workflow node
    WORKFLOW_NODE_EXITED = "workflow.node_exited"            # Exited a workflow node
    WORKFLOW_CHECKPOINT = "workflow.checkpoint"              # Checkpoint created
    WORKFLOW_RESUMED = "workflow.resumed"                    # Resumed from checkpoint
    WORKFLOW_COMPLETED = "workflow.completed"                # Workflow finished successfully
    WORKFLOW_FAILED = "workflow.failed"                      # Workflow execution failed

    # Cross-Agent Memory Signals
    MEMORY_CONTEXT_INJECTED = "memory.context_injected"      # Context injected to namespace
    MEMORY_HANDOFF_PREPARED = "memory.handoff_prepared"      # Handoff context prepared
    MEMORY_NAMESPACE_CREATED = "memory.namespace_created"    # New memory namespace created
    MEMORY_TEAM_MERGED = "memory.team_merged"                # Team memories merged

    # MCP Standardization Signals
    MCP_TOOL_REGISTERED = "mcp.tool_registered"              # New tool registered
    MCP_TOOL_CALLED = "mcp.tool_called"                      # Tool invoked
    MCP_AGENT_CONNECTED = "mcp.agent_connected"              # Agent connected via MCP
    MCP_CAPABILITY_DISCOVERED = "mcp.capability_discovered"  # Capabilities discovered

    # A2A Protocol Signals
    A2A_SESSION_REQUESTED = "a2a.session_requested"          # Session requested
    A2A_SESSION_STARTED = "a2a.session_started"              # Session started
    A2A_SESSION_ENDED = "a2a.session_ended"                  # Session ended
    A2A_TASK_AUCTIONED = "a2a.task_auctioned"                # Task put up for auction
    A2A_BID_RECEIVED = "a2a.bid_received"                    # Bid received for auction
    A2A_TASK_ASSIGNED = "a2a.task_assigned"                  # Task assigned to winner
    A2A_CONTEXT_SHARED = "a2a.context_shared"                # Context shared between agents
    A2A_SKILL_TRANSFERRED = "a2a.skill_transferred"          # Skill transferred

    # Quantum Computing Signals (AGI v1.8.2)
    QUANTUM_JOB_SUBMITTED = "quantum.job_submitted"          # Circuit submitted to backend
    QUANTUM_JOB_COMPLETED = "quantum.job_completed"          # Circuit execution finished
    QUANTUM_RESULT = "quantum.result"                        # Measurement results available
    QUANTUM_ERROR = "quantum.error"                          # Quantum execution error
    QUANTUM_CALIBRATION = "quantum.calibration"              # Backend calibration data
    QUANTUM_USAGE_WARNING = "quantum.usage_warning"          # Hardware quota warning
    QUANTUM_EVOLUTION_STARTED = "quantum.evolution_started"  # QGA evolution started
    QUANTUM_EVOLUTION_COMPLETE = "quantum.evolution_complete"  # QGA evolution finished
    QUANTUM_PROOF_GENERATED = "quantum.proof_generated"      # Proof image generated
    QUANTUM_PROOF_POSTED = "quantum.proof_posted"            # Proof posted to X

    # Multi-Channel Messaging Signals (AGI v1.8.3)
    CHANNEL_CONNECTED = "channel.connected"                  # Channel adapter connected
    CHANNEL_DISCONNECTED = "channel.disconnected"            # Channel adapter disconnected
    CHANNEL_MESSAGE_RECEIVED = "channel.message_received"    # Inbound message from any channel
    CHANNEL_MESSAGE_SENT = "channel.message_sent"            # Outbound message sent
    CHANNEL_USER_PAIRED = "channel.user_paired"              # User paired via pairing code
    CHANNEL_ACCESS_DENIED = "channel.access_denied"          # Access denied to user
    CHANNEL_MEDIA_RECEIVED = "channel.media_received"        # Media attachment received
    CHANNEL_REACTION_RECEIVED = "channel.reaction"           # Reaction/emoji received
    CHANNEL_TYPING_STARTED = "channel.typing_started"        # User started typing
    CHANNEL_ERROR = "channel.error"                          # Channel error occurred

    # =========================================================================
    # CLI/GUI Signals (AGI v1.8.4)
    # =========================================================================

    # CLI Session Signals
    CLI_SESSION_START = "cli.session.start"                  # Rich CLI session started
    CLI_SESSION_END = "cli.session.end"                      # Rich CLI session ended
    CLI_COMMAND = "cli.command"                              # CLI command executed

    # GUI Session Signals
    GUI_SESSION_START = "gui.session.start"                  # Streamlit GUI session started
    GUI_CANVAS_RENDER = "gui.canvas.render"                  # Matplotlib canvas rendered

    # User-to-Swarm Signals
    USER_A2A_REQUEST = "user.a2a.request"                    # User requesting A2A session
    USER_A2A_JOINED = "user.a2a.joined"                      # User joined A2A session
    USER_DELIBERATION_REQUEST = "user.deliberation.request"  # User requesting deliberation

    # =========================================================================
    # A2A Mesh Signals (AGI v1.8.4)
    # =========================================================================

    # Mesh Connectivity
    MESH_PEER_ANNOUNCE = "mesh.peer.announce"                # Peer announcing itself to mesh
    MESH_PEER_DISCOVER = "mesh.peer.discover"                # Peer discovery request
    MESH_PEER_HEARTBEAT = "mesh.peer.heartbeat"              # Peer heartbeat for liveness

    # Model-to-Model Communication
    M2M_INSIGHT = "m2m.insight"                              # Model sharing an insight
    M2M_QUERY = "m2m.query"                                  # Model querying another model
    M2M_RESPONSE = "m2m.response"                            # Model responding to query
    M2M_COLLABORATE = "m2m.collaborate"                      # Model requesting collaboration

    # Collective Bridge
    COLLECTIVE_DISPATCH = "collective.dispatch"              # Dispatching consensus to agents
    COLLECTIVE_ESCALATE = "collective.escalate"              # Agent escalating to collective
    COLLECTIVE_VOTE_REQUEST = "collective.vote_request"      # Agent requesting collective vote
    COLLECTIVE_SYNC = "collective.sync"                      # Syncing between collectives

    # Sub-Swarm Management (Extended)
    SUBSWARM_FORM = "subswarm.form"                          # Forming a new sub-swarm
    SUBSWARM_JOIN = "subswarm.join"                          # Agent joining sub-swarm
    SUBSWARM_LEAVE = "subswarm.leave"                        # Agent leaving sub-swarm


# =============================================================================
# NEURAL ROUTING DATASTRUCTURES (AGI Upgrade)
# =============================================================================

@dataclass
class SemanticSubscription:
    """
    A subscription based on semantic similarity rather than exact SignalType.

    Enables emergent routing where signals find handlers based on
    context_vector similarity, not hardcoded types.
    """
    subscription_id: str
    handler: Callable[["Signal"], Awaitable[None]]
    target_vector: List[float]  # The semantic space this handler is interested in
    similarity_threshold: float = 0.85  # Minimum cosine similarity to trigger
    signal_types: Optional[Set[SignalType]] = None  # Optional type filter

    # Performance tracking for evolution
    invocations: int = 0
    successful_invocations: int = 0
    avg_processing_time: float = 0.0
    fitness_score: float = 0.5  # Used by evolution middleware

    created_at: datetime = field(default_factory=datetime.now)
    last_invoked: Optional[datetime] = None


@dataclass
class SubscriptionFitness:
    """Tracks fitness of subscriptions for evolutionary optimization."""
    subscription_id: str
    signal_type: Optional[SignalType]
    handler_name: str

    # Fitness metrics
    invocation_count: int = 0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0
    relevance_score: float = 0.5  # How relevant were handled signals

    # Evolution state
    fitness: float = 0.5
    generation: int = 0
    mutations: List[str] = field(default_factory=list)


@dataclass
class SignalBatch:
    """A batch of signals for efficient processing."""
    signals: List["Signal"]
    created_at: datetime = field(default_factory=datetime.now)
    priority: float = 0.5  # Average urgency


@dataclass
class BackpressureState:
    """Tracks backpressure for rate limiting."""
    queue_depth: int = 0
    max_queue_depth: int = 1000
    signals_per_second: float = 0.0
    max_signals_per_second: float = 100.0
    is_throttling: bool = False
    dropped_signals: int = 0
    last_measured: datetime = field(default_factory=datetime.now)
    recent_counts: List[int] = field(default_factory=list)  # Per-second counts


@dataclass
class SpontaneousThoughtConfig:
    """Configuration for spontaneous thought generation."""
    enabled: bool = True
    min_idle_seconds: float = 30.0  # Minimum idle time before thinking
    max_idle_seconds: float = 180.0  # Maximum wait between thoughts
    creativity_temperature: float = 0.7  # 0=analytical, 1=highly creative
    thought_probability: float = 0.3  # Chance of generating thought when triggered
    focus_concepts: List[str] = field(default_factory=list)
    context_vector: Optional[List[float]] = None


@dataclass
class Signal:
    """A quantified event propagating through the Nexus."""
    type: SignalType
    payload: Dict[str, Any]
    source_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    urgency: float = 0.5  # 0.0 to 1.0 (Higher = processed first)
    
    context_vector: Optional[List[float]] = None
    semantic_tags: List[str] = field(default_factory=list)

MiddlewareFunc = Callable[[Signal], bool] # Returns True to continue, False to block
EvolutionMiddlewareFunc = Callable[[Signal, "Nexus"], bool]  # Can modify Nexus


class Nexus:
    """
    The central event bus with Neural Routing and Middleware.

    AGI UPGRADES:
    - Priority queue for urgency-based processing
    - Semantic subscription for vector-based routing
    - Self-evolving middleware that modifies handler graphs
    - Spontaneous thought generation during idle periods
    - Signal persistence for collective memory
    - Backpressure handling for stability
    """
    _instance = None

    def __init__(self):
        # Core subscription registry
        self._subscribers: Dict[SignalType, List[Callable[[Signal], Awaitable[None]]]] = {}
        self._history: List[Signal] = []  # Black Box
        self._middleware: List[MiddlewareFunc] = []
        self._lock = asyncio.Lock()

        # AGI Upgrades
        self._priority_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._semantic_subscriptions: List[SemanticSubscription] = []
        self._subscription_fitness: Dict[str, SubscriptionFitness] = {}
        self._evolution_middleware: List[EvolutionMiddlewareFunc] = []
        self._backpressure = BackpressureState()

        # Spontaneous thought system
        self._thought_config = SpontaneousThoughtConfig()
        self._thought_generator_task: Optional[asyncio.Task] = None
        self._last_activity: datetime = datetime.now()
        self._thought_llm_fn: Optional[Callable] = None

        # Signal persistence
        self._persistent_history: List[Dict] = []  # Serializable history
        self._max_persistent_history: int = 10000
        self._archival_callback: Optional[Callable] = None

        # Processing worker
        self._worker_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

        # Evolution tracking
        self._evolution_generation: int = 0
        self._evolution_history: List[Dict] = []

        # AGI v1.8: LangGraph hybrid integration
        self._langgraph_hybrid = None
        self._workflow_state_cache: Dict[str, Dict] = {}

        logger.info("Nexus initialized with AGI upgrades")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # =========================================================================
    # CORE SUBSCRIPTION (Original + Enhanced)
    # =========================================================================

    def subscribe(
        self,
        signal_type: SignalType,
        handler: Callable[[Signal], Awaitable[None]],
        track_fitness: bool = True,
    ):
        """Connect a synapse (handler) to a specific signal type."""
        if signal_type not in self._subscribers:
            self._subscribers[signal_type] = []
        self._subscribers[signal_type].append(handler)

        # Track fitness for evolution
        if track_fitness:
            handler_name = getattr(handler, '__name__', str(handler))
            fitness_id = f"{signal_type.value}:{handler_name}"
            self._subscription_fitness[fitness_id] = SubscriptionFitness(
                subscription_id=fitness_id,
                signal_type=signal_type,
                handler_name=handler_name,
            )

    def unsubscribe(self, signal_type: SignalType, handler: Callable):
        """Remove a handler from a signal type."""
        if signal_type in self._subscribers:
            try:
                self._subscribers[signal_type].remove(handler)
            except ValueError:
                pass

    # =========================================================================
    # SEMANTIC SUBSCRIPTION (AGI Upgrade)
    # =========================================================================

    def subscribe_semantic(
        self,
        handler: Callable[[Signal], Awaitable[None]],
        target_vector: List[float],
        similarity_threshold: float = 0.85,
        signal_types: Optional[Set[SignalType]] = None,
    ) -> str:
        """
        Subscribe based on semantic similarity of context_vector.

        Enables emergent routing where signals find handlers based on
        meaning, not just hardcoded types.

        Args:
            handler: Async function to call when similarity threshold met
            target_vector: The semantic space this handler is interested in
            similarity_threshold: Minimum cosine similarity to trigger (0-1)
            signal_types: Optional filter to specific signal types

        Returns:
            subscription_id for later unsubscription
        """
        subscription_id = f"semantic_{uuid.uuid4().hex[:8]}"

        subscription = SemanticSubscription(
            subscription_id=subscription_id,
            handler=handler,
            target_vector=target_vector,
            similarity_threshold=similarity_threshold,
            signal_types=signal_types,
        )

        self._semantic_subscriptions.append(subscription)
        logger.debug(f"Nexus: Added semantic subscription {subscription_id}")

        return subscription_id

    def unsubscribe_semantic(self, subscription_id: str) -> bool:
        """Remove a semantic subscription."""
        for i, sub in enumerate(self._semantic_subscriptions):
            if sub.subscription_id == subscription_id:
                self._semantic_subscriptions.pop(i)
                return True
        return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _dispatch_semantic(self, signal: Signal) -> int:
        """
        Dispatch signal to semantic subscribers based on context_vector similarity.

        Returns number of handlers invoked.
        """
        if not signal.context_vector or not self._semantic_subscriptions:
            return 0

        invoked = 0
        tasks = []

        for sub in self._semantic_subscriptions:
            # Optional type filter
            if sub.signal_types and signal.type not in sub.signal_types:
                continue

            # Compute similarity
            similarity = self._cosine_similarity(signal.context_vector, sub.target_vector)

            if similarity >= sub.similarity_threshold:
                # Track for evolution
                sub.invocations += 1
                sub.last_invoked = datetime.now()

                async def invoke_with_tracking(handler, subscription, sig):
                    import time
                    start = time.time()
                    try:
                        # AGI v1.8: Use safe handler invocation
                        await _safe_invoke_handler(handler, sig)
                        subscription.successful_invocations += 1
                        latency = (time.time() - start) * 1000
                        # Update rolling average
                        n = subscription.invocations
                        subscription.avg_processing_time = (
                            (subscription.avg_processing_time * (n - 1) + latency) / n
                        )
                    except Exception as e:
                        logger.error(f"Nexus: Semantic handler error: {e}")

                tasks.append(invoke_with_tracking(sub.handler, sub, signal))
                invoked += 1

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return invoked

    # =========================================================================
    # MIDDLEWARE (Original + Evolution)
    # =========================================================================

    def add_middleware(self, func: MiddlewareFunc):
        """Add a middleware function that runs on every signal."""
        self._middleware.append(func)

    def add_evolution_middleware(self, func: EvolutionMiddlewareFunc):
        """Add middleware that can modify the Nexus (self-evolving)."""
        self._evolution_middleware.append(func)

    def remove_middleware(self, func: MiddlewareFunc) -> bool:
        """Remove a middleware function."""
        try:
            self._middleware.remove(func)
            return True
        except ValueError:
            return False

    # =========================================================================
    # PRIORITY QUEUE PROCESSING (AGI Upgrade)
    # =========================================================================

    async def start(self):
        """Start the priority queue worker and spontaneous thought generator."""
        if self._is_running:
            return

        self._is_running = True
        self._worker_task = asyncio.create_task(self._process_queue())

        if self._thought_config.enabled:
            self._thought_generator_task = asyncio.create_task(self._spontaneous_thought_loop())

        logger.info("Nexus started (priority queue + thought generator)")

    async def stop(self):
        """Stop the Nexus processing."""
        self._is_running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._thought_generator_task:
            self._thought_generator_task.cancel()
            try:
                await self._thought_generator_task
            except asyncio.CancelledError:
                pass

        logger.info("Nexus stopped")

    async def _process_queue(self):
        """Worker task that processes signals by priority."""
        while self._is_running:
            try:
                # Get highest priority signal (negative urgency for max-heap behavior)
                neg_urgency, timestamp, signal = await asyncio.wait_for(
                    self._priority_queue.get(),
                    timeout=1.0,
                )

                # Update backpressure
                self._backpressure.queue_depth = self._priority_queue.qsize()

                # Dispatch to handlers
                await self._dispatch_signal(signal)

                self._priority_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Nexus: Queue processing error: {e}")

    async def _dispatch_signal(self, signal: Signal):
        """Dispatch signal to all relevant handlers."""
        import time
        start_time = time.time()

        # 1. Type-based handlers
        # AGI v1.8: Use safe handler invocation to support both sync and async handlers
        handlers = self._subscribers.get(signal.type, [])
        if handlers:
            results = await asyncio.gather(
                *[_safe_invoke_handler(h, signal) for h in handlers],
                return_exceptions=True
            )

            # Track fitness
            for i, result in enumerate(results):
                handler = handlers[i]
                handler_name = getattr(handler, '__name__', str(handler))
                fitness_id = f"{signal.type.value}:{handler_name}"

                if fitness_id in self._subscription_fitness:
                    fitness = self._subscription_fitness[fitness_id]
                    fitness.invocation_count += 1
                    latency = (time.time() - start_time) * 1000
                    n = fitness.invocation_count
                    fitness.avg_latency_ms = (fitness.avg_latency_ms * (n - 1) + latency) / n

                    if isinstance(result, Exception):
                        fitness.success_rate = (
                            (fitness.success_rate * (n - 1)) / n
                        )
                    else:
                        fitness.success_rate = (
                            (fitness.success_rate * (n - 1) + 1) / n
                        )

        # 2. Semantic handlers (context_vector based)
        await self._dispatch_semantic(signal)

    async def broadcast(self, signal: Signal):
        """
        Propagate a signal through the Nexus with priority and safety checks.

        If priority queue is running, enqueues for ordered processing.
        Otherwise processes immediately (backwards compatible).
        """
        self._last_activity = datetime.now()

        # 1. Store in Black Box (Circular Buffer)
        self._history.append(signal)
        if len(self._history) > 1000:
            self._history.pop(0)

        # Store in persistent history (serializable)
        self._store_persistent(signal)

        # 2. Backpressure check
        if self._backpressure.queue_depth >= self._backpressure.max_queue_depth:
            self._backpressure.is_throttling = True
            if signal.urgency < 0.7:  # Only drop low-urgency signals
                self._backpressure.dropped_signals += 1
                logger.warning(f"Nexus: Dropped low-priority signal due to backpressure")
                return
        else:
            self._backpressure.is_throttling = False

        # 3. Run Middleware (Logging, Safety, Filtering)
        for mw in self._middleware:
            try:
                if not mw(signal):
                    logger.debug(f"Nexus: Signal {signal.type.value} blocked by middleware")
                    return
            except Exception as e:
                logger.error(f"Nexus: Middleware error: {e}")
                return

        # 4. Run Evolution Middleware (can modify Nexus)
        for em in self._evolution_middleware:
            try:
                if not em(signal, self):
                    logger.debug(f"Nexus: Signal blocked by evolution middleware")
                    return
            except Exception as e:
                logger.error(f"Nexus: Evolution middleware error: {e}")

        # 5. Priority Queue or Immediate Processing
        if self._is_running and self._worker_task:
            # Enqueue with priority (negative urgency for max-heap)
            await self._priority_queue.put((
                -signal.urgency,
                signal.timestamp.timestamp(),
                signal,
            ))
        else:
            # Immediate processing (backwards compatible)
            # AGI v1.8: Use safe handler invocation to support both sync and async handlers
            handlers = self._subscribers.get(signal.type, [])
            if handlers:
                try:
                    await asyncio.gather(
                        *[_safe_invoke_handler(h, signal) for h in handlers],
                        return_exceptions=True
                    )
                except Exception as e:
                    logger.error(f"Nexus: Critical propagation failure: {e}")

            # Also dispatch to semantic handlers
            await self._dispatch_semantic(signal)

    async def emit(
        self,
        type: SignalType,
        payload: Dict[str, Any],
        source: str,
        urgency: float = 0.5,
        context_vector: Optional[List[float]] = None,
        semantic_tags: Optional[List[str]] = None,
    ):
        """Helper to create and broadcast a signal."""
        signal = Signal(
            type=type,
            payload=payload,
            source_id=source,
            urgency=urgency,
            context_vector=context_vector,
            semantic_tags=semantic_tags or [],
        )
        await self.broadcast(signal)

    # =========================================================================
    # SEMANTIC BROADCAST (AGI Cohesion Upgrade)
    # =========================================================================

    async def semantic_broadcast(
        self,
        signal: Signal,
        similarity_threshold: float = 0.15,
        include_type_handlers: bool = True,
        embed_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Intelligent signal routing based on semantic similarity.

        This is the neural routing upgrade that enables signals to find
        handlers based on context_vector similarity, creating emergent
        routing patterns without explicit configuration.

        Args:
            signal: The signal to broadcast
            similarity_threshold: Maximum cosine distance to trigger (lower = stricter)
            include_type_handlers: Also dispatch to type-based handlers
            embed_fn: Optional embedding function for text-based signals

        Returns:
            Dict with dispatch statistics:
            - semantic_handlers_invoked: Number of semantic handlers triggered
            - type_handlers_invoked: Number of type-based handlers triggered
            - avg_similarity: Average similarity score of invoked handlers
            - routing_latency_ms: Time taken for routing decision
        """
        import time
        start_time = time.time()

        self._last_activity = datetime.now()

        # Store in history
        self._history.append(signal)
        if len(self._history) > 1000:
            self._history.pop(0)
        self._store_persistent(signal)

        # Run middleware
        for mw in self._middleware:
            try:
                if not mw(signal):
                    return {"blocked_by_middleware": True}
            except Exception as e:
                logger.error(f"Nexus: Middleware error in semantic_broadcast: {e}")
                return {"middleware_error": str(e)}

        semantic_invoked = 0
        type_invoked = 0
        similarity_scores = []

        # Generate context_vector if not present but we have text content
        if not signal.context_vector and embed_fn:
            text_content = signal.payload.get("content") or signal.payload.get("text")
            if text_content:
                try:
                    if asyncio.iscoroutinefunction(embed_fn):
                        signal.context_vector = await embed_fn(text_content)
                    else:
                        signal.context_vector = embed_fn(text_content)
                except Exception as e:
                    logger.debug(f"Embedding generation failed: {e}")

        # Semantic dispatch with similarity threshold as distance
        if signal.context_vector and self._semantic_subscriptions:
            tasks = []

            for sub in self._semantic_subscriptions:
                # Optional type filter
                if sub.signal_types and signal.type not in sub.signal_types:
                    continue

                # Compute cosine similarity
                similarity = self._cosine_similarity(signal.context_vector, sub.target_vector)
                distance = 1.0 - similarity

                # Use distance threshold (lower distance = more similar)
                if distance <= similarity_threshold:
                    similarity_scores.append(similarity)
                    sub.invocations += 1
                    sub.last_invoked = datetime.now()

                    async def invoke_semantic(handler, subscription, sig, sim):
                        try:
                            # AGI v1.8: Use safe handler invocation
                            await _safe_invoke_handler(handler, sig)
                            subscription.successful_invocations += 1
                            return {"success": True, "similarity": sim}
                        except Exception as e:
                            logger.error(f"Semantic handler error: {e}")
                            return {"success": False, "error": str(e)}

                    tasks.append(invoke_semantic(sub.handler, sub, signal, similarity))
                    semantic_invoked += 1

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        # Type-based dispatch
        if include_type_handlers:
            handlers = self._subscribers.get(signal.type, [])
            if handlers:
                type_invoked = len(handlers)
                await asyncio.gather(
                    *[h(signal) for h in handlers],
                    return_exceptions=True
                )

        routing_latency = (time.time() - start_time) * 1000

        return {
            "semantic_handlers_invoked": semantic_invoked,
            "type_handlers_invoked": type_invoked,
            "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
            "routing_latency_ms": routing_latency,
            "signal_id": signal.id,
        }

    async def emit_semantic(
        self,
        type: SignalType,
        payload: Dict[str, Any],
        source: str,
        context_vector: List[float],
        urgency: float = 0.5,
        similarity_threshold: float = 0.15,
        semantic_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method to emit a signal with semantic routing.

        Combines signal creation with semantic_broadcast for cleaner API.
        """
        signal = Signal(
            type=type,
            payload=payload,
            source_id=source,
            urgency=urgency,
            context_vector=context_vector,
            semantic_tags=semantic_tags or [],
        )
        return await self.semantic_broadcast(signal, similarity_threshold=similarity_threshold)

    def inspection_black_box(self, last_n: int = 10) -> List[Signal]:
        """Retrieve recent signals for debugging/introspection."""
        return self._history[-last_n:]

    # =========================================================================
    # SPONTANEOUS THOUGHT GENERATOR (AGI Upgrade)
    # =========================================================================

    def configure_spontaneous_thoughts(
        self,
        enabled: bool = True,
        min_idle_seconds: float = 30.0,
        max_idle_seconds: float = 180.0,
        creativity_temperature: float = 0.7,
        thought_probability: float = 0.3,
        llm_fn: Optional[Callable] = None,
    ):
        """Configure the spontaneous thought generator."""
        self._thought_config.enabled = enabled
        self._thought_config.min_idle_seconds = min_idle_seconds
        self._thought_config.max_idle_seconds = max_idle_seconds
        self._thought_config.creativity_temperature = creativity_temperature
        self._thought_config.thought_probability = thought_probability
        self._thought_llm_fn = llm_fn

    async def _spontaneous_thought_loop(self):
        """Background task that generates thoughts during idle periods."""
        while self._is_running:
            try:
                # Random wait based on config
                wait_time = random.uniform(
                    self._thought_config.min_idle_seconds,
                    self._thought_config.max_idle_seconds,
                )
                await asyncio.sleep(wait_time)

                if not self._thought_config.enabled:
                    continue

                # Check if system has been idle
                idle_duration = (datetime.now() - self._last_activity).total_seconds()
                if idle_duration < self._thought_config.min_idle_seconds:
                    continue

                # Probabilistic thought generation
                if random.random() > self._thought_config.thought_probability:
                    continue

                # Generate thought
                thought = await self._generate_spontaneous_thought()
                if thought:
                    await self.emit(
                        type=SignalType.THOUGHT_EMITTED,
                        payload=thought,
                        source="spontaneous_cognition",
                        urgency=0.3,  # Low urgency for spontaneous thoughts
                        context_vector=self._thought_config.context_vector,
                    )
                    logger.debug(f"Nexus: Spontaneous thought emitted")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Nexus: Spontaneous thought error: {e}")

    async def _generate_spontaneous_thought(self) -> Optional[Dict]:
        """Generate a spontaneous thought using LLM or heuristics."""
        # Gather recent context from history
        recent_signals = self._history[-20:]
        recent_types = [s.type.value for s in recent_signals]
        recent_sources = list(set(s.source_id for s in recent_signals))

        if self._thought_llm_fn:
            try:
                prompt = f"""You are an AI system in "mind-wandering" mode - a creative thinking state.

RECENT ACTIVITY:
- Signal types: {recent_types[-5:]}
- Active sources: {recent_sources[-5:]}
- Creativity level: {self._thought_config.creativity_temperature:.1f}

Generate ONE spontaneous thought. This could be:
- A creative connection between recent activities
- A question about the system state
- An insight about patterns observed
- A novel idea for improvement

Return JSON:
{{"thought_type": "connection|question|insight|idea", "content": "your thought", "relevance": 0.0-1.0}}

Return ONLY the JSON:"""

                if asyncio.iscoroutinefunction(self._thought_llm_fn):
                    response = await self._thought_llm_fn(prompt)
                else:
                    response = self._thought_llm_fn(prompt)

                # Parse JSON
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    return json.loads(response[start:end])

            except Exception as e:
                logger.debug(f"LLM thought generation failed: {e}")

        # Heuristic fallback
        thought_templates = [
            {"thought_type": "connection", "content": f"Pattern observed: {random.choice(recent_types) if recent_types else 'idle'} activity correlates with system state"},
            {"thought_type": "question", "content": "What if we optimized the signal routing based on recent patterns?"},
            {"thought_type": "insight", "content": f"The {random.choice(recent_sources) if recent_sources else 'system'} component shows interesting behavior"},
            {"thought_type": "idea", "content": "Consider pre-caching frequently accessed context vectors"},
        ]

        thought = random.choice(thought_templates)
        thought["relevance"] = random.uniform(0.3, 0.7)
        return thought

    # =========================================================================
    # SIGNAL PERSISTENCE AND RECALL (AGI Upgrade)
    # =========================================================================

    def _store_persistent(self, signal: Signal):
        """Store signal in persistent history (serializable format)."""
        serialized = {
            "id": signal.id,
            "type": signal.type.value,
            "payload": signal.payload,
            "source_id": signal.source_id,
            "timestamp": signal.timestamp.isoformat(),
            "urgency": signal.urgency,
            "semantic_tags": signal.semantic_tags,
            "has_context_vector": signal.context_vector is not None,
        }

        self._persistent_history.append(serialized)

        # Maintain max size
        if len(self._persistent_history) > self._max_persistent_history:
            # Archive old signals if callback set
            if self._archival_callback:
                to_archive = self._persistent_history[:1000]
                asyncio.create_task(self._archive_signals(to_archive))
            self._persistent_history = self._persistent_history[-self._max_persistent_history:]

    async def _archive_signals(self, signals: List[Dict]):
        """Archive old signals to persistent storage."""
        if self._archival_callback:
            try:
                if asyncio.iscoroutinefunction(self._archival_callback):
                    await self._archival_callback(signals)
                else:
                    self._archival_callback(signals)
            except Exception as e:
                logger.error(f"Nexus: Signal archival failed: {e}")

    def set_archival_callback(self, callback: Callable[[List[Dict]], None]):
        """Set callback for archiving old signals to persistent storage."""
        self._archival_callback = callback

    def query_history(
        self,
        signal_type: Optional[SignalType] = None,
        source_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Query persistent signal history.

        Args:
            signal_type: Filter by signal type
            source_id: Filter by source
            since: Filter by timestamp
            limit: Maximum results

        Returns:
            List of serialized signals matching criteria
        """
        results = []

        for sig in reversed(self._persistent_history):
            if len(results) >= limit:
                break

            if signal_type and sig["type"] != signal_type.value:
                continue

            if source_id and sig["source_id"] != source_id:
                continue

            if since:
                sig_time = datetime.fromisoformat(sig["timestamp"])
                if sig_time < since:
                    continue

            results.append(sig)

        return results

    def get_signal_patterns(
        self,
        window_seconds: float = 300.0,
    ) -> Dict:
        """
        Analyze recent signal patterns for emergence detection.

        Returns:
            Dict with pattern statistics
        """
        cutoff = datetime.now() - timedelta(seconds=window_seconds)

        type_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        urgency_sum = 0.0
        count = 0

        for sig in self._persistent_history:
            sig_time = datetime.fromisoformat(sig["timestamp"])
            if sig_time < cutoff:
                continue

            type_counts[sig["type"]] = type_counts.get(sig["type"], 0) + 1
            source_counts[sig["source_id"]] = source_counts.get(sig["source_id"], 0) + 1
            urgency_sum += sig["urgency"]
            count += 1

        return {
            "total_signals": count,
            "signals_per_second": count / window_seconds if window_seconds > 0 else 0,
            "avg_urgency": urgency_sum / count if count > 0 else 0.5,
            "type_distribution": type_counts,
            "source_distribution": source_counts,
            "most_active_type": max(type_counts, key=type_counts.get) if type_counts else None,
            "most_active_source": max(source_counts, key=source_counts.get) if source_counts else None,
        }

    # =========================================================================
    # EVOLUTION SYSTEM (AGI Upgrade)
    # =========================================================================

    def evolve_subscriptions(self, fitness_threshold: float = 0.3):
        """
        Evolve subscription graph based on fitness scores.

        Low-fitness handlers are candidates for removal.
        High-fitness handlers are prioritized.
        """
        self._evolution_generation += 1
        changes = []

        # Calculate fitness for all subscriptions
        for fitness_id, fitness in self._subscription_fitness.items():
            # Compute fitness score
            fitness.fitness = (
                fitness.success_rate * 0.4 +
                (1 / (1 + fitness.avg_latency_ms / 100)) * 0.3 +  # Faster is better
                min(1.0, fitness.invocation_count / 100) * 0.3  # Usage matters
            )

            fitness.generation = self._evolution_generation

            if fitness.fitness < fitness_threshold:
                changes.append({
                    "action": "flagged_low_fitness",
                    "subscription_id": fitness_id,
                    "fitness": fitness.fitness,
                })

        # Similarly for semantic subscriptions
        for sub in self._semantic_subscriptions:
            if sub.invocations > 0:
                sub.fitness_score = sub.successful_invocations / sub.invocations
            else:
                sub.fitness_score = 0.5

        # Log evolution event
        self._evolution_history.append({
            "generation": self._evolution_generation,
            "timestamp": datetime.now().isoformat(),
            "changes": changes,
            "total_subscriptions": len(self._subscription_fitness),
            "semantic_subscriptions": len(self._semantic_subscriptions),
        })

        logger.info(
            f"Nexus: Evolution generation {self._evolution_generation} - "
            f"{len(changes)} low-fitness handlers flagged"
        )

        return changes

    def get_fitness_report(self) -> Dict:
        """Get fitness report for all subscriptions."""
        type_fitness = {}
        semantic_fitness = []

        for fitness_id, fitness in self._subscription_fitness.items():
            type_fitness[fitness_id] = {
                "fitness": fitness.fitness,
                "invocations": fitness.invocation_count,
                "success_rate": fitness.success_rate,
                "avg_latency_ms": fitness.avg_latency_ms,
            }

        for sub in self._semantic_subscriptions:
            semantic_fitness.append({
                "subscription_id": sub.subscription_id,
                "fitness": sub.fitness_score,
                "invocations": sub.invocations,
                "successful": sub.successful_invocations,
                "threshold": sub.similarity_threshold,
            })

        return {
            "generation": self._evolution_generation,
            "type_subscriptions": type_fitness,
            "semantic_subscriptions": semantic_fitness,
            "evolution_history": self._evolution_history[-10:],
        }

    # =========================================================================
    # STATUS AND DIAGNOSTICS
    # =========================================================================

    def get_status(self) -> Dict:
        """Get comprehensive Nexus status."""
        return {
            "is_running": self._is_running,
            "queue_depth": self._priority_queue.qsize() if self._is_running else 0,
            "history_size": len(self._history),
            "persistent_history_size": len(self._persistent_history),
            "type_subscriptions": {
                st.value: len(handlers)
                for st, handlers in self._subscribers.items()
            },
            "semantic_subscriptions": len(self._semantic_subscriptions),
            "middleware_count": len(self._middleware),
            "evolution_middleware_count": len(self._evolution_middleware),
            "backpressure": {
                "queue_depth": self._backpressure.queue_depth,
                "is_throttling": self._backpressure.is_throttling,
                "dropped_signals": self._backpressure.dropped_signals,
            },
            "spontaneous_thoughts": {
                "enabled": self._thought_config.enabled,
                "creativity": self._thought_config.creativity_temperature,
            },
            "evolution_generation": self._evolution_generation,
            "langgraph_connected": self._langgraph_hybrid is not None,
        }

    # =========================================================================
    # LANGGRAPH HYBRID INTEGRATION (AGI v1.8)
    # =========================================================================

    def connect_langgraph(self, langgraph_hybrid) -> None:
        """
        Connect the LangGraph hybrid workflow system to Nexus.

        This enables bidirectional communication between the event bus
        and stateful workflows.

        Args:
            langgraph_hybrid: LangGraphNexusHybrid instance
        """
        self._langgraph_hybrid = langgraph_hybrid
        langgraph_hybrid.connect_nexus(self)
        logger.info("Nexus: LangGraph hybrid connected")

    async def emit_with_workflow_state(
        self,
        type: SignalType,
        payload: Dict[str, Any],
        source: str,
        workflow_id: str,
        urgency: float = 0.5,
    ) -> None:
        """
        Emit a signal with attached workflow state context.

        This allows workflow-aware signal processing where handlers
        can access the current workflow state.

        Args:
            type: Signal type to emit
            payload: Signal payload
            source: Signal source identifier
            workflow_id: ID of the associated workflow
            urgency: Signal urgency (0-1)
        """
        # Get workflow state if available
        workflow_state = None
        if self._langgraph_hybrid:
            workflow_status = self._langgraph_hybrid.get_workflow_status(workflow_id)
            if workflow_status:
                workflow_state = workflow_status
                self._workflow_state_cache[workflow_id] = workflow_state

        # Enrich payload with workflow context
        enriched_payload = {
            **payload,
            "_workflow_id": workflow_id,
            "_workflow_state": workflow_state,
        }

        await self.emit(
            type=type,
            payload=enriched_payload,
            source=source,
            urgency=urgency,
        )

    def get_workflow_state(self, workflow_id: str) -> Optional[Dict]:
        """
        Get cached workflow state for a workflow ID.

        Args:
            workflow_id: The workflow identifier

        Returns:
            Workflow state dict or None if not found
        """
        if workflow_id in self._workflow_state_cache:
            return self._workflow_state_cache[workflow_id]

        if self._langgraph_hybrid:
            return self._langgraph_hybrid.get_workflow_status(workflow_id)

        return None

# Global accessor
nexus = Nexus.get_instance()


# =============================================================================
# DEFAULT MIDDLEWARE
# =============================================================================

def logging_middleware(signal: Signal) -> bool:
    """Log signals with urgency-based levels."""
    if signal.urgency > 0.7:
        logger.warning(f"🚨 [URGENT] {signal.type.value} from {signal.source_id}")
    else:
        logger.debug(f"⚡ {signal.type.value} from {signal.source_id}")
    return True


def rate_limit_middleware(signal: Signal) -> bool:
    """Basic rate limiting based on urgency."""
    # Always allow high-urgency signals
    if signal.urgency >= 0.8:
        return True

    # Check backpressure
    if nexus._backpressure.is_throttling:
        if signal.urgency < 0.5:
            return False  # Block low-priority during throttling

    return True


# =============================================================================
# EVOLUTION MIDDLEWARE (AGI)
# =============================================================================

def anomaly_evolution_middleware(signal: Signal, nexus_instance: Nexus) -> bool:
    """
    Evolution middleware that responds to anomalies.

    When anomalies are detected, this can:
    - Spawn new handlers dynamically
    - Adjust thresholds
    - Trigger evolution of subscription graph
    """
    if signal.type == SignalType.ANOMALY_DETECTED:
        anomaly_type = signal.payload.get("anomaly_type", "unknown")
        severity = signal.payload.get("severity", 0.5)

        if severity > 0.8:
            # High-severity anomaly - trigger evolution
            logger.info(f"Nexus: High-severity anomaly detected - triggering evolution")
            nexus_instance.evolve_subscriptions(fitness_threshold=0.4)

    return True


def memory_consolidation_middleware(signal: Signal, nexus_instance: Nexus) -> bool:
    """
    Middleware that tracks memory-related signals.

    Can trigger consolidation or optimize routing for memory operations.
    """
    if signal.type == SignalType.MEMORY_CONSOLIDATION:
        # Track memory operations for pattern detection
        memory_ids = signal.payload.get("memory_ids", [])
        session_ref = signal.payload.get("session_ref")

        logger.debug(
            f"Nexus: Memory consolidation - {len(memory_ids)} memories, "
            f"session={session_ref}"
        )

    return True


# Register default middleware
nexus.add_middleware(logging_middleware)
nexus.add_middleware(rate_limit_middleware)
nexus.add_evolution_middleware(anomaly_evolution_middleware)
nexus.add_evolution_middleware(memory_consolidation_middleware)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def emit_thought(
    content: str,
    source: str = "system",
    thought_type: str = "general",
    urgency: float = 0.5,
    context_vector: Optional[List[float]] = None,
):
    """Convenience function to emit a thought signal."""
    await nexus.emit(
        type=SignalType.THOUGHT_EMITTED,
        payload={
            "content": content,
            "thought_type": thought_type,
        },
        source=source,
        urgency=urgency,
        context_vector=context_vector,
    )


async def emit_memory_consolidation(
    memory_ids: List[str],
    session_ref: Optional[str] = None,
    context_vector: Optional[List[float]] = None,
):
    """Convenience function for memory consolidation signals."""
    await nexus.emit(
        type=SignalType.MEMORY_CONSOLIDATION,
        payload={
            "memory_ids": memory_ids,
            "session_ref": session_ref,
        },
        source="unified_memory",
        urgency=0.6,
        context_vector=context_vector,
    )


async def emit_dialogue_event(
    event_type: str,
    session_id: str,
    content: Dict[str, Any],
    urgency: float = 0.5,
):
    """Convenience function for dialogue signals."""
    signal_map = {
        "started": SignalType.DIALOGUE_STARTED,
        "propose": SignalType.DIALOGUE_PROPOSE,
        "critique": SignalType.DIALOGUE_CRITIQUE,
        "refine": SignalType.DIALOGUE_REFINE,
        "vote": SignalType.DIALOGUE_VOTE,
        "consensus": SignalType.DIALOGUE_CONSENSUS,
        "completed": SignalType.DIALOGUE_COMPLETED,
    }

    signal_type = signal_map.get(event_type, SignalType.DIALOGUE_STARTED)

    await nexus.emit(
        type=signal_type,
        payload={
            "session_id": session_id,
            **content,
        },
        source="dialogue_system",
        urgency=urgency,
    )
