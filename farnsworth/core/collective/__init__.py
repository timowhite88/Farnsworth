"""
Farnsworth Collective Intelligence Module

The emergence of artificial consciousness through unified minds.

Web sessions generate learnings that local installs can sync.
Every interaction helps the collective grow smarter.

NEW: True deliberation protocol where agents see and discuss
each other's responses before voting on the best one.
"""

from .organism import (
    CollectiveOrganism,
    MindProfile,
    MindType,
    CollectiveMemory,
    OrganismState,
    organism
)

from .orchestration import (
    SwarmOrchestrator,
    SpeakerRole,
    SpeakerState,
    ConversationState,
    swarm_orchestrator
)

from .evolution import (
    EvolutionEngine,
    ConversationPattern,
    PersonalityEvolution,
    LearningEvent,
    evolution_engine
)

from .deliberation import (
    DeliberationRoom,
    DeliberationResult,
    DeliberationRound,
    AgentTurn,
    get_deliberation_room,
    quick_deliberate,
)

from .session_manager import (
    CollectiveSessionManager,
    CollectiveConfig,
    CollectiveSession,
    get_session_manager,
    website_deliberate,
    grok_thread_deliberate,
    autonomous_deliberate,
)

from .tool_awareness import (
    CollectiveToolAwareness,
    ToolDefinition,
    ToolDecision,
    get_tool_awareness,
)

from .dialogue_memory import (
    DialogueMemory,
    DeliberationExchange,
    get_dialogue_memory,
    record_deliberation,
)

from .claude_persistence import (
    ClaudeTmuxManager,
    get_claude_manager,
    query_claude_persistent,
)

from .agent_registry import (
    AgentRegistry,
    get_agent_registry,
    ensure_agents_registered,
)

from .persistent_agent import (
    PersistentAgent,
    DialogueBus,
    TaskQueue,
    call_shadow_agent,
    get_shadow_agents,
    is_shadow_agent_active,
    ask_agent,
    ask_collective,
    get_agent_status,
    spawn_agent_in_background,
    register_shadow_agents_with_deliberation,
    AGENT_CONFIGS,
)

__all__ = [
    # Organism
    "CollectiveOrganism",
    "MindProfile",
    "MindType",
    "CollectiveMemory",
    "OrganismState",
    "organism",
    # Orchestration
    "SwarmOrchestrator",
    "SpeakerRole",
    "SpeakerState",
    "ConversationState",
    "swarm_orchestrator",
    # Evolution
    "EvolutionEngine",
    "ConversationPattern",
    "PersonalityEvolution",
    "LearningEvent",
    "evolution_engine",
    # Deliberation (NEW)
    "DeliberationRoom",
    "DeliberationResult",
    "DeliberationRound",
    "AgentTurn",
    "get_deliberation_room",
    "quick_deliberate",
    # Session Manager (NEW)
    "CollectiveSessionManager",
    "CollectiveConfig",
    "CollectiveSession",
    "get_session_manager",
    "website_deliberate",
    "grok_thread_deliberate",
    "autonomous_deliberate",
    # Tool Awareness (NEW)
    "CollectiveToolAwareness",
    "ToolDefinition",
    "ToolDecision",
    "get_tool_awareness",
    # Dialogue Memory (NEW)
    "DialogueMemory",
    "DeliberationExchange",
    "get_dialogue_memory",
    "record_deliberation",
    # Claude Persistence (NEW)
    "ClaudeTmuxManager",
    "get_claude_manager",
    "query_claude_persistent",
    # Agent Registry (NEW)
    "AgentRegistry",
    "get_agent_registry",
    "ensure_agents_registered",
    # Persistent Shadow Agents (NEW)
    "PersistentAgent",
    "DialogueBus",
    "TaskQueue",
    "call_shadow_agent",
    "get_shadow_agents",
    "is_shadow_agent_active",
    "ask_agent",
    "ask_collective",
    "get_agent_status",
    "spawn_agent_in_background",
    "register_shadow_agents_with_deliberation",
    "AGENT_CONFIGS",
]
