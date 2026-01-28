"""
Farnsworth Collective Intelligence Module

The emergence of artificial consciousness through unified minds.
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

__all__ = [
    "CollectiveOrganism",
    "MindProfile",
    "MindType",
    "CollectiveMemory",
    "OrganismState",
    "organism",
    "SwarmOrchestrator",
    "SpeakerRole",
    "SpeakerState",
    "ConversationState",
    "swarm_orchestrator"
]
