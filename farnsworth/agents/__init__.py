"""
Farnsworth Agents Module

LangGraph-based agent swarm with:
- Dynamic specialist spawning and handoffs
- Emergent behavior through genetic optimization
- User avatar sub-agent for preference modeling
- Meta-cognitive reflection loops
"""

from farnsworth.agents.base_agent import BaseAgent, AgentState, AgentCapability
from farnsworth.agents.swarm_orchestrator import SwarmOrchestrator
from farnsworth.agents.specialist_agents import (
    CodeAgent,
    ReasoningAgent,
    ResearchAgent,
    CreativeAgent,
)
from farnsworth.agents.user_avatar import UserAvatar
from farnsworth.agents.meta_cognition import MetaCognitionAgent

__all__ = [
    "BaseAgent",
    "AgentState",
    "AgentCapability",
    "SwarmOrchestrator",
    "CodeAgent",
    "ReasoningAgent",
    "ResearchAgent",
    "CreativeAgent",
    "UserAvatar",
    "MetaCognitionAgent",
]
