"""
Farnsworth MCP Server Module

Hybrid MCP server for Claude Code integration with:
- Memory access and manipulation tools
- Agent delegation and monitoring
- Evolution control and feedback
- Resource streaming for context awareness
"""

from farnsworth.mcp_server.server import FarnsworthMCPServer, run_server
from farnsworth.mcp_server.memory_tools import MemoryTools
from farnsworth.mcp_server.agent_tools import AgentTools
from farnsworth.mcp_server.evolution_tools import EvolutionTools
from farnsworth.mcp_server.resources import FarnsworthResources

__all__ = [
    "FarnsworthMCPServer",
    "run_server",
    "MemoryTools",
    "AgentTools",
    "EvolutionTools",
    "FarnsworthResources",
]
