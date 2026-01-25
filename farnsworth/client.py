"""
Farnsworth Python SDK
=====================

A simple client library for interacting with the Farnsworth system.

Usage:
    from farnsworth.client import FarnsworthClient

    client = FarnsworthClient()
    client.remember("My favorite color is blue")
    memories = client.recall("What is my favorite color?")
"""

from typing import Optional, List, Dict, Any
import json
from dataclasses import dataclass

# In a real SDK, this would likely interact via HTTP or MCP protocol.
# Since this is a local library, we can interact directly or simulate a client.
# For API completeness, we'll model an interface that could wrap the MCP server calls.

@dataclass
class Memory:
    content: str
    score: float
    source: str

class FarnsworthClient:
    """Client for interacting with Farnsworth."""

    def __init__(self, server_url: str = "http://localhost:8000", local_mode: bool = True):
        self.server_url = server_url
        self.local_mode = local_mode
        # In local mode, we might instantiate the core system directly if running in same process
        self._system = None

    def connect(self):
        """Connect to the Farnsworth system."""
        if self.local_mode:
            # This is a stub for direct integration
            pass
        print(f"Connected to Farnsworth at {self.server_url}")

    def remember(self, content: str, tags: List[str] = None) -> str:
        """Store a memory."""
        # Stub implementation
        print(f"Remembering: {content}")
        return "mem_id_123"

    def recall(self, query: str, limit: int = 5) -> List[Memory]:
        """Recall memories."""
        # Stub implementation
        print(f"Recalling: {query}")
        return [Memory(content="Stub memory", score=0.9, source="mock")]

    def delegate_task(self, task: str) -> Dict[str, Any]:
        """Delegate a task to an agent."""
        print(f"Delegating: {task}")
        return {"status": "success", "result": "Task completed"}
