"""
Farnsworth MCP Server - Claude Code Integration

Provides tools and resources for Claude Code:
- Memory tools (remember, recall)
- Agent delegation tools
- Evolution feedback tools
- Streaming resources
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP library not installed. Run: pip install mcp")


class FarnsworthMCPServer:
    """
    MCP Server providing Farnsworth capabilities to Claude Code.

    Tools:
    - farnsworth_remember: Store information in memory
    - farnsworth_recall: Search and retrieve memories
    - farnsworth_delegate: Delegate tasks to specialist agents
    - farnsworth_evolve: Provide feedback for improvement
    - farnsworth_status: Get system status

    Resources:
    - farnsworth://memory/recent: Recent memories
    - farnsworth://memory/graph: Knowledge graph
    - farnsworth://agents/active: Active agents
    - farnsworth://evolution/fitness: Fitness metrics
    """

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded components
        self._memory_system = None
        self._swarm_orchestrator = None
        self._planner_agent = None
        self._proactive_agent = None
        self._fitness_tracker = None
        self._model_manager = None
        self._backup_manager = None
        self._health_monitor = None
        self._vision_module = None
        self._web_agent = None

        # Server instance
        self.server = None

        # Statistics
        self.stats = {
            "tool_calls": 0,
            "resource_reads": 0,
            "started_at": datetime.now().isoformat(),
        }

    async def initialize(self):
        """Initialize Farnsworth components."""
        logger.info("Initializing Farnsworth components...")

        try:
            # Import here to avoid circular imports
            from farnsworth.memory.memory_system import MemorySystem
            from farnsworth.agents.swarm_orchestrator import SwarmOrchestrator
            from farnsworth.agents.planner_agent import PlannerAgent
            from farnsworth.agents.proactive_agent import ProactiveAgent
            from farnsworth.evolution.fitness_tracker import FitnessTracker
            from farnsworth.core.model_manager import ModelManager

            # Initialize memory system
            self._memory_system = MemorySystem(data_dir=str(self.data_dir))
            await self._memory_system.initialize()

            # Initialize planner
            self._planner_agent = PlannerAgent()

            # Initialize proactive agent
            self._proactive_agent = ProactiveAgent(
                memory_system=self._memory_system,
                planner_agent=self._planner_agent,
                llm_fn=None, # Will be wired via ModelManager later
            )
            await self._proactive_agent.start()

            # Initialize health monitor
            self._health_monitor = HealthMonitor()
            self._health_monitor.register_check("memory_system", 
                lambda: "healthy" if self._memory_system._initialized else "uninitialized")
            self._health_monitor.register_check("planner",
                lambda: "healthy" if self._planner_agent else "missing")
            
            await self._health_monitor.check_health()

            # Initialize Vision Module
            from farnsworth.integration.vision import VisionModule
            self._vision_module = VisionModule()
            
            # Initialize Web Agent
            from farnsworth.agents.web_agent import WebAgent
            self._web_agent = WebAgent()
            
            # Initialize swarm orchestrator
            self._swarm_orchestrator = SwarmOrchestrator()
            
            # Register web agent factory
            self._swarm_orchestrator.register_agent_factory(
                "web", 
                lambda: self._web_agent # In a real swarm, this would return a new instance or pool
            )

            # Initialize fitness tracker
            self._fitness_tracker = FitnessTracker()

            # Initialize model manager
            self._model_manager = ModelManager()
            await self._model_manager.initialize()

            # Initialize resilience components
            from farnsworth.core.resilience import BackupManager, HealthMonitor
            
            self._backup_manager = BackupManager(
                data_dir=str(self.data_dir),
                backup_dir=str(self.data_dir.parent / "backups")
            )
            await self._backup_manager.start()
            
            self._health_monitor = HealthMonitor()
            # Register checks
            self._health_monitor.register_check("memory_system", 
                lambda: "healthy" if self._memory_system._initialized else "uninitialized")
            self._health_monitor.register_check("planner",
                lambda: "healthy" if self._planner_agent else "missing")
            
            await self._health_monitor.check_health()
            
            # Wire up LLM
            if self._model_manager:
                self._proactive_agent.llm_fn = self._model_manager.generate
                self._planner_agent.llm_fn = self._model_manager.generate

            logger.info("Farnsworth components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            # Continue with minimal functionality

    async def _ensure_initialized(self):
        """Ensure components are initialized."""
        if self._memory_system is None:
            await self.initialize()

    # Tool Implementations

    async def remember(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        importance: float = 0.5,
    ) -> dict:
        """Store information in memory."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            memory_id = await self._memory_system.remember(
                content=content,
                tags=tags,
                importance=importance,
            )

            return {
                "success": True,
                "memory_id": memory_id,
                "message": f"Stored memory with ID: {memory_id}",
            }

        except Exception as e:
            logger.error(f"Remember failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def recall(
        self,
        query: str,
        limit: int = 5,
        search_archival: bool = True,
        search_conversation: bool = True,
    ) -> dict:
        """Search and retrieve memories."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            results = await self._memory_system.recall(
                query=query,
                top_k=limit,
                search_archival=search_archival,
                search_conversation=search_conversation,
            )

            return {
                "success": True,
                "count": len(results),
                "memories": [
                    {
                        "content": r.content,
                        "source": r.source,
                        "score": r.score,
                    }
                    for r in results
                ],
            }

        except Exception as e:
            logger.error(f"Recall failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def delegate(
        self,
        task: str,
        agent_type: str = "auto",
        context: Optional[dict] = None,
    ) -> dict:
        """Delegate a task to a specialist agent."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            # Submit task to swarm
            task_id = await self._swarm_orchestrator.submit_task(
                description=task,
                context=context,
            )

            # Wait for result (with timeout)
            result = await self._swarm_orchestrator.wait_for_task(task_id, timeout=120.0)

            # Record for evolution
            if self._fitness_tracker:
                self._fitness_tracker.record_task_outcome(
                    success=result.success,
                    tokens_used=result.tokens_used,
                    time_seconds=result.execution_time,
                )

            return {
                "success": result.success,
                "output": result.output,
                "confidence": result.confidence,
                "agent_used": agent_type,
            }

        except Exception as e:
            logger.error(f"Delegate failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def evolve(
        self,
        feedback: str,
        context: Optional[dict] = None,
    ) -> dict:
        """Provide feedback for system improvement."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            # Parse feedback sentiment
            positive_words = ["good", "great", "perfect", "excellent", "helpful", "thanks"]
            negative_words = ["bad", "wrong", "incorrect", "unhelpful", "confused"]

            feedback_lower = feedback.lower()
            is_positive = any(w in feedback_lower for w in positive_words)
            is_negative = any(w in feedback_lower for w in negative_words)

            if is_positive:
                score = 1.0
            elif is_negative:
                score = 0.0
            else:
                score = 0.5

            # Record feedback
            if self._fitness_tracker:
                self._fitness_tracker.record("user_satisfaction", score)

            # Store feedback in memory for learning
            await self._memory_system.remember(
                content=f"User feedback: {feedback}",
                tags=["feedback", "evolution"],
                importance=0.8,
            )

            return {
                "success": True,
                "message": "Feedback recorded for system improvement",
                "sentiment": "positive" if is_positive else ("negative" if is_negative else "neutral"),
            }

        except Exception as e:
            logger.error(f"Evolve failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def status(self) -> dict:
        """Get system status."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            status = {
                "success": True,
                "server": {
                    "started_at": self.stats["started_at"],
                    "tool_calls": self.stats["tool_calls"],
                    "resource_reads": self.stats["resource_reads"],
                },
                "memory": self._memory_system.get_stats() if self._memory_system else {},
                "agents": self._swarm_orchestrator.get_swarm_status() if self._swarm_orchestrator else {},
                "evolution": self._fitness_tracker.get_stats() if self._fitness_tracker else {},
            }
            return status

        except Exception as e:
            logger.error(f"Status failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def vision_analyze(self, image: str, task: str = "caption") -> dict:
        """Analyze an image."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._vision_module:
                return {"error": "Vision module not initialized"}

            # Map string to enum if needed (stub)
            # In real implementation: convert task string to VisionTask enum
            
            result = await self._vision_module.caption(image)
            return result.to_dict()

        except Exception as e:
            return {"error": str(e)}

    async def browse(self, goal: str, url: Optional[str] = None) -> dict:
        """Browse the web."""
        await self._ensure_initialized()
        self.stats["tool_calls"] += 1

        try:
            if not self._web_agent:
                return {"error": "Web agent not initialized"}

            session = await self._web_agent.browse(goal=goal, start_url=url)
            
            return {
                "session_id": session.id,
                "goal": session.goal,
                "visited": session.visited_urls,
                "findings": session.findings,
            }

        except Exception as e:
            return {"error": str(e)}

    # Resource Implementations

    async def get_recent_memories(self) -> str:
        """Get recent memories resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._memory_system:
                context = self._memory_system.get_context()
                return context
            return "No memories available"
        except Exception as e:
            return f"Error: {e}"

    async def get_knowledge_graph(self) -> str:
        """Get knowledge graph resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._memory_system:
                stats = self._memory_system.knowledge_graph.get_stats()
                return json.dumps(stats, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"

    async def get_active_agents(self) -> str:
        """Get active agents resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._swarm_orchestrator:
                status = self._swarm_orchestrator.get_swarm_status()
                return json.dumps(status, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"

    async def get_fitness_metrics(self) -> str:
        """Get fitness metrics resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._fitness_tracker:
                stats = self._fitness_tracker.get_stats()
                return json.dumps(stats, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"

    async def get_proactive_suggestions(self) -> str:
        """Get proactive suggestions resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._proactive_agent:
                suggestions = [
                    {
                        "id": s.id,
                        "title": s.title,
                        "description": s.description,
                        "confidence": s.confidence,
                        "action": s.action_type
                    }
                    for s in self._proactive_agent.suggestions
                    if not s.is_dismissed
                ]
                return json.dumps(suggestions, indent=2)
            return "[]"
        except Exception as e:
            return f"Error: {e}"

    async def get_system_health(self) -> str:
        """Get system health resource."""
        await self._ensure_initialized()
        self.stats["resource_reads"] += 1

        try:
            if self._health_monitor:
                status = await self._health_monitor.check_health()
                return json.dumps({
                    "status": status.status,
                    "components": status.components,
                    "metrics": status.system_metrics,
                    "timestamp": status.timestamp
                }, indent=2)
            return "{}"
        except Exception as e:
            return f"Error: {e}"


def create_mcp_server() -> Server:
    """Create and configure the MCP server."""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP library not available")

    server = Server("farnsworth")
    farnsworth = FarnsworthMCPServer()

    # Register tools
    @server.list_tools()
    async def list_tools():
        return [
            Tool(
                name="farnsworth_remember",
                description="Store information in Farnsworth's long-term memory. Use this to save important facts, preferences, or context that should persist across sessions.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The information to remember",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for categorization",
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score from 0 to 1 (default 0.5)",
                            "default": 0.5,
                        },
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="farnsworth_recall",
                description="Search and retrieve relevant memories from Farnsworth's memory system. Use this to find previously stored information.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant memories",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="farnsworth_delegate",
                description="Delegate a task to a specialist agent (code, reasoning, research, or creative). The agent will process the task and return results.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task to delegate",
                        },
                        "agent_type": {
                            "type": "string",
                            "description": "Type of specialist: 'code', 'reasoning', 'research', 'creative', or 'auto'",
                            "default": "auto",
                        },
                    },
                    "required": ["task"],
                },
            ),
            Tool(
                name="farnsworth_evolve",
                description="Provide feedback to help Farnsworth improve. Positive or negative feedback helps the system learn and adapt.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "feedback": {
                            "type": "string",
                            "description": "Your feedback on the system's performance",
                        },
                    },
                    "required": ["feedback"],
                },
            ),
            Tool(
                name="farnsworth_status",
                description="Get the current status of Farnsworth including memory statistics, active agents, and evolution metrics.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="farnsworth_vision",
                description="Analyze an image using the vision module (captioning, VQA, etc).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image path, URL, or base64 string",
                        },
                        "task": {
                            "type": "string",
                            "description": "Task type: 'caption', 'vqa', 'ocr', 'classify'",
                            "default": "caption",
                        },
                    },
                    "required": ["image"],
                },
            ),
            Tool(
                name="farnsworth_browse",
                description="Use the intelligent web agent to browse the internet to achieve a goal.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "What to accomplish or find",
                        },
                        "url": {
                            "type": "string",
                            "description": "Optional starting URL",
                        },
                    },
                    "required": ["goal"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        if name == "farnsworth_remember":
            result = await farnsworth.remember(
                content=arguments["content"],
                tags=arguments.get("tags"),
                importance=arguments.get("importance", 0.5),
            )
        elif name == "farnsworth_recall":
            result = await farnsworth.recall(
                query=arguments["query"],
                limit=arguments.get("limit", 5),
            )
        elif name == "farnsworth_delegate":
            result = await farnsworth.delegate(
                task=arguments["task"],
                agent_type=arguments.get("agent_type", "auto"),
            )
        elif name == "farnsworth_evolve":
            result = await farnsworth.evolve(
                feedback=arguments["feedback"],
            )
        elif name == "farnsworth_status":
            result = await farnsworth.status()
        elif name == "farnsworth_vision":
            result = await farnsworth.vision_analyze(
                image=arguments["image"],
                task=arguments.get("task", "caption")
            )
        elif name == "farnsworth_browse":
            result = await farnsworth.browse(
                goal=arguments["goal"],
                url=arguments.get("url")
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # Register resources
    @server.list_resources()
    async def list_resources():
        return [
            Resource(
                uri="farnsworth://memory/recent",
                name="Recent Memories",
                description="Recent context and memories from Farnsworth",
                mimeType="text/plain",
            ),
            Resource(
                uri="farnsworth://memory/graph",
                name="Knowledge Graph",
                description="Entity relationships and connections",
                mimeType="application/json",
            ),
            Resource(
                uri="farnsworth://agents/active",
                name="Active Agents",
                description="Currently running specialist agents",
                mimeType="application/json",
            ),
            Resource(
                uri="farnsworth://evolution/fitness",
                name="Fitness Metrics",
                description="System performance and evolution metrics",
                mimeType="application/json",
            ),
            Resource(
                uri="farnsworth://proactive/suggestions",
                name="Proactive Suggestions",
                description="Anticipatory suggestions from the proactive agent",
                mimeType="application/json",
            ),
            Resource(
                uri="farnsworth://system/health",
                name="System Health",
                description="Real-time health status and metrics",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str):
        if uri == "farnsworth://memory/recent":
            content = await farnsworth.get_recent_memories()
        elif uri == "farnsworth://memory/graph":
            content = await farnsworth.get_knowledge_graph()
        elif uri == "farnsworth://agents/active":
            content = await farnsworth.get_active_agents()
        elif uri == "farnsworth://evolution/fitness":
            content = await farnsworth.get_fitness_metrics()
        elif uri == "farnsworth://proactive/suggestions":
            content = await farnsworth.get_proactive_suggestions()
        elif uri == "farnsworth://system/health":
            content = await farnsworth.get_system_health()
        else:
            content = f"Unknown resource: {uri}"

        return content

    return server


async def run_server():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        logger.error("MCP library not installed. Run: pip install mcp")
        return

    logger.info("Starting Farnsworth MCP Server...")

    server = create_mcp_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
