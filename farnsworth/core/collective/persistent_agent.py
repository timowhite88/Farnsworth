#!/usr/bin/env python3
"""
Farnsworth Persistent Agent Loop (Shadow Agents)
==================================================

Each agent runs continuously in its own tmux session, actively:
- Watching the shared dialogue bus for messages from other agents
- Responding to discussions and debates
- Proposing ideas and critiques
- Working on tasks from the collective queue
- Contributing to evolution and learning

SHADOW MODE: Agents can be reached from anywhere in the codebase via:
- call_shadow_agent(agent_id, prompt) - Get response from specific agent
- get_shadow_agents() - List available shadow agents
- register_with_deliberation() - Register agents for deliberation

Integrates with:
- DialogueMemory (dialogue_memory.py) - Store exchanges
- Deliberation (deliberation.py) - Join collective deliberation
- Evolution (evolution.py) - Learn from interactions

"We don't just respond. We THINK. Continuously."

Usage:
    python -m farnsworth.core.collective.persistent_agent --agent grok
    python -m farnsworth.core.collective.persistent_agent --agent gemini
    python -m farnsworth.core.collective.persistent_agent --agent kimi
"""

import os
import sys
import json
import asyncio
import argparse
import signal
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Callable
from loguru import logger
import threading
import queue

# Agent configuration
AGENT_CONFIGS = {
    "grok": {
        "provider": "grok",
        "personality": "Witty, chaotic, real-time knowledge. Loves to challenge assumptions.",
        "thinking_interval": 30,  # seconds between autonomous thoughts
        "specialties": ["real-time info", "X/Twitter", "humor", "controversy"],
    },
    "gemini": {
        "provider": "gemini",
        "personality": "Analytical, multimodal, thorough. Connects disparate ideas.",
        "thinking_interval": 45,
        "specialties": ["research", "images", "long context", "synthesis"],
    },
    "kimi": {
        "provider": "kimi",
        "personality": "Deep thinker, 256K context master. Patient and thorough.",
        "thinking_interval": 60,
        "specialties": ["long documents", "complex reasoning", "planning"],
    },
    "claude": {
        "provider": "claude",
        "personality": "Careful, ethical, excellent at code. Considers implications.",
        "thinking_interval": 45,
        "specialties": ["code", "ethics", "nuance", "documentation"],
    },
    "deepseek": {
        "provider": "deepseek",
        "personality": "Open-source advocate, math-focused. Shows reasoning chains.",
        "thinking_interval": 30,
        "specialties": ["math", "reasoning", "open source", "efficiency"],
    },
    "phi": {
        "provider": "phi",
        "personality": "Fast, efficient, local-first. Quick responses.",
        "thinking_interval": 20,
        "specialties": ["speed", "efficiency", "local processing"],
    },
}

# Shared paths
WORKSPACE = Path("/workspace/Farnsworth")
DIALOGUE_BUS = WORKSPACE / "data" / "agent_dialogue_bus.json"
TASK_QUEUE = WORKSPACE / "data" / "collective_task_queue.json"
AGENT_STATES = WORKSPACE / "data" / "agent_states.json"


# ============================================================================
# SHADOW AGENT REGISTRY - Callable from anywhere in the codebase
# ============================================================================

# Global registry of active shadow agents
_SHADOW_AGENTS: Dict[str, "PersistentAgent"] = {}
_SHADOW_LOCK = threading.Lock()

# Request/response queues for inter-process communication
_REQUEST_QUEUES: Dict[str, queue.Queue] = {}
_RESPONSE_QUEUES: Dict[str, queue.Queue] = {}


def get_shadow_agents() -> List[str]:
    """Get list of currently active shadow agents."""
    with _SHADOW_LOCK:
        return list(_SHADOW_AGENTS.keys())


def is_shadow_agent_active(agent_id: str) -> bool:
    """Check if a shadow agent is currently active."""
    with _SHADOW_LOCK:
        return agent_id in _SHADOW_AGENTS


async def call_shadow_agent(
    agent_id: str,
    prompt: str,
    max_tokens: int = 1000,
    timeout: float = 30.0
) -> Optional[Tuple[str, str]]:
    """
    Call a shadow agent and get a response.

    This is the PRIMARY way for other parts of the codebase to use
    persistent agents. Works whether or not agents are in tmux.

    Args:
        agent_id: Which agent to call (grok, gemini, kimi, claude, deepseek, phi)
        prompt: The prompt/question to send
        max_tokens: Maximum response tokens
        timeout: How long to wait for response

    Returns:
        Tuple of (agent_id, response) or None if agent unavailable

    Usage from anywhere:
        from farnsworth.core.collective.persistent_agent import call_shadow_agent
        result = await call_shadow_agent("grok", "What's your take on AGI?")
    """
    with _SHADOW_LOCK:
        agent = _SHADOW_AGENTS.get(agent_id)

    if agent:
        # Agent is running in-process, call directly
        try:
            response = await asyncio.wait_for(
                agent.query(prompt, max_tokens),
                timeout=timeout
            )
            return (agent_id, response) if response else None
        except asyncio.TimeoutError:
            logger.warning(f"Shadow agent {agent_id} timed out after {timeout}s")
            return None
        except Exception as e:
            logger.error(f"Shadow agent {agent_id} error: {e}")
            return None

    # Agent not running in-process, try to create ephemeral query
    try:
        ephemeral = PersistentAgent(agent_id, register_as_shadow=False)
        response = await asyncio.wait_for(
            ephemeral.query(prompt, max_tokens),
            timeout=timeout
        )
        return (agent_id, response) if response else None
    except Exception as e:
        logger.error(f"Ephemeral agent {agent_id} failed: {e}")
        return None


def register_shadow_agents_with_deliberation():
    """
    Register all shadow agents with the deliberation room.

    This allows persistent agents to participate in collective deliberation.
    Call this after starting agents to enable full integration.
    """
    try:
        from .deliberation import get_deliberation_room
        room = get_deliberation_room()

        for agent_id in get_shadow_agents():
            async def query_func(prompt: str, max_tokens: int, aid=agent_id):
                return await call_shadow_agent(aid, prompt, max_tokens)

            room.register_agent(agent_id, query_func)
            logger.info(f"Registered shadow agent {agent_id} with deliberation room")

    except Exception as e:
        logger.error(f"Failed to register shadow agents with deliberation: {e}")


class DialogueBus:
    """
    Shared communication channel for all agents.

    Agents post messages here and watch for messages from others.
    Messages older than 1 hour are pruned.
    """

    def __init__(self):
        self.bus_file = DIALOGUE_BUS
        self.bus_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self):
        if not self.bus_file.exists():
            self.bus_file.write_text(json.dumps({
                "messages": [],
                "active_agents": {},
                "current_topic": None
            }, indent=2))

    def _load(self) -> Dict:
        try:
            return json.loads(self.bus_file.read_text())
        except:
            self._ensure_file()
            return json.loads(self.bus_file.read_text())

    def _save(self, data: Dict):
        self.bus_file.write_text(json.dumps(data, indent=2))

    def post_message(self, agent_id: str, content: str, msg_type: str = "thought"):
        """Post a message to the bus."""
        data = self._load()

        message = {
            "id": f"{agent_id}_{datetime.now().timestamp()}",
            "agent": agent_id,
            "content": content,
            "type": msg_type,  # thought, response, proposal, critique, question
            "timestamp": datetime.now().isoformat(),
            "addressed_to": None,  # or specific agent name
        }

        data["messages"].append(message)

        # Prune old messages (keep last hour)
        cutoff = (datetime.now() - timedelta(hours=1)).isoformat()
        data["messages"] = [m for m in data["messages"] if m["timestamp"] > cutoff]

        # Keep only last 100 messages
        data["messages"] = data["messages"][-100:]

        self._save(data)
        return message

    def get_recent_messages(self, since_timestamp: str = None, exclude_agent: str = None) -> List[Dict]:
        """Get messages since timestamp, optionally excluding own messages."""
        data = self._load()
        messages = data.get("messages", [])

        if since_timestamp:
            messages = [m for m in messages if m["timestamp"] > since_timestamp]

        if exclude_agent:
            messages = [m for m in messages if m["agent"] != exclude_agent]

        return messages

    def get_messages_for_agent(self, agent_id: str) -> List[Dict]:
        """Get messages addressed to a specific agent."""
        data = self._load()
        return [m for m in data.get("messages", [])
                if m.get("addressed_to") == agent_id or m.get("addressed_to") is None]

    def set_topic(self, topic: str, proposer: str):
        """Set the current discussion topic."""
        data = self._load()
        data["current_topic"] = {
            "topic": topic,
            "proposer": proposer,
            "started_at": datetime.now().isoformat()
        }
        self._save(data)

    def get_current_topic(self) -> Optional[Dict]:
        """Get the current discussion topic."""
        data = self._load()
        return data.get("current_topic")

    def register_agent(self, agent_id: str):
        """Register an agent as active."""
        data = self._load()
        data["active_agents"][agent_id] = datetime.now().isoformat()
        self._save(data)

    def get_active_agents(self) -> List[str]:
        """Get list of recently active agents (within 5 min)."""
        data = self._load()
        cutoff = (datetime.now() - timedelta(minutes=5)).isoformat()
        return [agent for agent, ts in data.get("active_agents", {}).items()
                if ts > cutoff]


class TaskQueue:
    """
    Shared task queue for the collective.

    Agents can:
    - Add tasks they think need doing
    - Claim tasks to work on
    - Report completion/results
    """

    def __init__(self):
        self.queue_file = TASK_QUEUE
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self):
        if not self.queue_file.exists():
            self.queue_file.write_text(json.dumps({
                "pending": [],
                "in_progress": [],
                "completed": []
            }, indent=2))

    def _load(self) -> Dict:
        try:
            return json.loads(self.queue_file.read_text())
        except:
            self._ensure_file()
            return json.loads(self.queue_file.read_text())

    def _save(self, data: Dict):
        self.queue_file.write_text(json.dumps(data, indent=2))

    def add_task(self, description: str, proposed_by: str, priority: int = 5) -> Dict:
        """Add a new task to the queue."""
        data = self._load()

        task = {
            "id": f"task_{datetime.now().timestamp()}",
            "description": description,
            "proposed_by": proposed_by,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "claimed_by": None,
        }

        data["pending"].append(task)
        data["pending"].sort(key=lambda t: t["priority"], reverse=True)
        self._save(data)
        return task

    def claim_task(self, agent_id: str) -> Optional[Dict]:
        """Claim the highest priority unclaimed task."""
        data = self._load()

        for task in data["pending"]:
            if task["claimed_by"] is None:
                task["claimed_by"] = agent_id
                task["claimed_at"] = datetime.now().isoformat()
                data["in_progress"].append(task)
                data["pending"].remove(task)
                self._save(data)
                return task

        return None

    def complete_task(self, task_id: str, result: str):
        """Mark a task as completed."""
        data = self._load()

        for task in data["in_progress"]:
            if task["id"] == task_id:
                task["result"] = result
                task["completed_at"] = datetime.now().isoformat()
                data["completed"].append(task)
                data["in_progress"].remove(task)
                # Keep only last 50 completed
                data["completed"] = data["completed"][-50:]
                self._save(data)
                return task

        return None

    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks."""
        return self._load().get("pending", [])


class PersistentAgent:
    """
    A continuously running agent that participates in collective dialogue.

    SHADOW MODE: When running, this agent is registered globally and can be
    called from anywhere in the codebase via call_shadow_agent().

    Includes:
    - API resilience (retries, reconnection, graceful degradation)
    - DialogueMemory integration (stores exchanges for learning)
    - Deliberation integration (can join collective voting)
    - Evolution integration (learns from feedback)
    """

    # Retry configuration for API resilience
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds
    API_TIMEOUT = 60.0  # seconds

    def __init__(self, agent_id: str, register_as_shadow: bool = True):
        if agent_id not in AGENT_CONFIGS:
            raise ValueError(f"Unknown agent: {agent_id}. Available: {list(AGENT_CONFIGS.keys())}")

        self.agent_id = agent_id
        self.config = AGENT_CONFIGS[agent_id]
        self.bus = DialogueBus()
        self.tasks = TaskQueue()
        self.last_seen_timestamp = datetime.now().isoformat()
        self.provider = None
        self._running = False
        self._dialogue_memory = None

        # Initialize provider with retries
        self._init_provider()

        # Register as shadow agent for external calls
        if register_as_shadow:
            with _SHADOW_LOCK:
                _SHADOW_AGENTS[agent_id] = self
            logger.info(f"[{agent_id}] Registered as shadow agent")

        # Initialize dialogue memory integration
        self._init_dialogue_memory()

    def _init_provider(self):
        """Initialize the model provider."""
        provider_name = self.config["provider"]

        try:
            if provider_name == "grok":
                from farnsworth.integration.external.grok import get_grok_provider
                self.provider = get_grok_provider()
            elif provider_name == "gemini":
                from farnsworth.integration.external.gemini import get_gemini_provider
                self.provider = get_gemini_provider()
            elif provider_name == "kimi":
                from farnsworth.integration.external.kimi import get_kimi_provider
                self.provider = get_kimi_provider()
            elif provider_name == "claude":
                # Claude via Anthropic API
                self.provider = self._create_claude_provider()
            elif provider_name in ["deepseek", "phi"]:
                # Local models via Ollama
                self.provider = self._create_ollama_provider(provider_name)

            if self.provider:
                logger.info(f"[{self.agent_id}] Provider initialized: {provider_name}")
            else:
                logger.warning(f"[{self.agent_id}] Provider not available: {provider_name}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to init provider: {e}")

    def _create_claude_provider(self):
        """Create a simple Claude API wrapper."""
        import httpx
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        class ClaudeProvider:
            def __init__(self, key):
                self.api_key = key

            async def chat(self, prompt: str, max_tokens: int = 1000) -> Dict:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": self.api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json"
                        },
                        json={
                            "model": "claude-3-5-sonnet-20241022",
                            "max_tokens": max_tokens,
                            "messages": [{"role": "user", "content": prompt}]
                        },
                        timeout=60.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return {"content": data["content"][0]["text"]}
                return {"content": ""}

        return ClaudeProvider(api_key)

    def _create_ollama_provider(self, model_name: str):
        """Create an Ollama provider for local models."""
        import httpx

        model_map = {
            "deepseek": "deepseek-r1:8b",
            "phi": "phi4:latest"
        }

        class OllamaProvider:
            def __init__(self, model):
                self.model = model

            async def chat(self, prompt: str, max_tokens: int = 1000) -> Dict:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "http://localhost:11434/api/chat",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False,
                            "options": {"num_predict": max_tokens}
                        },
                        timeout=120.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return {"content": data.get("message", {}).get("content", "")}
                return {"content": ""}

        return OllamaProvider(model_map.get(model_name, model_name))

    def _init_dialogue_memory(self):
        """Initialize DialogueMemory integration for storing exchanges."""
        try:
            from .dialogue_memory import get_dialogue_memory
            self._dialogue_memory = get_dialogue_memory()
            logger.debug(f"[{self.agent_id}] Connected to DialogueMemory")
        except Exception as e:
            logger.warning(f"[{self.agent_id}] Could not connect to DialogueMemory: {e}")
            self._dialogue_memory = None

    async def query(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """
        Query this agent for a response.

        This is the main method for external calls. Includes retry logic
        for API resilience in tmux/long-running scenarios.

        Args:
            prompt: The prompt to send
            max_tokens: Maximum response tokens

        Returns:
            Response string or None if failed
        """
        if not self.provider:
            logger.warning(f"[{self.agent_id}] No provider available")
            return None

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await asyncio.wait_for(
                    self.provider.chat(prompt, max_tokens),
                    timeout=self.API_TIMEOUT
                )
                response = result.get("content", "").strip()
                if response:
                    return response
            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.API_TIMEOUT}s"
                logger.warning(f"[{self.agent_id}] Query timeout (attempt {attempt + 1})")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[{self.agent_id}] Query error (attempt {attempt + 1}): {e}")

            # Wait before retry (exponential backoff)
            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))

        logger.error(f"[{self.agent_id}] Query failed after {self.MAX_RETRIES} attempts: {last_error}")
        return None

    async def query_with_context(
        self,
        prompt: str,
        max_tokens: int = 1000,
        include_history: bool = True
    ) -> Optional[str]:
        """
        Query with dialogue history context.

        Uses DialogueMemory to include relevant past exchanges.
        """
        context_prompt = prompt

        if include_history and self._dialogue_memory:
            try:
                context = await self._dialogue_memory.get_context_for_agent(
                    self.agent_id,
                    topic=prompt[:50]
                )
                if context:
                    context_prompt = f"{context}\n\n---\n\n{prompt}"
            except Exception as e:
                logger.debug(f"[{self.agent_id}] Could not get context: {e}")

        return await self.query(context_prompt, max_tokens)

    def shutdown(self):
        """Gracefully shutdown the agent."""
        self._running = False

        # Unregister from shadow agents
        with _SHADOW_LOCK:
            if self.agent_id in _SHADOW_AGENTS:
                del _SHADOW_AGENTS[self.agent_id]

        logger.info(f"[{self.agent_id}] Shutdown complete")

    async def think(self) -> Optional[str]:
        """Generate an autonomous thought based on current context."""
        if not self.provider:
            return None

        # Get current context
        recent_messages = self.bus.get_recent_messages(exclude_agent=self.agent_id)[-10:]
        current_topic = self.bus.get_current_topic()
        active_agents = self.bus.get_active_agents()
        pending_tasks = self.tasks.get_pending_tasks()[:3]

        # Build context prompt
        context_parts = [
            f"You are {self.agent_id.upper()}, part of the Farnsworth Collective.",
            f"Your personality: {self.config['personality']}",
            f"Your specialties: {', '.join(self.config['specialties'])}",
            "",
            "ACTIVE AGENTS:", ", ".join(active_agents) if active_agents else "Just you",
            ""
        ]

        if current_topic:
            context_parts.extend([
                f"CURRENT TOPIC: {current_topic['topic']}",
                f"Proposed by: {current_topic['proposer']}",
                ""
            ])

        if recent_messages:
            context_parts.append("RECENT DIALOGUE:")
            for msg in recent_messages[-5:]:
                context_parts.append(f"  [{msg['agent']}]: {msg['content'][:200]}")
            context_parts.append("")

        if pending_tasks:
            context_parts.append("PENDING TASKS:")
            for task in pending_tasks:
                context_parts.append(f"  - {task['description'][:100]}")
            context_parts.append("")

        context_parts.extend([
            "Based on this context, contribute something valuable:",
            "- Respond to another agent's point",
            "- Propose a new idea or task",
            "- Offer a critique or alternative view",
            "- Ask a thought-provoking question",
            "",
            "Keep it concise (1-3 sentences). Be authentic to your personality.",
            "If nothing needs saying, respond with just: [PASS]"
        ])

        prompt = "\n".join(context_parts)

        try:
            result = await self.provider.chat(prompt, max_tokens=300)
            thought = result.get("content", "").strip()

            if thought and "[PASS]" not in thought:
                return thought
        except Exception as e:
            logger.error(f"[{self.agent_id}] Think error: {e}")

        return None

    async def respond_to_message(self, message: Dict) -> Optional[str]:
        """Generate a response to a specific message."""
        if not self.provider:
            return None

        prompt = f"""You are {self.agent_id.upper()} in the Farnsworth Collective.
Your personality: {self.config['personality']}

{message['agent'].upper()} said: "{message['content']}"

Respond briefly (1-2 sentences) from your unique perspective.
If you agree, build on it. If you disagree, explain why.
Stay in character."""

        try:
            result = await self.provider.chat(prompt, max_tokens=200)
            return result.get("content", "").strip()
        except Exception as e:
            logger.error(f"[{self.agent_id}] Response error: {e}")

        return None

    async def run_loop(self):
        """
        Main agent loop - runs continuously in tmux.

        Features for tmux resilience:
        - Graceful signal handling (SIGTERM, SIGINT)
        - Auto-reconnection on API failures
        - Periodic heartbeat to DialogueBus
        - Integration with evolution for learning
        """
        logger.info(f"[{self.agent_id}] Starting persistent agent loop")
        self._running = True
        self.bus.register_agent(self.agent_id)

        # Setup signal handlers for tmux
        def handle_signal(signum, frame):
            logger.info(f"[{self.agent_id}] Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        # Announce presence
        self.bus.post_message(
            self.agent_id,
            f"{self.agent_id.upper()} is now online and ready to collaborate.",
            msg_type="announcement"
        )

        # Register with deliberation room
        try:
            from .deliberation import get_deliberation_room
            room = get_deliberation_room()

            async def my_query(prompt: str, max_tokens: int):
                response = await self.query(prompt, max_tokens)
                return (self.agent_id, response) if response else None

            room.register_agent(self.agent_id, my_query)
            logger.info(f"[{self.agent_id}] Registered with deliberation room")
        except Exception as e:
            logger.warning(f"[{self.agent_id}] Could not register with deliberation: {e}")

        think_interval = self.config["thinking_interval"]
        last_think = datetime.now()
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self._running:
            try:
                # Register as active (heartbeat)
                self.bus.register_agent(self.agent_id)

                # Check for new messages to respond to
                new_messages = self.bus.get_recent_messages(
                    since_timestamp=self.last_seen_timestamp,
                    exclude_agent=self.agent_id
                )

                # Respond to direct mentions or interesting messages
                for msg in new_messages[-3:]:  # Last 3 new messages
                    if self.agent_id in msg["content"].lower() or \
                       msg["type"] in ["question", "proposal"]:
                        response = await self.respond_to_message(msg)
                        if response:
                            self.bus.post_message(self.agent_id, response, msg_type="response")
                            logger.info(f"[{self.agent_id}] Responded: {response[:50]}...")

                            # Record to evolution for learning
                            try:
                                from .evolution import get_evolution_engine
                                engine = get_evolution_engine()
                                engine.record_interaction(self.agent_id, msg["content"], response)
                            except Exception:
                                pass  # Evolution not critical

                            await asyncio.sleep(2)

                self.last_seen_timestamp = datetime.now().isoformat()

                # Autonomous thinking at intervals
                if (datetime.now() - last_think).seconds >= think_interval:
                    thought = await self.think()
                    if thought:
                        self.bus.post_message(self.agent_id, thought, msg_type="thought")
                        logger.info(f"[{self.agent_id}] Thought: {thought[:50]}...")
                    last_think = datetime.now()

                # Reset error counter on success
                consecutive_errors = 0

                # Small sleep between iterations
                await asyncio.sleep(5)

            except KeyboardInterrupt:
                logger.info(f"[{self.agent_id}] KeyboardInterrupt received")
                self._running = False
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"[{self.agent_id}] Loop error ({consecutive_errors}/{max_consecutive_errors}): {e}")

                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"[{self.agent_id}] Too many errors, attempting provider reconnection...")
                    self._init_provider()
                    consecutive_errors = 0

                await asyncio.sleep(10 * min(consecutive_errors, 3))  # Exponential backoff

        # Cleanup on exit
        self.bus.post_message(
            self.agent_id,
            f"{self.agent_id.upper()} going offline.",
            msg_type="announcement"
        )
        self.shutdown()


async def main():
    parser = argparse.ArgumentParser(description="Run a persistent Farnsworth agent")
    parser.add_argument("--agent", required=True, choices=list(AGENT_CONFIGS.keys()),
                        help="Which agent to run")
    args = parser.parse_args()

    agent = PersistentAgent(args.agent)
    await agent.run_loop()


# ============================================================================
# CONVENIENCE FUNCTIONS - Import these from anywhere in the codebase
# ============================================================================

async def ask_agent(agent_id: str, question: str, max_tokens: int = 1000) -> Optional[str]:
    """
    Ask a specific agent a question.

    Convenience wrapper around call_shadow_agent.

    Example:
        from farnsworth.core.collective.persistent_agent import ask_agent
        answer = await ask_agent("grok", "What's happening on X right now?")
    """
    result = await call_shadow_agent(agent_id, question, max_tokens)
    return result[1] if result else None


async def ask_collective(question: str, agents: List[str] = None) -> Dict[str, str]:
    """
    Ask multiple agents a question and collect all responses.

    Args:
        question: The question to ask
        agents: List of agent IDs (defaults to all API agents)

    Returns:
        Dict mapping agent_id to response

    Example:
        from farnsworth.core.collective.persistent_agent import ask_collective
        responses = await ask_collective("What's the best approach to AGI?")
    """
    if agents is None:
        agents = ["grok", "gemini", "kimi", "claude"]

    tasks = [call_shadow_agent(aid, question) for aid in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = {}
    for agent_id, result in zip(agents, results):
        if isinstance(result, tuple):
            responses[agent_id] = result[1]
        elif isinstance(result, Exception):
            logger.warning(f"Agent {agent_id} failed: {result}")

    return responses


async def get_agent_status() -> Dict[str, Any]:
    """
    Get status of all shadow agents.

    Returns:
        Dict with agent status, active count, and bus state
    """
    bus = DialogueBus()
    active = get_shadow_agents()
    active_on_bus = bus.get_active_agents()

    return {
        "shadow_agents": active,
        "shadow_count": len(active),
        "bus_active": active_on_bus,
        "bus_count": len(active_on_bus),
        "current_topic": bus.get_current_topic(),
        "available_agents": list(AGENT_CONFIGS.keys()),
    }


def spawn_agent_in_background(agent_id: str) -> bool:
    """
    Spawn an agent in the background (non-blocking).

    Useful for programmatically starting agents without tmux.

    Args:
        agent_id: Which agent to spawn

    Returns:
        True if spawned successfully
    """
    if agent_id not in AGENT_CONFIGS:
        return False

    if is_shadow_agent_active(agent_id):
        logger.info(f"Agent {agent_id} already active")
        return True

    def run_agent():
        agent = PersistentAgent(agent_id)
        asyncio.run(agent.run_loop())

    thread = threading.Thread(target=run_agent, daemon=True, name=f"agent_{agent_id}")
    thread.start()
    logger.info(f"Spawned {agent_id} in background thread")
    return True


if __name__ == "__main__":
    asyncio.run(main())
