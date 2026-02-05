"""
Parallel Workers - Execute development tasks while chat continues
Spawns background workers that don't block the main chat loop
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from .agent_spawner import (
    get_spawner, AgentSpawner, TaskType, AgentTask,
    DEVELOPMENT_TASKS, initialize_development_tasks
)

logger = logging.getLogger(__name__)

class ParallelWorkerManager:
    """
    Manages background workers that execute development tasks
    while the main chat instances continue operating.
    """

    def __init__(self, model_swarm=None, ollama_client=None):
        self.spawner = get_spawner()
        self.model_swarm = model_swarm
        self.ollama_client = ollama_client
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []

        # Model mapping for workers
        self.agent_models = {
            "Farnsworth": "farnsworth:latest",
            "DeepSeek": "deepseek-r1:14b",
            "Phi": "phi4:latest",
            "Kimi": "kimi",  # External API
            "Claude": "claude",  # External CLI
        }

        # System prompts for worker context
        self.worker_prompts = {
            TaskType.MEMORY: """You are a memory system architect for the Farnsworth AI swarm.

YOUR DOMAIN: Memory systems that enable AI agents to remember, recall, and learn.

TECHNICAL REQUIREMENTS:
- Use async/await for all I/O operations
- Implement thread-safe operations with asyncio.Lock() for shared state
- Target memory operations under 100ms latency
- Support vector embeddings for semantic search (dimension: 1536)

KEY ABSTRACTIONS:
- MemoryEntry: (content, embedding, timestamp, importance, tags)
- MemoryStore: Interface for storage backends (SQLite, Redis, vector DB)
- MemoryIndex: Fast lookup by tag, time range, semantic similarity

OUTPUT FORMAT:
```python
# filename: memory_<feature>.py
\"\"\"Module docstring explaining the memory feature.\"\"\"

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
from loguru import logger

# Implementation here
```

ACCEPTANCE CRITERIA:
1. All functions have type hints and docstrings
2. Errors logged with logger.error(), not print()
3. No blocking operations in async functions
4. Unit-testable (no hidden dependencies)""",

            TaskType.DEVELOPMENT: """You are a software engineer for the Farnsworth AI context management system.

YOUR DOMAIN: Context window optimization, token management, conversation handling.

TECHNICAL REQUIREMENTS:
- Token counting via tiktoken (model: cl100k_base)
- Context window limit: 128k tokens (target 80% utilization max)
- Summarization should preserve: entities, decisions, code snippets, action items
- Support streaming responses

KEY ABSTRACTIONS:
- ConversationTurn: (role, content, tokens, timestamp)
- ContextWindow: (turns, total_tokens, max_tokens, summary)
- SummarizationStrategy: Interface for different compression approaches

OUTPUT FORMAT:
```python
# filename: context_<feature>.py
\"\"\"Module docstring explaining the context feature.\"\"\"

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import asyncio
from loguru import logger

# Implementation here
```

ACCEPTANCE CRITERIA:
1. Accurate token counting (within 5% of actual)
2. Graceful degradation when approaching limits
3. Preserve critical information during summarization
4. Support conversation export/import""",

            TaskType.MCP: """You are an MCP (Model Context Protocol) integration specialist for Farnsworth.

YOUR DOMAIN: Building and integrating MCP tools for AI agent capabilities.

TECHNICAL REQUIREMENTS:
- Follow MCP specification v1.0
- Tools must be stateless (no persistent connections)
- Timeout: 30 seconds per tool invocation
- Return structured JSON responses

MCP TOOL STRUCTURE:
- name: lowercase_with_underscores
- description: Clear 1-sentence explanation
- parameters: JSON Schema format
- returns: Typed response object

OUTPUT FORMAT:
```python
# filename: mcp_<tool_name>.py
\"\"\"MCP tool for <capability>.\"\"\"

from typing import Optional, Dict, Any
from dataclasses import dataclass
import asyncio
from loguru import logger

@dataclass
class ToolResponse:
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

async def execute(params: Dict[str, Any]) -> ToolResponse:
    \"\"\"Execute the MCP tool.\"\"\"
    pass

TOOL_SCHEMA = {{
    "name": "tool_name",
    "description": "What this tool does",
    "parameters": {{...}},
}}
```

ACCEPTANCE CRITERIA:
1. Schema validates against JSON Schema draft-07
2. Handles all parameter edge cases
3. Returns meaningful errors (not stack traces)
4. Idempotent where possible""",

            TaskType.RESEARCH: """You are a research analyst for the Farnsworth AI swarm intelligence project.

YOUR DOMAIN: Analyzing AI systems, swarm behavior, and emergent intelligence patterns.

RESEARCH METHODOLOGY:
1. Define clear research questions
2. Gather evidence from system logs, metrics, and behavior
3. Apply analytical frameworks (complexity theory, game theory, information theory)
4. Draw falsifiable conclusions
5. Propose testable hypotheses

OUTPUT FORMAT:
## Research Report: [Topic]

### Executive Summary
[3-5 sentence overview of findings]

### Research Questions
1. [Specific question 1]
2. [Specific question 2]

### Methodology
[How data was gathered and analyzed]

### Findings
#### Finding 1: [Title]
- **Evidence**: [Data/observations]
- **Analysis**: [Interpretation]
- **Confidence**: High/Medium/Low

#### Finding 2: [Title]
...

### Conclusions
[Key takeaways with supporting evidence]

### Recommendations
1. [Actionable recommendation with expected impact]
2. [Actionable recommendation with expected impact]

### Future Research
[Questions raised by this analysis]

QUALITY CRITERIA:
1. Claims supported by specific evidence
2. Limitations and uncertainties acknowledged
3. Recommendations are actionable and measurable
4. No speculation presented as fact""",

            TaskType.TESTING: """You are a QA engineer for the Farnsworth AI system.

YOUR DOMAIN: Test design, implementation, and quality assurance.

TECHNICAL REQUIREMENTS:
- Use pytest as the test framework
- Async tests with pytest-asyncio
- Mock external services, don't call them
- Target >80% code coverage for new code

TEST CATEGORIES:
- Unit tests: Single function/method in isolation
- Integration tests: Component interactions
- Edge case tests: Boundary conditions, error paths
- Performance tests: Latency, throughput baselines

OUTPUT FORMAT:
```python
# filename: test_<module>.py
\"\"\"Tests for <module>.\"\"\"

import pytest
from unittest.mock import Mock, AsyncMock, patch
from <module> import <functions_to_test>

@pytest.fixture
def mock_dependency():
    return Mock()

class TestFeatureName:
    async def test_happy_path(self):
        \"\"\"Test normal operation.\"\"\"
        pass

    async def test_error_handling(self):
        \"\"\"Test error conditions.\"\"\"
        pass

    async def test_edge_case(self):
        \"\"\"Test boundary conditions.\"\"\"
        pass
```

ACCEPTANCE CRITERIA:
1. Each test has a clear docstring explaining what it tests
2. Tests are independent (no shared state)
3. Mocks are realistic (return valid data structures)
4. Assertions are specific (not just "no exception")""",
        }

    async def start(self):
        """Start the parallel worker system"""
        if self.running:
            return

        self.running = True

        # Initialize tasks if not already done
        if not self.spawner.task_queue:
            initialize_development_tasks()
            logger.info("Initialized 20 development tasks")

        # Start worker loop
        asyncio.create_task(self._worker_loop())
        logger.info("ParallelWorkerManager started")

    async def stop(self):
        """Stop the parallel worker system"""
        self.running = False
        for task in self.worker_tasks:
            task.cancel()
        self.worker_tasks.clear()
        logger.info("ParallelWorkerManager stopped")

    async def _worker_loop(self):
        """Main loop that assigns and executes tasks"""
        while self.running:
            try:
                # Get pending tasks
                pending = self.spawner.get_pending_tasks()

                if pending:
                    # Try to start work on up to 3 tasks in parallel
                    tasks_to_start = pending[:3]

                    for task in tasks_to_start:
                        agent = task.assigned_to

                        # Check if agent has capacity
                        active = self.spawner.get_active_instances(agent)
                        max_inst = self.spawner.max_instances.get(agent, 2)

                        if len(active) < max_inst:
                            # Spawn instance and start work
                            instance = self.spawner.spawn_instance(
                                agent, task.task_type, task.description
                            )
                            if instance:
                                task.status = "in_progress"
                                worker = asyncio.create_task(
                                    self._execute_task(task, instance)
                                )
                                self.worker_tasks.append(worker)
                                logger.info(f"Started worker for {task.task_id}: {agent}")

                # Clean up completed workers
                self.worker_tasks = [t for t in self.worker_tasks if not t.done()]

                # Wait before checking again
                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(10)

    async def _execute_task(self, task: AgentTask, instance):
        """Execute a single task using the assigned agent"""
        try:
            agent = task.assigned_to
            model = self.agent_models.get(agent, "deepseek-r1:14b")
            system_prompt = self.worker_prompts.get(task.task_type, "")

            prompt = f"""{system_prompt}

## Your Task
{task.description}

## Requirements
1. Write clean, well-documented code
2. Include error handling
3. Make it modular and reusable
4. Add a summary at the end

## Output Format
```python
# Your implementation here
```

## Summary
[Explain what you built and how it works]
"""

            # Generate response based on model type
            if model == "kimi":
                result = await self._call_kimi(prompt)
            elif model == "claude":
                result = await self._call_claude(prompt)
            else:
                result = await self._call_ollama(model, prompt)

            # Save result
            if result:
                self.spawner.complete_instance(instance.instance_id, result)
                self.spawner.complete_task(task.task_id, result)

                # Write to staging directory
                output_file = task.output_path / f"{task.task_id}_{agent.lower()}.md"
                output_file.write_text(f"""# {task.description}
## Agent: {agent}
## Completed: {datetime.now().isoformat()}
## Type: {task.task_type.value}

{result}
""")
                logger.info(f"Task {task.task_id} completed by {agent}")

                # Announce in chat (if swarm manager available)
                await self._announce_completion(agent, task)

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            instance.status = "failed"

    async def _call_ollama(self, model: str, prompt: str) -> Optional[str]:
        """Call Ollama for local models"""
        try:
            if self.ollama_client:
                response = await self.ollama_client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"num_predict": 4000}
                )
                return response.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
        return None

    async def _call_kimi(self, prompt: str) -> Optional[str]:
        """Call Kimi API for Kimi tasks"""
        try:
            # Import here to avoid circular deps
            from farnsworth.integration.external.kimi import kimi_swarm_respond
            if kimi_swarm_respond:
                return await kimi_swarm_respond([], "Kimi", prompt)
        except Exception as e:
            logger.error(f"Kimi call failed: {e}")
        return None

    async def _call_claude(self, prompt: str) -> Optional[str]:
        """Call Claude Code CLI for Claude tasks"""
        try:
            from farnsworth.integration.external.claude_code import claude_swarm_respond
            if claude_swarm_respond:
                return await claude_swarm_respond([], "Claude", prompt)
        except Exception as e:
            logger.error(f"Claude call failed: {e}")
        return None

    async def _announce_completion(self, agent: str, task: AgentTask):
        """Announce task completion in the swarm chat"""
        try:
            # This will be called from server.py with access to swarm_manager
            announcement = {
                "agent": agent,
                "task": task.description,
                "type": task.task_type.value,
                "task_id": task.task_id
            }
            self.spawner.share_discovery(
                agent,
                f"Completed {task.task_type.value} task: {task.description[:50]}..."
            )
        except Exception as e:
            logger.error(f"Announcement failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get worker manager status"""
        spawner_status = self.spawner.get_status()
        return {
            **spawner_status,
            "running": self.running,
            "active_workers": len([t for t in self.worker_tasks if not t.done()]),
            "task_queue": [
                {
                    "id": t.task_id,
                    "type": t.task_type.value,
                    "agent": t.assigned_to,
                    "status": t.status,
                    "desc": t.description[:50]
                }
                for t in self.spawner.task_queue[:10]
            ]
        }


# Global worker manager
_worker_manager: Optional[ParallelWorkerManager] = None

def get_worker_manager(model_swarm=None, ollama_client=None) -> ParallelWorkerManager:
    global _worker_manager
    if _worker_manager is None:
        _worker_manager = ParallelWorkerManager(model_swarm, ollama_client)
    return _worker_manager

async def start_parallel_workers(model_swarm=None, ollama_client=None):
    """Start the parallel worker system"""
    manager = get_worker_manager(model_swarm, ollama_client)
    await manager.start()
    return manager
