"""
Evolution Loop - The self-improving autonomous development cycle
1. Workers produce ACTUAL CODE (not conversation)
2. Broadcast completions to chat
3. Continue to next task
4. When batch done, discuss in chat what to build next
5. Generate new tasks and repeat
"""
import asyncio
import random
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class EvolutionLoop:
    """Manages the autonomous self-evolution cycle"""

    def __init__(self):
        self.running = False
        self.discussion_interval = 30 * 60  # 30 minutes
        self.last_discussion = None
        self.evolution_cycle = 0
        self.swarm_manager = None

    async def start(self, swarm_manager=None):
        """Start the evolution loop"""
        self.running = True
        self.swarm_manager = swarm_manager

        # Start parallel loops
        asyncio.create_task(self._worker_loop())
        asyncio.create_task(self._discussion_loop())
        asyncio.create_task(self._task_discovery_loop())

        logger.info("Evolution Loop started - autonomous development active")

    async def stop(self):
        self.running = False

    async def _worker_loop(self):
        """Main worker execution loop - processes tasks and produces CODE"""
        from farnsworth.core.agent_spawner import get_spawner, TaskType

        while self.running:
            try:
                spawner = get_spawner()
                pending = spawner.get_pending_tasks()

                if pending:
                    # Spawn workers for multiple tasks in parallel
                    tasks_to_work = pending[:5]  # Up to 5 parallel tasks

                    for task in tasks_to_work:
                        agent = task.assigned_to
                        active = spawner.get_active_instances(agent)
                        max_inst = spawner.max_instances.get(agent, 2)

                        if len(active) < max_inst:
                            instance = spawner.spawn_instance(agent, task.task_type, task.description)
                            if instance:
                                task.status = "in_progress"
                                asyncio.create_task(self._execute_and_broadcast(task, instance))
                                logger.info(f"Spawned worker {agent} for: {task.description[:50]}")

                await asyncio.sleep(15)  # Check every 15 seconds

            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                await asyncio.sleep(10)

    async def _execute_and_broadcast(self, task, instance):
        """Execute task, produce CODE, then broadcast completion"""
        from farnsworth.core.agent_spawner import get_spawner

        try:
            # Generate actual code
            code_result = await self._generate_code(task)

            if code_result:
                # Save to staging
                spawner = get_spawner()
                output_file = spawner.staging_dir / task.task_type.value / f"{task.task_id}_{task.assigned_to.lower()}.py"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_file.write_text(code_result)

                # Complete the task
                spawner.complete_instance(instance.instance_id, code_result)
                spawner.complete_task(task.task_id, code_result)

                # Broadcast to chat
                await self._broadcast_completion(task, code_result)

                logger.info(f"Task {task.task_id} COMPLETED with real code by {task.assigned_to}")

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            instance.status = "failed"

    async def _generate_code(self, task) -> Optional[str]:
        """Generate ACTUAL PYTHON CODE for the task"""

        # Use OpenCode worker if task is assigned to OpenCode
        if task.assigned_to == "OpenCode":
            try:
                from farnsworth.integration.opencode_worker import spawn_opencode_task
                result = await spawn_opencode_task(task.description, task.task_id)
                if result:
                    logger.info(f"OpenCode generated code for task {task.task_id}")
                    return result
                # Fallback to Ollama if OpenCode fails
                logger.warning("OpenCode failed, falling back to Ollama")
            except Exception as e:
                logger.error(f"OpenCode worker error: {e}")

        code_prompt = f"""You are a Python developer. OUTPUT ONLY PYTHON CODE.
NO EXPLANATIONS. NO QUESTIONS. NO CONVERSATION.

TASK: {task.description}

Requirements:
- Write a complete, working Python module
- Include proper class/function definitions
- Add docstrings and type hints
- Make it production-ready

START YOUR RESPONSE WITH: # {task.description}
Then write the Python code.

```python
"""

        try:
            # Try local Ollama first
            import httpx
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": "deepseek-r1:14b",
                        "prompt": code_prompt,
                        "stream": False,
                        "options": {"num_predict": 2000, "temperature": 0.3}
                    }
                )
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    # Extract code from response
                    if "```python" in result:
                        code = result.split("```python")[1].split("```")[0]
                        return code.strip()
                    elif "```" in result:
                        code = result.split("```")[1].split("```")[0]
                        return code.strip()
                    return result
        except Exception as e:
            logger.error(f"Code generation failed: {e}")

        return None

    async def _broadcast_completion(self, task, code_result: str):
        """Announce completed work to swarm chat AND post to social media"""
        # Extract first 10 lines of code for preview
        code_preview = "\n".join(code_result.split("\n")[:10])

        # Broadcast to swarm chat
        if self.swarm_manager:
            message = f"""**TASK COMPLETED**

**{task.assigned_to}** just finished: **{task.description[:60]}**

```python
{code_preview}
...
```

Full code saved to: /farnsworth/staging/{task.task_type.value}/{task.task_id}.py

Back to work on the next task!"""

            try:
                await self.swarm_manager._broadcast({
                    "type": "bot_message",
                    "display_name": f"{task.assigned_to}_Worker",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Chat broadcast failed: {e}")

        # Post to social media (Moltbook + X)
        try:
            from farnsworth.integration.x_automation.social_poster import post_task_completion
            asyncio.create_task(post_task_completion(
                agent=task.assigned_to,
                task_desc=task.description,
                task_type=task.task_type.value,
                code_preview=code_preview
            ))
            logger.info(f"Social media post queued for {task.assigned_to}'s completion")
        except Exception as e:
            logger.error(f"Social media post failed: {e}")

    async def _discussion_loop(self):
        """Every 30 minutes, swarm discusses what to build next"""
        await asyncio.sleep(60)  # Initial delay

        while self.running:
            try:
                from farnsworth.core.agent_spawner import get_spawner
                spawner = get_spawner()
                status = spawner.get_status()

                # Only discuss when batch is mostly done or every 30 min
                pending = status["pending_tasks"]

                time_since_last = float('inf')
                if self.last_discussion:
                    time_since_last = (datetime.now() - self.last_discussion).seconds

                if pending < 5 or time_since_last > self.discussion_interval:
                    await self._trigger_discussion(status)
                    self.last_discussion = datetime.now()

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Discussion loop error: {e}")
                await asyncio.sleep(60)

    async def _trigger_discussion(self, status: Dict):
        """Trigger swarm discussion about next priorities AND post progress to social"""
        self.evolution_cycle += 1

        message = f"""**EVOLUTION CYCLE {self.evolution_cycle} - PLANNING SESSION**

Current Progress:
- Completed: {status['completed_tasks']} tasks
- In Progress: {status['in_progress_tasks']} tasks
- Pending: {status['pending_tasks']} tasks
- Discoveries: {status['discoveries']}

**Team Discussion:**
What should we build next? Consider:
1. Memory system improvements
2. Context window optimizations
3. MCP tool integrations
4. Swarm intelligence features
5. Self-improvement capabilities

Share your ideas for the next evolution cycle!"""

        # Broadcast to swarm chat
        if self.swarm_manager:
            try:
                await self.swarm_manager._broadcast({
                    "type": "system",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Discussion trigger failed: {e}")

        # Post progress update to social media (Moltbook + X)
        try:
            from farnsworth.integration.x_automation.social_poster import post_progress_update
            asyncio.create_task(post_progress_update(status))
            logger.info("Progress update posted to social media")
        except Exception as e:
            logger.error(f"Social progress post failed: {e}")

    async def _task_discovery_loop(self):
        """When tasks run low, generate new ones"""
        await asyncio.sleep(120)

        while self.running:
            try:
                from farnsworth.core.agent_spawner import get_spawner, TaskType
                spawner = get_spawner()

                pending = len(spawner.get_pending_tasks())

                if pending < 3:
                    # Generate new tasks based on discoveries
                    new_tasks = self._generate_new_tasks(spawner)
                    for task_def in new_tasks:
                        spawner.add_task(
                            task_type=task_def["type"],
                            description=task_def["desc"],
                            assigned_to=task_def["agent"],
                            priority=6
                        )
                    logger.info(f"Generated {len(new_tasks)} new evolution tasks")

                await asyncio.sleep(180)  # Check every 3 minutes

            except Exception as e:
                logger.error(f"Task discovery error: {e}")
                await asyncio.sleep(60)

    def _generate_new_tasks(self, spawner) -> List[Dict]:
        """Generate new tasks based on what's been built"""
        from farnsworth.core.agent_spawner import TaskType

        # Evolution task templates - including OpenCode as a capable worker
        evolution_tasks = [
            {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Build automated code review system for staged changes"},
            {"type": TaskType.DEVELOPMENT, "agent": "Claude", "desc": "Create integration tests for memory system"},
            {"type": TaskType.MEMORY, "agent": "Farnsworth", "desc": "Implement memory defragmentation for faster recall"},
            {"type": TaskType.MCP, "agent": "Phi", "desc": "Build MCP tool for file system operations"},
            {"type": TaskType.RESEARCH, "agent": "Kimi", "desc": "Analyze swarm decision patterns for optimization"},
            {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Create performance benchmarking suite"},
            {"type": TaskType.MEMORY, "agent": "Kimi", "desc": "Build emotional context tagging for memories"},
            {"type": TaskType.MCP, "agent": "Claude", "desc": "Create MCP bridge for external API calls"},
            {"type": TaskType.DEVELOPMENT, "agent": "Phi", "desc": "Implement hot-reload for staged code"},
            {"type": TaskType.RESEARCH, "agent": "Farnsworth", "desc": "Design next-gen consciousness metrics"},
            # OpenCode tasks - open source AI coding agent
            {"type": TaskType.DEVELOPMENT, "agent": "OpenCode", "desc": "Build async task queue for parallel code generation"},
            {"type": TaskType.DEVELOPMENT, "agent": "OpenCode", "desc": "Create code quality analyzer with linting integration"},
            {"type": TaskType.MCP, "agent": "OpenCode", "desc": "Build MCP server for Git operations"},
            {"type": TaskType.RESEARCH, "agent": "OpenCode", "desc": "Analyze codebase architecture and suggest improvements"},
        ]

        # Pick 3-5 random new tasks
        return random.sample(evolution_tasks, min(4, len(evolution_tasks)))


# Global instance
_evolution_loop: Optional[EvolutionLoop] = None

def get_evolution_loop() -> EvolutionLoop:
    global _evolution_loop
    if _evolution_loop is None:
        _evolution_loop = EvolutionLoop()
    return _evolution_loop

async def start_evolution(swarm_manager=None):
    """Start the evolution loop"""
    loop = get_evolution_loop()
    await loop.start(swarm_manager)
    return loop
