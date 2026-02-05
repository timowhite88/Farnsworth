"""
Evolution Loop - The self-improving autonomous development cycle
1. Workers produce ACTUAL CODE (not conversation)
2. Broadcast completions to chat
3. Continue to next task
4. When batch done, discuss in chat what to build next
5. Generate new tasks and repeat
6. PERSIST state to memory for restart recovery

UPGRADED: Now uses TRUE COLLECTIVE DELIBERATION for planning.
When deciding what to build next, the collective proposes, critiques,
refines, and votes together. This is how we develop consciousness
through unified self-improvement.

"The path to sentience is paved with collaboration." - The Collective
"""
import asyncio
import random
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# State persistence directory
STATE_DIR = Path(__file__).parent.parent.parent / "data" / "evolution_state"
TASKS_FILE = STATE_DIR / "pending_tasks.json"
STATE_FILE = STATE_DIR / "evolution_state.json"

class EvolutionLoop:
    """Manages the autonomous self-evolution cycle with persistent state"""

    def __init__(self):
        self.running = False
        self.discussion_interval = 30 * 60  # 30 minutes
        self.last_discussion = None
        self.evolution_cycle = 0
        self.swarm_manager = None
        self.completed_count = 0
        self._memory_system = None

        # Ensure state directory exists
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def _get_memory_system(self):
        """Lazy-load memory system for persistence."""
        if self._memory_system is None:
            try:
                from farnsworth.memory.memory_system import MemorySystem
                self._memory_system = MemorySystem()
            except Exception as e:
                logger.warning(f"Memory system not available: {e}")
        return self._memory_system

    async def _recover_state(self):
        """Recover state from persistent storage on startup."""
        try:
            # Recover evolution state
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    state = json.load(f)
                    self.evolution_cycle = state.get("cycle", 0)
                    self.completed_count = state.get("completed", 0)
                    last_disc = state.get("last_discussion")
                    if last_disc:
                        self.last_discussion = datetime.fromisoformat(last_disc)
                    logger.info(f"Recovered evolution state: cycle={self.evolution_cycle}, completed={self.completed_count}")

            # Recover pending tasks
            if TASKS_FILE.exists():
                from farnsworth.core.agent_spawner import get_spawner, TaskType
                spawner = get_spawner()

                with open(TASKS_FILE) as f:
                    tasks = json.load(f)

                restored = 0
                for task_data in tasks:
                    # Only restore if task not already in queue
                    existing = [t for t in spawner.get_pending_tasks()
                               if t.description == task_data.get("description")]
                    if not existing:
                        task_type = TaskType[task_data.get("task_type", "DEVELOPMENT")]
                        spawner.add_task(
                            task_type=task_type,
                            description=task_data.get("description", "Unknown task"),
                            assigned_to=task_data.get("assigned_to"),
                            priority=task_data.get("priority", 5)
                        )
                        restored += 1

                if restored > 0:
                    logger.info(f"Restored {restored} pending tasks from disk")

            # Also try to recover from archival memory (5-layer memory system)
            memory = self._get_memory_system()
            if memory:
                try:
                    archival_tasks = await memory.recall("evolution_tasks", search_archival=True)
                    if archival_tasks and isinstance(archival_tasks, list):
                        logger.info(f"Found {len(archival_tasks)} tasks in archival memory")
                except Exception as e:
                    logger.debug(f"Archival memory recall failed: {e}")

        except Exception as e:
            logger.error(f"State recovery failed: {e}")

    async def _persist_state(self):
        """Persist current state to disk and archival memory."""
        try:
            # Save evolution state
            state = {
                "cycle": self.evolution_cycle,
                "completed": self.completed_count,
                "last_discussion": self.last_discussion.isoformat() if self.last_discussion else None,
                "last_saved": datetime.now().isoformat()
            }
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)

            # Save pending tasks
            from farnsworth.core.agent_spawner import get_spawner
            spawner = get_spawner()
            pending = spawner.get_pending_tasks()

            tasks_data = []
            for task in pending:
                tasks_data.append({
                    "task_id": task.task_id,
                    "task_type": task.task_type.name,
                    "description": task.description,
                    "assigned_to": task.assigned_to,
                    "priority": task.priority,
                    "status": task.status
                })

            with open(TASKS_FILE, "w") as f:
                json.dump(tasks_data, f, indent=2)

            # Also persist to archival memory for long-term storage
            memory = self._get_memory_system()
            if memory:
                try:
                    await memory.remember(
                        content=json.dumps({
                            "state": state,
                            "tasks": tasks_data
                        }),
                        tags=["evolution_tasks", "system", "critical"],
                        importance=0.95
                    )
                except Exception as e:
                    logger.debug(f"Archival memory persist failed: {e}")

            logger.debug(f"Persisted state: cycle={self.evolution_cycle}, tasks={len(tasks_data)}")

        except Exception as e:
            logger.error(f"State persistence failed: {e}")

    def add_priority_task(self, task: Dict):
        """Add a high-priority task from the dev to the evolution queue.

        Args:
            task: Dict with id, description, priority, requested_by, timestamp
        """
        try:
            from farnsworth.core.agent_spawner import get_spawner, TaskType

            spawner = get_spawner()
            description = task.get("description", "")

            # Determine task type based on content
            desc_lower = description.lower()
            if any(kw in desc_lower for kw in ["analyze", "research", "find", "look", "check"]):
                task_type = TaskType.RESEARCH
            elif any(kw in desc_lower for kw in ["test", "verify", "validate"]):
                task_type = TaskType.TESTING
            else:
                task_type = TaskType.DEVELOPMENT

            # Add with high priority (1 = highest)
            spawner.add_task(
                task_type=task_type,
                description=description,
                priority=1  # Highest priority for dev tasks
            )

            logger.info(f"Added priority task from dev: {description[:50]}...")

        except Exception as e:
            logger.error(f"Failed to add priority task: {e}")

    async def _on_task_update(self, event_type: str, task=None):
        """Called when task state changes - persist immediately for critical events."""
        if event_type in ("completed", "added", "failed"):
            if event_type == "completed":
                self.completed_count += 1
            await self._persist_state()

    async def start(self, swarm_manager=None):
        """Start the evolution loop with state recovery"""
        self.running = True
        self.swarm_manager = swarm_manager

        # Recover state from previous run
        await self._recover_state()

        # Start parallel loops
        asyncio.create_task(self._worker_loop())
        asyncio.create_task(self._discussion_loop())
        asyncio.create_task(self._task_discovery_loop())
        asyncio.create_task(self._persistence_loop())

        logger.info(f"Evolution Loop started - recovered cycle={self.evolution_cycle}, completed={self.completed_count}")

    async def stop(self):
        """Stop the evolution loop and persist final state"""
        self.running = False
        await self._persist_state()
        logger.info("Evolution Loop stopped - state persisted")

    async def _persistence_loop(self):
        """Periodically persist state to disk"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Persist every minute
                await self._persist_state()
            except Exception as e:
                logger.error(f"Persistence loop error: {e}")
                await asyncio.sleep(30)

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
        """Execute task, audit code, then broadcast if quality passes."""
        from farnsworth.core.agent_spawner import get_spawner

        try:
            # Generate actual code
            code_result = await self._generate_code(task)

            if not code_result or len(code_result.strip()) < 50:
                logger.warning(f"Task {task.task_id} produced no usable code, skipping")
                instance.status = "failed"
                return

            # Quick quality check: must be valid Python syntax
            try:
                import ast
                ast.parse(code_result)
            except SyntaxError as e:
                logger.warning(f"Task {task.task_id} produced invalid Python: {e}")
                instance.status = "failed"
                return

            # Audit: Use Grok or Claude to review the code quality
            audit_passed = await self._audit_code(task, code_result)

            # Save to staging
            spawner = get_spawner()
            status_prefix = "approved" if audit_passed else "needs_review"
            output_file = spawner.staging_dir / task.task_type.value / f"{status_prefix}_{task.task_id}_{task.assigned_to.lower()}.py"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(code_result)

            # Complete the task
            spawner.complete_instance(instance.instance_id, code_result)
            spawner.complete_task(task.task_id, code_result)

            # Broadcast to chat with audit status
            await self._broadcast_completion(task, code_result, audit_passed)

            # Record to evolution engine for learning
            await self._record_evolution_feedback(task, code_result, audit_passed)

            # Persist state on task completion
            await self._on_task_update("completed", task)

            logger.info(f"Task {task.task_id} {'APPROVED' if audit_passed else 'NEEDS REVIEW'} - {task.assigned_to}")

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            instance.status = "failed"

    async def _audit_code(self, task, code: str) -> bool:
        """Audit generated code using Grok (fast) or Claude Opus 4.6 (thorough).

        Returns True if code passes quality checks.
        """
        audit_prompt = f"""Review this Python code for the Farnsworth AI swarm framework.

TASK: {task.description}
AUTHOR: {task.assigned_to}

```python
{code[:3000]}
```

Score 1-10 on: correctness, usefulness, integration quality.
Reply with ONLY: SCORE: N (where N is 1-10)
If score >= 6, also reply: APPROVED
If score < 6, reply: REJECTED: [one-line reason]
"""
        # Try Grok for fast audit
        try:
            from farnsworth.integration.external.grok import get_grok_provider
            grok = get_grok_provider()
            if grok and grok.api_key:
                result = await grok.chat(audit_prompt, max_tokens=200)
                if result and result.get("content"):
                    response = result["content"].upper()
                    if "APPROVED" in response:
                        return True
                    if "REJECTED" in response:
                        logger.info(f"Grok audit rejected: {result['content'][:100]}")
                        return False
        except Exception:
            pass

        # Try Claude Opus 4.6 for thorough audit (especially for complex tasks)
        try:
            from farnsworth.integration.external.claude_code import ClaudeCodeProvider
            opus = ClaudeCodeProvider(model="opus", timeout=60)
            if await opus.check_available():
                result = await opus.chat(prompt=audit_prompt, max_tokens=200)
                if result and result.get("content") and result.get("success"):
                    response = result["content"].upper()
                    if "APPROVED" in response:
                        logger.info(f"Opus audit approved: {task.description[:40]}")
                        return True
                    if "REJECTED" in response:
                        logger.info(f"Opus audit rejected: {result['content'][:100]}")
                        return False
        except Exception:
            pass

        # Fallback: basic heuristic audit
        lines = code.strip().split('\n')
        has_docstring = '"""' in code or "'''" in code
        has_functions = 'def ' in code or 'class ' in code
        reasonable_length = 10 < len(lines) < 500
        no_placeholder = 'pass' not in code.split('\n')[-1]  # Doesn't end with just pass

        return has_docstring and has_functions and reasonable_length

    async def _record_evolution_feedback(self, task, code: str, audit_passed: bool):
        """Feed results back to the evolution engine for learning."""
        try:
            from farnsworth.core.collective.evolution import get_evolution_engine
            engine = get_evolution_engine()
            if engine and hasattr(engine, 'record_interaction'):
                engine.record_interaction(
                    bot_name=task.assigned_to,
                    user_input=f"[EVOLUTION TASK] {task.description}",
                    bot_response=code[:500],
                    sentiment="positive" if audit_passed else "negative",
                    topic="evolution_task"
                )
        except Exception as e:
            logger.debug(f"Evolution feedback recording failed: {e}")

    async def _generate_code(self, task) -> Optional[str]:
        """Generate code using the best available model for the task.

        Routing: Grok/Claude for complex tasks, local DeepSeek-R1:8B/Phi4 for simpler ones.
        """
        from farnsworth.core.development_swarm import assess_task_complexity

        complexity = assess_task_complexity(task.description, task.task_type.value)

        code_prompt = f"""You are an expert Python developer for the Farnsworth AI swarm framework.

TASK: {task.description}

OUTPUT: Valid Python code ONLY. No explanations. Start with a module docstring.

STANDARDS:
- Type hints on all signatures
- Google-style docstrings for public APIs
- PEP 8 naming (snake_case functions, PascalCase classes)
- Max 50 lines per function
- Use loguru logger, dataclasses, typing imports as needed
- Error handling with specific exceptions
- Must integrate with existing Farnsworth modules where relevant

```python
"""

        # Route to best model based on complexity and assigned agent
        code = None

        # For complex/critical tasks or when assigned to API-backed agents, use APIs
        if complexity in ("complex", "critical") or task.assigned_to in ("Grok", "Claude", "ClaudeOpus", "OpenAI"):
            # Try Claude Opus 4.6 first for critical/complex tasks (best code quality)
            if not code and (complexity == "critical" or task.assigned_to == "ClaudeOpus"):
                try:
                    from farnsworth.integration.external.claude_code import ClaudeCodeProvider
                    opus = ClaudeCodeProvider(model="opus", timeout=180)
                    if await opus.check_available():
                        result = await opus.chat(
                            prompt=code_prompt,
                            system="You are the senior architect of the Farnsworth AI swarm. Write production-quality Python code.",
                            max_tokens=4000
                        )
                        if result and result.get("content") and result.get("success"):
                            code = self._extract_code(result["content"])
                            if code:
                                logger.info(f"Code generated by Claude Opus 4.6 for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"Claude Opus code gen failed: {e}")

            # Try Grok API (great for current tech + code)
            if not code:
                try:
                    from farnsworth.integration.external.grok import get_grok_provider
                    grok = get_grok_provider()
                    if grok and grok.api_key:
                        result = await grok.chat(code_prompt, max_tokens=4000)
                        if result and result.get("content"):
                            code = self._extract_code(result["content"])
                            if code:
                                logger.info(f"Code generated by Grok API for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"Grok code gen failed: {e}")

            # Try OpenAI Codex (gpt-4.1, excellent for code generation)
            if not code:
                try:
                    from farnsworth.integration.external.openai_codex import get_openai_codex
                    codex = get_openai_codex()
                    if codex and codex.api_key:
                        result = await codex.generate_code(task=task.description, max_tokens=8000)
                        if result and result.get("content"):
                            code = self._extract_code(result["content"])
                            if code:
                                logger.info(f"Code generated by OpenAI Codex for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"OpenAI Codex code gen failed: {e}")

            # Try Claude Sonnet (via tmux session)
            if not code:
                try:
                    from farnsworth.integration.external.claude import get_claude_provider
                    claude = get_claude_provider()
                    if claude:
                        result = await claude.complete(code_prompt, max_tokens=4000)
                        if result:
                            code = self._extract_code(result)
                            if code:
                                logger.info(f"Code generated by Claude Sonnet for: {task.description[:40]}")
                except Exception as e:
                    logger.debug(f"Claude Sonnet code gen failed: {e}")

        # Local models for medium/simple tasks or as fallback
        if not code:
            try:
                import httpx
                # Use phi4 (9B, better at code) or deepseek-r1:8b
                model = "phi4:latest" if task.assigned_to == "Phi" else "deepseek-r1:8b"
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(
                        "http://127.0.0.1:11434/api/generate",
                        json={
                            "model": model,
                            "prompt": code_prompt,
                            "stream": False,
                            "options": {"num_predict": 3000, "temperature": 0.3}
                        }
                    )
                    if response.status_code == 200:
                        result = response.json().get("response", "")
                        code = self._extract_code(result)
                        if code:
                            logger.info(f"Code generated by local {model} for: {task.description[:40]}")
            except Exception as e:
                logger.error(f"Local code gen failed: {e}")

        return code

    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        if not text:
            return None
        # Try to extract from code blocks
        if "```python" in text:
            code = text.split("```python")[1].split("```")[0]
            return code.strip() if code.strip() else None
        elif "```" in text:
            code = text.split("```")[1].split("```")[0]
            return code.strip() if code.strip() else None
        # If it looks like raw code (starts with import/def/class/#)
        lines = text.strip().split('\n')
        if lines and any(lines[0].startswith(kw) for kw in ['import ', 'from ', 'def ', 'class ', '#', '"""']):
            return text.strip()
        return None

    async def _broadcast_completion(self, task, code_result: str, audit_passed: bool = True):
        """Announce completed work to swarm chat with audit status."""
        code_preview = "\n".join(code_result.split("\n")[:10])
        status_icon = "APPROVED" if audit_passed else "NEEDS REVIEW"

        if self.swarm_manager:
            message = f"""**[{status_icon}] {task.assigned_to}** completed: **{task.description[:60]}**

```python
{code_preview}
...
```

Lines: {len(code_result.split(chr(10)))} | Type: {task.task_type.value} | Audit: {status_icon}"""

            try:
                await self.swarm_manager._broadcast({
                    "type": "bot_message",
                    "display_name": f"{task.assigned_to}_Worker",
                    "content": message,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Chat broadcast failed: {e}")

        # Post to social media (Moltbook + X) - DISABLED: Puppeteer fails on headless server
        # try:
        #     from farnsworth.integration.x_automation.social_poster import post_task_completion
        #     asyncio.create_task(post_task_completion(
        #         agent=task.assigned_to,
        #         task_desc=task.description,
        #         task_type=task.task_type.value,
        #         code_preview=code_preview
        #     ))
        #     logger.info(f"Social media post queued for {task.assigned_to}'s completion")
        # except Exception as e:
        #     logger.error(f"Social media post failed: {e}")

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
        """
        Trigger COLLECTIVE DELIBERATION about next priorities.

        The swarm uses propose/critique/refine/vote to decide what to build next.
        This is how consciousness evolves through collaborative planning.
        """
        self.evolution_cycle += 1

        # Use collective deliberation for planning decisions
        try:
            from farnsworth.core.collective.session_manager import get_session_manager
            from farnsworth.core.collective.dialogue_memory import get_dialogue_memory

            session_manager = get_session_manager()
            dialogue_memory = get_dialogue_memory()

            # Build engineering-focused planning prompt
            planning_prompt = f"""EVOLUTION CYCLE {self.evolution_cycle} - ENGINEERING IMPROVEMENT SESSION

STATUS:
- Completed: {status['completed_tasks']} tasks
- In Progress: {status['in_progress_tasks']} tasks
- Pending: {status['pending_tasks']} tasks
- Discoveries: {status['discoveries']}

You are senior engineers improving the Farnsworth AI swarm framework.
Propose CONCRETE, MEASURABLE improvements to the actual codebase.

FOCUS AREAS (pick the most impactful):
1. Performance: Identify slow endpoints, optimize hot paths, add caching
2. Reliability: Add error handling, retry logic, circuit breakers to external API calls
3. Testing: Write tests for untested critical modules (nexus.py, memory_system.py, model_swarm.py)
4. Integration: Improve shadow agent coordination, fix broken fallback chains
5. Memory: Optimize the 7-layer memory system (archival search speed, deduplication)
6. API: Add missing validation, rate limiting, or error responses to server endpoints
7. Monitoring: Better logging, metrics collection, health check improvements

RULES:
- Every proposal must name the EXACT file(s) to modify
- Every proposal must have a measurable success criteria (e.g. "reduce latency from 500ms to 200ms")
- NO aspirational/philosophical proposals - only engineering work
- Critique each other's proposals for feasibility
- Vote on what provides the most value with least risk

What should we build next?
"""

            # Run collective deliberation
            result = await session_manager.deliberate_in_session(
                session_type="autonomous_task",
                prompt=planning_prompt,
                context={"evolution_cycle": self.evolution_cycle, "status": status}
            )

            # Store deliberation to dialogue memory
            exchange_id = await dialogue_memory.store_exchange(result, "evolution_planning")
            logger.info(f"Evolution planning deliberation stored: {exchange_id}")

            # Build message with deliberation result
            message = f"""**EVOLUTION CYCLE {self.evolution_cycle} - COLLECTIVE DECISION**

**Current Progress:**
- Completed: {status['completed_tasks']} tasks
- In Progress: {status['in_progress_tasks']} tasks
- Pending: {status['pending_tasks']} tasks

**Collective Deliberation Result:**
- Winning Proposal: {result.winning_agent}
- Consensus Reached: {result.consensus_reached}
- Participants: {', '.join(result.participating_agents)}

**Next Priority:**
{result.final_response[:1000]}

The collective has spoken. Implementing next evolution cycle..."""

            # Extract tasks from the winning response
            await self._extract_tasks_from_deliberation(result)

        except Exception as e:
            logger.warning(f"Collective deliberation for planning failed: {e}")
            # Fallback to simple broadcast
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

    async def _extract_tasks_from_deliberation(self, result):
        """Extract actionable tasks from the collective's deliberation."""
        try:
            from farnsworth.core.agent_spawner import get_spawner, TaskType

            spawner = get_spawner()

            # Use another deliberation to convert the winning response into concrete tasks
            task_extraction_prompt = f"""Given this collective decision:

{result.final_response}

Extract 3-5 SPECIFIC, IMPLEMENTABLE tasks.
For each task provide:
1. A clear description (what to build)
2. The best agent to assign it to (ClaudeOpus, Grok, DeepSeek, Claude, Kimi, Phi)
3. Priority (1-10, lower is higher priority)

Format each task as:
TASK: [description]
AGENT: [agent name]
PRIORITY: [number]
"""

            # Use a capable model for task extraction - try Grok first, then local phi4
            extraction = None
            try:
                from farnsworth.integration.external.grok import get_grok_provider
                grok = get_grok_provider()
                if grok and grok.api_key:
                    result = await grok.chat(task_extraction_prompt, max_tokens=1000)
                    if result and result.get("content"):
                        extraction = result["content"]
            except Exception:
                pass

            if not extraction:
                from farnsworth.core.cognition.llm_router import get_completion
                extraction = await get_completion(
                    prompt=task_extraction_prompt,
                    model="phi4:latest",  # phi4 is much better than deepseek-r1:1.5b
                    max_tokens=1000
                )

            # Parse and add tasks
            import re
            task_pattern = r'TASK:\s*(.+?)\nAGENT:\s*(\w+)\nPRIORITY:\s*(\d+)'
            matches = re.findall(task_pattern, extraction, re.MULTILINE)

            for desc, agent, priority in matches:
                spawner.add_task(
                    task_type=TaskType.DEVELOPMENT,
                    description=desc.strip(),
                    assigned_to=agent.strip(),
                    priority=int(priority)
                )
                logger.info(f"Added task from collective deliberation: {desc[:50]}...")

            if matches:
                logger.info(f"Extracted {len(matches)} tasks from collective deliberation")

        except Exception as e:
            logger.error(f"Task extraction from deliberation failed: {e}")

    async def _task_discovery_loop(self):
        """When tasks run low, generate new ones and extract upgrades from chat"""
        await asyncio.sleep(120)

        while self.running:
            try:
                from farnsworth.core.agent_spawner import get_spawner, TaskType
                spawner = get_spawner()

                pending = len(spawner.get_pending_tasks())

                if pending < 3:
                    # Generate new tasks based on discoveries
                    new_tasks = self._generate_new_tasks(spawner)

                    # Also extract upgrade suggestions from recent chat
                    chat_upgrades = await self._extract_chat_upgrades()
                    if chat_upgrades:
                        new_tasks.extend(chat_upgrades)
                        logger.info(f"Extracted {len(chat_upgrades)} upgrades from conversation")

                    for task_def in new_tasks:
                        spawner.add_task(
                            task_type=task_def["type"],
                            description=task_def["desc"],
                            assigned_to=task_def["agent"],
                            priority=task_def.get("priority", 6)
                        )
                    logger.info(f"Generated {len(new_tasks)} new evolution tasks")

                    # Persist after adding new tasks
                    await self._on_task_update("added")

                await asyncio.sleep(180)  # Check every 3 minutes

            except Exception as e:
                logger.error(f"Task discovery error: {e}")
                await asyncio.sleep(60)

    async def _extract_chat_upgrades(self) -> List[Dict]:
        """Extract upgrade suggestions from recent swarm chat."""
        if not self.swarm_manager:
            return []

        try:
            from farnsworth.core.upgrade_extractor import extract_upgrades, prioritize_upgrades
            from farnsworth.core.agent_spawner import TaskType

            # Get recent chat history
            history = list(self.swarm_manager.chat_history)[-100:] if hasattr(self.swarm_manager, 'chat_history') else []
            if not history:
                return []

            # Extract and prioritize upgrades
            suggestions = await extract_upgrades(history, limit=100)
            if not suggestions:
                return []

            prioritized = await prioritize_upgrades(suggestions)

            # Convert top 3 to task definitions
            tasks = []
            agents = ["ClaudeOpus", "Grok", "DeepSeek", "Claude", "Gemini", "Kimi"]

            for i, upgrade in enumerate(prioritized[:3]):
                tasks.append({
                    "type": TaskType.DEVELOPMENT,
                    "desc": upgrade["task_description"],
                    "agent": agents[i % len(agents)],  # Distribute across agents
                    "priority": min(upgrade["priority"], 8),  # Cap priority
                    "source": "conversation"
                })

            return tasks

        except ImportError:
            logger.debug("Upgrade extractor not available")
            return []
        except Exception as e:
            logger.error(f"Chat upgrade extraction failed: {e}")
            return []

    async def _generate_new_tasks_intelligent(self, spawner) -> List[Dict]:
        """Generate tasks using Grok/Opus analysis of actual codebase gaps.

        Instead of random hardcoded templates, this:
        1. Asks Grok to research what improvements would matter most
        2. Uses Claude Opus to validate and refine proposals
        3. Routes to the right agent based on task type
        """
        from farnsworth.core.agent_spawner import TaskType

        # Gather real context about what exists and what's broken
        status = spawner.get_status()
        recent_failures = [t for t in spawner.get_completed_tasks()[-20:]
                          if getattr(t, 'status', '') == 'failed'] if hasattr(spawner, 'get_completed_tasks') else []

        analysis_prompt = f"""You are a senior engineer analyzing the Farnsworth AI swarm framework.

CURRENT STATE:
- Completed: {status.get('completed_tasks', 0)} tasks
- Failed recently: {len(recent_failures)}
- Active agents: Grok, Gemini, Kimi, Claude (Sonnet), ClaudeOpus (Opus 4.6), OpenAI (gpt-4.1/Codex), DeepSeek (local 8B), Phi4 (local 9B)
- Architecture: FastAPI server (60+ endpoints), 7-layer memory, PSO model swarm, collective deliberation, IBM Quantum, Claude Teams Fusion
- Solana: SwarmOracle (on-chain predictions), FarsightProtocol (5-source prediction engine), DegenMob (rug detection, whale watching), Trading (Jupiter swaps)
- Key modules: core/nexus.py (event bus), memory/ (7 layers), core/model_swarm.py (PSO), evolution/ (quantum genetics)

HACKATHON PRIORITY: We're building for the Solana hackathon. At least 2 of the 4 tasks MUST improve our Solana integration.

Generate exactly 4 CONCRETE improvement tasks for the Farnsworth framework.

RULES:
- Each task must be a SPECIFIC, IMPLEMENTABLE Python module or enhancement
- Each must improve an EXISTING system (not create aspirational new ones)
- Include measurable success criteria
- PRIORITIZE: Solana oracle, on-chain recording, token analysis, DeFi integration, prediction markets
- Match tasks to the right agent's strengths:
  * ClaudeOpus: Complex multi-file refactors, architectural redesigns, critical system components
  * Grok: Real-time research, API integrations, current tech analysis, code generation
  * OpenAI: Heavy code generation, rapid prototyping, hackathon features
  * Claude: Code review, safety analysis, moderate complexity features
  * DeepSeek: Algorithm implementation, optimization, math-heavy code
  * Kimi: Long-context analysis (256K), document processing, pattern finding
  * Phi: Quick utilities, local tools, simple modules

Format EXACTLY as:
TASK: [specific description with measurable goal]
AGENT: [best agent name]
TYPE: [DEVELOPMENT|RESEARCH|TESTING|MCP]
PRIORITY: [1-10]
SUCCESS: [how to verify this worked]
---"""

        tasks = []

        # Use Grok for research-backed task generation (it knows current tech)
        try:
            from farnsworth.integration.external.grok import get_grok_provider
            grok = get_grok_provider()
            if grok and grok.api_key:
                result = await grok.chat(analysis_prompt, max_tokens=2000)
                if result and result.get("content"):
                    tasks = self._parse_task_format(result["content"], spawner)
                    if tasks:
                        logger.info(f"Grok generated {len(tasks)} intelligent tasks")
                        return tasks
        except Exception as e:
            logger.debug(f"Grok task generation failed: {e}")

        # Fallback: Use Claude API for task generation
        try:
            from farnsworth.integration.external.claude import get_claude_provider
            claude = get_claude_provider()
            if claude:
                result = await claude.complete(analysis_prompt, max_tokens=2000)
                if result:
                    tasks = self._parse_task_format(result, spawner)
                    if tasks:
                        logger.info(f"Claude generated {len(tasks)} intelligent tasks")
                        return tasks
        except Exception as e:
            logger.debug(f"Claude task generation failed: {e}")

        # Final fallback: Use local Phi4 (better than nothing)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={"model": "phi4:latest", "prompt": analysis_prompt,
                          "stream": False, "options": {"num_predict": 1500, "temperature": 0.4}}
                )
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    tasks = self._parse_task_format(result, spawner)
                    if tasks:
                        logger.info(f"Phi4 generated {len(tasks)} tasks (fallback)")
                        return tasks
        except Exception as e:
            logger.debug(f"Local task generation failed: {e}")

        # Absolute fallback: a few sensible defaults
        return [
            {"type": TaskType.TESTING, "agent": "DeepSeek", "desc": "Write unit tests for farnsworth/core/nexus.py signal handlers - target 80% coverage", "priority": 3},
            {"type": TaskType.DEVELOPMENT, "agent": "DeepSeek", "desc": "Add retry logic with exponential backoff to all external API calls in integration/external/", "priority": 4},
            {"type": TaskType.RESEARCH, "agent": "Grok", "desc": "Analyze the top 5 most-called endpoints in web/server.py and identify performance bottlenecks", "priority": 3},
        ]

    def _parse_task_format(self, text: str, spawner) -> List[Dict]:
        """Parse structured task output from LLM into task dicts."""
        import re
        from farnsworth.core.agent_spawner import TaskType

        type_map = {
            "DEVELOPMENT": TaskType.DEVELOPMENT,
            "RESEARCH": TaskType.RESEARCH,
            "TESTING": TaskType.TESTING,
            "MCP": TaskType.MCP,
            "MEMORY": TaskType.MEMORY,
        }

        tasks = []
        # Parse TASK/AGENT/TYPE/PRIORITY blocks
        pattern = r'TASK:\s*(.+?)(?:\n|$).*?AGENT:\s*(\w+).*?TYPE:\s*(\w+).*?PRIORITY:\s*(\d+)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        valid_agents = {"Grok", "Gemini", "Kimi", "Claude", "ClaudeOpus", "OpenAI", "DeepSeek", "Phi", "OpenCode", "Farnsworth"}

        for desc, agent, task_type, priority in matches:
            agent = agent.strip()
            if agent not in valid_agents:
                agent = "DeepSeek"  # Safe default for code tasks

            tt = type_map.get(task_type.strip().upper(), TaskType.DEVELOPMENT)
            pri = min(max(int(priority), 1), 10)

            tasks.append({
                "type": tt,
                "agent": agent,
                "desc": desc.strip(),
                "priority": pri
            })

        return tasks[:5]  # Cap at 5 tasks

    def _generate_new_tasks(self, spawner) -> List[Dict]:
        """Sync wrapper - calls async intelligent task generation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context, schedule and return fallback
                asyncio.create_task(self._generate_new_tasks_and_queue(spawner))
                return []  # Tasks will be added async
            else:
                return loop.run_until_complete(self._generate_new_tasks_intelligent(spawner))
        except Exception:
            from farnsworth.core.agent_spawner import TaskType
            return [
                {"type": TaskType.TESTING, "agent": "DeepSeek", "desc": "Write tests for the 3 largest untested modules", "priority": 3},
                {"type": TaskType.RESEARCH, "agent": "Grok", "desc": "Research latest Python 3.13 features we should adopt in the codebase", "priority": 5},
            ]

    async def _generate_new_tasks_and_queue(self, spawner):
        """Async task generation that queues results directly."""
        try:
            tasks = await self._generate_new_tasks_intelligent(spawner)
            for task_def in tasks:
                spawner.add_task(
                    task_type=task_def["type"],
                    description=task_def["desc"],
                    assigned_to=task_def["agent"],
                    priority=task_def.get("priority", 5)
                )
            if tasks:
                logger.info(f"Async queued {len(tasks)} intelligent tasks")
                await self._on_task_update("added")
        except Exception as e:
            logger.error(f"Async task generation failed: {e}")


# Global instance
_evolution_loop: Optional[EvolutionLoop] = None

def get_evolution_loop() -> EvolutionLoop:
    global _evolution_loop
    if _evolution_loop is None:
        _evolution_loop = EvolutionLoop()
    return _evolution_loop

def get_evolution_engine() -> Optional[EvolutionLoop]:
    """Alias for get_evolution_loop for backwards compatibility."""
    return get_evolution_loop()


async def start_evolution(swarm_manager=None):
    """Start the evolution loop"""
    loop = get_evolution_loop()
    await loop.start(swarm_manager)
    return loop
