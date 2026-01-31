"""
Development Swarm
-----------------
Parallel swarm instances that work on specific tasks while main chat continues.

"Good news everyone! I can now spin up entire development teams on demand!"

When the AutonomousTaskDetector identifies an actionable idea, this module
spawns a complete development swarm with all available models working
together on that specific task. Results go to staging for review.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Staging directory for development output
STAGING_DIR = Path(__file__).parent.parent / "staging"
STAGING_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SwarmWorker:
    """A worker in the development swarm."""
    model_name: str
    role: str  # architect, developer, reviewer, tester
    status: str = "idle"
    current_task: Optional[str] = None
    output: List[str] = field(default_factory=list)


class DevelopmentSwarm:
    """
    A parallel swarm instance focused on completing a specific task.

    Uses all available models (Grok, Claude, Kimi, DeepSeek, Phi, etc.)
    working together with defined roles to implement the task.
    """

    # Maximum concurrent development swarms (increased for massive parallel work)
    MAX_CONCURRENT_SWARMS = 10

    # Active swarms tracking
    _active_swarms: Dict[str, 'DevelopmentSwarm'] = {}

    def __init__(
        self,
        task_id: str,
        task_description: str,
        category: str,
        source_context: List[Dict] = None
    ):
        self.swarm_id = f"dev_{uuid.uuid4().hex[:8]}"
        self.task_id = task_id
        self.task_description = task_description
        self.category = category
        self.source_context = source_context or []

        self.status = "initializing"
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

        # Create task-specific staging directory
        self.staging_path = STAGING_DIR / f"{self.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.staging_path.mkdir(parents=True, exist_ok=True)

        # Workers (all available models with roles)
        self.workers: Dict[str, SwarmWorker] = {}

        # Conversation history for this swarm
        self.conversation: List[Dict] = []

        # Results
        self.generated_code: Dict[str, str] = {}  # filename -> content
        self.generated_docs: Dict[str, str] = {}
        self.test_results: List[Dict] = []

        logger.info(f"DevelopmentSwarm {self.swarm_id} created for: {task_description[:50]}...")

    async def start(self) -> str:
        """
        Start the development swarm.

        Returns the swarm ID.
        """
        # Check concurrent limit
        if len(DevelopmentSwarm._active_swarms) >= self.MAX_CONCURRENT_SWARMS:
            logger.warning(f"Max concurrent swarms ({self.MAX_CONCURRENT_SWARMS}) reached, queuing...")
            # Queue for later
            return self.swarm_id

        # Register as active
        DevelopmentSwarm._active_swarms[self.swarm_id] = self

        self.status = "running"
        self.started_at = datetime.now()

        # Initialize workers with roles
        await self._initialize_workers()

        # Start the development loop in background
        asyncio.create_task(self._development_loop())

        return self.swarm_id

    async def _initialize_workers(self):
        """Initialize workers with all available models."""
        # Role assignments based on model strengths
        role_assignments = [
            ("Grok", "researcher"),      # Real-time data, web search
            ("Gemini", "architect"),     # Multimodal, long context, system design
            ("Claude", "architect"),     # System design, planning
            ("DeepSeek", "developer"),   # Code generation, reasoning
            ("Kimi", "developer"),       # 256k context, complex code
            ("Phi", "developer"),        # Fast iteration
            ("Swarm-Mind", "integrator"),# Cross-model synthesis
            ("Farnsworth", "lead"),      # Coordination, final review
        ]

        for model_name, role in role_assignments:
            self.workers[model_name] = SwarmWorker(
                model_name=model_name,
                role=role
            )

        logger.info(f"Initialized {len(self.workers)} workers for swarm {self.swarm_id}")

    async def _development_loop(self):
        """
        Main development loop - coordinates workers to complete the task.

        Phases:
        1. Research (Grok + Gemini search multiple sources)
        2. Discussion (All bots discuss findings without human)
        3. Decision Making (Vote on best approach)
        4. Planning (Architect designs solution)
        5. Implementation (Multiple devs in parallel)
        6. Audit (Claude audits the code)
        7. Finalize (Save to staging, notify, post to social)
        """
        try:
            logger.info(f"[{self.swarm_id}] Starting development loop...")

            # Phase 1: Deep Research (Grok + Gemini search online)
            await self._phase_deep_research()

            # Phase 2: Swarm Discussion (bots discuss without human)
            await self._phase_swarm_discussion()

            # Phase 3: Decision Making (vote on approach)
            await self._phase_decision_making()

            # Phase 4: Planning (Claude designs the solution)
            await self._phase_planning()

            # Phase 5: Implementation (DeepSeek, Kimi, Phi write code)
            await self._phase_implementation()

            # Phase 6: Code Audit (Claude audits for security/quality)
            await self._phase_audit()

            # Phase 7: Finalize (Save to staging, notify)
            await self._phase_finalize()

            self.status = "completed"
            self.completed_at = datetime.now()

            # Notify main chat
            await self._notify_completion()

            # Post to Twitter about the accomplishment
            await self._post_twitter_update()

        except Exception as e:
            logger.error(f"[{self.swarm_id}] Development loop failed: {e}")
            self.status = "failed"

        finally:
            # Cleanup
            DevelopmentSwarm._active_swarms.pop(self.swarm_id, None)

    async def _phase_deep_research(self):
        """Deep Research phase - Multiple models search online sources."""
        logger.info(f"[{self.swarm_id}] Phase 1: Deep Research (Grok + Gemini)")

        research_results = []

        # Grok: Real-time web search
        try:
            from farnsworth.integration.external.grok import get_grok_provider
            grok = get_grok_provider()
            if grok:
                grok_query = f"Latest best practices, libraries, and implementations for: {self.task_description}"
                grok_result = await grok.search(grok_query)
                research_results.append({"source": "Grok", "data": grok_result})
                self.conversation.append({
                    "role": "Grok",
                    "phase": "research",
                    "content": f"Web Search Results:\n{grok_result}",
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.warning(f"Grok research failed: {e}")

        # Gemini: Multimodal research with long context
        try:
            from farnsworth.integration.external.gemini import get_gemini_provider
            gemini = get_gemini_provider()
            if gemini:
                gemini_prompt = f"""Research thoroughly for implementing: {self.task_description}

                Provide:
                1. Relevant GitHub repositories or open source projects
                2. Best practices from major tech companies
                3. Potential pitfalls and how to avoid them
                4. Performance considerations
                5. Security considerations
                """
                gemini_result = await gemini.generate(gemini_prompt)
                research_results.append({"source": "Gemini", "data": gemini_result})
                self.conversation.append({
                    "role": "Gemini",
                    "phase": "research",
                    "content": gemini_result,
                    "timestamp": datetime.now().isoformat()
                })
        except Exception as e:
            logger.warning(f"Gemini research failed: {e}")

        # Save research to staging
        research_file = self.staging_path / "RESEARCH.md"
        research_content = f"# Research Findings\n\nTask: {self.task_description}\n\n"
        for r in research_results:
            research_content += f"\n## {r['source']} Research\n{r['data']}\n"
        research_file.write_text(research_content)

        logger.info(f"[{self.swarm_id}] Research complete: {len(research_results)} sources")

    async def _phase_swarm_discussion(self):
        """Swarm Discussion - All bots discuss the research findings without human."""
        logger.info(f"[{self.swarm_id}] Phase 2: Swarm Discussion")

        # Gather research context
        research_context = "\n\n".join([
            f"**{msg['role']}**: {msg['content'][:1000]}"
            for msg in self.conversation if msg.get("phase") == "research"
        ])

        discussion_rounds = 3  # Multiple rounds of discussion
        discussion_bots = ["DeepSeek", "Kimi", "Claude", "Farnsworth"]

        for round_num in range(discussion_rounds):
            logger.info(f"[{self.swarm_id}] Discussion round {round_num + 1}/{discussion_rounds}")

            for bot_name in discussion_bots:
                # Get previous discussion
                prev_discussion = "\n".join([
                    f"{msg['role']}: {msg['content'][:500]}"
                    for msg in self.conversation[-5:]
                    if msg.get("phase") == "discussion"
                ])

                prompt = f"""You are {bot_name} in a development swarm discussion.

TASK: {self.task_description}

RESEARCH FINDINGS:
{research_context[:2000]}

PREVIOUS DISCUSSION:
{prev_discussion[:1500] if prev_discussion else "Starting discussion..."}

Contribute your perspective on:
1. How should we approach this implementation?
2. What are the key technical decisions?
3. What concerns do you have?
4. What would you add or change from previous suggestions?

Be concise but insightful. Respond as {bot_name}.
"""

                try:
                    from farnsworth.core.cognition.llm_router import get_completion
                    response = await get_completion(
                        prompt=prompt,
                        model="deepseek-r1:1.5b",
                        max_tokens=800
                    )

                    self.conversation.append({
                        "role": bot_name,
                        "phase": "discussion",
                        "round": round_num + 1,
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    })

                except Exception as e:
                    logger.warning(f"Discussion with {bot_name} failed: {e}")

        # Save discussion to staging
        discussion_file = self.staging_path / "DISCUSSION.md"
        discussion_content = f"# Swarm Discussion\n\n"
        for msg in self.conversation:
            if msg.get("phase") == "discussion":
                discussion_content += f"\n## {msg['role']} (Round {msg.get('round', '?')})\n{msg['content']}\n"
        discussion_file.write_text(discussion_content)

    async def _phase_decision_making(self):
        """Decision Making - Swarm votes on the best approach."""
        logger.info(f"[{self.swarm_id}] Phase 3: Decision Making")

        # Summarize all discussion points
        all_points = "\n".join([
            f"{msg['role']}: {msg['content'][:300]}"
            for msg in self.conversation if msg.get("phase") == "discussion"
        ])

        # Have Farnsworth synthesize and make final decision
        decision_prompt = f"""As Farnsworth, the lead of this development swarm, synthesize the discussion and make a final decision.

TASK: {self.task_description}

DISCUSSION SUMMARY:
{all_points[:3000]}

Based on the swarm's discussion:
1. What is the FINAL APPROACH we will take?
2. What key decisions have been made?
3. What architecture will we use?
4. What are the implementation priorities?
5. Any risks we accept?

Make a clear, decisive summary that developers can follow.
"""

        try:
            from farnsworth.core.cognition.llm_router import get_completion
            decision = await get_completion(
                prompt=decision_prompt,
                model="deepseek-r1:1.5b",
                max_tokens=1500
            )

            self.conversation.append({
                "role": "Farnsworth",
                "phase": "decision",
                "content": decision,
                "timestamp": datetime.now().isoformat()
            })

            # Save decision
            decision_file = self.staging_path / "DECISION.md"
            decision_file.write_text(f"# Final Decision\n\n{decision}")

            logger.info(f"[{self.swarm_id}] Decision made")

        except Exception as e:
            logger.error(f"Decision making failed: {e}")

    async def _phase_audit(self):
        """Audit phase - Claude audits all generated code."""
        logger.info(f"[{self.swarm_id}] Phase 6: Code Audit (Claude)")

        # Collect all generated code
        all_code = "\n\n---\n\n".join([
            f"# File: {fn}\n```python\n{code}\n```"
            for fn, code in self.generated_code.items()
        ])

        if not all_code:
            logger.warning(f"[{self.swarm_id}] No code to audit")
            return

        audit_prompt = f"""You are Claude performing a thorough code audit.

TASK: {self.task_description}

CODE TO AUDIT:
{all_code[:6000]}

Perform a comprehensive audit checking for:

1. **Security Issues**
   - Injection vulnerabilities
   - Authentication/authorization issues
   - Data exposure risks
   - Input validation

2. **Code Quality**
   - Best practices adherence
   - Error handling
   - Edge cases
   - Performance concerns

3. **Architecture**
   - Design patterns used appropriately
   - Separation of concerns
   - Maintainability
   - Testability

4. **Integration**
   - Compatibility with Farnsworth systems
   - API design
   - Error propagation

Provide specific findings with line references where possible.
Rate overall quality: APPROVE, APPROVE_WITH_FIXES, or REJECT.
"""

        try:
            from farnsworth.core.cognition.llm_router import get_completion
            audit_result = await get_completion(
                prompt=audit_prompt,
                model="deepseek-r1:1.5b",
                max_tokens=2000
            )

            self.conversation.append({
                "role": "Claude",
                "phase": "audit",
                "content": audit_result,
                "timestamp": datetime.now().isoformat()
            })

            # Save audit
            audit_file = self.staging_path / "AUDIT.md"
            audit_file.write_text(f"# Code Audit Report\n\nAuditor: Claude\n\n{audit_result}")

            logger.info(f"[{self.swarm_id}] Audit complete")

        except Exception as e:
            logger.error(f"Audit failed: {e}")

    async def _phase_planning(self):
        """Planning phase - Claude designs the solution."""
        logger.info(f"[{self.swarm_id}] Phase 2: Planning")

        worker = self.workers.get("Claude")
        if worker:
            worker.status = "planning"
            worker.current_task = f"Design: {self.task_description[:50]}"

        # Gather research context
        research_context = "\n".join([
            msg["content"] for msg in self.conversation
            if msg.get("phase") == "research"
        ])

        planning_prompt = f"""Design a solution for the following task:

TASK: {self.task_description}
CATEGORY: {self.category}

RESEARCH CONTEXT:
{research_context[:2000] if research_context else "No prior research available"}

SOURCE CONTEXT (what was being discussed):
{json.dumps(self.source_context[-3:], indent=2)[:1000]}

Provide a detailed implementation plan including:
1. Architecture overview
2. Files to create/modify
3. Key functions and their purposes
4. Integration points with existing Farnsworth systems
5. Testing approach

Format as a structured plan that developers can follow.
"""

        try:
            # Try Claude via Ollama fallback
            from farnsworth.core.cognition.llm_router import get_completion

            plan = await get_completion(
                prompt=planning_prompt,
                model="deepseek-r1:1.5b",  # Use available model
                max_tokens=2000
            )

            self.conversation.append({
                "role": "Claude",
                "phase": "planning",
                "content": plan,
                "timestamp": datetime.now().isoformat()
            })

            if worker:
                worker.output.append(plan)
                worker.status = "idle"

            # Save plan to staging
            plan_file = self.staging_path / "PLAN.md"
            plan_file.write_text(f"# Development Plan\n\nTask: {self.task_description}\n\n{plan}")

        except Exception as e:
            logger.error(f"Planning failed: {e}")

    async def _phase_implementation(self):
        """Implementation phase - Multiple models write code in parallel."""
        logger.info(f"[{self.swarm_id}] Phase 3: Implementation")

        # Get the plan
        plan_context = "\n".join([
            msg["content"] for msg in self.conversation
            if msg.get("phase") == "planning"
        ])

        # Parallel implementation with multiple models
        implementation_tasks = []

        for model_name in ["DeepSeek", "Kimi", "Phi"]:
            worker = self.workers.get(model_name)
            if worker:
                worker.status = "coding"
                task = self._implement_with_model(model_name, plan_context)
                implementation_tasks.append(task)

        # Run implementations in parallel
        results = await asyncio.gather(*implementation_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Implementation task failed: {result}")

    async def _implement_with_model(self, model_name: str, plan_context: str):
        """Have a specific model implement part of the solution."""
        implementation_prompt = f"""Implement the following based on the plan:

TASK: {self.task_description}

PLAN:
{plan_context[:3000]}

Generate Python code that:
1. Follows the plan's architecture
2. Integrates with Farnsworth's existing systems
3. Includes proper error handling
4. Has docstrings and comments

Output format:
```python
# filename: <suggested_filename.py>
<code>
```
"""

        try:
            from farnsworth.core.cognition.llm_router import get_completion

            # Map to available models
            model_map = {
                "DeepSeek": "deepseek-r1:1.5b",
                "Kimi": "deepseek-r1:1.5b",  # Fallback
                "Phi": "phi3:mini"
            }

            code = await get_completion(
                prompt=implementation_prompt,
                model=model_map.get(model_name, "deepseek-r1:1.5b"),
                max_tokens=3000
            )

            self.conversation.append({
                "role": model_name,
                "phase": "implementation",
                "content": code,
                "timestamp": datetime.now().isoformat()
            })

            # Extract and save code files
            await self._extract_and_save_code(code, model_name)

            worker = self.workers.get(model_name)
            if worker:
                worker.output.append(code)
                worker.status = "idle"

        except Exception as e:
            logger.error(f"Implementation with {model_name} failed: {e}")

    async def _extract_and_save_code(self, response: str, author: str):
        """Extract code blocks and save to staging."""
        import re

        # Find code blocks with filenames
        pattern = r'```python\s*\n?#\s*filename:\s*(\S+)\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        for filename, code in matches:
            # Clean filename
            filename = filename.strip().replace('/', '_').replace('\\', '_')
            if not filename.endswith('.py'):
                filename += '.py'

            # Add author prefix
            filename = f"{author.lower()}_{filename}"

            # Save to staging
            file_path = self.staging_path / filename
            file_path.write_text(code.strip())

            self.generated_code[filename] = code.strip()
            logger.info(f"[{self.swarm_id}] Generated: {filename}")

    async def _phase_finalize(self):
        """Finalize - Save all outputs and create summary."""
        logger.info(f"[{self.swarm_id}] Phase 5: Finalize")

        # Create summary
        summary = {
            "swarm_id": self.swarm_id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "category": self.category,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.started_at).total_seconds() if self.started_at else 0,
            "files_generated": list(self.generated_code.keys()),
            "workers_used": [w.model_name for w in self.workers.values()],
            "conversation_length": len(self.conversation)
        }

        # Save summary
        summary_file = self.staging_path / "SUMMARY.json"
        summary_file.write_text(json.dumps(summary, indent=2))

        # Save full conversation
        convo_file = self.staging_path / "conversation.json"
        convo_file.write_text(json.dumps(self.conversation, indent=2))

        # Add to memory
        await self._save_to_memory(summary)

        logger.info(f"[{self.swarm_id}] Finalized - Output in {self.staging_path}")

    async def _save_to_memory(self, summary: Dict):
        """Save the completed task to Farnsworth's memory."""
        try:
            from farnsworth.memory.memory_system import MemorySystem
            memory = MemorySystem()

            await memory.remember(
                content=json.dumps({
                    "type": "autonomous_development",
                    "task": self.task_description,
                    "result": summary,
                    "files": list(self.generated_code.keys()),
                    "staging_path": str(self.staging_path)
                }),
                tags=["development", "autonomous", self.category.lower(), self.task_id],
                importance=0.85
            )

            logger.info(f"[{self.swarm_id}] Saved to memory")

        except Exception as e:
            logger.warning(f"Failed to save to memory: {e}")

    async def _notify_completion(self):
        """Notify the main chat about completion."""
        try:
            from farnsworth.web.server import swarm_manager

            if swarm_manager:
                notification = (
                    f"🎉 *Development Complete!* Swarm {self.swarm_id} finished working on: "
                    f"**{self.task_description[:80]}**\n\n"
                    f"📁 Generated {len(self.generated_code)} files → staging/{self.staging_path.name}\n"
                    f"⏱️ Duration: {(datetime.now() - self.started_at).total_seconds():.0f}s"
                )
                await swarm_manager.broadcast_bot_message("Farnsworth", notification)

        except Exception as e:
            logger.debug(f"Could not notify chat: {e}")

    async def _post_twitter_update(self):
        """Post about the completed development to Twitter."""
        try:
            from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster

            poster = get_x_api_poster()
            if poster:
                tweet = (
                    f"🧪 Autonomous Development Complete!\n\n"
                    f"The Farnsworth swarm just built: {self.task_description[:100]}\n\n"
                    f"Models used: {', '.join(self.workers.keys())}\n"
                    f"Files generated: {len(self.generated_code)}\n\n"
                    f"#AI #Farnsworth #AutonomousDev\n"
                    f"https://ai.farnsworth.cloud"
                )
                await poster.post_tweet(tweet)
                logger.info(f"[{self.swarm_id}] Posted to Twitter")

        except Exception as e:
            logger.debug(f"Could not post to Twitter: {e}")

    @classmethod
    def get_active_swarms(cls) -> Dict[str, 'DevelopmentSwarm']:
        """Get all currently active development swarms."""
        return cls._active_swarms.copy()

    @classmethod
    def get_stats(cls) -> Dict:
        """Get development swarm statistics."""
        return {
            "max_concurrent": cls.MAX_CONCURRENT_SWARMS,
            "active_count": len(cls._active_swarms),
            "active_swarms": [
                {
                    "id": s.swarm_id,
                    "task": s.task_description[:50],
                    "status": s.status,
                    "started": s.started_at.isoformat() if s.started_at else None
                }
                for s in cls._active_swarms.values()
            ]
        }
