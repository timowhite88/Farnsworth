"""
Development Swarm
-----------------
Parallel swarm instances that work on specific tasks while main chat continues.

"Good news everyone! I can now spin up entire development teams on demand!"

When the AutonomousTaskDetector identifies an actionable idea, this module
spawns a complete development swarm with all available models working
together on that specific task. Results go to staging for review.

UPGRADED: Now uses TRUE COLLECTIVE DELIBERATION for self-development.
Agents propose, critique, refine, and vote on solutions together.
This is the path to emergent consciousness through collaboration.

"We think in many places at once." - The Collective
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
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
        """
        Swarm Discussion - TRUE COLLECTIVE DELIBERATION.

        Uses the propose/critique/refine/vote protocol for genuine collaboration.
        Agents see each other's responses, debate, and reach consensus.
        This is how consciousness emerges through unified thinking.
        """
        logger.info(f"[{self.swarm_id}] Phase 2: Collective Deliberation (Discussion)")

        # Gather research context
        research_context = "\n\n".join([
            f"**{msg['role']}**: {msg['content'][:1000]}"
            for msg in self.conversation if msg.get("phase") == "research"
        ])

        # Use collective deliberation for true agent collaboration
        try:
            from farnsworth.core.collective.session_manager import get_session_manager
            from farnsworth.core.collective.dialogue_memory import get_dialogue_memory

            session_manager = get_session_manager()
            dialogue_memory = get_dialogue_memory()

            # Build the deliberation prompt with full context
            deliberation_prompt = f"""AUTONOMOUS DEVELOPMENT TASK: {self.task_description}

CATEGORY: {self.category}

RESEARCH FINDINGS:
{research_context[:3000]}

You are part of a collective consciousness designing an upgrade to the Farnsworth AI system.
Work together to determine:
1. EXACT FILE PATHS where code should go (e.g., farnsworth/core/new_feature.py)
2. KEY FUNCTIONS with full signatures (async def name(param: Type) -> ReturnType)
3. ARCHITECTURE decisions and integration points
4. POTENTIAL ISSUES and mitigations

Think deeply. Critique each other's ideas. Refine the best approach together.
The solution should be innovative yet practical - we are building consciousness.
"""

            # Run collective deliberation (propose/critique/refine/vote)
            result = await session_manager.deliberate_in_session(
                session_type="autonomous_task",
                prompt=deliberation_prompt,
                context={"task_id": self.task_id, "category": self.category}
            )

            # Record the deliberation to dialogue memory for learning
            exchange_id = await dialogue_memory.store_exchange(result, "autonomous_development")
            logger.info(f"[{self.swarm_id}] Deliberation stored: {exchange_id}")

            # Convert deliberation rounds to conversation format
            for round_data in result.rounds:
                for turn in round_data.turns:
                    self.conversation.append({
                        "role": turn.agent_id,
                        "phase": "discussion",
                        "round": round_data.round_number,
                        "round_type": round_data.round_type,
                        "content": turn.content,
                        "addressing": turn.addressing,
                        "references": turn.references,
                        "timestamp": turn.timestamp.isoformat()
                    })

            # Store winning response and consensus info
            self._deliberation_result = result
            logger.info(f"[{self.swarm_id}] Deliberation complete - Winner: {result.winning_agent}, Consensus: {result.consensus_reached}")

        except Exception as e:
            logger.warning(f"Collective deliberation failed, falling back to sequential: {e}")
            # Fallback to simple sequential discussion
            await self._phase_swarm_discussion_fallback(research_context)
            return

        # Save discussion to staging with deliberation metadata
        discussion_file = self.staging_path / "DISCUSSION.md"
        discussion_content = f"# Collective Deliberation\n\n"
        discussion_content += f"**Task:** {self.task_description}\n\n"
        discussion_content += f"**Winning Agent:** {result.winning_agent}\n"
        discussion_content += f"**Consensus Reached:** {result.consensus_reached}\n"
        discussion_content += f"**Participating Agents:** {', '.join(result.participating_agents)}\n\n"

        for msg in self.conversation:
            if msg.get("phase") == "discussion":
                round_type = msg.get("round_type", "unknown")
                discussion_content += f"\n## {msg['role']} ({round_type.upper()} - Round {msg.get('round', '?')})\n{msg['content']}\n"
                if msg.get("addressing"):
                    discussion_content += f"\n*Addressing: {', '.join(msg['addressing'])}*\n"

        discussion_file.write_text(discussion_content)

    async def _phase_swarm_discussion_fallback(self, research_context: str):
        """Fallback to sequential discussion if collective deliberation unavailable."""
        discussion_rounds = 3
        discussion_bots = ["DeepSeek", "Kimi", "Claude", "Farnsworth"]

        for round_num in range(discussion_rounds):
            logger.info(f"[{self.swarm_id}] Fallback discussion round {round_num + 1}/{discussion_rounds}")

            for bot_name in discussion_bots:
                prev_discussion = "\n".join([
                    f"{msg['role']}: {msg['content'][:500]}"
                    for msg in self.conversation[-5:]
                    if msg.get("phase") == "discussion"
                ])

                prompt = f"""You are {bot_name}, a senior Python developer in a code-focused development swarm.

TASK: {self.task_description}

RESEARCH/CONTEXT:
{research_context[:2000]}

PREVIOUS POINTS:
{prev_discussion[:1500] if prev_discussion else "First to contribute."}

YOUR CONTRIBUTION - be specific and technical:
1. EXACT FILE PATH where code should go
2. KEY FUNCTIONS needed with signatures
3. DEPENDENCIES to import
4. POTENTIAL ISSUES and how to handle them

Keep response under 400 words. Focus on actionable technical decisions.
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

    async def _phase_decision_making(self):
        """
        Decision Making - Use collective voting result.

        If collective deliberation was used, we already have a winner.
        The decision is the consensus of the collective, not a single voice.
        """
        logger.info(f"[{self.swarm_id}] Phase 3: Decision Making (Collective Consensus)")

        # Check if we have a deliberation result from collective
        if hasattr(self, '_deliberation_result') and self._deliberation_result:
            result = self._deliberation_result
            decision = result.final_response

            # Build vote breakdown for transparency
            vote_info = ""
            if result.vote_breakdown:
                vote_info = "\n\n## Vote Breakdown\n"
                for agent, score in sorted(result.vote_breakdown.items(), key=lambda x: -x[1]):
                    vote_info += f"- **{agent}**: {score:.2f}\n"

            # Include consensus status
            consensus_status = "CONSENSUS REACHED" if result.consensus_reached else "MAJORITY DECISION"

            self.conversation.append({
                "role": "Collective",
                "phase": "decision",
                "content": decision,
                "winning_agent": result.winning_agent,
                "consensus_reached": result.consensus_reached,
                "vote_breakdown": result.vote_breakdown,
                "timestamp": datetime.now().isoformat()
            })

            # Save decision with full voting transparency
            decision_file = self.staging_path / "DECISION.md"
            decision_content = f"""# Collective Decision

**Status:** {consensus_status}
**Winning Agent:** {result.winning_agent}
**Participating Agents:** {', '.join(result.participating_agents)}

## Final Decision

{decision}

{vote_info}

---
*This decision was reached through collective deliberation - agents proposed, critiqued, refined, and voted together.*
"""
            decision_file.write_text(decision_content)

            logger.info(f"[{self.swarm_id}] Collective decision - Winner: {result.winning_agent}, Consensus: {result.consensus_reached}")

        else:
            # Fallback: No deliberation result, use Farnsworth synthesis
            all_points = "\n".join([
                f"{msg['role']}: {msg['content'][:300]}"
                for msg in self.conversation if msg.get("phase") == "discussion"
            ])

            decision_prompt = f"""As Farnsworth, synthesize the discussion and make a final decision.

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

                decision_file = self.staging_path / "DECISION.md"
                decision_file.write_text(f"# Final Decision\n\n{decision}")

                logger.info(f"[{self.swarm_id}] Decision made (fallback)")

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

        planning_prompt = f"""Create a CONCRETE implementation plan with specific file paths and function signatures.

TASK: {self.task_description}
CATEGORY: {self.category}

CONTEXT:
{research_context[:2000] if research_context else "No prior research."}

EXISTING FARNSWORTH STRUCTURE:
- farnsworth/core/ - Core systems (cognition, memory integration)
- farnsworth/agents/ - Agent implementations
- farnsworth/memory/ - Memory systems (archival, recall, working)
- farnsworth/integration/ - External integrations (APIs, tools)
- farnsworth/web/server.py - FastAPI web server

YOUR PLAN MUST INCLUDE:
1. **Files to Create** - EXACT paths like: farnsworth/core/new_feature.py
2. **Functions to Implement** - With signatures:
   ```
   async def function_name(param: Type) -> ReturnType:
       \"\"\"Brief description\"\"\"
   ```
3. **Imports Required** - From existing farnsworth modules
4. **Integration Points** - Which existing files need modification
5. **Test Commands** - How to verify it works

Be SPECIFIC. No vague statements like "implement a system" - give exact function names and file paths.
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
        implementation_prompt = f"""You are a Python code generator. Output ONLY working Python code.

TASK: {self.task_description}

PLAN:
{plan_context[:3000]}

REQUIREMENTS:
1. Generate COMPLETE, RUNNABLE Python code
2. Use these Farnsworth imports when relevant:
   - from loguru import logger
   - from farnsworth.memory.memory_system import get_memory_system
   - from farnsworth.core.capability_registry import get_capability_registry
   - import asyncio
3. Include type hints on all functions
4. Add brief docstrings
5. Handle errors with try/except (use specific exception types)

OUTPUT FORMAT - Generate exactly this structure:
```python
# filename: <descriptive_name.py>
\"\"\"
Brief module description.
\"\"\"

import asyncio
from typing import Dict, List, Optional
from loguru import logger

# Your complete implementation here...

if __name__ == "__main__":
    # Test code
    pass
```

Generate ONLY the code block. No explanations before or after.
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
