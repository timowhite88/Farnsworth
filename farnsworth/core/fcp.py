"""
Farnsworth Cognitive Projection (FCP) Engine.

"Good news, everyone! I've invented a device that projects my internal state directly into markdown!"

FCP replaces static context files with a live "Holographic Projection" of the Agent Swarm's cognitive state.
It ensures that the context injected into the LLM is always fresh, resonant, and relevant.

Artifacts (The Hologram):
- VISION.md: The immutable core intent and axioms.
- FOCUS.md: The dynamic working memory and active synapses.
- HORIZON.md: Probabilistic milestones and computed trajectories.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from loguru import logger

from farnsworth.core.nexus import nexus, Signal, SignalType
from farnsworth.memory.project_tracking import ProjectTracker, Project, TaskStatus

class FCPEngine:
    """
    The engine responsible for the Farnsworth Cognitive Projection protocol.
    It listens to the Nexus and projects the internal state into human/AI-readable artifacts.
    """
    
    def __init__(self, project_root: str, tracker: ProjectTracker):
        self.project_root = Path(project_root)
        self.tracker = tracker
        
        # The Holographic Surface (Files)
        self.vision_file = self.project_root / "VISION.md"
        self.focus_file = self.project_root / "FOCUS.md"
        self.horizon_file = self.project_root / "HORIZON.md"
        
        # Connect to the Nexus
        nexus.subscribe(SignalType.TASK_UPDATED, self._on_state_change)
        nexus.subscribe(SignalType.TASK_COMPLETED, self._on_state_change)
        nexus.subscribe(SignalType.DECISION_REACHED, self._on_state_change)
        
        logger.info("FCP Engine initialized. Holographic emitters ready.")

    async def _on_state_change(self, signal: Signal):
        """
        When the Nexus detects a cognitive shift, update the hologram.
        """
        project_id = signal.payload.get("project_id")
        if project_id:
            await self.project_hologram(project_id)

    async def project_hologram(self, project_id: str):
        """
        Project the internal high-dimensional state into the markdown artifacts.
        """
        project = self.tracker.projects.get(project_id)
        if not project:
            return

        # Parallel emission
        await asyncio.gather(
            self._project_vision(project),
            self._project_focus(project),
            self._project_horizon(project)
        )
        logger.debug(f"FCP: Hologram refreshed for {project.name}")

    async def get_resonant_context(self, query: str) -> str:
        """
        Constructs the "Thought Vector" (Context String) for the LLM.
        This performs 'Context Resonance' to find relevant info.
        """
        context_blocks = []
        
        # 1. Vision (The Axioms)
        if self.vision_file.exists():
            content = self.vision_file.read_text(encoding="utf-8")
            context_blocks.append(f"<fcp_vision>\n{content}\n</fcp_vision>")
            
        # 2. Focus (The Working Memory)
        if self.focus_file.exists():
            content = self.focus_file.read_text(encoding="utf-8")
            context_blocks.append(f"<fcp_focus>\n{content}\n</fcp_focus>")
            
        return "\n\n".join(context_blocks)

    async def _project_vision(self, project: Project):
        """Updates VISION.md - The crystallized intent."""
        content = f"""# 👁️ Vision: {project.name}

> "The architectural axioms and crystallized intent."

## Core Purpose
{project.description}

## DNA (Tags)
{', '.join([f'`{t}`' for t in project.tags])}

## Domain Constraints
*Generated from architectural decisions*
"""
        # In a real system, we'd query the 'Decision' memory layer here
        await self._write_artifact(self.vision_file, content)

    async def _project_focus(self, project: Project):
        """Updates FOCUS.md - The dynamic working memory."""
        tasks = await self.tracker.list_tasks(project.id)
        
        active = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
        pending = [t for t in tasks if t.status == TaskStatus.PENDING]
        blocked = [t for t in tasks if t.status == TaskStatus.BLOCKED]
        
        # Calculate 'Neural Activity' (Cognitive Load)
        load = len(active) / 5.0 * 100
        
        content = f"""# 🎯 Focus: {project.name}

**Neural Activity**: {load:.0f}% | **Swarm Status**: ACTIVE

## ⚡ Active Synapses (In Progress)
"""
        if active:
            for t in active:
                content += f"- **{t.title}** (Priority {t.priority})\n  - {t.description}\n"
        else:
            content += "- *Network Idle - Awaiting Signal*\n"
            
        content += "\n## ⏳ Immediate Queue\n"
        for t in pending[:5]:
            content += f"- {t.title}\n"
            
        content += "\n## 🚧 Neural Blockers\n"
        if blocked:
            for t in blocked:
                content += f"- 🛑 {t.title} (Blocked by dependencies)\n"
        else:
            content += "- *Flow Optimal*\n"
            
        await self._write_artifact(self.focus_file, content)

    async def _project_horizon(self, project: Project):
        """Updates HORIZON.md - The probabilistic future."""
        content = f"""# 🌅 Horizon: {project.name}

> "Probabilistic milestones and computed trajectories."

## Trajectory
*Calculated based on current velocity and Task completion rates.*

## Milestones
"""
        # Would inject actual milestones here
        await self._write_artifact(self.horizon_file, content)

    async def _write_artifact(self, path: Path, content: str):
        path.write_text(content, encoding="utf-8")
