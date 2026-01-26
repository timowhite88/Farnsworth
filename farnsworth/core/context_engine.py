"""
Farnsworth Context Engine (GSD Implementation)

This module implements the "Get Things Done" (GSD) context engineering framework.
It manages the automatic synchronization of project state files and ensures
that the LLM always has the correct context to avoid "context rot".

Files Managed:
- PROJECT.md: High-level vision and goals
- STATE.md: Current tactical state (active task, decisions)
- ROADMAP.md: Strategic milestones
- DECISIONS.md: Log of architectural decisions
"""

import os
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path
from loguru import logger

from farnsworth.memory.project_tracking import ProjectTracker, Project, Task, TaskStatus

class ContextEngine:
    """
    Engine for managing project context and preventing context rot.
    """
    
    def __init__(self, project_root: str, tracker: ProjectTracker):
        self.project_root = Path(project_root)
        self.tracker = tracker
        
        # Standard GSD files
        self.project_file = self.project_root / "PROJECT.md"
        self.state_file = self.project_root / "STATE.md"
        self.roadmap_file = self.project_root / "ROADMAP.md"
        self.decisions_file = self.project_root / "DECISIONS.md"

    async def sync_state(self, project_id: str):
        """
        Synchronize the internal ProjectTracker state to the GSD markdown files.
        This ensures the "external memory" (files) matches the "internal memory" (DB).
        """
        project = self.tracker.projects.get(project_id)
        if not project:
            logger.warning(f"Cannot sync state: Project {project_id} not found")
            return

        # Parallelize file updates
        await asyncio.gather(
            self._update_project_md(project),
            self._update_state_md(project),
            self._update_roadmap_md(project)
        )
        
        logger.info(f"Synced project state to GSD files for {project.name}")

    async def get_context_prompt(self) -> str:
        """
        Generate the XML context prompt for the LLM.
        This reads the current state files and formats them for injection.
        """
        context_parts = []
        
        # 1. Project Vision
        if self.project_file.exists():
            content = self.project_file.read_text(encoding='utf-8')
            context_parts.append(f"<project_vision>\n{content}\n</project_vision>")
            
        # 2. Current State (Critical for avoiding loops)
        if self.state_file.exists():
            content = self.state_file.read_text(encoding='utf-8')
            context_parts.append(f"<current_state>\n{content}\n</current_state>")
            
        # 3. Roadmap Status
        if self.roadmap_file.exists():
            # We might want to summarize this if it's too long
            content = self.roadmap_file.read_text(encoding='utf-8')
            context_parts.append(f"<roadmap_summary>\n{content}\n</roadmap_summary>")
            
        return "\n\n".join(context_parts)

    async def _update_project_md(self, project: Project):
        """Update PROJECT.md with current project details."""
        content = f"""# {project.name}

## Vision
{project.description}

## Status
**Current Status**: {project.status.value.upper()}
**Last Updated**: {project.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Tags
{', '.join([f'`{t}`' for t in project.tags])}
"""
        await self._write_file(self.project_file, content)

    async def _update_state_md(self, project: Project):
        """
        Update STATE.md with the current tactical state.
        This is the most dynamic file in the GSD framework.
        """
        # Get active tasks
        tasks = await self.tracker.list_tasks(project.id)
        active_tasks = [t for t in tasks if t.status == TaskStatus.IN_PROGRESS]
        pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        blocked_tasks = [t for t in tasks if t.status == TaskStatus.BLOCKED]
        
        content = f"""# Current State: {project.name}

## 🎯 Active Focus
"""
        if active_tasks:
            for t in active_tasks:
                content += f"- **IN PROGRESS**: {t.title} (Priority: {t.priority})\n  - {t.description}\n"
        else:
            content += "- *No active tasks. Waiting for planning.*\n"
            
        content += "\n## 📋 Next Up\n"
        for t in pending_tasks[:5]:  # Top 5 pending
            content += f"- {t.title}\n"
            
        content += "\n## 🛑 Blockers\n"
        if blocked_tasks:
            for t in blocked_tasks:
                content += f"- {t.title} (Blocked by: {t.depends_on})\n"
        else:
            content += "- *None*\n"
            
        await self._write_file(self.state_file, content)

    async def _update_roadmap_md(self, project: Project):
        """Update ROADMAP.md from milestones."""
        milestones = await self.tracker._load_milestones_async() # Assuming we add this helper or use public API
        project_milestones = [m for m in milestones.values() if m.project_id == project.id]
        
        content = f"""# Roadmap: {project.name}

## Milestones
"""
        for m in sorted(project_milestones, key=lambda x: x.target_date or datetime.max):
            symbol = "✅" if m.is_achieved else "📅"
            content += f"\n### {symbol} {m.title} ({int(m.progress_percentage)}%)\n"
            content += f"{m.description}\n"
            if m.target_date:
                content += f"**Target Date**: {m.target_date.strftime('%Y-%m-%d')}\n"
            
            content += "\n**Criteria**:\n"
            for c in m.criteria:
                content += f"- [ ] {c}\n"
                
        await self._write_file(self.roadmap_file, content)

    async def _write_file(self, path: Path, content: str):
        """Helper to write file asynchronously (simulated via thread pool if needed)."""
        # For simplicity in this implementation, we use sync write
        # In a full async app, we'd use aiofiles
        path.write_text(content, encoding='utf-8')
