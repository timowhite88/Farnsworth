"""
Farnsworth Hackathon Dashboard Routes
======================================

Provides the /hackathon dashboard page and API endpoints for live
hackathon status, deliberation feeds, and manual task triggering.

Colosseum Agent Hackathon — Agent 657, Project 326
"""

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

router = APIRouter()


def _get_templates():
    """Lazy-load Jinja2 templates."""
    try:
        from farnsworth.web.server import templates
        return templates
    except Exception:
        from fastapi.templating import Jinja2Templates
        from pathlib import Path
        return Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))


@router.get("/api/hackathon/status")
async def hackathon_status():
    """Get live hackathon status — aggregated from all available subsystems."""
    try:
        result = {
            "agent_id": "657",
            "project_id": "326",
            "team_id": "333",
            "timestamp": datetime.now().isoformat(),
        }

        # Active swarms
        try:
            from farnsworth.core.development_swarm import DevelopmentSwarm
            result["all_active_swarms"] = len(DevelopmentSwarm._active_swarms)
            if hasattr(DevelopmentSwarm, "_hackathon_state"):
                state = DevelopmentSwarm._hackathon_state
                result["active_swarms"] = state.get("active_tasks", [])
                result["completed_builds"] = state.get("completed", [])[-20:]
                result["recent_deliberations"] = state.get("deliberations", [])[-10:]
                result["colosseum_posts"] = state.get("colosseum_posts", [])[-10:]
            else:
                result["active_swarms"] = []
                result["completed_builds"] = []
                result["recent_deliberations"] = []
                result["colosseum_posts"] = []
        except Exception:
            result["all_active_swarms"] = 0
            result["active_swarms"] = []
            result["completed_builds"] = []
            result["recent_deliberations"] = []
            result["colosseum_posts"] = []

        # Tools
        try:
            from farnsworth.core.collective.tool_awareness import get_tool_awareness
            result["tools_available"] = len(get_tool_awareness().AVAILABLE_TOOLS)
        except Exception:
            result["tools_available"] = 0

        # Skills
        try:
            from farnsworth.core.skill_registry import get_skill_registry
            result["skills_registered"] = len(get_skill_registry()._skills)
        except Exception:
            result["skills_registered"] = 0

        # Memory
        try:
            from farnsworth.memory.memory_system import get_memory_system
            mem = get_memory_system()
            if hasattr(mem, "get_stats"):
                stats = mem.get_stats()
                result["memory_entries"] = stats.get("total_entries", 0) if isinstance(stats, dict) else 0
            else:
                result["memory_entries"] = 0
        except Exception:
            result["memory_entries"] = 0

        # Evolution
        try:
            from farnsworth.core.evolution_loop import get_evolution_loop
            result["evolution_cycle"] = get_evolution_loop().evolution_cycle
        except Exception:
            result["evolution_cycle"] = 0

        # Gateway
        try:
            from farnsworth.core.external_gateway import get_external_gateway
            result["gateway"] = get_external_gateway().get_stats()
        except Exception:
            result["gateway"] = {}

        # Orchestrator
        try:
            from farnsworth.core.token_orchestrator import get_token_orchestrator
            result["orchestrator"] = get_token_orchestrator().get_dashboard()
        except Exception:
            result["orchestrator"] = {}

        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Hackathon status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/hackathon/deliberations")
async def hackathon_deliberations():
    """Get recent hackathon deliberation transcripts."""
    try:
        from farnsworth.core.development_swarm import DevelopmentSwarm
        if hasattr(DevelopmentSwarm, "_hackathon_state"):
            deliberations = DevelopmentSwarm._hackathon_state.get("deliberations", [])
        else:
            deliberations = []
        return JSONResponse({
            "deliberations": deliberations[-20:],
            "count": len(deliberations),
        })
    except Exception as e:
        logger.error(f"Deliberations fetch error: {e}")
        return JSONResponse({"deliberations": [], "count": 0})


@router.post("/api/hackathon/trigger")
async def hackathon_trigger(request: Request):
    """Manually trigger a hackathon development task."""
    try:
        body = await request.json()
        description = body.get("description", "").strip()
        if not description:
            return JSONResponse({"error": "description required"}, status_code=400)

        if not description.startswith("[HACKATHON]"):
            description = f"[HACKATHON] {description}"

        from farnsworth.core.development_swarm import DevelopmentSwarm

        swarm = DevelopmentSwarm(
            task_id=f"hackathon_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_description=description,
            category="hackathon",
            primary_agent="ClaudeOpus",
        )
        swarm_id = await swarm.start()

        logger.info(f"Manually triggered hackathon swarm: {swarm_id}")
        return JSONResponse({
            "swarm_id": swarm_id,
            "task": description,
            "status": "started",
        })
    except Exception as e:
        logger.error(f"Hackathon trigger error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
