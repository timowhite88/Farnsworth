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


@router.get("/hackathon", response_class=HTMLResponse)
async def hackathon_page(request: Request):
    """Serve the hackathon dashboard page."""
    try:
        templates = _get_templates()
        return templates.TemplateResponse("hackathon.html", {"request": request})
    except Exception as e:
        logger.error(f"Hackathon page error: {e}")
        return HTMLResponse(f"<h1>Hackathon Dashboard</h1><p>Template error: {e}</p>", status_code=500)


@router.get("/api/hackathon/status")
async def hackathon_status():
    """Get live hackathon status — active swarms, completed builds, Colosseum posts."""
    try:
        from farnsworth.core.development_swarm import DevelopmentSwarm

        state = DevelopmentSwarm._hackathon_state

        # Count tools and skills
        tools_available = 0
        try:
            from farnsworth.core.collective.tool_awareness import get_tool_awareness
            tools_available = len(get_tool_awareness().AVAILABLE_TOOLS)
        except Exception:
            pass

        skills_registered = 0
        try:
            from farnsworth.core.skill_registry import get_skill_registry
            registry = get_skill_registry()
            skills_registered = len(registry._skills)
        except Exception:
            pass

        memory_entries = 0
        try:
            from farnsworth.memory.memory_system import get_memory_system
            mem = get_memory_system()
            if hasattr(mem, 'get_stats'):
                stats = mem.get_stats()
                memory_entries = stats.get("total_entries", 0) if isinstance(stats, dict) else 0
        except Exception:
            pass

        evolution_cycle = 0
        try:
            from farnsworth.core.evolution_loop import get_evolution_loop
            evolution_cycle = get_evolution_loop().evolution_cycle
        except Exception:
            pass

        # Gateway stats
        gateway_stats = {}
        try:
            from farnsworth.core.external_gateway import get_external_gateway
            gw = get_external_gateway()
            gateway_stats = gw.get_stats()
        except Exception:
            pass

        # Orchestrator stats
        orchestrator_stats = {}
        try:
            from farnsworth.core.token_orchestrator import get_token_orchestrator
            orch = get_token_orchestrator()
            orchestrator_stats = orch.get_dashboard()
        except Exception:
            pass

        return JSONResponse({
            "agent_id": "657",
            "project_id": "326",
            "active_swarms": state.get("active_tasks", []),
            "completed_builds": state.get("completed", [])[-20:],
            "recent_deliberations": state.get("deliberations", [])[-10:],
            "colosseum_posts": state.get("colosseum_posts", [])[-10:],
            "evolution_cycle": evolution_cycle,
            "tools_available": tools_available,
            "skills_registered": skills_registered,
            "memory_entries": memory_entries,
            "all_active_swarms": len(DevelopmentSwarm._active_swarms),
            "gateway": gateway_stats,
            "orchestrator": orchestrator_stats,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f"Hackathon status error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/api/hackathon/deliberations")
async def hackathon_deliberations():
    """Get recent hackathon deliberation transcripts."""
    try:
        from farnsworth.core.development_swarm import DevelopmentSwarm
        deliberations = DevelopmentSwarm._hackathon_state.get("deliberations", [])
        return JSONResponse({
            "deliberations": deliberations[-20:],
            "count": len(deliberations),
        })
    except Exception as e:
        logger.error(f"Deliberations fetch error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/api/hackathon/trigger")
async def hackathon_trigger(request: Request):
    """Manually trigger a hackathon development task."""
    try:
        body = await request.json()
        description = body.get("description", "").strip()
        if not description:
            return JSONResponse({"error": "description required"}, status_code=400)

        # Tag as hackathon
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
