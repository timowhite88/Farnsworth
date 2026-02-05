"""
Admin Routes - Workers, Staging, Cognition, Heartbeat

Endpoints:
- GET /api/workers/status - Parallel worker system status
- POST /api/workers/init-tasks - Initialize development tasks
- POST /api/workers/start - Start parallel workers
- GET /api/staging/files - List staging files
- GET /api/evolution/status - Evolution loop status (separate from engine)
- GET /api/cognition/status - Cognitive system status
- GET /api/heartbeat - Swarm health vitals
- GET /api/heartbeat/history - Heartbeat history
"""

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_shared():
    """Import shared state from server module lazily."""
    from farnsworth.web import server
    return server


# ============================================
# PARALLEL WORKER API
# ============================================

@router.get("/api/workers/status")
async def get_workers_status():
    """Get parallel worker system status."""
    from farnsworth.core.agent_spawner import get_spawner
    spawner = get_spawner()
    return {
        "spawner": spawner.get_status(),
        "tasks": [
            {
                "id": t.task_id,
                "type": t.task_type.value,
                "agent": t.assigned_to,
                "status": t.status,
                "description": t.description
            }
            for t in spawner.task_queue
        ],
        "discoveries": spawner.shared_state.get("discoveries", [])[-10:],
        "proposals": len(spawner.shared_state.get("proposals", [])),
    }


@router.post("/api/workers/init-tasks")
async def init_tasks():
    """Initialize the 20 development tasks."""
    from farnsworth.core.agent_spawner import initialize_development_tasks
    status = initialize_development_tasks()
    return {"status": "initialized", "info": status}


@router.post("/api/workers/start")
async def start_workers():
    """Start the parallel worker system."""
    try:
        from farnsworth.core.parallel_workers import start_parallel_workers
        manager = await start_parallel_workers()
        return {"status": "started", "info": manager.get_status()}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/api/staging/files")
async def get_staging_files():
    """List files in the staging directory."""
    import os
    staging_dir = Path("/workspace/Farnsworth/farnsworth/staging")
    files = []
    if staging_dir.exists():
        for root, dirs, filenames in os.walk(staging_dir):
            for f in filenames:
                path = Path(root) / f
                files.append({
                    "path": str(path.relative_to(staging_dir)),
                    "size": path.stat().st_size,
                    "modified": path.stat().st_mtime
                })
    return {"files": files[:50]}


# ============================================
# EVOLUTION LOOP STATUS
# ============================================

@router.get("/api/evolution/loop-status")
async def get_evolution_loop_status():
    """Get evolution loop status."""
    from farnsworth.core.agent_spawner import get_spawner
    from farnsworth.core.evolution_loop import get_evolution_loop
    loop = get_evolution_loop()
    spawner = get_spawner()
    return {
        "running": loop.running,
        "evolution_cycle": loop.evolution_cycle,
        "last_discussion": loop.last_discussion.isoformat() if loop.last_discussion else None,
        "spawner": spawner.get_status()
    }


# ============================================
# HUMAN-LIKE COGNITION SYSTEMS
# ============================================

@router.get("/api/cognition/status")
async def get_cognition_status():
    """Get cognitive system status."""
    try:
        from farnsworth.core.temporal_awareness import get_temporal_awareness
        from farnsworth.core.spontaneous_cognition import get_spontaneous_cognition
        from farnsworth.core.capability_registry import get_capability_registry

        temporal = get_temporal_awareness()
        cognition = get_spontaneous_cognition()
        registry = get_capability_registry()

        return {
            "temporal": temporal.get_status(),
            "emotional_state": cognition.get_emotional_state(),
            "recent_thoughts": [t.to_dict() for t in cognition.get_recent_thoughts(5)],
            "capabilities_count": len(registry.capabilities),
            "available_capabilities": len(registry.get_available()),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================
# SWARM HEARTBEAT
# ============================================

@router.get("/api/heartbeat")
async def get_heartbeat_status():
    """Get current swarm health vitals."""
    try:
        from farnsworth.core.swarm_heartbeat import get_current_vitals
        vitals = await get_current_vitals()
        return vitals.to_dict() if vitals else {"error": "No vitals available"}
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/heartbeat/history")
async def get_heartbeat_history():
    """Get recent heartbeat history."""
    try:
        from farnsworth.core.swarm_heartbeat import get_heartbeat
        heartbeat = get_heartbeat()
        return {"history": [v.to_dict() for v in heartbeat.health_history[-20:]]}
    except Exception as e:
        return {"error": str(e)}
