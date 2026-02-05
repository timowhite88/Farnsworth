"""
Polymarket Prediction Routes

Endpoints:
- GET /api/polymarket/predictions - Get recent predictions
- GET /api/polymarket/stats - Prediction accuracy statistics
- POST /api/polymarket/generate - Manually trigger predictions
"""

import logging
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/polymarket/predictions")
async def get_polymarket_predictions(limit: int = 10):
    """Get recent Polymarket predictions."""
    try:
        from farnsworth.core.polymarket_predictor import get_predictor
        predictor = get_predictor()
        predictions = predictor.get_recent_predictions(limit)

        return {
            "predictions": [p.to_dict() for p in predictions],
            "count": len(predictions),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        return {"error": str(e), "predictions": []}


@router.get("/api/polymarket/stats")
async def get_polymarket_stats():
    """Get prediction accuracy statistics."""
    try:
        from farnsworth.core.polymarket_predictor import get_predictor
        from dataclasses import asdict

        predictor = get_predictor()
        stats = predictor.get_stats()

        return {
            "stats": asdict(stats),
            "updated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"error": str(e), "stats": {}}


@router.post("/api/polymarket/generate")
async def trigger_polymarket_predictions():
    """Manually trigger prediction generation."""
    try:
        from farnsworth.core.polymarket_predictor import get_predictor
        predictor = get_predictor()
        predictions = await predictor.generate_predictions(count=2)

        return {
            "success": True,
            "predictions": [p.to_dict() for p in predictions],
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        return {"success": False, "error": str(e)}
