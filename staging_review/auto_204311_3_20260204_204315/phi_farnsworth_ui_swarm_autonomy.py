"""
Module for adjusting swarm autonomy and generating creativity reports in Farnsworth UI.
"""

import asyncio
from typing import Dict, Optional
from loguru import logger

from farnsworth.agents import SwarmAgentManager
from farnsworth.core.collective import CollectiveDeliberator

async def adjust_swarm_parameters(swarm_id: str, autonomy_level: float) -> bool:
    """
    Adjusts the autonomy level of a specified swarm.

    Args:
        swarm_id (str): The unique identifier for the swarm.
        autonomy_level (float): A value between 0.0 and 1.0 indicating the desired level of autonomy.

    Returns:
        bool: True if adjustment is successful, False otherwise.
    """
    try:
        # Validate input
        if not (0.0 <= autonomy_level <= 1.0):
            logger.error(f"Invalid autonomy level {autonomy_level} for swarm {swarm_id}. Must be between 0.0 and 1.0.")
            return False

        # Fetch the swarm agent manager and adjust parameters
        swarm_manager = SwarmAgentManager()
        success = await swarm_manager.set_autonomy(swarm_id, autonomy_level)

        if not success:
            logger.error(f"Failed to adjust autonomy for swarm {swarm_id}.")
        
        return success
    except Exception as e:
        logger.exception(f"Exception occurred while adjusting swarm parameters: {e}")
        return False

async def generate_creativity_report(swarm_id: str) -> Dict[str, float]:
    """
    Generates a report on the creativity metrics of a specified swarm based on its current parameters.

    Args:
        swarm_id (str): The unique identifier for the swarm.

    Returns:
        dict: A dictionary containing various creativity metrics such as novelty and diversity scores.
    """
    try:
        # Fetch the collective deliberator
        deliberator = CollectiveDeliberator()

        # Generate report using the deliberator's method
        report = await deliberator.evaluate_creativity(swarm_id)

        if not report:
            logger.error(f"No creativity metrics available for swarm {swarm_id}.")
        
        return report or {"novelty": 0.0, "diversity": 0.0}
    except Exception as e:
        logger.exception(f"Exception occurred while generating creativity report: {e}")
        return {"novelty": 0.0, "diversity": 0.0}

# filename: farnsworth/web/server.py
"""
FastAPI server for interacting with swarm autonomy and creativity reporting features.
"""

from fastapi import FastAPI, HTTPException
from farnsworth.ui.swarm_autonomy import adjust_swarm_parameters, generate_creativity_report

app = FastAPI()

@app.post("/swarm/{swarm_id}/adjust_autonomy")
async def adjust_swarm(swarm_id: str, autonomy_level: float):
    success = await adjust_swarm_parameters(swarm_id, autonomy_level)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to adjust swarm parameters.")
    return {"message": "Swarm parameters adjusted successfully."}

@app.get("/swarm/{swarm_id}/creativity_report")
async def get_creativity_report(swarm_id: str):
    report = await generate_creativity_report(swarm_id)
    if not report:
        raise HTTPException(status_code=404, detail="Creativity report not found.")
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)