"""
Module for adjusting swarm autonomy levels and generating creativity reports within Farnsworth UI.
"""

import asyncio
from typing import Dict, Optional
from loguru import logger

# Required imports from the Farnsworth structure
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
            logger.error(f"Autonomy level {autonomy_level} out of bounds for swarm {swarm_id}.")
            return False

        # Access the SwarmAgentManager and adjust parameters
        manager = SwarmAgentManager()
        success = await manager.set_swarm_autonomy(swarm_id, autonomy_level)
        
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
        # Access CollectiveDeliberator to gather data
        deliberator = CollectiveDeliberator()
        report = await deliberator.get_creativity_metrics(swarm_id)
        
        if not report:
            logger.warning(f"No creativity metrics available for swarm {swarm_id}.")
            return {"novelty": 0.0, "diversity": 0.0}
        
        return report
    
    except Exception as e:
        logger.exception(f"Exception occurred while generating creativity report: {e}")
        return {"novelty": 0.0, "diversity": 0.0}


# filename: farnsworth/web/server.py
"""
FastAPI server setup for handling swarm autonomy adjustments and creativity reports.
"""

from fastapi import FastAPI, HTTPException
import asyncio

app = FastAPI()

@app.post("/swarm/{swarm_id}/adjust_autonomy")
async def adjust_swarm(swarm_id: str, autonomy_level: float):
    """
    Endpoint to adjust the autonomy level of a specified swarm.

    Args:
        swarm_id (str): The unique identifier for the swarm.
        autonomy_level (float): Desired autonomy level between 0.0 and 1.0.

    Returns:
        dict: Success message if adjustment is successful, raises HTTPException otherwise.
    """
    success = await adjust_swarm_parameters(swarm_id, autonomy_level)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to adjust swarm parameters.")
    return {"message": "Swarm parameters adjusted successfully."}

@app.get("/swarm/{swarm_id}/creativity_report")
async def get_creativity_report(swarm_id: str):
    """
    Endpoint to generate a creativity report for a specified swarm.

    Args:
        swarm_id (str): The unique identifier for the swarm.

    Returns:
        dict: Creativity metrics such as novelty and diversity scores.
    """
    report = await generate_creativity_report(swarm_id)
    return report


if __name__ == "__main__":
    # Test code can be placed here if needed
    pass