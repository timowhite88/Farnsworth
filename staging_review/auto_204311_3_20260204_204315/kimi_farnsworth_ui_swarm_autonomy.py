"""
Module to adjust autonomy levels in swarms and generate creativity reports for fostering creativity within the Farnsworth structure.
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
            raise ValueError("Autonomy level must be between 0.0 and 1.0")

        swarm_agent_manager = SwarmAgentManager()
        swarm_info = await swarm_agent_manager.get_swarm(swarm_id)
        
        if not swarm_info:
            logger.error(f"Swarm with ID {swarm_id} does not exist.")
            return False

        # Adjust autonomy level
        await swarm_agent_manager.set_autonomy_level(swarm_id, autonomy_level)
        logger.info(f"Autonomy level for swarm {swarm_id} set to {autonomy_level}.")
        
        return True

    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error adjusting swarm parameters: {e}")

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
        collective_deliberator = CollectiveDeliberator()
        creativity_metrics = await collective_deliberator.evaluate_creativity(swarm_id)
        
        if not creativity_metrics:
            logger.error(f"No creativity metrics found for swarm {swarm_id}.")
            return {"error": "No data available"}

        logger.info(f"Creativity report generated for swarm {swarm_id}: {creativity_metrics}")
        
        return creativity_metrics

    except Exception as e:
        logger.error(f"Unexpected error generating creativity report: {e}")

    return {"error": "Failed to generate report"}

# filename: farnsworth/web/server.py

"""
FastAPI server module to handle requests for adjusting swarm autonomy and retrieving creativity reports.
"""

from fastapi import FastAPI, HTTPException
import asyncio
from typing import Dict
from loguru import logger
from .swarm_autonomy import adjust_swarm_parameters, generate_creativity_report

app = FastAPI()

@app.post("/swarm/{swarm_id}/adjust_autonomy")
async def adjust_swarm(swarm_id: str, autonomy_level: float) -> Dict[str, str]:
    success = await adjust_swarm_parameters(swarm_id, autonomy_level)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to adjust swarm parameters.")
    return {"message": "Swarm parameters adjusted successfully."}

@app.get("/swarm/{swarm_id}/creativity_report")
async def get_creativity_report(swarm_id: str) -> Dict[str, Optional[float]]:
    report = await generate_creativity_report(swarm_id)
    if "error" in report:
        raise HTTPException(status_code=404, detail="Creativity report not available.")
    return report

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)