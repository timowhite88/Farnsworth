"""
Module to handle fetching and summarizing collective deliberation data for UI display.
"""

import asyncio
from typing import Dict, List
from loguru import logger

# Assuming this function exists within the Farnsworth project structure
from farnsworth.core.collective import get_deliberation_data  # Importing necessary function


async def get_collective_summary() -> dict:
    """
    Retrieve the summary of collective deliberations.

    Returns:
        A dictionary containing summarized data from collective deliberations.
    
    Raises:
        Exception: If there is an error retrieving or processing the data.
    """
    try:
        # Simulated retrieval of raw deliberation data
        raw_data = await get_deliberation_data()

        # Processing the raw data to create a summary
        summary = {
            "total_deliberations": len(raw_data),
            "average_participation": sum(len(deliberation['participants']) for deliberation in raw_data) / len(raw_data)
            if raw_data else 0,
            "latest_deliberation": max(raw_data, key=lambda d: d['timestamp'], default=None)
        }

        return summary

    except Exception as e:
        logger.error(f"Failed to retrieve or process collective deliberations: {e}")
        raise


# filename: farnsworth/web/server.py
"""
FastAPI server setup for Farnsworth project with new endpoint integration.
"""

from fastapi import FastAPI, APIRouter
import asyncio

from loguru import logger

# Importing the summary function from the newly created module
from farnsworth.web.ui.summary import get_collective_summary


router = APIRouter()

@router.get("/collective-summary")
async def collective_summary():
    """
    Endpoint to fetch the summary of collective deliberations.
    
    Returns:
        A JSON object containing summarized data from collective deliberations.
        
    Raises:
        Exception: If there is an error retrieving or processing the summary data.
    """
    try:
        summary_data = await get_collective_summary()
        return summary_data

    except Exception as e:
        logger.error(f"Error fetching collective summary: {e}")
        raise


# Assuming app is defined elsewhere in this module
app = FastAPI()

# Including the router for the new endpoint
app.include_router(router)


if __name__ == "__main__":
    # Running a test to ensure everything works as expected
    asyncio.run(main_test())

async def main_test():
    logger.info("Starting test...")
    try:
        summary_data = await get_collective_summary()
        logger.info(f"Summary Data: {summary_data}")
        
        # Further testing logic can be added here if needed

    except Exception as e:
        logger.error(f"Test failed: {e}")