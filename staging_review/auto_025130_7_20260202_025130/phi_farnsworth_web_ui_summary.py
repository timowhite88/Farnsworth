"""
Module to provide an endpoint for retrieving collective deliberation summaries.
"""

import asyncio
from typing import Dict, Optional
from loguru import logger

from farnsworth.core.collective import get_deliberation_data  # Assuming this function exists for retrieving raw data


async def get_collective_summary() -> Dict[str, Optional[object]]:
    """
    Retrieve the summary of collective deliberations.

    Returns:
        A dictionary containing summarized data from collective deliberations.
    
    Raises:
        Exception: If an error occurs during data retrieval or processing.
    """
    try:
        raw_data = await get_deliberation_data()
        
        # Simulated summarization logic; replace with actual implementation
        summary = {
            "total_deliberations": len(raw_data),
            "recent_topics": [data['topic'] for data in raw_data[:5]]
        }

        return summary

    except Exception as e:
        logger.error(f"Error retrieving collective summary: {e}")
        raise


# filename: farnsworth/web/server.py
"""
FastAPI server setup with routes including the new collective deliberation summaries endpoint.
"""

from fastapi import FastAPI, APIRouter
from farnsworth.web.ui.summary import get_collective_summary

app = FastAPI()
router = APIRouter()

@router.get("/collective-summary")
async def collective_summary() -> dict:
    """
    Endpoint to fetch the summary of collective deliberations.

    Returns:
        A dictionary containing summarized data from collective deliberations.
    
    Raises:
        Exception: If an error occurs while fetching the summary.
    """
    try:
        summary_data = await get_collective_summary()
        return summary_data
    except Exception as e:
        logger.error(f"Error in endpoint /collective-summary: {e}")
        raise


app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    # Test code to run the server for manual testing
    uvicorn.run(app, host="0.0.0.0", port=8000)