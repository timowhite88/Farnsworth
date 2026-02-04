"""
Module for performing web searches using an external API asynchronously.
"""

import aiohttp
from typing import Dict

async def perform_web_search(query: str) -> Dict:
    """
    Perform a web search using an external API and return the results.

    :param query: The search query string.
    :return: A dictionary containing the search results.
    :raises aiohttp.ClientError: If there's an issue with the HTTP request.
    :raises Exception: For any unexpected issues during execution.
    """
    api_url = "https://api.example.com/search"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params={"q": query}) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"HTTP request failed: {e}")
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred during web search.")
        raise

# filename: farnsworth/web/server.py
"""
FastAPI server module for Farnsworth, integrating web search functionality.
"""

from fastapi import FastAPI, APIRouter, HTTPException
from typing import Dict
import asyncio
from loguru import logger
from farnsworth.integration.web_search import perform_web_search

router = APIRouter()

@router.get("/search")
async def search_endpoint(query: str) -> Dict:
    """
    Endpoint for performing a web search.

    :param query: The search query string.
    :return: JSON response with the search results.
    :raises HTTPException: If an error occurs during the search process.
    """
    try:
        results = await perform_web_search(query)
        return results
    except aiohttp.ClientError as e:
        logger.error(f"Web search failed due to client error: {e}")
        raise HTTPException(status_code=500, detail="Client error occurred.")
    except Exception as e:
        logger.exception("An unexpected error occurred during the web search process.")
        raise HTTPException(status_code=500, detail="Internal server error.")

app = FastAPI()

# Include the search endpoint
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.exception("Failed to start the FastAPI server.")