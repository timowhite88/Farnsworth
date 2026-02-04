"""
Module for integrating web search capabilities into Farnsworth infrastructure,
allowing asynchronous external searches via an API.
"""

import asyncio
from typing import Dict, Any
import aiohttp
from loguru import logger

async def perform_web_search(query: str) -> Dict[str, Any]:
    """
    Perform a web search using an external API and return the results.

    :param query: The search query string.
    :return: A dictionary containing the search results.
    """
    api_url = "https://api.example.com/search"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, params={"q": query}) as response:
                # Check for a successful response
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Search API returned status code {response.status}")
                    response.raise_for_status()
    except aiohttp.ClientError as e:
        logger.exception("A network-related error occurred during the web search.")
        raise RuntimeError("Network-related error") from e
    except Exception as e:
        logger.exception("An unexpected error occurred during the web search.")
        raise RuntimeError("Unexpected error") from e

# filename: farnsworth_web_server.py
"""
Integration of a new web search endpoint into Farnsworth's FastAPI server.
"""

from fastapi import APIRouter, HTTPException
from farnsworth_integration_web_search import perform_web_search
from loguru import logger

router = APIRouter()

@router.get("/search")
async def search_endpoint(query: str) -> Dict[str, Any]:
    """
    Endpoint for performing a web search.

    :param query: The search query string.
    :return: JSON response with the search results.
    """
    try:
        results = await perform_web_search(query)
        return results
    except RuntimeError as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# filename: farnsworth_web_app.py
"""
Main FastAPI application setup for Farnsworth including the web search integration.
"""

from fastapi import FastAPI
from farnsworth_web_server import router

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Farnsworth API!"}

# Include the search endpoint
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    # Run the server for testing purposes (not production-ready)
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)