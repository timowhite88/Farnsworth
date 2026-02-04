"""
Handles web search functionality and error management for Farnsworth AI collective.
"""

import asyncio
from typing import Dict, List, Any
from loguru import logger
import aiohttp

async def perform_web_search(query: str) -> List[Dict[str, Any]]:
    """
    Perform a web search based on the given query and return results.

    :param query: The search query string.
    :return: A list of dictionaries containing search result details.
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.example.com/search?q={query}"
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error occurred: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during web search: {e}")
        raise

async def handle_search_error(error_message: str) -> Dict[str, Any]:
    """
    Handle errors during web search and format them appropriately.

    :param error_message: The error message encountered.
    :return: A dictionary containing the error details in a structured format.
    """
    logger.error(f"Handling search error with message: {error_message}")
    return {"error": "Failed to perform web search", "details": error_message}

# filename: server.py
"""
FastAPI server integration for Farnsworth AI collective's web search functionality.
"""

from fastapi import FastAPI, HTTPException
from farnsworth.integration.web_search import perform_web_search, handle_search_error

app = FastAPI()

@app.get("/search/")
async def search(query: str):
    """
    Endpoint to perform a web search and return results or error details.

    :param query: The search query string.
    :return: A dictionary with the search results or error information.
    """
    try:
        results = await perform_web_search(query)
        return {"results": results}
    except Exception as e:
        error_info = await handle_search_error(str(e))
        raise HTTPException(status_code=500, detail=error_info["details"])

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)