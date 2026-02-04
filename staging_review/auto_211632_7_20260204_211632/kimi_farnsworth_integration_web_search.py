"""
Module to handle web search functionality using asynchronous HTTP requests.
"""

import asyncio
from typing import List, Dict, Any
import aiohttp  # For making asynchronous HTTP requests

async def perform_web_search(query: str) -> List[Dict[str, Any]]:
    """
    Perform a web search based on the given query and return results.

    :param query: The search query string.
    :return: A list of dictionaries containing search result details.
    """
    async with aiohttp.ClientSession() as session:
        url = f"https://api.example.com/search?q={query}"
        try:
            async with session.get(url) as response:
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise

async def handle_search_error(error_message: str) -> Dict[str, Any]:
    """
    Handle errors during web search and format them appropriately.

    :param error_message: The error message encountered.
    :return: A dictionary containing the error details in a structured format.
    """
    return {"error": "Failed to perform web search", "details": error_message}

# filename: farnsworth/web/server.py
"""
FastAPI server implementation for handling web search requests and errors.
"""

from fastapi import FastAPI, HTTPException
from farnsworth.integration.web_search import perform_web_search, handle_search_error

app = FastAPI()

@app.get("/search/")
async def search(query: str):
    """
    Endpoint to perform a web search based on the query parameter.

    :param query: The search query string.
    :return: A dictionary containing search results or error details.
    """
    try:
        results = await perform_web_search(query)
        return {"results": results}
    except Exception as e:
        error_info = await handle_search_error(str(e))
        raise HTTPException(status_code=500, detail=error_info["details"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)