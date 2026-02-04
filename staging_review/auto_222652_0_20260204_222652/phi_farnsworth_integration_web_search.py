"""
Module for handling web search functionality using aiohttp.
"""

import asyncio
from typing import Dict, Any
from loguru import logger

async def perform_web_search(query: str) -> Dict[str, Any]:
    """
    Perform a web search and return results as a dictionary.

    Args:
        query (str): The search query string.

    Returns:
        dict: A dictionary containing the search results.
    
    Raises:
        Exception: If the web search fails due to network issues or invalid responses.
    """
    import aiohttp

    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.example.com/search?q={query}"
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Web search failed with status {response.status}")
    except aiohttp.ClientError as e:
        logger.error("Network error occurred during web search", exc_info=e)
        raise
    except asyncio.TimeoutError as e:
        logger.error("Timeout error during web search", exc_info=e)
        raise

if __name__ == "__main__":
    async def test_web_search():
        try:
            results = await perform_web_search("example query")
            print(results)
        except Exception as e:
            logger.error("Failed to perform web search", exc_info=e)

    asyncio.run(test_web_search())

# filename: farnsworth/agents/web_search_agent.py
"""
Agent implementation that utilizes the web search module.
"""

import asyncio
from typing import Dict, Any

from farnsworth.integration.web_search import perform_web_search
from loguru import logger

async def fetch_information(agent_id: int, query: str) -> Dict[str, Any]:
    """
    Fetch information based on a web search for the given agent.

    Args:
        agent_id (int): The ID of the agent requesting information.
        query (str): The search query string.

    Returns:
        dict: A dictionary containing the agent ID and search results.
    
    Raises:
        Exception: If an error occurs during the web search process.
    """
    try:
        results = await perform_web_search(query)
        return {"agent_id": agent_id, "results": results}
    except Exception as e:
        logger.error(f"Failed to fetch information for agent {agent_id}", exc_info=e)
        raise

if __name__ == "__main__":
    async def test_fetch_information():
        try:
            result = await fetch_information(1, "example query")
            print(result)
        except Exception as e:
            logger.error("Test failed", exc_info=e)

    asyncio.run(test_fetch_information())