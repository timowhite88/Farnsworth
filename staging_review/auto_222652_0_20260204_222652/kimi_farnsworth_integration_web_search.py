"""
Module for handling web search functionality within the Farnsworth AI framework.
"""

import asyncio
from typing import Dict, Any
from aiohttp import ClientSession, ClientError
from loguru import logger

async def perform_web_search(query: str) -> Dict[str, Any]:
    """
    Perform a web search and return results as a dictionary.

    Args:
        query (str): The search query string.

    Returns:
        dict: A dictionary containing the search results.
    
    Raises:
        Exception: If an error occurs during the web search request.
    """
    url = f"https://api.example.com/search?q={query}"
    try:
        async with ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Web search failed with status {response.status}")
                    raise Exception(f"Web search failed with status {response.status}")
    except ClientError as e:
        logger.exception("A client error occurred during the web search.")
        raise

# filename: farnsworth/agents/web_search_agent.py
"""
Agent implementation for performing web searches within the Farnsworth AI framework.
"""

from typing import Dict, Any
import asyncio
from loguru import logger
from farnsworth.integration.web_search import perform_web_search

async def fetch_information(agent_id: int, query: str) -> Dict[str, Any]:
    """
    Fetch information based on a web search for the given agent.

    Args:
        agent_id (int): The ID of the agent performing the search.
        query (str): The search query string.

    Returns:
        dict: A dictionary containing the search results and agent ID.
    
    Raises:
        Exception: If an error occurs during the fetching process.
    """
    try:
        logger.info(f"Agent {agent_id} is performing a web search for '{query}'.")
        results = await perform_web_search(query)
        return {"agent_id": agent_id, "results": results}
    except Exception as e:
        logger.exception("An error occurred while fetching information.")
        raise

if __name__ == "__main__":
    # Test code
    async def main():
        try:
            result = await fetch_information(agent_id=1, query="example query")
            print(result)
        except Exception as e:
            print(f"Error: {e}")

    asyncio.run(main())