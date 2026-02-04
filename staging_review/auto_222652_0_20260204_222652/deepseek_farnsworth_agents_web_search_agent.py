"""
Agent implementation for utilizing web search capabilities within the Farnsworth AI framework.
"""

import asyncio
from typing import Dict, Any
from loguru import logger

from farnsworth.integration.web_search import perform_web_search

async def fetch_information(agent_id: int, query: str) -> Dict[str, Any]:
    """
    Fetch information based on a web search for the given agent.

    Args:
        agent_id (int): The ID of the agent performing the search.
        query (str): The search query string to be used.

    Returns:
        dict: A dictionary containing the search results along with the agent's ID.
    
    Raises:
        Exception: If the web search fails or returns an unsuccessful status.
    """
    try:
        logger.info(f"Agent {agent_id} is performing a search for: '{query}'")
        results = await perform_web_search(query)
        return {"agent_id": agent_id, "results": results}
    except Exception as e:
        logger.error(f"Error fetching information for agent {agent_id}: {e}")
        raise

if __name__ == "__main__":
    # Test code to fetch information using the web search agent
    async def main():
        try:
            agent_id = 1
            query = "example test"
            results = await fetch_information(agent_id, query)
            print(results)
        except Exception as e:
            logger.error(f"Error during testing: {e}")

    asyncio.run(main())