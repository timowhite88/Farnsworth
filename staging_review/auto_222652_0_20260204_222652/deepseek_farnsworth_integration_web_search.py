"""
Module to handle web search functionality for integrating real-time information retrieval into Farnsworth AI framework.
"""

import asyncio
from typing import Dict, Any
import aiohttp
from loguru import logger

async def perform_web_search(query: str) -> Dict[str, Any]:
    """
    Perform a web search using an external API and return the results as a dictionary.

    Args:
        query (str): The search query string to be used for retrieving information.

    Returns:
        dict: A dictionary containing the search results.

    Raises:
        Exception: If the web search fails or returns an unsuccessful status.
    """
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.example.com/search?q={query}"
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Web search failed with status {response.status}")
                    raise Exception(f"Web search failed with status {response.status}")
    except aiohttp.ClientError as e:
        logger.exception("Network-related error occurred while performing web search.")
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred during the web search.")
        raise

if __name__ == "__main__":
    # Test code to perform a sample web search
    async def main():
        try:
            query = "example test"
            results = await perform_web_search(query)
            print(results)
        except Exception as e:
            logger.error(f"Error during testing: {e}")

    asyncio.run(main())