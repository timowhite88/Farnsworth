"""
Module for integrating and fetching data from external APIs asynchronously using aiohttp.
"""

import asyncio
from typing import Dict, Optional
import aiohttp
from loguru import logger

async def fetch_external_data(api_url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Fetch data from an external API.

    Parameters:
    - api_url (str): The URL of the external API endpoint.
    - headers (Optional[dict]): Optional HTTP headers for the request.

    Returns:
    - dict: A dictionary containing the response JSON data or error message.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to fetch data from {api_url}. Status: {response.status}")
                    return {"error": f"HTTP error {response.status}"}
    except aiohttp.ClientError as e:
        logger.exception("A client error occurred while fetching external data.")
        return {"error": "Client error during API request"}
    except asyncio.TimeoutError:
        logger.error(f"Timeout when connecting to {api_url}")
        return {"error": "Request timed out"}
    except Exception as e:
        logger.exception("An unexpected error occurred while fetching external data.")
        return {"error": "Unexpected error during API request"}

# filename: server.py
"""
FastAPI web server module that integrates with the Farnsworth system and includes an endpoint for
fetching data from external APIs.
"""

from fastapi import FastAPI, HTTPException
from farnsworth.integration.api_integration import fetch_external_data

app = FastAPI()

@app.get("/external-data/{api_url}")
async def get_external_data(api_url: str):
    """
    Endpoint to retrieve data from an external API.

    Parameters:
    - api_url (str): The URL of the external API endpoint.

    Returns:
    - dict: A dictionary containing the response JSON data or an error message.
    """
    try:
        result = await fetch_external_data(api_url)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except Exception as e:
        logger.exception("An unexpected error occurred while processing the API request.")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn

    # Test code: Run the FastAPI server with Uvicorn for testing purposes
    uvicorn.run(app, host="0.0.0.0", port=8000)