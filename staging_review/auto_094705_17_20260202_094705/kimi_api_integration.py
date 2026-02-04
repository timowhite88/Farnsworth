"""
Module to integrate and fetch data from external APIs asynchronously using aiohttp.
"""

import asyncio
from typing import Dict, Optional
import aiohttp

async def fetch_external_data(api_url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Fetch data from an external API.

    Parameters:
    - api_url (str): The URL of the external API endpoint.
    - headers (dict): Optional HTTP headers for the request.

    Returns:
    - dict: A dictionary containing the response JSON data or error message.
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(api_url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to fetch external data")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

# filename: server.py
"""
FastAPI server module for the Farnsworth application, integrating external API functionality.
"""

from fastapi import FastAPI, HTTPException
from farnsworth.integration.api_integration import fetch_external_data

app = FastAPI()

@app.get("/external-data/{api_url}")
async def get_external_data(api_url: str) -> Dict[str, Any]:
    """
    Endpoint to retrieve data from an external API.

    Parameters:
    - api_url (str): The URL of the external API endpoint.

    Returns:
    - dict: A dictionary containing the response JSON data or an error message.
    """
    try:
        data = await fetch_external_data(api_url)
        return {"data": data}
    except HTTPException as e:
        logger.error(f"Error fetching data: {e.detail}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)