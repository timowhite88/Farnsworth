"""
Module for integrating external data via API calls in Farnsworth.
"""

import aiohttp
from typing import Dict, Optional

async def fetch_external_data(api_url: str, headers: Optional[Dict[str, str]] = None) -> Dict:
    """
    Fetch data from an external API.

    Parameters:
    - api_url (str): The URL of the external API endpoint.
    - headers (Optional[Dict[str, str]]): Optional HTTP headers for the request.

    Returns:
    - Dict: A dictionary containing the response JSON data.

    Raises:
    - HTTPError: If an error occurs during the API call.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(api_url, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"HTTP request error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during API call: {e}")
        raise

# filename: server.py
"""
FastAPI server for Farnsworth with external data integration.
"""

from fastapi import FastAPI, HTTPException
from farnsworth.integration.api_integration import fetch_external_data
from loguru import logger

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
        data = await fetch_external_data(api_url)
        return {"data": data}
    except Exception as e:
        logger.error(f"Error fetching external data: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve external data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)