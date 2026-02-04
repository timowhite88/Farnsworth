"""
Module for parsing and validating web search results JSON data to prevent deserialization errors.
"""

import asyncio
from typing import Any, Dict
import json

async def parse_web_search_result(data: str) -> Dict[str, Any]:
    """
    Parses the JSON string from a web search result.

    :param data: The raw JSON string received from the web search.
    :return: A dictionary representing the parsed JSON structure if valid.
    :raises ValueError: If the incoming data contains an unexpected variant.
    """
    try:
        json_data = json.loads(data)
        
        # Validate expected keys and types
        tools = json_data.get('tools', [])
        for tool in tools:
            tool_type = tool.get('type')
            if tool_type not in ['function', 'live_search']:
                raise ValueError(f"Unexpected tool type: {tool_type}")
                
        return json_data

    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError("Failed to parse web search result") from e


# filename: farnsworth/web/server.py
"""
Module handling server interactions for web search features.
"""

import asyncio
from typing import Dict, Any
from loguru import logger
from farnsworth.integration.web_search_parser import parse_web_search_result

async def handle_web_search(data: str) -> Dict[str, Any]:
    """
    Endpoint function to process incoming web search data.

    :param data: Raw JSON string from web search.
    :return: Processed and validated web search result or an error message.
    """
    try:
        parsed_data = await parse_web_search_result(data)
        # Continue with further processing using parsed_data
        return {"success": True, "data": parsed_data}

    except ValueError as e:
        logger.error(f"Error handling web search: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test code
    async def test_handle_web_search():
        valid_data = '{"tools": [{"type": "function"}, {"type": "live_search"}]}'
        invalid_data = '{"tools": [{"type": "web_search"}]}'

        print("Testing with valid data:")
        response = await handle_web_search(valid_data)
        assert "success" in response, f"Expected success message but got {response}"

        print("\nTesting with invalid tool type:")
        response = await handle_web_search(invalid_data)
        assert "error" in response and "Unexpected tool type: web_search" in response["error"], \
            f"Expected error message but got {response}"

    asyncio.run(test_handle_web_search())