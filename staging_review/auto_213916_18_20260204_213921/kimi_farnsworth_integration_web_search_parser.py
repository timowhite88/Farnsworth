"""
Module for parsing and validating JSON data from web search results.

Includes functions to handle deserialization errors by ensuring the expected
data structure matches the actual incoming data.
"""

import asyncio
from typing import Any, Dict
import json
from loguru import logger

# Function to parse the web search result and handle deserialization errors
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
        logger.error("Failed to parse web search result", exc_info=e)
        raise ValueError("Failed to parse web search result") from e


# filename: farnsworth/web/server.py
"""
Module for handling web server operations related to processing web search data.

Includes functions that integrate with web search parsers to ensure valid data.
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
        # Placeholder for further processing using parsed_data
        return {"status": "success", "data": parsed_data}

    except ValueError as e:
        logger.error("Error handling web search data", exc_info=e)
        return {"error": str(e)}

if __name__ == "__main__":
    # Test code to demonstrate functionality
    async def main():
        test_valid_data = '{"tools": [{"type": "function"}, {"type": "live_search"}]}'
        response = await handle_web_search(test_valid_data)
        logger.info(f"Valid data response: {response}")

        test_invalid_data = '{"tools": [{"type": "web_search"}]}'
        response = await handle_web_search(test_invalid_data)
        logger.info(f"Invalid data response: {response}")

    asyncio.run(main())