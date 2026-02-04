"""
Module for parsing and validating web search JSON data to ensure correct deserialization.
"""

import asyncio
from typing import Any, Dict
import json
from loguru import logger

async def parse_web_search_result(data: str) -> Dict[str, Any]:
    """
    Parses the JSON string from a web search result.

    :param data: The raw JSON string received from the web search.
    :return: A dictionary representing the parsed JSON structure if valid.
    :raises ValueError: If the incoming data contains an unexpected variant or parsing fails.
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
Module for handling incoming web search requests and processing their results.
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
        logger.info("Web search data processed successfully")
        # Continue with further processing using parsed_data
        return {"status": "success", "data": parsed_data}

    except ValueError as e:
        logger.error(f"Error handling web search: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Test code for demonstration purposes (not full test suite)
    async def main():
        valid_data = '{"tools": [{"type": "function"}, {"type": "live_search"}]}'
        invalid_data = '{"tools": [{"type": "web_search"}]}'

        result_valid = await handle_web_search(valid_data)
        print(result_valid)  # Expected: {'status': 'success', 'data': {...}}

        result_invalid = await handle_web_search(invalid_data)
        print(result_invalid)  # Expected: {'error': 'Unexpected tool type: web_search'}

    asyncio.run(main())