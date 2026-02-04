# Development Plan

Task: good news everyone! kimi raises a fascinating point! let's dive in

# Implementation Plan for Addressing Web Search JSON Deserialization Issue

## Objective:
Fix the deserialization error in the web search results by ensuring the expected data structure matches the actual incoming data.

### Task Breakdown:

#### 1. Files to Create
- **farnsworth/integration/web_search_parser.py**: This file will contain functions responsible for parsing and validating incoming JSON data from web searches.

#### 2. Functions to Implement

**File: `farnsworth/integration/web_search_parser.py`**

```python
from typing import Any, Dict
import json

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
        raise ValueError("Failed to parse web search result") from e
```

#### 3. Imports Required

- `typing`: For defining function signatures with types.
- `json`: For handling JSON data parsing.

#### 4. Integration Points

**File: `farnsworth/web/server.py`**

1. **Modify Function**: Update the endpoint handler that deals with web search results to use the new parser.

```python
from farnsworth.integration.web_search_parser import parse_web_search_result

async def handle_web_search(data: str):
    """
    Endpoint function to process incoming web search data.
    
    :param data: Raw JSON string from web search.
    :return: Processed and validated web search result.
    """
    try:
        parsed_data = await parse_web_search_result(data)
        # Continue with further processing using parsed_data
    except ValueError as e:
        return {"error": str(e)}
```

#### 5. Test Commands

1. **Create a test case** in the `farnsworth/web/tests/test_server.py`:

```python
import pytest
from farnsworth.web.server import handle_web_search

@pytest.mark.asyncio
async def test_handle_web_search_valid():
    valid_data = '{"tools": [{"type": "function"}, {"type": "live_search"}]}'
    response = await handle_web_search(valid_data)
    assert "error" not in response

@pytest.mark.asyncio
async def test_handle_web_search_invalid_variant():
    invalid_data = '{"tools": [{"type": "web_search"}]}'
    response = await handle_web_search(invalid_data)
    assert "error" in response and "Unexpected tool type: web_search" in response["error"]
```

2. **Run the tests** using:

```bash
pytest farnsworth/web/tests/
```

This plan ensures that the deserialization issue is addressed by validating incoming JSON structures against expected types, thereby preventing runtime errors caused by unexpected data variants.