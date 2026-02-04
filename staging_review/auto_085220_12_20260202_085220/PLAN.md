# Development Plan

Task: 🧪 *task detected!* i noticed swarm-mind suggested something actionable

To implement a simple trading feature within the existing Farnsworth structure, we will create a basic order execution system that allows an agent to place buy or sell orders. The implementation plan includes creating new files, defining functions with signatures, specifying necessary imports, integration points, and test commands.

### Implementation Plan

#### 1. Files to Create
- **File Path**: `farnsworth/agents/trading.py`
  
#### 2. Functions to Implement

**Function: place_order**

```python
async def place_order(order_type: str, quantity: int, price: float) -> dict:
    """
    Place a buy or sell order.
    
    Args:
        order_type (str): Type of the order ('buy' or 'sell').
        quantity (int): Number of units to trade.
        price (float): Price per unit for the trade.
        
    Returns:
        dict: A dictionary containing order details and status.
    """
```

**Function: execute_order**

```python
async def execute_order(order_id: str) -> dict:
    """
    Execute an existing order by its ID.
    
    Args:
        order_id (str): Unique identifier for the order.
        
    Returns:
        dict: A dictionary containing execution details and status.
    """
```

#### 3. Imports Required

- `farnsworth.core.collective`: For integrating collective deliberation if needed in future enhancements.
- `farnsworth.integration.api_clients`: Assuming a module exists or will be created for interacting with external trading APIs.

```python
from farnsworth.integration import api_clients
```

#### 4. Integration Points

**Modify Existing Files:**

- **File**: `farnsworth/agents/__init__.py`
  
  - Import the new functions from `trading.py` to make them available as part of the agents module.

```python
from .trading import place_order, execute_order
```

- **File**: `farnsworth/web/server.py`
  
  - Add endpoints for placing and executing orders. This will involve modifying the FastAPI server to handle new routes.

```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/orders/place")
async def place_order_endpoint(order_type: str, quantity: int, price: float):
    return await place_order(order_type, quantity, price)

@router.get("/orders/execute/{order_id}")
async def execute_order_endpoint(order_id: str):
    return await execute_order(order_id)
```

#### 5. Test Commands

**Test the Trading System:**

- **Command to Run Tests**: Assuming a test framework like `pytest` is used.

```bash
pytest farnsworth/agents/test_trading.py
```

**Sample Test File: `farnsworth/agents/test_trading.py`**

```python
import pytest
from farnsworth.agents.trading import place_order, execute_order

@pytest.mark.asyncio
async def test_place_order():
    response = await place_order('buy', 10, 100.0)
    assert 'status' in response
    assert response['status'] == 'pending'

@pytest.mark.asyncio
async def test_execute_order():
    # Assuming an order_id is returned from a successful place_order call
    mock_order_id = "12345"
    response = await execute_order(mock_order_id)
    assert 'execution_status' in response
    assert response['execution_status'] == 'completed'
```

This plan outlines the creation of a simple trading feature within the Farnsworth architecture, ensuring clarity and specificity at each step.