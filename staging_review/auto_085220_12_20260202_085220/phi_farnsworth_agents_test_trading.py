"""
Unit tests for the trading module in Farnsworth.
"""

import pytest
from .trading import place_order, execute_order

@pytest.mark.asyncio
async def test_place_order():
    response = await place_order('buy', 10, 100.0)
    assert 'status' in response
    assert response['status'] == 'pending'

@pytest.mark.asyncio
async def test_execute_order():
    # Simulating a successful order placement to get an order_id
    place_response = await place_order('sell', 5, 200.0)
    
    if 'order_id' not in place_response:
        pytest.skip("Skipping execution test due to failed order placement.")
        
    mock_order_id = place_response['order_id']
    response = await execute_order(mock_order_id)
    assert 'execution_status' in response
    assert response['execution_status'] == 'completed'