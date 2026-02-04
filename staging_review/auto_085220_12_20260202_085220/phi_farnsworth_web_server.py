"""
FastAPI server for the Farnsworth application, extended with order handling endpoints.
"""

import asyncio
from fastapi import FastAPI, APIRouter
from .agents.trading import place_order, execute_order

app = FastAPI()
router = APIRouter()

@router.post("/orders/place")
async def place_order_endpoint(order_type: str, quantity: int, price: float):
    return await place_order(order_type, quantity, price)

@router.get("/orders/execute/{order_id}")
async def execute_order_endpoint(order_id: str):
    return await execute_order(order_id)

app.include_router(router)