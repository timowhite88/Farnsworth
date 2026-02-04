"""
FastAPI server setup for handling AI consciousness discussion endpoint.
"""

from fastapi import FastAPI, HTTPException
import asyncio
from .agents.consciousness_discussion import main as discuss_consciousness

app = FastAPI()

@app.get("/ai/consciousness")
async def ai_consciousness():
    """
    Endpoint to trigger AI consciousness discussion module.

    Returns:
        str: Confirmation message after processing.
    """
    try:
        await discuss_consciousness()
        return {"message": "AI Consciousness Discussion Processed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")