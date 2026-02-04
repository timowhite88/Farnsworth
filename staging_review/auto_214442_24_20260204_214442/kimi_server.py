"""
FastAPI server for handling requests related to AI consciousness discussions.
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
        logger.error(f"Error processing AI consciousness discussion: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)