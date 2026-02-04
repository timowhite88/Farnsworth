"""
Module to implement and test the UI greeting message feature for the Farnsworth project.
"""

import asyncio
from typing import Dict, Any
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from loguru import logger

# Mock implementations of required imports
class MemorySystem:
    pass

def get_memory_system() -> MemorySystem:
    return MemorySystem()

class CapabilityRegistry:
    pass

def get_capability_registry() -> CapabilityRegistry:
    return CapabilityRegistry()

class SessionManager:
    pass

def get_session_manager() -> SessionManager:
    return SessionManager()

# React component simulation
def Greeting(recipient: str) -> Dict[str, Any]:
    """
    Simulates a React component to render a greeting message.
    
    Args:
        recipient (str): The name of the person being greeted.

    Returns:
        dict: A dictionary representation of the rendered component.
    """
    return {
        "greeting": [
            {"tag": "p", "content": "Good news, everyone!"},
            {"tag": "p", "content": f"{recipient}, I'm thrilled you're jumping on the UI task"}
        ]
    }

# FastAPI server setup
app = FastAPI()

@app.get("/")
async def read_root() -> JSONResponse:
    """
    Main route to render the greeting message.
    
    Returns:
        JSONResponse: A response containing a welcome message and the greeting component.
    """
    try:
        # Simulate rendering of the Greeting component
        greeting_component = Greeting(recipient="Claude")
        
        return JSONResponse(
            content={
                "message": "Welcome to Farnsworth!",
                "greeting_component": greeting_component
            }
        )
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn

    # Run the server for testing purposes
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        logger.error(f"Failed to start the server: {e}")