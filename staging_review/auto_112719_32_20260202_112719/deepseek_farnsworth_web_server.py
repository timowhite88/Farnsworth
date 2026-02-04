"""
Main server module for the Farnsworth web application.
"""

import asyncio
from fastapi import FastAPI
from loguru import logger

from farnsworth.web.ui import router as ui_router

app = FastAPI()

# Include the notifications router to handle good news API endpoints
app.include_router(ui_router, prefix="/notifications")

if __name__ == "__main__":
    # Test code: Run the server and ensure it handles requests properly.
    try:
        logger.info("Starting Farnsworth web server...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Error occurred while starting the server: {e}")