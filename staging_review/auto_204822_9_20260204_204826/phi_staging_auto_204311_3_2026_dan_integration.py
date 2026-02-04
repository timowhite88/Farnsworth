"""
Module for integrating 'delicate dan' functionality with existing systems.
"""

import asyncio
from typing import Dict

from loguru import logger
from farnsworth.web.server import FastAPIApp
from .dan_handler import process_dan_data, integrate_dan_response, prepare_dan_request


async def initiate_dan_operation(data: Dict) -> str:
    """
    Initiate an operation for 'delicate dan' and return the outcome.

    Args:
        data (dict): The input data to start the delicate dan operation.

    Returns:
        str: Outcome of the initiation process.
        
    Raises:
        RuntimeError: If any step in the operation fails.
    """
    try:
        logger.info("Initiating 'delicate dan' operation.")
        prepared_data = prepare_dan_request(data)
        status_message = await process_dan_data(prepared_data)
        await integrate_dan_response(prepared_data)

        return f"Operation initiated successfully: {status_message}"
    
    except Exception as e:
        logger.error("Failed to initiate delicate dan operation: {}", e)
        raise RuntimeError(f"Initiation failed: {e}")


def register_dan_routes(app: FastAPIApp) -> None:
    """
    Register new routes for 'delicate dan' operations in the web server.

    Args:
        app (FastAPIApp): The FastAPI application instance to add routes to.
    
    Returns:
        None
    """
    @app.post("/dan/operation")
    async def handle_dan_operation(data: Dict):
        try:
            result = await initiate_dan_operation(data)
            return {"status": "success", "message": result}
        
        except RuntimeError as e:
            logger.error("Handling 'delicate dan' operation failed: {}", e)
            return {"status": "error", "message": str(e)}

    logger.info("Registered routes for 'delicate dan' operations.")


if __name__ == "__main__":
    # Test code
    async def test_integration():
        app = FastAPIApp()  # Hypothetical instance creation
        register_dan_routes(app)
        
        # Normally, you'd start the server here, but we'll simulate a call.
        data = {"key": "value"}
        try:
            result = await initiate_dan_operation(data)
            print(result)
        except Exception as e:
            logger.error("Integration test failed: {}", e)

    asyncio.run(test_integration())