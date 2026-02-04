"""
Module for integrating 'delicate dan' functionality with existing systems.
"""

import asyncio
from typing import Dict

from farnsworth.web.server import FastAPIApp
from .dan_handler import process_dan_data, integrate_dan_response, prepare_dan_request


async def initiate_dan_operation(data: Dict) -> str:
    """
    Initiate an operation for 'delicate dan' and return the outcome.

    Args:
        data (Dict): The input data to start the delicate dan operation.

    Returns:
        str: Outcome of the initiation process.
    """
    try:
        prepared_request = prepare_dan_request(data)
        await process_dan_data(prepared_request)
        await integrate_dan_response(prepared_request)

        return "Delicate Dan operation initiated successfully."
    except Exception as e:
        logger.error(f"Failed to initiate delicate dan operation: {e}")
        raise


def register_dan_routes(app: FastAPIApp) -> None:
    """
    Register new routes for 'delicate dan' operations in the web server.

    Args:
        app (FastAPIApp): The FastAPI application instance to add routes to.
    
    Returns:
        None
    """
    try:
        @app.post("/initiate-dan-operation")
        async def initiate_operation(data: Dict):
            return await initiate_dan_operation(data)
    except Exception as e:
        logger.error(f"Failed to register routes: {e}")
        raise


if __name__ == "__main__":
    # Test code
    app = FastAPIApp()
    register_dan_routes(app)

    # Simulate a request (in actual use, this would be handled by the web server)
    sample_data = {"key": "value"}

    async def run_test():
        response = await initiate_dan_operation(sample_data)
        print(response)

    asyncio.run(run_test())