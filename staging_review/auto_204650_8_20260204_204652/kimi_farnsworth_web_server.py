"""
A server module for integrating a new Greeting component into the Farnsworth web interface using FastAPI and React.
"""

import asyncio
from typing import Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel

class GreetingComponent(BaseModel):
    recipient: str

    def render(self) -> Dict[str, str]:
        """
        Render the greeting message for the specified recipient.

        Returns:
            A dictionary containing the greeting messages.
        """
        return {
            "greeting": f"Good news, everyone! {self.recipient}, I'm thrilled you're jumping on the UI task"
        }

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        An instance of FastAPI configured for this project.
    """
    app = FastAPI()

    @app.get("/", response_model=Dict[str, Optional[Dict[str, str]]])
    async def read_root(recipient: Optional[str] = "Claude") -> Dict[str, Optional[Dict[str, str]]]:
        """
        Main route that returns a welcome message and the greeting component.

        Args:
            recipient (Optional[str]): The name of the recipient for the greeting. Defaults to "Claude".

        Returns:
            A dictionary containing a welcome message and the rendered greeting.
        """
        try:
            greeting_component = GreetingComponent(recipient=recipient)
            return {
                "message": "Welcome to Farnsworth!",
                "greeting_component": greeting_component.render()
            }
        except Exception as e:
            logger.error(f"Error in rendering greeting: {e}")
            raise

    return app

if __name__ == "__main__":
    # Test code
    import uvicorn

    try:
        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=8000)
    except Exception as e:
        logger.error(f"Failed to start the server: {e}")