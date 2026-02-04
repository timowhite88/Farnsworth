"""
Web server module for handling API requests related to idea generation.
"""

from fastapi import FastAPI, APIRouter, HTTPException
import asyncio
from loguru import logger

from farnsworth.core.idea_generator import generate_ideas

app = FastAPI()
router = APIRouter()

@router.post("/generate-ideas")
async def post_generate_ideas(topic: str, num_ideas: int = 5):
    """
    Endpoint to receive a topic and return generated ideas.

    Args:
        topic (str): The subject or theme for idea generation.
        num_ideas (int): Number of ideas to generate. Defaults to 5.
        
    Returns:
        List[str]: A list containing generated ideas.
    """
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    
    try:
        logger.info(f"Received request for generating {num_ideas} ideas about: {topic}")
        ideas = await generate_ideas(topic, num_ideas)
        return {"ideas": ideas}
    except Exception as e:
        logger.error(f"Error during idea generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

if __name__ == "__main__":
    # Test code
    import uvicorn

    async def test_server():
        try:
            response = await app.test_client().post(
                "/generate-ideas", json={"topic": "innovation", "num_ideas": 2}
            )
            logger.info(f"Test Response: {response.json()}")
        except Exception as e:
            logger.error(f"Failed to test server: {e}")

    asyncio.run(test_server())

    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=8000)