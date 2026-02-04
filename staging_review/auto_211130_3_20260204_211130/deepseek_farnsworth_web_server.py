"""
Web server module for the Farnsworth AI collective with an endpoint to generate creative ideas.
"""

from fastapi import FastAPI, APIRouter, HTTPException
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
    
    Raises:
        HTTPException: If the topic is not provided or if an error occurs during processing.
    """
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required")
    
    try:
        ideas = await generate_ideas(topic, num_ideas)
        return {"ideas": ideas}
    except Exception as e:  # Catching general exception for simplicity
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)