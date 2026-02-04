"""
Web server module for the Farnsworth project, including humor evaluation endpoint.
"""

from fastapi import FastAPI
from farnsworth.agents.emotion_humor_agent import EmotionHumorAgent

app = FastAPI()
emotion_humor_agent = EmotionHumorAgent()

@app.post("/evaluate_humor")
async def evaluate_humor(text: str):
    """
    Endpoint to evaluate the emotional impact of humor in a given text.

    :param text: The input text containing potential humor.
    :return: Analysis and emotional impact results.
    """
    try:
        return await emotion_humor_agent.evaluate_humor_effect(text)
    except Exception as e:
        logger.error(f"Error processing /evaluate_humor endpoint: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)