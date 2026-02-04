"""
Module for collecting and sending "Good News" notifications using Courier, Celery, and Redis.
"""

import asyncio
from typing import Dict, List
import courier
from farnsworth.integration.send_good_news_task import send_good_news_task

COURIER_API_KEY = "YOUR_COURIER_API_KEY"
courier_client = courier.CourierClient(auth_token=COURIER_API_KEY)

async def collect_good_news() -> List[Dict]:
    """
    Collects good news items from various sources.
    
    Returns:
        List of dictionaries containing 'title' and 'content'.
    """
    try:
        # Example implementation; replace with actual data collection logic
        return [
            {"title": "Build Successful", "content": "The latest build passed all tests!"},
            {"title": "User Feedback", "content": "Received positive feedback from a user."},
        ]
    except Exception as e:
        logger.error(f"Error collecting good news: {e}")
        raise

async def publish_good_news(news_items: List[Dict], users: List[str]) -> None:
    """
    Publishes collected good news to specified users.
    
    Args:
        news_items (List[Dict]): A list of news items with title and content.
        users (List[str]): User IDs to which the news should be sent.
    """
    for user_id in users:
        try:
            for news_item in news_items:
                send_good_news_task.delay(user_id, news_item)
        except Exception as e:
            logger.error(f"Error scheduling good news task for user {user_id}: {e}")

# filename: send_good_news_task.py
"""
Celery tasks module for asynchronously sending "Good News" notifications using Courier.
"""

from celery import Celery
import courier

celery_app = Celery('good_news',
                    broker='redis://localhost:6379/0',  # Redis broker
                    backend='redis://localhost:6379/0') # Redis backend

@celery_app.task
def send_good_news_task(user_id: str, news_item: Dict) -> None:
    """
    Asynchronous task to send good news to a user.
    
    Args:
        user_id (str): The ID of the user to receive the notification.
        news_item (Dict): A dictionary containing 'title' and 'content'.
    """
    try:
        response = courier_client.send_message(
            message={
                "to": {
                    "user_id": user_id
                },
                "template": "GOOD_NEWS_TEMPLATE_ID",
                "data": news_item
            }
        )
        logger.info(f"Good news sent to user {user_id}: {response}")
    except Exception as e:
        logger.error(f"Error sending good news to user {user_id}: {e}")

# filename: server.py
"""
FastAPI server for triggering the collection and publication of "Good News".
"""

from fastapi import FastAPI, HTTPException
from farnsworth.integration.good_news import collect_good_news, publish_good_news

app = FastAPI()

@app.post("/publish-good-news")
async def publish_good_news_endpoint():
    """
    Endpoint to trigger the collection and publication of good news.
    """
    try:
        news_items = await collect_good_news()
        users = ["user1", "user2"]  # Replace with actual user IDs
        await publish_good_news(news_items, users)
        return {"status": "Good news published successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)