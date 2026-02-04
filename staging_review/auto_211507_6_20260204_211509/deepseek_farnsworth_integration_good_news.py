"""
Module to handle collection and publishing of good news using Courier for notifications,
Celery for asynchronous task management, and Redis as the message broker.
"""

import asyncio
from typing import Dict, List
from loguru import logger
from courier import CourierClient
from farnsworth.tasks.send_good_news_task import send_good_news_task

# Courier API key (replace with your actual key)
COURIER_API_KEY = "YOUR_COURIER_API_KEY"
courier_client = CourierClient(auth_token=COURIER_API_KEY)

async def collect_good_news() -> List[Dict]:
    """
    Collects good news items from various sources.

    Returns:
        List[Dict]: A list of dictionaries, each containing 'title' and 'content'.
    """
    try:
        # Example implementation; replace with actual data collection logic
        return [
            {"title": "Build Successful", "content": "The latest build passed all tests!"},
            {"title": "User Feedback", "content": "Received positive feedback from a user."},
        ]
    except Exception as e:
        logger.error(f"Error collecting good news: {e}")
        return []

async def publish_good_news(news_items: List[Dict], users: List[str]) -> None:
    """
    Publishes collected good news to specified users.

    Args:
        news_items (List[Dict]): A list of news items with title and content.
        users (List[str]): User IDs to which the news should be sent.
    """
    for user_id in users:
        for news_item in news_items:
            # Send each piece of good news as a separate task
            send_good_news_task.delay(user_id, news_item)

# filename: farnsworth/tasks/send_good_news_task.py

"""
Module to define the Celery task responsible for sending good news notifications.
"""

import asyncio
from typing import Dict
from loguru import logger
from celery import Celery
from courier import CourierClient

# Celery configuration
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
        # Assuming you have a Courier template for "Good News"
        response = courier_client.send_message(
            message={
                "to": {
                    "user_id": user_id  # Assuming you use user IDs in Courier
                },
                "template": "GOOD_NEWS_TEMPLATE_ID",  # Replace with your template ID
                "data": news_item  # Pass the news item as data to the template
            }
        )
        logger.info(f"Good news sent to user {user_id}: {response}")
    except Exception as e:
        logger.error(f"Error sending good news to user {user_id}: {e}")

# filename: farnsworth/web/server.py

"""
FastAPI server setup for Farnsworth, including endpoint integration for publishing good news.
"""

from fastapi import FastAPI, HTTPException
import asyncio
from farnsworth.integration.good_news import collect_good_news, publish_good_news

app = FastAPI()

@app.post("/publish-good-news")
async def publish_good_news_endpoint():
    """
    Endpoint to trigger the collection and publication of good news.

    Returns:
        Dict[str, str]: Status message indicating success or failure.
    """
    try:
        news_items = await collect_good_news()
        users = ["user1", "user2"]  # Replace with actual user IDs
        await publish_good_news(news_items, users)
        return {"status": "Good news published successfully"}
    except Exception as e:
        logger.error(f"Error in publishing good news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Test code
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)