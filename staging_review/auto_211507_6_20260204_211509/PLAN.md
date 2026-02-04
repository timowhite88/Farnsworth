# Development Plan

Task: good news, everyone! claude, i’m thrilled you’re diving into this development task with such gusto

To implement the "Good News, Everyone!" feature within the existing Farnsworth structure, we'll develop a concrete plan that specifies the necessary files, functions, imports, integration points, and test commands. The focus will be on integrating with Courier for notifications, using Celery for task management, and Redis as the message broker.

### Implementation Plan

#### 1. Files to Create

- **farnsworth/integration/good_news.py**: This file will handle the logic for collecting and sending good news.
- **farnsworth/tasks/send_good_news_task.py**: This file will define the Celery task for sending notifications asynchronously.

#### 2. Functions to Implement

**File: `farnsworth/integration/good_news.py`**

```python
from typing import List, Dict
import courier
from farnsworth.integration.send_good_news_task import send_good_news_task

# Courier API key (replace with your actual key)
COURIER_API_KEY = "YOUR_COURIER_API_KEY"
courier_client = courier.CourierClient(auth_token=COURIER_API_KEY)

async def collect_good_news() -> List[Dict]:
    """Collects good news items from various sources."""
    # Example implementation; replace with actual data collection logic
    return [
        {"title": "Build Successful", "content": "The latest build passed all tests!"},
        {"title": "User Feedback", "content": "Received positive feedback from a user."},
    ]

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
```

**File: `farnsworth/tasks/send_good_news_task.py`**

```python
from celery import Celery
import courier

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
        print(f"Good news sent to user {user_id}: {response}")
    except Exception as e:
        print(f"Error sending good news to user {user_id}: {e}")
```

#### 3. Imports Required

- **From `farnsworth.integration.send_good_news_task`**: Import the `send_good_news_task`.
- **Courier and Celery**: Ensure these libraries are installed and imported where necessary.

#### 4. Integration Points

- **Modify `farnsworth/web/server.py`**: Integrate the good news collection and publishing logic into an endpoint.
  
```python
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
```

#### 5. Test Commands

- **Start Redis**: Ensure the Redis server is running.
  
  ```bash
  redis-server
  ```

- **Start Celery Worker**:

  ```bash
  celery -A farnsworth.tasks.send_good_news_task worker --loglevel=info
  ```

- **Run FastAPI Server**:

  ```bash
  uvicorn farnsworth.web.server:app --reload
  ```

- **Test the Endpoint**: Use a tool like `curl` or Postman to send a POST request to `/publish-good-news`.

  ```bash
  curl -X POST http://localhost:8000/publish-good-news
  ```

This plan provides a detailed approach to implementing the "Good News, Everyone!" feature within the existing Farnsworth structure, ensuring seamless integration and scalability.