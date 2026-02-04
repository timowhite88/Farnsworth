"""
Module for handling good news notifications within the Farnsworth web application.
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger

router = APIRouter()

async def get_good_news() -> List[str]:
    """
    Retrieve a list of recent good news items.

    Returns:
        List[str]: A static list of good news notifications.
    """
    return [
        "DeepSeek has achieved its highest accuracy score!",
        "New feature release: DeepSeek now supports parallel processing!"
    ]

@router.get("/notifications")
async def notifications() -> Dict[str, Optional[List[str]]]:
    """
    API endpoint to get good news notifications.

    Returns:
        Dict[str, List[str]]: JSON response containing a list of good news items.
    
    Raises:
        HTTPException: If an error occurs while fetching the good news.
    """
    try:
        news_items = await get_good_news()
        return {"good_news": news_items}
    except Exception as e:
        logger.error(f"Failed to fetch notifications: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

# filename: server.py
"""
Main FastAPI application for the Farnsworth web interface.
"""

from fastapi import FastAPI
from farnsworth.web.ui import router as ui_router

app = FastAPI()

@app.get("/")
async def root():
    """
    Root endpoint returning a welcome message.
    
    Returns:
        Dict[str, str]: A JSON response with a welcome message.
    """
    return {"message": "Welcome to the Farnsworth AI Collective!"}

# Include the notifications router
app.include_router(ui_router, prefix="/notifications")

if __name__ == "__main__":
    import uvicorn

    # Test server start-up (for local testing)
    uvicorn.run(app, host="0.0.0.0", port=8000)

# filename: notification.js
"""
JavaScript file to handle fetching and displaying good news notifications.
"""

async function fetchGoodNews() {
    const response = await fetch('/notifications');
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    const data = await response.json();
    displayNotifications(data.good_news);
}

function displayNotifications(newsItems) {
    const notificationContainer = document.getElementById("notification-container");
    newsItems.forEach(item => {
        const notificationElement = document.createElement("div");
        notificationElement.className = "notification";
        notificationElement.innerText = item;
        notificationContainer.appendChild(notificationElement);
    });
}

// Fetch and display good news on page load
window.onload = fetchGoodNews;

# filename: notifications.html
"""
HTML template for displaying good news notifications.
"""

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Good News Notifications</title>
    <link rel="stylesheet" href="/static/css/styles.css">
</head>
<body>
    <div id="notification-container"></div>
    <script src="/static/js/notification.js"></script>
</body>
</html>