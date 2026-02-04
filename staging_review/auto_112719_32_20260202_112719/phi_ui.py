"""
Module for handling UI updates related to good news notifications in Farnsworth.
"""

import asyncio
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()

async def get_good_news() -> List[str]:
    """
    Retrieve a list of recent good news items.

    Returns:
        List[str]: A list containing static good news messages.
    """
    # For simplicity, return static news items. In practice, fetch from a database or service.
    return [
        "DeepSeek has achieved its highest accuracy score!",
        "New feature release: DeepSeek now supports parallel processing!"
    ]

@router.get("/notifications")
async def notifications() -> Dict[str, List[str]]:
    """
    API endpoint to get good news notifications.

    Returns:
        dict: JSON object containing a list of good news items.
        
    Raises:
        HTTPException: If an error occurs during fetching of the news.
    """
    try:
        news_items = await get_good_news()
        return {"good_news": news_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# server.py
"""
Main FastAPI server module integrating UI components for good news notifications.
"""

from fastapi import FastAPI
from farnsworth.web.ui import router as ui_router

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to Farnsworth!"}

app.include_router(ui_router, prefix="/notifications")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# filename: notification.js
"""
JavaScript file for handling the display of good news notifications on the client side.
"""

async function fetchGoodNews() {
    try {
        const response = await fetch('/notifications');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        displayNotifications(data.good_news);
    } catch (error) {
        console.error("Error fetching good news:", error);
    }
}

function displayNotifications(newsItems) {
    const notificationContainer = document.getElementById("notification-container");
    if (!notificationContainer) return;

    // Clear existing notifications
    notificationContainer.innerHTML = '';

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

# filename: styles.css
"""
Basic CSS for styling the good news notifications.
"""

.notification {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
}