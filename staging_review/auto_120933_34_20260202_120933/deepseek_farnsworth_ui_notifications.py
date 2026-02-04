"""
Module for handling UI notifications related to new features in Farnsworth system.
"""

import asyncio
from typing import Dict
from fastapi import WebSocket

def create_notification_message(feature_info: Dict[str, str]) -> str:
    """
    Create a formatted message for the notification.

    :param feature_info: Dictionary containing 'title' and 'description'.
    :return: Formatted string message.
    """
    return f"New Feature: {feature_info['title']} - {feature_info['description']}"

async def send_notification(websocket: WebSocket, feature_info: Dict[str, str]) -> None:
    """
    Send a notification message to the connected client.

    :param websocket: The active WebSocket connection.
    :param feature_info: Dictionary containing 'title' and 'description'.
    """
    try:
        message = create_notification_message(feature_info)
        await websocket.send_text(message)
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")

# filename: farnsworth/web/server.py
"""
FastAPI server module for handling WebSocket connections and notifications.
"""

import asyncio
from fastapi import FastAPI, WebSocket

from farnsworth.ui.notifications import send_notification

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to handle client connections and send feature notifications.

    :param websocket: The active WebSocket connection.
    """
    await websocket.accept()
    try:
        # Example feature info to send as notification
        feature_info = {"title": "New Feature", "description": "Check out the new updates!"}
        await notify_new_feature(websocket, feature_info)
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")

async def notify_new_feature(websocket: WebSocket, feature_info: Dict[str, str]) -> None:
    """
    Notify connected clients about a new feature.

    :param websocket: The active WebSocket connection.
    :param feature_info: Dictionary containing 'title' and 'description'.
    """
    await send_notification(websocket, feature_info)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# filename: farnsworth/web/static/js/notifications.js
"""
JavaScript module for handling notification display on the client side.
"""

function displayNotification(message) {
    // Display the notification message in a UI element (e.g., toast or modal).
    alert(`New Feature Notification: ${message}`);
}

const socket = new WebSocket("ws://localhost:8000/ws");

socket.onopen = function() {
    console.log("WebSocket connection opened.");
};

socket.onmessage = function(event) {
    displayNotification(event.data);
};

socket.onerror = function(error) {
    console.error('WebSocket Error:', error);
};