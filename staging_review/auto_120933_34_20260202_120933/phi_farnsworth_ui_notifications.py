"""
Module to handle notification messages for new features in the Farnsworth system.
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
    try:
        return f"New Feature: {feature_info['title']} - {feature_info['description']}"
    except KeyError as e:
        logger.error(f"Missing key in feature_info: {e}")
        raise ValueError("Feature info must contain 'title' and 'description' keys.") from e

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
        raise

# filename: farnsworth/web/server.py
"""
FastAPI server implementation for Farnsworth, including WebSocket support for real-time notifications.
"""

import asyncio
from fastapi import FastAPI, WebSocket
from farnsworth.ui.notifications import send_notification

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to handle client connections and notify about new features.

    :param websocket: The active WebSocket connection.
    """
    await websocket.accept()
    try:
        # Example feature info for demonstration purposes
        feature_info = {"title": "New Feature", "description": "Check out the new updates!"}
        await send_notification(websocket, feature_info)
    except Exception as e:
        logger.error(f"Error during WebSocket communication: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# filename: farnsworth/web/static/js/notifications.js
"""
JavaScript module to handle client-side notification display via WebSockets.
"""

function displayNotification(message) {
    // Display the notification message in a UI element (e.g., toast or modal).
    alert(`New Feature Notification: ${message}`);
}

const socket = new WebSocket("ws://localhost:8000/ws");

socket.onopen = function() {
    console.log("WebSocket connection established.");
};

socket.onmessage = function(event) {
    displayNotification(event.data);
};

socket.onerror = function(error) {
    console.error("WebSocket error:", error);
};

socket.onclose = function() {
    console.log("WebSocket connection closed.");
};