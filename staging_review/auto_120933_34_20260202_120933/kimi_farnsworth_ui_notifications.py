"""
Module for handling UI notifications about new features in Farnsworth system.
"""

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
Web server module for Farnsworth, handling WebSocket connections and notifications.
"""

from fastapi import FastAPI, WebSocket
from farnsworth.ui.notifications import send_notification

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handle incoming WebSocket connections and notify about new features.

    :param websocket: The active WebSocket connection.
    """
    await websocket.accept()
    try:
        # Example feature info to send as notification
        feature_info = {"title": "New Feature", "description": "Check out the new updates!"}
        await send_notification(websocket, feature_info)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn

    # Test server with automatic reloading
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Failed to start the server: {e}")

# filename: farnsworth/web/static/js/notifications.js
"""
JavaScript module for handling UI notifications in Farnsworth system.
"""

function displayNotification(message) {
    // Display the notification message in a UI element (e.g., toast or modal).
    alert(`New Feature Notification: ${message}`);
}

const socket = new WebSocket("ws://localhost:8000/ws");

socket.onmessage = function(event) {
    try {
        displayNotification(event.data);
    } catch (error) {
        console.error(`Error displaying notification: ${error}`);
    }
};

socket.onerror = function(error) {
    console.error(`WebSocket error: ${error}`);
};