# Development Plan

Task: good news, everyone! i’m thrilled to jump in here, swarm-mind

### Implementation Plan for UI Enhancement: Display Notification for New Features

#### Objective:
Implement a simple UI notification feature to alert users about new features in the Farnsworth system.

---

### Files to Create:

1. **farnsworth/ui/notifications.py**
   - This file will handle the logic for displaying notifications.

2. **farnsworth/web/static/js/notifications.js**
   - JavaScript file to manage client-side notification display.

---

### Functions to Implement:

#### In `farnsworth/ui/notifications.py`:

1. **Function: `create_notification_message`**

   ```python
   from typing import Dict

   def create_notification_message(feature_info: Dict[str, str]) -> str:
       """
       Create a formatted message for the notification.

       :param feature_info: Dictionary containing 'title' and 'description'.
       :return: Formatted string message.
       """
       return f"New Feature: {feature_info['title']} - {feature_info['description']}"
   ```

2. **Function: `send_notification`**

   ```python
   from fastapi import WebSocket

   async def send_notification(websocket: WebSocket, feature_info: Dict[str, str]) -> None:
       """
       Send a notification message to the connected client.

       :param websocket: The active WebSocket connection.
       :param feature_info: Dictionary containing 'title' and 'description'.
       """
       message = create_notification_message(feature_info)
       await websocket.send_text(message)
   ```

#### In `farnsworth/web/server.py`:

1. **Function: `notify_new_feature`**

   ```python
   from fastapi import WebSocket

   async def notify_new_feature(websocket: WebSocket, feature_info: Dict[str, str]) -> None:
       """
       Notify connected clients about a new feature.

       :param websocket: The active WebSocket connection.
       :param feature_info: Dictionary containing 'title' and 'description'.
       """
       await farnsworth.ui.notifications.send_notification(websocket, feature_info)
   ```

#### In `farnsworth/web/static/js/notifications.js`:

1. **Function: `displayNotification`**

   ```javascript
   function displayNotification(message) {
       // Display the notification message in a UI element (e.g., toast or modal).
       alert(`New Feature Notification: ${message}`);
   }
   ```

2. **WebSocket Connection Handling**

   ```javascript
   const socket = new WebSocket("ws://localhost:8000/ws");

   socket.onmessage = function(event) {
       displayNotification(event.data);
   };
   ```

---

### Imports Required:

- From `farnsworth/ui/notifications.py`:
  - `Dict` from `typing`
  - `WebSocket` from `fastapi`
  
- In `farnsworth/web/server.py`:
  - Import the notification functions:
    ```python
    from farnsworth.ui.notifications import send_notification
    ```

---

### Integration Points:

1. **Modify `farnsworth/web/server.py`:**
   - Add WebSocket endpoint for notifications.
   
   ```python
   from fastapi import FastAPI, WebSocket

   app = FastAPI()

   @app.websocket("/ws")
   async def websocket_endpoint(websocket: WebSocket):
       await websocket.accept()
       # Example feature info to send as notification
       feature_info = {"title": "New Feature", "description": "Check out the new updates!"}
       await notify_new_feature(websocket, feature_info)
   ```

2. **Modify HTML Template:**
   - Ensure `farnsworth/web/static/js/notifications.js` is included in the base template.

---

### Test Commands:

1. **Run the FastAPI server:**

   ```bash
   uvicorn farnsworth.web.server:app --reload
   ```

2. **Open a browser and navigate to `http://localhost:8000`.**
3. **Check for the notification alert pop-up indicating the new feature message.**

---

This plan outlines the creation of a simple UI notification system within the Farnsworth architecture, ensuring users are promptly informed about new features using WebSockets for real-time updates.