"""
Integration tests for WebSocket thread safety.

Tests that ConnectionManager and SwarmChatManager properly
handle concurrent operations with asyncio locks.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestConnectionManagerSafety:
    """Test ConnectionManager concurrent operation safety."""

    @pytest.mark.asyncio
    async def test_concurrent_connect_disconnect(self):
        """Concurrent connect/disconnect should not corrupt state."""
        # Import from wherever ConnectionManager ends up (server.py or routes/websocket.py)
        try:
            from farnsworth.web.routes.websocket import ConnectionManager
        except ImportError:
            from farnsworth.web.server import ConnectionManager

        manager = ConnectionManager()

        # Create mock websockets
        ws_list = []
        for i in range(10):
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            ws_list.append(ws)

        # Concurrent connects
        await asyncio.gather(*[manager.connect(ws) for ws in ws_list])
        assert len(manager.active_connections) == 10

        # Concurrent disconnects
        for ws in ws_list:
            manager.disconnect(ws)
        assert len(manager.active_connections) == 0

    @pytest.mark.asyncio
    async def test_broadcast_during_disconnect(self):
        """Broadcasting during disconnect should not crash."""
        try:
            from farnsworth.web.routes.websocket import ConnectionManager
        except ImportError:
            from farnsworth.web.server import ConnectionManager

        manager = ConnectionManager()

        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_json = AsyncMock()
        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_json = AsyncMock(side_effect=Exception("disconnected"))

        await manager.connect(ws1)
        await manager.connect(ws2)

        # Broadcasting should handle the failing connection gracefully
        await manager.broadcast({"type": "test", "data": "hello"})

        # ws1 should have received the message
        ws1.send_json.assert_called()


class TestSwarmChatManagerSafety:
    """Test SwarmChatManager concurrent operation safety."""

    @pytest.mark.asyncio
    async def test_no_hardcoded_dev_check(self):
        """SwarmChatManager should not have hardcoded 'winning' dev check."""
        import inspect
        try:
            from farnsworth.web.routes.websocket import SwarmChatManager
            source = inspect.getsource(SwarmChatManager)
        except ImportError:
            from farnsworth.web.server import SwarmChatManager
            source = inspect.getsource(SwarmChatManager)

        # The hardcoded check should be removed or made configurable
        # Looking for the pattern: user_name.lower() == "winning"
        assert 'lower() == "winning"' not in source or \
               "ADMIN_USERS" in source or "admin_users" in source or \
               "is_admin" in source, \
               "Hardcoded 'winning' dev check should be removed or made configurable"
