"""
Integration tests for Nexus signal handler safety.

Tests that both sync and async handlers are safely invoked
via _safe_invoke_handler() - fixes the AGI v1.8 bug where
type-based handlers were called directly without the wrapper.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


@pytest.fixture
def safe_invoke_handler():
    """Import _safe_invoke_handler from nexus."""
    from farnsworth.core.nexus import _safe_invoke_handler
    return _safe_invoke_handler


class TestSafeInvokeHandler:
    """Test _safe_invoke_handler correctly handles sync/async handlers."""

    @pytest.mark.asyncio
    async def test_async_handler_invoked(self, safe_invoke_handler):
        """Async handlers should be awaited properly."""
        handler = AsyncMock(return_value="async_result")
        signal = MagicMock()

        result = await safe_invoke_handler(handler, signal)

        handler.assert_called_once_with(signal)
        assert result == "async_result"

    @pytest.mark.asyncio
    async def test_sync_handler_invoked(self, safe_invoke_handler):
        """Sync handlers should be called without error."""
        handler = MagicMock(return_value="sync_result")
        signal = MagicMock()

        result = await safe_invoke_handler(handler, signal)

        handler.assert_called_once_with(signal)
        assert result == "sync_result"

    @pytest.mark.asyncio
    async def test_handler_exception_caught(self, safe_invoke_handler):
        """Exceptions in handlers should be caught, not propagated."""
        handler = MagicMock(side_effect=ValueError("test error"))
        signal = MagicMock()

        result = await safe_invoke_handler(handler, signal)

        # Should return None on error, not raise
        assert result is None

    @pytest.mark.asyncio
    async def test_async_handler_exception_caught(self, safe_invoke_handler):
        """Exceptions in async handlers should be caught."""
        handler = AsyncMock(side_effect=RuntimeError("async error"))
        signal = MagicMock()

        result = await safe_invoke_handler(handler, signal)

        assert result is None


class TestSemanticBroadcastFix:
    """Test that semantic_broadcast uses safe handler invocation for type handlers."""

    @pytest.mark.asyncio
    async def test_type_handlers_use_safe_invoke(self):
        """Verify type-based handlers go through _safe_invoke_handler."""
        # Read the source to confirm the fix is applied
        import inspect
        from farnsworth.core import nexus

        source = inspect.getsource(nexus)

        # The bug was: *[h(signal) for h in handlers]
        # The fix is: *[_safe_invoke_handler(h, signal) for h in handlers]
        # Check that the pattern with type-based dispatch uses safe invoke
        assert "_safe_invoke_handler(h, signal) for h in handlers" in source, (
            "Type-based handlers in semantic_broadcast should use _safe_invoke_handler"
        )
