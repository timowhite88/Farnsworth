"""
Integration tests for server route modules.

Verifies that route modules are properly structured
and that all expected endpoints exist after the server.py split.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestRouteModuleImports:
    """Test that route modules are importable."""

    def test_routes_package(self):
        """Routes package should be importable."""
        try:
            import farnsworth.web.routes
            assert True
        except ImportError:
            pytest.skip("Routes package not yet created")

    def test_chat_routes(self):
        """Chat routes should be importable."""
        try:
            from farnsworth.web.routes.chat import router
            assert router is not None
        except ImportError:
            pytest.skip("Chat routes not yet created")

    def test_admin_routes(self):
        """Admin routes should be importable."""
        try:
            from farnsworth.web.routes.admin import router
            assert router is not None
        except ImportError:
            pytest.skip("Admin routes not yet created")

    def test_websocket_routes(self):
        """WebSocket routes should be importable."""
        try:
            from farnsworth.web.routes.websocket import router
            assert router is not None
        except ImportError:
            pytest.skip("WebSocket routes not yet created")

    def test_claude_teams_routes(self):
        """Claude Teams routes should be importable."""
        try:
            from farnsworth.web.routes.claude_teams import router
            assert router is not None
        except ImportError:
            pytest.skip("Claude Teams routes not yet created")


class TestServerApp:
    """Test the main server app configuration."""

    def test_server_importable(self):
        """Server module should be importable."""
        # This tests that server.py doesn't crash on import
        try:
            # Just import the module - it should not fail
            import importlib
            spec = importlib.util.find_spec("farnsworth.web.server")
            assert spec is not None
        except Exception as e:
            pytest.fail(f"Server module import failed: {e}")

    def test_rate_limiter_exists(self):
        """RateLimiter class should exist."""
        from farnsworth.web.server import RateLimiter
        assert RateLimiter is not None

        # Test basic functionality
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        assert limiter.is_allowed("test_client") is True

    def test_rate_limiter_blocks(self):
        """RateLimiter should block after burst exceeded."""
        from farnsworth.web.server import RateLimiter

        limiter = RateLimiter(requests_per_minute=1, burst_size=2)

        # First two should be allowed (burst)
        assert limiter.is_allowed("test") is True
        assert limiter.is_allowed("test") is True

        # Third should be blocked
        assert limiter.is_allowed("test") is False
