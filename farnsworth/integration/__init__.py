"""
Farnsworth Integration Module

External tool and capability integration with:
- Dynamic tool routing and capability matching
- Multimodal input processing (vision, audio)
- Composio integration for 500+ tools
"""

from farnsworth.integration.tool_router import ToolRouter
from farnsworth.integration.multimodal import MultimodalProcessor

__all__ = [
    "ToolRouter",
    "MultimodalProcessor",
]
