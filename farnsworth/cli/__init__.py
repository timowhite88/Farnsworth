"""
Farnsworth CLI Module

Direct-to-user interfaces for easy interaction with Farnsworth.

AGI v1.8.4 Additions:
- RichCLI: Enhanced TUI with Rich/Textual
- SwarmSession: CLI-to-Swarm A2A sessions
"""

from .user_cli import FarnsworthCLI, run_user_cli
from .interactive import InteractiveShell
from .quick_actions import QuickActions
from .rich_cli import RichCLI, run_rich_cli
from .swarm_session import (
    SwarmSession,
    SwarmSessionManager,
    create_swarm_session,
    get_session_manager,
)

__all__ = [
    # Original
    "FarnsworthCLI",
    "run_user_cli",
    "InteractiveShell",
    "QuickActions",
    # AGI v1.8.4
    "RichCLI",
    "run_rich_cli",
    "SwarmSession",
    "SwarmSessionManager",
    "create_swarm_session",
    "get_session_manager",
]
