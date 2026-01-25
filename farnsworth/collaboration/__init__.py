"""
Farnsworth Collaboration Module

Team collaboration features for multi-user Farnsworth deployments:
- Shared Memory Pools
- Multi-User Support
- Permission-Based Access Control
- Collaborative Sessions
"""

from farnsworth.collaboration.shared_memory import (
    SharedMemoryPool,
    MemoryPermission,
    MemoryAccess,
)
from farnsworth.collaboration.multi_user import (
    UserManager,
    UserProfile,
    UserSession,
    UserRole,
)
from farnsworth.collaboration.permissions import (
    PermissionManager,
    Permission,
    PermissionLevel,
    AccessControl,
)
from farnsworth.collaboration.sessions import (
    CollaborativeSession,
    SessionManager,
    SessionEvent,
    SessionState,
)

__all__ = [
    # Shared Memory
    "SharedMemoryPool",
    "MemoryPermission",
    "MemoryAccess",
    # Multi-User
    "UserManager",
    "UserProfile",
    "UserSession",
    "UserRole",
    # Permissions
    "PermissionManager",
    "Permission",
    "PermissionLevel",
    "AccessControl",
    # Sessions
    "CollaborativeSession",
    "SessionManager",
    "SessionEvent",
    "SessionState",
]
