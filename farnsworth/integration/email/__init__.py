"""
Farnsworth Email Integration Package

"Good news, everyone! I can now manage your email across all providers!"

Comprehensive email integration for Office 365, Google Workspace, and more.
"""

from farnsworth.integration.email.office365 import (
    Office365Integration,
    office365_integration,
)
from farnsworth.integration.email.google_workspace import (
    GoogleWorkspaceIntegration,
    google_workspace_integration,
)
from farnsworth.integration.email.mailbox_filter import (
    MailboxFilter,
    FilterRule,
    mailbox_filter,
)


__all__ = [
    "Office365Integration",
    "office365_integration",
    "GoogleWorkspaceIntegration",
    "google_workspace_integration",
    "MailboxFilter",
    "FilterRule",
    "mailbox_filter",
]
