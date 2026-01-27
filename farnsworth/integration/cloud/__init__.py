"""
Farnsworth Cloud Integration Package

"Good news, everyone! I can manage all your clouds from one place!"

Comprehensive cloud provider integrations for Azure, AWS, GCP.
"""

from farnsworth.integration.cloud.azure_manager import (
    AzureManager,
    azure_manager,
)
from farnsworth.integration.cloud.aws_manager import (
    AWSManager,
    aws_manager,
)


__all__ = [
    "AzureManager",
    "azure_manager",
    "AWSManager",
    "aws_manager",
]
