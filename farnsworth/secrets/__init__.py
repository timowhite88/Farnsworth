"""
Farnsworth Secrets and Credentials Vault

"I've hidden my most dangerous inventions in a vault...
 but I can never remember the combination!"

Multi-provider secrets management with rotation and audit logging.
"""

from farnsworth.secrets.vault_manager import (
    VaultManager,
    Secret,
    SecretVersion,
    SecretType,
)
from farnsworth.secrets.hashicorp_vault import HashiCorpVaultProvider
from farnsworth.secrets.aws_secrets import AWSSecretsProvider
from farnsworth.secrets.azure_keyvault import AzureKeyVaultProvider

__all__ = [
    "VaultManager",
    "Secret",
    "SecretVersion",
    "SecretType",
    "HashiCorpVaultProvider",
    "AWSSecretsProvider",
    "AzureKeyVaultProvider",
]
