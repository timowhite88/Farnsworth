"""
x402 Configuration Module.
"""

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class X402Config:
    """Configuration for x402 protocol."""

    # Enable/disable
    enabled: bool = False

    # Network (Base is primary for x402)
    network: str = "base"

    # Client-side (paying for APIs via Bankr)
    bankr_api_key: str = ""
    cost_per_request: Decimal = Decimal("0.01")  # USDC
    daily_spend_limit: Decimal = Decimal("10.00")  # USD

    # Server-side (receiving payments)
    receiver_wallet: str = ""  # Your wallet address on Base
    default_endpoint_price: Decimal = Decimal("0.001")  # USDC

    # Verification
    require_verification: bool = True
    verification_timeout: float = 30.0  # seconds

    @classmethod
    def from_env(cls) -> "X402Config":
        """Create config from environment variables."""
        return cls(
            enabled=os.environ.get("X402_ENABLED", "false").lower() == "true",
            network=os.environ.get("X402_NETWORK", "base"),
            bankr_api_key=os.environ.get("BANKR_API_KEY", ""),
            cost_per_request=Decimal(os.environ.get("X402_COST_PER_REQUEST", "0.01")),
            daily_spend_limit=Decimal(os.environ.get("X402_DAILY_SPEND_LIMIT", "10.00")),
            receiver_wallet=os.environ.get("X402_RECEIVER_WALLET", ""),
            default_endpoint_price=Decimal(os.environ.get("X402_DEFAULT_PRICE", "0.001")),
        )

    def validate(self) -> list:
        """Validate configuration. Returns list of errors."""
        errors = []

        if self.enabled:
            if not self.bankr_api_key:
                errors.append("BANKR_API_KEY required for x402 client")

            if not self.receiver_wallet and self.enabled:
                logger.warning("X402_RECEIVER_WALLET not set - server-side monetization disabled")

        return errors


_config: Optional[X402Config] = None


def get_x402_config() -> X402Config:
    """Get or create the global x402 config."""
    global _config
    if _config is None:
        _config = X402Config.from_env()
    return _config
