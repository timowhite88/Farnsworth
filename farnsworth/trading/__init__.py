"""Farnsworth Degen Trader v3 - Collective Intelligence Memecoin Trading."""

from .degen_trader import (
    DegenTrader, create_wallet, start_trader,
    CopyTradeEngine, XSentinelMonitor, QuantumTradeOracle, TradingMemory,
)

__all__ = [
    "DegenTrader", "create_wallet", "start_trader",
    "CopyTradeEngine", "XSentinelMonitor", "QuantumTradeOracle", "TradingMemory",
]
