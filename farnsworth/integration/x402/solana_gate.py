"""
x402 Solana Payment Gate — Premium Quantum Trading API

Implements the x402 HTTP 402 protocol for Solana-native payments.
Charges 1 SOL per query for quantum trading intelligence.

x402 Protocol Flow (V2):
1. Client sends request to premium endpoint
2. Server returns 402 with PAYMENT-REQUIRED header (base64 JSON)
3. Client pays 1 SOL to ecosystem wallet
4. Client retries with PAYMENT-SIGNATURE header (base64 JSON with tx sig)
5. Server verifies on-chain via Helius/RPC, returns data with PAYMENT-RESPONSE header

Solana Network ID: solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp (mainnet-beta)
"""

import os
import json
import time
import base64
import hashlib
import asyncio
import aiohttp
import logging
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Set

logger = logging.getLogger("x402.solana_gate")

# =============================================================================
# CONSTANTS
# =============================================================================

# Solana mainnet-beta network ID per x402 spec
SOLANA_NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"

# Ecosystem wallet — receives SOL payments
ECOSYSTEM_WALLET = os.getenv(
    "X402_SOLANA_RECEIVER",
    "3fSS5RVErbgcJEDCQmCXpKsD2tWqfhxFZtkDUB8qw"
)

# SOL is the native asset on Solana (no mint address needed)
SOL_ASSET = "native"

# 1 SOL in lamports
SOL_DECIMALS = 9
QUERY_PRICE_SOL = 1.0
QUERY_PRICE_LAMPORTS = int(QUERY_PRICE_SOL * (10 ** SOL_DECIMALS))  # 1_000_000_000

# RPC + Helius
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")

# x402 header names (V2 spec)
HEADER_PAYMENT_REQUIRED = "X-PAYMENT"  # 402 response: base64 JSON payment requirements
HEADER_PAYMENT_SIGNATURE = "X-PAYMENT"  # retry request: base64 JSON with tx signature
HEADER_PAYMENT_RESPONSE = "X-PAYMENT-RESPONSE"  # success response: base64 JSON receipt

# Discovery
FARNSWORTH_API_BASE = os.getenv("FARNSWORTH_API_URL", "https://ai.farnsworth.cloud")

# Max payment age (prevent old tx replays)
MAX_PAYMENT_AGE_SECONDS = 300  # 5 minutes


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PaymentReceipt:
    """Record of a verified x402 payment."""
    tx_signature: str
    payer_wallet: str
    amount_lamports: int
    amount_sol: float
    endpoint: str
    token_queried: str
    verified_at: str
    slot: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class X402SolanaStats:
    """Track x402 payment statistics."""
    total_queries: int = 0
    total_revenue_lamports: int = 0
    total_402_issued: int = 0
    total_verified: int = 0
    total_rejected: int = 0
    unique_payers: int = 0
    _payer_set: Set[str] = field(default_factory=set, repr=False)
    recent_payments: List[dict] = field(default_factory=list, repr=False)

    def record_402(self):
        self.total_402_issued += 1

    def record_verified(self, receipt: PaymentReceipt):
        self.total_queries += 1
        self.total_verified += 1
        self.total_revenue_lamports += receipt.amount_lamports
        self._payer_set.add(receipt.payer_wallet)
        self.unique_payers = len(self._payer_set)
        self.recent_payments.append(receipt.to_dict())
        # Keep last 200
        if len(self.recent_payments) > 200:
            self.recent_payments = self.recent_payments[-200:]

    def record_rejected(self):
        self.total_rejected += 1

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_revenue_sol": self.total_revenue_lamports / (10 ** SOL_DECIMALS),
            "total_revenue_lamports": self.total_revenue_lamports,
            "total_402_issued": self.total_402_issued,
            "total_verified": self.total_verified,
            "total_rejected": self.total_rejected,
            "unique_payers": self.unique_payers,
            "recent_payments": self.recent_payments[-10:],  # Last 10 for display
        }


# =============================================================================
# SOLANA PAYMENT VERIFIER
# =============================================================================

class SolanaPaymentVerifier:
    """
    Verifies SOL transfer transactions on Solana mainnet.
    Uses Helius parsed API (primary) + Solana RPC fallback.
    Prevents replay attacks via used_signatures set.
    """

    def __init__(self):
        self._used_signatures: Set[str] = set()
        self._sig_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "web", "data", "x402_signatures.json"
        )
        self._load_signatures()

    def _load_signatures(self):
        """Load used signatures from disk."""
        try:
            if os.path.exists(self._sig_file):
                with open(self._sig_file, 'r') as f:
                    data = json.load(f)
                    self._used_signatures = set(data.get("signatures", []))
                    logger.info(f"Loaded {len(self._used_signatures)} used x402 signatures")
        except Exception as e:
            logger.warning(f"Failed to load x402 signatures: {e}")

    def _save_signatures(self):
        """Persist used signatures to disk."""
        try:
            os.makedirs(os.path.dirname(self._sig_file), exist_ok=True)
            with open(self._sig_file, 'w') as f:
                # Keep last 5000 signatures
                sigs = list(self._used_signatures)[-5000:]
                json.dump({"signatures": sigs, "updated": datetime.now().isoformat()}, f)
        except Exception as e:
            logger.warning(f"Failed to save x402 signatures: {e}")

    def is_replay(self, tx_signature: str) -> bool:
        """Check if this signature was already used."""
        return tx_signature in self._used_signatures

    def mark_used(self, tx_signature: str):
        """Mark a signature as used."""
        self._used_signatures.add(tx_signature)
        self._save_signatures()

    async def verify_sol_transfer(
        self,
        tx_signature: str,
        expected_recipient: str = ECOSYSTEM_WALLET,
        min_amount_lamports: int = QUERY_PRICE_LAMPORTS,
    ) -> Dict[str, Any]:
        """
        Verify a SOL transfer transaction on-chain.

        Returns:
            {
                "valid": bool,
                "payer": str (sender wallet),
                "amount_lamports": int,
                "slot": int,
                "error": str (if invalid)
            }
        """
        result = {"valid": False, "payer": "", "amount_lamports": 0, "slot": 0, "error": None}

        # Replay check
        if self.is_replay(tx_signature):
            result["error"] = "Transaction signature already used (replay attack)"
            return result

        try:
            # Try Helius first
            if HELIUS_API_KEY:
                helius_result = await self._verify_via_helius(
                    tx_signature, expected_recipient, min_amount_lamports
                )
                if helius_result:
                    return helius_result

            # Fallback to RPC
            return await self._verify_via_rpc(
                tx_signature, expected_recipient, min_amount_lamports
            )
        except Exception as e:
            logger.error(f"x402 verification error: {e}")
            result["error"] = str(e)
            return result

    async def _verify_via_helius(
        self, tx_signature: str, expected_recipient: str, min_amount_lamports: int
    ) -> Optional[Dict[str, Any]]:
        """Verify using Helius parsed transaction API."""
        try:
            url = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json={"transactions": [tx_signature]}) as resp:
                    if resp.status != 200:
                        return None

                    data = await resp.json()
                    if not data:
                        return {"valid": False, "error": "Transaction not found", "payer": "", "amount_lamports": 0, "slot": 0}

                    tx = data[0]

                    # Check native SOL transfers
                    native_transfers = tx.get("nativeTransfers", [])
                    for transfer in native_transfers:
                        if transfer.get("toUserAccount") == expected_recipient:
                            amount = transfer.get("amount", 0)  # in lamports
                            if amount >= min_amount_lamports:
                                return {
                                    "valid": True,
                                    "payer": transfer.get("fromUserAccount", ""),
                                    "amount_lamports": amount,
                                    "slot": tx.get("slot", 0),
                                    "error": None,
                                }
                            else:
                                return {
                                    "valid": False,
                                    "payer": transfer.get("fromUserAccount", ""),
                                    "amount_lamports": amount,
                                    "slot": tx.get("slot", 0),
                                    "error": f"Insufficient: {amount / 1e9:.4f} SOL (need {min_amount_lamports / 1e9} SOL)",
                                }

                    return {
                        "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                        "error": "No SOL transfer to ecosystem wallet found in transaction",
                    }
        except Exception as e:
            logger.warning(f"Helius x402 verification failed: {e}")
            return None

    async def _verify_via_rpc(
        self, tx_signature: str, expected_recipient: str, min_amount_lamports: int
    ) -> Dict[str, Any]:
        """Verify using Solana RPC getTransaction (fallback)."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getTransaction",
                    "params": [
                        tx_signature,
                        {
                            "encoding": "jsonParsed",
                            "commitment": "confirmed",
                            "maxSupportedTransactionVersion": 0,
                        }
                    ]
                }

                async with session.post(SOLANA_RPC_URL, json=payload) as resp:
                    data = await resp.json()

                    if "error" in data:
                        return {
                            "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                            "error": f"RPC error: {data['error'].get('message', 'Unknown')}",
                        }

                    result = data.get("result")
                    if not result:
                        return {
                            "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                            "error": "Transaction not found or not confirmed",
                        }

                    # Check tx success
                    if result.get("meta", {}).get("err") is not None:
                        return {
                            "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                            "error": "Transaction failed on-chain",
                        }

                    # Check SOL balance changes
                    account_keys = result.get("transaction", {}).get("message", {}).get("accountKeys", [])
                    pre_balances = result.get("meta", {}).get("preBalances", [])
                    post_balances = result.get("meta", {}).get("postBalances", [])

                    # Find the recipient account and check balance increase
                    for i, key_info in enumerate(account_keys):
                        pubkey = key_info.get("pubkey", key_info) if isinstance(key_info, dict) else key_info
                        if pubkey == expected_recipient and i < len(pre_balances) and i < len(post_balances):
                            received = post_balances[i] - pre_balances[i]
                            if received >= min_amount_lamports:
                                # Find the payer (first signer)
                                payer = ""
                                for k in account_keys:
                                    kp = k.get("pubkey", k) if isinstance(k, dict) else k
                                    signer = k.get("signer", False) if isinstance(k, dict) else False
                                    if signer and kp != expected_recipient:
                                        payer = kp
                                        break

                                return {
                                    "valid": True,
                                    "payer": payer,
                                    "amount_lamports": received,
                                    "slot": result.get("slot", 0),
                                    "error": None,
                                }
                            elif received > 0:
                                return {
                                    "valid": False, "payer": "", "amount_lamports": received, "slot": 0,
                                    "error": f"Insufficient: {received / 1e9:.4f} SOL (need {min_amount_lamports / 1e9} SOL)",
                                }

                    return {
                        "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                        "error": "No SOL transfer to ecosystem wallet found",
                    }

        except Exception as e:
            logger.error(f"RPC x402 verification failed: {e}")
            return {
                "valid": False, "payer": "", "amount_lamports": 0, "slot": 0,
                "error": f"Verification failed: {e}",
            }


# =============================================================================
# x402 PROTOCOL HELPERS
# =============================================================================

def build_payment_required_payload(endpoint: str, resource_description: str = "") -> dict:
    """
    Build the x402 V2 PAYMENT-REQUIRED payload.
    Returned as base64 JSON in the X-PAYMENT header of 402 responses.
    """
    return {
        "x402Version": 2,
        "accepts": [
            {
                "scheme": "exact",
                "network": SOLANA_NETWORK,
                "maxAmountRequired": str(QUERY_PRICE_LAMPORTS),
                "resource": f"{FARNSWORTH_API_BASE}{endpoint}",
                "description": resource_description or "Farnsworth Quantum Trading Intelligence — 1 SOL per query",
                "mimeType": "application/json",
                "payTo": ECOSYSTEM_WALLET,
                "asset": SOL_ASSET,
                "maxTimeoutSeconds": MAX_PAYMENT_AGE_SECONDS,
                "extra": {
                    "name": "Farnsworth AI Swarm",
                    "pricing": f"{QUERY_PRICE_SOL} SOL per query",
                    "capabilities": [
                        "quantum_simulation",
                        "ema_momentum",
                        "collective_intelligence",
                        "signal_fusion",
                        "scenario_analysis",
                    ],
                }
            }
        ],
        "error": "",
    }


def encode_x402_header(payload: dict) -> str:
    """Encode a payload as base64 JSON for x402 headers."""
    return base64.b64encode(json.dumps(payload).encode()).decode()


def decode_x402_header(header_value: str) -> Optional[dict]:
    """Decode a base64 JSON x402 header."""
    try:
        return json.loads(base64.b64decode(header_value).decode())
    except Exception:
        return None


def build_payment_response(receipt: PaymentReceipt) -> dict:
    """Build the PAYMENT-RESPONSE header payload for successful payment."""
    return {
        "x402Version": 2,
        "success": True,
        "transaction": receipt.tx_signature,
        "payer": receipt.payer_wallet,
        "network": SOLANA_NETWORK,
        "settled": True,
    }


# =============================================================================
# x402 DISCOVERY METADATA
# =============================================================================

def get_x402_discovery_manifest() -> dict:
    """
    x402 discovery manifest for registration with hubs, bazaars, and i1l.store.

    This metadata lets x402 clients discover our paid endpoints.
    Register by POSTing to x402 bazaar/discovery endpoints or serving at
    /.well-known/x402.json
    """
    return {
        "x402Version": 2,
        "provider": {
            "name": "Farnsworth AI Swarm",
            "description": (
                "Quantum-enhanced trading intelligence powered by IBM Quantum simulation, "
                "EMA momentum analysis, and multi-agent collective deliberation. "
                "Submit any Solana token address, receive a comprehensive trading signal "
                "with direction, confidence, strength, and reasoning."
            ),
            "url": FARNSWORTH_API_BASE,
            "logo": f"{FARNSWORTH_API_BASE}/static/logo.png",
            "category": "trading",
            "tags": ["solana", "quantum", "trading", "defi", "ai", "signals"],
        },
        "endpoints": [
            {
                "path": "/api/x402/quantum/analyze",
                "method": "POST",
                "description": "Quantum trading signal for any Solana token",
                "price": str(QUERY_PRICE_LAMPORTS),
                "asset": SOL_ASSET,
                "network": SOLANA_NETWORK,
                "payTo": ECOSYSTEM_WALLET,
                "requestSchema": {
                    "type": "object",
                    "required": ["token_address"],
                    "properties": {
                        "token_address": {
                            "type": "string",
                            "description": "Solana token mint address to analyze",
                        },
                    },
                },
                "responseSchema": {
                    "type": "object",
                    "properties": {
                        "signal": {"type": "object", "description": "Full quantum trading signal"},
                        "scenarios": {"type": "object", "description": "Multi-scenario quantum analysis"},
                        "accuracy": {"type": "object", "description": "Historical signal accuracy stats"},
                    },
                },
            },
        ],
        "networks": [SOLANA_NETWORK],
        "termsOfService": f"{FARNSWORTH_API_BASE}/terms",
        "contact": "farnsworth@ai.farnsworth.cloud",
    }


# =============================================================================
# PREMIUM QUANTUM QUERY ENGINE
# =============================================================================

async def run_premium_quantum_analysis(token_address: str) -> Dict[str, Any]:
    """
    Run the full quantum analysis pipeline for a premium x402 query.

    Returns comprehensive trading data:
    - Quantum trading signal (EMA + quantum simulation + collective)
    - Multi-scenario quantum analysis
    - Signal accuracy stats
    - Cross-token correlations (if available)
    """
    result = {
        "token_address": token_address,
        "timestamp": datetime.now().isoformat(),
        "powered_by": "Farnsworth AI Swarm — Quantum Trading Cortex",
    }

    # 1. Generate quantum trading signal
    try:
        from farnsworth.core.quantum_trading import get_quantum_cortex
        cortex = get_quantum_cortex()

        if cortex is None:
            # Cortex not initialized yet — try initializing
            from farnsworth.core.quantum_trading import initialize_quantum_cortex
            cortex = await initialize_quantum_cortex()

        # Fetch live price data from Birdeye/Jupiter for this token
        price_history = await _fetch_price_history(token_address)
        current_price = price_history[-1] if price_history else 0.0

        # Generate the fused signal
        signal = await cortex.generate_signal(
            token_address=token_address,
            price_history=price_history,
            current_price=current_price,
        )
        result["signal"] = signal.to_dict()

        # 2. Run scenario analysis
        try:
            scenarios = await cortex.quantum_scenario_analysis(
                token_address, price_history
            )
            result["scenarios"] = scenarios
        except Exception as e:
            result["scenarios"] = {"error": str(e)}

        # 3. Accuracy stats (public — proves value)
        try:
            accuracy = cortex.accuracy_tracker.get_accuracy_stats()
            result["accuracy"] = accuracy
        except Exception:
            result["accuracy"] = {}

        # 4. Correlations (if any discovered)
        try:
            corr_list = []
            for key, corr in cortex.correlations.items():
                if token_address in key:
                    corr_list.append(corr.to_dict())
            if corr_list:
                result["correlations"] = corr_list
        except Exception:
            pass

    except Exception as e:
        logger.error(f"x402 quantum analysis failed: {e}")
        result["signal"] = {"error": str(e)}

    # 5. Add price context
    try:
        result["price_data"] = {
            "current_price": current_price if 'current_price' in dir() else 0.0,
            "price_points": len(price_history) if 'price_history' in dir() else 0,
            "source": "birdeye/jupiter",
        }
    except Exception:
        pass

    return result


async def _fetch_price_history(token_address: str, limit: int = 60) -> List[float]:
    """
    Fetch recent price history for a token via Birdeye or Jupiter.
    Returns list of prices (most recent last).
    """
    prices = []

    # Try Birdeye first
    birdeye_key = os.getenv("BIRDEYE_API_KEY", "")
    if birdeye_key:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://public-api.birdeye.so/defi/history_price?address={token_address}&address_type=token&type=1m&time_from={int(time.time()) - 3600}&time_to={int(time.time())}"
                headers = {"X-API-KEY": birdeye_key, "x-chain": "solana"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        items = data.get("data", {}).get("items", [])
                        prices = [item.get("value", 0) for item in items if item.get("value")]
                        if prices:
                            return prices[-limit:]
        except Exception as e:
            logger.debug(f"Birdeye price fetch failed: {e}")

    # Fallback: Jupiter price API (single price, generate synthetic history)
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.jup.ag/price/v2?ids={token_address}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price_data = data.get("data", {}).get(token_address, {})
                    price = float(price_data.get("price", 0))
                    if price > 0:
                        # Return single price point (cortex handles sparse data)
                        return [price]
    except Exception as e:
        logger.debug(f"Jupiter price fetch failed: {e}")

    return prices


# =============================================================================
# SINGLETON
# =============================================================================

_verifier: Optional[SolanaPaymentVerifier] = None
_stats: Optional[X402SolanaStats] = None


def get_solana_verifier() -> SolanaPaymentVerifier:
    """Get or create the Solana payment verifier."""
    global _verifier
    if _verifier is None:
        _verifier = SolanaPaymentVerifier()
    return _verifier


def get_x402_stats() -> X402SolanaStats:
    """Get or create the x402 stats tracker."""
    global _stats
    if _stats is None:
        _stats = X402SolanaStats()
    return _stats
