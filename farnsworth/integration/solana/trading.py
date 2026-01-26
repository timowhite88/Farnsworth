"""
Farnsworth Solana Trading Core.

"Money is just data that the community has agreed is important."
"""

import os
import json
import aiohttp
import base58
from loguru import logger
from typing import Dict, Any, List, Optional
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair # type: ignore
from solders.pubkey import Pubkey # type: ignore
from solders.transaction import VersionedTransaction # type: ignore

class SolanaTradingSkill:
    def __init__(self, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        self.rpc_url = rpc_url
        self.client = AsyncClient(rpc_url)
        self.keypair: Optional[Keypair] = None
        self._load_wallet()

    def _load_wallet(self):
        """Load Solana private key from environment."""
        pk_str = os.environ.get("SOLANA_PRIVATE_KEY")
        if pk_str:
            try:
                # Expecting base58 encoded private key
                self.keypair = Keypair.from_base58_string(pk_str)
                logger.info(f"Solana: Wallet loaded: {self.keypair.pubkey()}")
            except Exception as e:
                logger.error(f"Solana: Failed to load wallet: {e}")

    async def get_balance(self, pubkey: Optional[str] = None) -> float:
        """Get SOL balance for a pubkey or the loaded wallet."""
        address = Pubkey.from_string(pubkey) if pubkey else self.keypair.pubkey()
        resp = await self.client.get_balance(address)
        return resp.value / 10**9

    # --- Jupiter Aggregator ---
    async def jupiter_swap(self, input_mint: str, output_mint: str, amount_indices: int, slippage_bps: int = 50) -> Dict:
        """Create and execute a swap via Jupiter V6 API."""
        if not self.keypair:
            return {"error": "Wallet not configured."}

        quote_url = f"https://quote-api.jup.ag/v6/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount_indices}&slippageBps={slippage_bps}"
        
        async with aiohttp.ClientSession() as session:
            # 1. Get Quote
            async with session.get(quote_url) as resp:
                if resp.status != 200:
                    return {"error": f"Jupiter Quote Error: {await resp.text()}"}
                quote_data = await resp.json()

            # 2. Get Swap Transaction
            swap_url = "https://quote-api.jup.ag/v6/swap"
            payload = {
                "quoteResponse": quote_data,
                "userPublicKey": str(self.keypair.pubkey()),
                "wrapAndUnwrapSol": True
            }
            async with session.post(swap_url, json=payload) as resp:
                if resp.status != 200:
                    return {"error": f"Jupiter Swap API Error: {await resp.text()}"}
                swap_data = await resp.json()

            # 3. Sign and Send
            raw_tx = base58.b58decode(swap_data["swapTransaction"])
            tx = VersionedTransaction.from_bytes(raw_tx)
            
            # Note: Implementation for signing and sending versioned transactions
            # Requires solders and solana-py 0.30+
            # This is a high-level representation
            return {"status": "Transaction generated", "tx_base58": swap_data["swapTransaction"], "quote": quote_data}

    # --- Pump.fun Trading ---
    async def pump_fun_trade(self, action: str, mint: str, amount: float, denominated_in_sol: bool = True) -> Dict:
        """Execute a trade on Pump.fun via PumpPortal."""
        if not self.keypair:
            return {"error": "Wallet not configured."}

        url = "https://pumpportal.fun/api/trade-local"
        payload = {
            "publicKey": str(self.keypair.pubkey()),
            "action": action, # "buy" or "sell"
            "mint": mint,
            "amount": amount,
            "denominatedInSol": denominated_in_sol,
            "slippage": 10,
            "priorityFee": 0.0001,
            "pool": "pump"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    return {"error": f"PumpPortal Error: {await resp.text()}"}
                tx_data = await resp.read()
                # Sign and send logic here
                return {"status": "Pump trade tx generated", "mint": mint, "action": action}

    # --- Meteora LP ---
    async def meteora_info(self, pair_address: str) -> Dict:
        """Fetch Meteora pool/pair information."""
        # Meteora Dynamic Pools / DLMM
        url = f"https://api.meteora.ag/pair/{pair_address}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"error": f"Meteora API error: {resp.status}"}

solana_trader = SolanaTradingSkill()
