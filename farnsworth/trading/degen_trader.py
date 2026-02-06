"""
Farnsworth Degen Trader v3 - Collective Intelligence Memecoin Trading

High-frequency Solana memecoin trader powered by the Farnsworth swarm.
- Pump.fun WebSocket for instant new launch detection
- Wallet graph analysis to detect cabals and insider coordination
- Quantum-enhanced analysis: IBM Quantum QAOA, quantum random timing, FarsightProtocol
- Deep swarm integration: Grok (X sentiment), DeepSeek (TA), Gemini (multi-factor)
- X/Twitter sentinel: real-time cabal movement detection via Grok
- Copy trading: track most profitable wallets from GMGN/Birdeye, pre-buy on their moves
- Trading memory: learns from every trade via MemorySystem + KnowledgeGraph
- Alchemy RPC for fast on-chain reads/sends
- Jupiter v6 execution with priority fees
- DexScreener + GMGN + Birdeye for comprehensive market data

Requires: pip install solders solana aiohttp websockets
"""

import asyncio
import json
import time
import logging
import os
import base64
import hashlib
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Set
from pathlib import Path
from datetime import datetime, timedelta

import aiohttp

logger = logging.getLogger("farnsworth.trading")

# ============================================================
# CONSTANTS
# ============================================================
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000

JUPITER_QUOTE_URL = "https://quote-api.jup.ag/v6/quote"
JUPITER_SWAP_URL = "https://quote-api.jup.ag/v6/swap"
JUPITER_PRICE_URL = "https://api.jup.ag/price/v2"

DEXSCREENER_TOKENS = "https://api.dexscreener.com/latest/dex/tokens"
DEXSCREENER_BOOSTS = "https://api.dexscreener.com/token-boosts/latest/v1"
DEXSCREENER_PROFILES = "https://api.dexscreener.com/token-profiles/latest/v1"

PUMPFUN_WS_URL = "wss://pumpportal.fun/api/data"
RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

DEFAULT_RPC = "https://api.mainnet-beta.solana.com"

# Copy trading / smart money sources
GMGN_SMART_MONEY_URL = "https://gmgn.ai/defi/quotation/v1/rank/sol/swaps/24h"
BIRDEYE_BASE_URL = "https://public-api.birdeye.so"
PUMPFUN_API_URL = "https://frontend-api-v3.pump.fun"

WALLET_DIR = Path(__file__).parent / ".wallets"
STATE_FILE = Path(__file__).parent / ".trader_state.json"


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class TokenInfo:
    address: str
    symbol: str
    name: str
    pair_address: str
    price_usd: float
    liquidity_usd: float
    volume_24h: float
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    age_minutes: float = 0.0
    holders: int = 0
    score: float = 0.0
    fdv: float = 0.0
    buy_count_5m: int = 0
    sell_count_5m: int = 0
    # Enhanced fields
    cabal_score: float = 0.0       # 0=no cabal, 100=definite cabal
    rug_probability: float = 0.0   # 0-1 from quantum Monte Carlo
    swarm_sentiment: str = ""      # BUY/SKIP/STRONG_BUY
    source: str = "dexscreener"    # dexscreener/pumpfun/raydium
    creator_wallet: str = ""
    top_holders_connected: bool = False


@dataclass
class Position:
    token_address: str
    symbol: str
    entry_price: float
    amount_tokens: float
    amount_sol_spent: float
    entry_time: float
    take_profit_levels: List[float] = field(default_factory=lambda: [2.0, 5.0, 10.0])
    stop_loss: float = 0.5
    partial_sells: int = 0
    source: str = ""  # what detected it


@dataclass
class Trade:
    timestamp: float
    action: str
    token_address: str
    symbol: str
    amount_sol: float
    price_usd: float
    tx_signature: str
    pnl_sol: float = 0.0
    reason: str = ""


@dataclass
class WalletProfile:
    """Profile of a wallet we're tracking."""
    address: str
    funded_by: str = ""           # who sent this wallet its first SOL
    total_sol_received: float = 0
    token_buys: int = 0
    connected_wallets: Set[str] = field(default_factory=set)
    last_seen: float = 0
    is_whale: bool = False
    win_rate: float = 0.0         # historical success


@dataclass
class TrackedWallet:
    """A profitable wallet we're copy trading."""
    address: str
    label: str = ""               # "gmgn_smart_money", "birdeye_top", "pumpfun_winner"
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_hold_minutes: float = 0.0
    last_trade_time: float = 0
    tokens_traded: int = 0
    source: str = ""
    last_checked: float = 0


@dataclass
class TradeMemoryEntry:
    """A single trade's learning data stored in memory."""
    token_address: str
    symbol: str
    action: str                   # "buy" or "sell"
    entry_score: float
    rug_probability: float
    swarm_sentiment: str
    cabal_score: float
    source: str
    outcome: str                  # "win", "loss", "rug", "timeout"
    pnl_multiple: float           # 2.0 = doubled, 0.5 = lost half
    hold_minutes: float
    liquidity_at_entry: float
    age_at_entry: float
    timestamp: float = 0.0


@dataclass
class TraderConfig:
    rpc_url: str = DEFAULT_RPC
    fast_rpc_url: str = ""          # Alchemy/Helius for speed-critical calls
    max_position_sol: float = 0.1
    max_positions: int = 10
    min_liquidity: float = 5000.0
    max_liquidity: float = 500000.0
    min_age_minutes: float = 1.0
    max_age_minutes: float = 60.0
    min_score: float = 60.0
    scan_interval: int = 8          # faster scanning
    slippage_bps: int = 500
    priority_fee_lamports: int = 100000
    reserve_sol: float = 0.05
    whale_wallets: List[str] = field(default_factory=list)
    use_swarm: bool = True
    use_quantum: bool = True        # quantum Monte Carlo for rug detection
    use_pumpfun: bool = True        # pump.fun WebSocket monitoring
    use_wallet_analysis: bool = True  # wallet graph/cabal detection
    cabal_is_bullish: bool = True   # treat coordinated wallets as positive signal
    max_rug_probability: float = 0.6  # skip tokens above this rug score
    # v3: Copy trading
    use_copy_trading: bool = True   # track and copy top wallets
    copy_trade_max_sol: float = 0.05  # smaller size for copy trades
    # v3: X sentinel
    use_x_sentinel: bool = True    # monitor X for cabal signals via Grok
    # v3: Trading memory
    use_trading_memory: bool = True  # learn from past trades


# ============================================================
# WALLET MANAGEMENT
# ============================================================
def create_wallet(name: str = "degen_trader") -> Tuple[str, str]:
    """Generate a new Solana wallet. Returns (pubkey, wallet_file_path)."""
    try:
        from solders.keypair import Keypair
    except ImportError:
        raise RuntimeError("Install solders: pip install solders")

    kp = Keypair()
    pubkey = str(kp.pubkey())

    WALLET_DIR.mkdir(parents=True, exist_ok=True)
    wallet_path = WALLET_DIR / f"{name}.json"

    keypair_bytes = list(bytes(kp))
    wallet_data = {
        "pubkey": pubkey,
        "keypair": keypair_bytes,
        "created_at": datetime.utcnow().isoformat(),
        "name": name,
    }

    wallet_path.write_text(json.dumps(wallet_data, indent=2))
    logger.info(f"Wallet created: {pubkey}")
    logger.info(f"Wallet file saved to: {wallet_path}")
    return pubkey, str(wallet_path)


def load_wallet(name: str = "degen_trader"):
    """Load wallet keypair from file."""
    try:
        from solders.keypair import Keypair
    except ImportError:
        raise RuntimeError("Install solders: pip install solders")

    wallet_path = WALLET_DIR / f"{name}.json"
    if not wallet_path.exists():
        raise FileNotFoundError(f"No wallet at {wallet_path}. Run create_wallet() first.")

    data = json.loads(wallet_path.read_text())
    return Keypair.from_bytes(bytes(data["keypair"]))


# ============================================================
# PUMP.FUN MONITOR
# ============================================================
class PumpFunMonitor:
    """Real-time pump.fun new token and trade monitoring via WebSocket."""

    def __init__(self):
        self.ws = None
        self.new_tokens: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.hot_tokens: Dict[str, dict] = {}  # mint -> trade stats
        self.running = False
        self._task = None

    async def start(self):
        """Connect to pump.fun WebSocket and monitor."""
        self.running = True
        self._task = asyncio.create_task(self._listen())
        logger.info("PumpFun monitor started")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        if self.ws:
            await self.ws.close()

    async def _listen(self):
        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, pump.fun monitor disabled")
            return

        while self.running:
            try:
                async with websockets.connect(PUMPFUN_WS_URL) as ws:
                    self.ws = ws
                    # Subscribe to new token creates
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    logger.info("Subscribed to pump.fun new tokens")

                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            await self._handle_message(data)
                        except json.JSONDecodeError:
                            continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"PumpFun WS error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _handle_message(self, data: dict):
        """Process pump.fun WebSocket messages."""
        if data.get("txType") == "create":
            # New token created on pump.fun
            token_data = {
                "mint": data.get("mint", ""),
                "name": data.get("name", ""),
                "symbol": data.get("symbol", ""),
                "creator": data.get("traderPublicKey", ""),
                "initial_buy_sol": data.get("solAmount", 0) / LAMPORTS_PER_SOL if data.get("solAmount") else 0,
                "timestamp": time.time(),
                "source": "pumpfun",
            }
            if token_data["mint"]:
                try:
                    self.new_tokens.put_nowait(token_data)
                except asyncio.QueueFull:
                    self.new_tokens.get_nowait()  # drop oldest
                    self.new_tokens.put_nowait(token_data)
                logger.info(f"PUMPFUN NEW: ${token_data['symbol']} by {token_data['creator'][:8]}...")

        elif data.get("txType") == "buy" or data.get("txType") == "sell":
            mint = data.get("mint", "")
            if mint:
                if mint not in self.hot_tokens:
                    self.hot_tokens[mint] = {"buys": 0, "sells": 0, "volume_sol": 0, "first_seen": time.time()}
                stats = self.hot_tokens[mint]
                sol_amount = data.get("solAmount", 0) / LAMPORTS_PER_SOL if data.get("solAmount") else 0
                if data["txType"] == "buy":
                    stats["buys"] += 1
                else:
                    stats["sells"] += 1
                stats["volume_sol"] += sol_amount

        # Cleanup old hot tokens (older than 30 min)
        cutoff = time.time() - 1800
        self.hot_tokens = {k: v for k, v in self.hot_tokens.items() if v.get("first_seen", 0) > cutoff}


# ============================================================
# WALLET GRAPH ANALYZER
# ============================================================
class WalletAnalyzer:
    """Analyze wallet connections to detect cabals and insider coordination."""

    def __init__(self, rpc_url: str, fast_rpc_url: str = ""):
        self.rpc_url = rpc_url
        self.fast_rpc = fast_rpc_url or rpc_url
        self.wallet_cache: Dict[str, WalletProfile] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def init_session(self, session: aiohttp.ClientSession):
        self.session = session

    async def analyze_token_holders(self, mint: str) -> Dict:
        """Analyze top holders of a token for connected wallets."""
        result = {
            "top_holders": [],
            "connected_groups": [],
            "cabal_score": 0.0,
            "concentration": 0.0,
            "dev_holds_pct": 0.0,
        }

        try:
            # Get largest token accounts
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTokenLargestAccounts",
                "params": [mint]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                accounts = data.get("result", {}).get("value", [])

            if not accounts:
                return result

            # Get top 10 holder addresses
            holder_addresses = []
            total_supply = 0
            top_10_amount = 0

            for acc in accounts[:10]:
                amount = float(acc.get("uiAmount", 0) or 0)
                top_10_amount += amount
                total_supply = max(total_supply, top_10_amount * 2)  # rough estimate

                # Resolve owner of token account
                owner = await self._get_token_account_owner(acc.get("address", ""))
                if owner:
                    holder_addresses.append({"address": owner, "amount": amount})

            if total_supply > 0:
                result["concentration"] = top_10_amount / total_supply

            # Check if top holders share funding sources
            funding_sources = {}
            for holder in holder_addresses[:5]:  # check top 5 for speed
                funder = await self._trace_funding_source(holder["address"])
                if funder:
                    if funder not in funding_sources:
                        funding_sources[funder] = []
                    funding_sources[funder].append(holder["address"])

            # Detect connected groups (wallets funded by same source)
            for source, wallets in funding_sources.items():
                if len(wallets) >= 2:
                    result["connected_groups"].append({
                        "funder": source[:12] + "...",
                        "wallets": len(wallets),
                        "addresses": [w[:12] + "..." for w in wallets],
                    })

            # Calculate cabal score
            connected_wallets = sum(len(g["addresses"]) for g in result["connected_groups"])
            if connected_wallets >= 4:
                result["cabal_score"] = 90
            elif connected_wallets >= 3:
                result["cabal_score"] = 70
            elif connected_wallets >= 2:
                result["cabal_score"] = 50
            else:
                result["cabal_score"] = 10

            # High concentration amplifies cabal score
            if result["concentration"] > 0.5:
                result["cabal_score"] = min(100, result["cabal_score"] * 1.3)

            result["top_holders"] = [
                {"address": h["address"][:12] + "...", "amount": h["amount"]}
                for h in holder_addresses[:5]
            ]

        except Exception as e:
            logger.debug(f"Wallet analysis error for {mint}: {e}")

        return result

    async def _get_token_account_owner(self, token_account: str) -> Optional[str]:
        """Get the owner wallet of a token account."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [token_account, {"encoding": "jsonParsed"}]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                info = data.get("result", {}).get("value", {})
                if info:
                    return info.get("data", {}).get("parsed", {}).get("info", {}).get("owner")
        except Exception:
            pass
        return None

    async def _trace_funding_source(self, wallet: str) -> Optional[str]:
        """Trace where a wallet got its initial SOL from."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet, {"limit": 5}]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                sigs = data.get("result", [])

            if not sigs:
                return None

            # Check the earliest transaction for funding source
            earliest_sig = sigs[-1].get("signature", "")
            if not earliest_sig:
                return None

            tx_payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTransaction",
                "params": [earliest_sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            }
            async with self.session.post(self.fast_rpc, json=tx_payload) as resp:
                tx_data = await resp.json()
                tx = tx_data.get("result", {})

            if not tx:
                return None

            # Look for SOL transfer to this wallet in the first transaction
            instructions = (
                tx.get("transaction", {}).get("message", {}).get("instructions", [])
            )
            for ix in instructions:
                parsed = ix.get("parsed", {})
                if parsed.get("type") == "transfer":
                    info = parsed.get("info", {})
                    if info.get("destination") == wallet:
                        return info.get("source", "")

        except Exception:
            pass
        return None

    async def check_whale_buys(self, mint: str) -> List[Dict]:
        """Check if known whale wallets are buying a token."""
        whale_buys = []
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [mint, {"limit": 30}]
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                sigs = data.get("result", [])

            # Check recent transactions for large swaps
            for sig_info in sigs[:10]:
                sig = sig_info.get("signature", "")
                if not sig:
                    continue
                tx_payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
                }
                async with self.session.post(self.fast_rpc, json=tx_payload) as resp:
                    tx_data = await resp.json()
                    tx = tx_data.get("result")

                if not tx:
                    continue

                # Check pre/post balances for large SOL movements
                pre_balances = tx.get("meta", {}).get("preBalances", [])
                post_balances = tx.get("meta", {}).get("postBalances", [])
                accounts = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])

                for i, (pre, post) in enumerate(zip(pre_balances, post_balances)):
                    sol_change = (post - pre) / LAMPORTS_PER_SOL
                    if sol_change < -1.0:  # spent > 1 SOL on this token
                        account_key = accounts[i] if i < len(accounts) else {}
                        addr = account_key.get("pubkey", "") if isinstance(account_key, dict) else str(account_key)
                        whale_buys.append({
                            "wallet": addr[:12] + "..." if addr else "unknown",
                            "sol_spent": abs(sol_change),
                            "tx": sig[:20] + "...",
                        })
                        break  # one whale per tx

                await asyncio.sleep(0.1)  # rate limiting

        except Exception as e:
            logger.debug(f"Whale buy check error: {e}")

        return whale_buys


# (QuantumTradeAnalyzer replaced by QuantumTradeOracle in v3)


# ============================================================
# SWARM TRADE INTELLIGENCE
# ============================================================
class SwarmTradeIntelligence:
    """Deep integration with shadow agents for trade decisions."""

    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self._shadow_available = None

    async def multi_agent_analysis(self, token: TokenInfo) -> Dict:
        """Run parallel analysis across specialized agents.

        - Grok: X/Twitter sentiment, trending status
        - DeepSeek: Technical analysis, chart patterns
        - Gemini: Multi-factor holistic review
        Returns combined verdict.
        """
        results = {"verdict": "SKIP", "confidence": 0, "reasons": []}

        try:
            from farnsworth.core.collective.persistent_agent import call_shadow_agent
            self._shadow_available = True
        except ImportError:
            self._shadow_available = False
            return await self._api_fallback(token)

        token_brief = (
            f"${token.symbol} | Price: ${token.price_usd:.8f} | "
            f"Liq: ${token.liquidity_usd:.0f} | Age: {token.age_minutes:.0f}m | "
            f"5m: {token.price_change_5m:+.1f}% | FDV: ${token.fdv:.0f} | "
            f"Buys/Sells(5m): {token.buy_count_5m}/{token.sell_count_5m}"
        )

        # Run agents in parallel for speed
        tasks = {
            "grok": call_shadow_agent(
                "grok",
                f"Quick degen scan: {token_brief}. "
                f"Check X/Twitter for ${token.symbol} mentions, hype, cabal activity. "
                f"Reply ONLY: BUY, STRONG_BUY, or SKIP with 1-line reason.",
                max_tokens=100, timeout=8.0,
            ),
            "deepseek": call_shadow_agent(
                "deepseek",
                f"Technical analysis: {token_brief}. "
                f"Analyze buy/sell ratio, liquidity depth, momentum. "
                f"Reply ONLY: BUY, STRONG_BUY, or SKIP with 1-line reason.",
                max_tokens=100, timeout=8.0,
            ),
            "gemini": call_shadow_agent(
                "gemini",
                f"Risk assessment: {token_brief}. "
                f"Evaluate rug risk, holder distribution signal, growth potential. "
                f"Reply ONLY: BUY, STRONG_BUY, or SKIP with 1-line reason.",
                max_tokens=100, timeout=8.0,
            ),
        }

        done = {}
        for name, task in tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=10.0)
                if result:
                    _, response = result
                    done[name] = response.upper() if response else ""
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Agent {name} timeout/error: {e}")
                done[name] = ""

        # Tally votes
        buy_votes = 0
        strong_buy_votes = 0
        reasons = []

        for agent, response in done.items():
            if "STRONG_BUY" in response:
                strong_buy_votes += 1
                buy_votes += 1
                reasons.append(f"{agent}: STRONG_BUY")
            elif "BUY" in response:
                buy_votes += 1
                reasons.append(f"{agent}: BUY")
            else:
                reasons.append(f"{agent}: SKIP")

        total = len(done) or 1
        results["reasons"] = reasons

        if strong_buy_votes >= 2:
            results["verdict"] = "STRONG_BUY"
            results["confidence"] = 95
        elif buy_votes >= 2:
            results["verdict"] = "BUY"
            results["confidence"] = 75
        elif buy_votes >= 1:
            results["verdict"] = "WEAK_BUY"
            results["confidence"] = 50
        else:
            results["verdict"] = "SKIP"
            results["confidence"] = 30

        logger.info(f"Swarm verdict on {token.symbol}: {results['verdict']} ({buy_votes}/{total} BUY)")
        return results

    async def _api_fallback(self, token: TokenInfo) -> Dict:
        """Fallback when shadow agents aren't available."""
        try:
            prompt = (
                f"Quick degen check on ${token.symbol}: "
                f"Liq ${token.liquidity_usd:.0f}, Age {token.age_minutes:.0f}m, "
                f"5m {token.price_change_5m:+.1f}%, B/S {token.buy_count_5m}/{token.sell_count_5m}, "
                f"FDV ${token.fdv:.0f}. Reply ONLY 'BUY' or 'SKIP'."
            )
            async with self.session.post(
                "http://localhost:8080/api/chat",
                json={"message": prompt, "bot": "Farnsworth", "mode": "quick"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    reply = (data.get("response") or "").upper()
                    if "BUY" in reply:
                        return {"verdict": "BUY", "confidence": 60, "reasons": ["swarm_api: BUY"]}
        except Exception:
            pass
        return {"verdict": "SKIP", "confidence": 0, "reasons": ["swarm_unavailable"]}


# ============================================================
# COPY TRADE ENGINE
# ============================================================
class CopyTradeEngine:
    """Track and copy-trade the most profitable Solana wallets."""

    def __init__(self, session: aiohttp.ClientSession, fast_rpc: str):
        self.session = session
        self.fast_rpc = fast_rpc
        self.tracked_wallets: Dict[str, TrackedWallet] = {}
        self.copy_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._running = False
        self._monitor_task = None
        self._birdeye_key = os.environ.get("BIRDEYE_API_KEY", "")

    async def start(self):
        self._running = True
        await self._discover_top_wallets()
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"CopyTrade engine started, tracking {len(self.tracked_wallets)} wallets")

    async def stop(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()

    async def _discover_top_wallets(self):
        """Find most profitable wallets from GMGN, Birdeye, and DexScreener top holders."""
        await self._fetch_gmgn_wallets()
        if self._birdeye_key:
            await self._fetch_birdeye_wallets()
        await self._fetch_top_holder_wallets()
        logger.info(f"Discovered {len(self.tracked_wallets)} wallets to track")

    async def _fetch_gmgn_wallets(self):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            params = {"orderby": "smartmoney", "direction": "desc"}
            async with self.session.get(
                GMGN_SMART_MONEY_URL, params=params, headers=headers,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"GMGN returned {resp.status}")
                    return
                data = await resp.json()
                rank = data.get("data", {}).get("rank", [])
                for item in rank[:30]:
                    # Extract creator/deployer addresses as smart money
                    creator = item.get("creator_address") or item.get("maker", "")
                    if creator and creator not in self.tracked_wallets:
                        self.tracked_wallets[creator] = TrackedWallet(
                            address=creator,
                            label="gmgn_smart_money",
                            source="gmgn",
                        )
        except Exception as e:
            logger.debug(f"GMGN fetch error: {e}")

    async def _fetch_birdeye_wallets(self):
        try:
            headers = {"X-API-KEY": self._birdeye_key}
            url = f"{BIRDEYE_BASE_URL}/trader/gainers-losers"
            params = {"chain": "solana", "type": "gainers", "sort_by": "PnL", "limit": 20}
            async with self.session.get(
                url, headers=headers, params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                traders = data.get("data", {}).get("items", [])
                for trader in traders:
                    addr = trader.get("address", "")
                    if addr and addr not in self.tracked_wallets:
                        self.tracked_wallets[addr] = TrackedWallet(
                            address=addr,
                            label="birdeye_top_gainer",
                            win_rate=float(trader.get("win_rate", 0)),
                            total_pnl=float(trader.get("pnl", 0)),
                            source="birdeye",
                        )
        except Exception as e:
            logger.debug(f"Birdeye fetch error: {e}")

    async def _fetch_top_holder_wallets(self):
        """Get wallets from top holders of recent DexScreener boosted tokens."""
        try:
            async with self.session.get(DEXSCREENER_BOOSTS) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
                items = data if isinstance(data, list) else []
                sol_tokens = [i for i in items if i.get("chainId") == "solana"][:5]

            for token_item in sol_tokens:
                addr = token_item.get("tokenAddress", "")
                if not addr:
                    continue
                payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTokenLargestAccounts",
                    "params": [addr],
                }
                try:
                    async with self.session.post(self.fast_rpc, json=payload) as resp:
                        data = await resp.json()
                        accounts = data.get("result", {}).get("value", [])
                    for acc in accounts[:3]:
                        owner_payload = {
                            "jsonrpc": "2.0", "id": 1,
                            "method": "getAccountInfo",
                            "params": [acc.get("address", ""), {"encoding": "jsonParsed"}],
                        }
                        async with self.session.post(self.fast_rpc, json=owner_payload) as resp:
                            od = await resp.json()
                            owner = (od.get("result", {}).get("value") or {}).get("data", {}).get("parsed", {}).get("info", {}).get("owner", "")
                        if owner and owner not in self.tracked_wallets:
                            self.tracked_wallets[owner] = TrackedWallet(
                                address=owner, label="top_holder", source="dexscreener_boost",
                            )
                    await asyncio.sleep(0.3)
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Top holder wallet fetch error: {e}")

    async def _monitor_loop(self):
        """Continuously monitor tracked wallets for new buys."""
        while self._running:
            try:
                wallets = list(self.tracked_wallets.values())
                for wallet in wallets:
                    if not self._running:
                        break
                    new_buys = await self._check_wallet_activity(wallet)
                    for buy in new_buys:
                        try:
                            self.copy_queue.put_nowait(buy)
                        except asyncio.QueueFull:
                            self.copy_queue.get_nowait()
                            self.copy_queue.put_nowait(buy)
                    await asyncio.sleep(0.3)
                # Re-discover every 15 minutes
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Copy trade monitor error: {e}")
                await asyncio.sleep(10)

    async def _check_wallet_activity(self, wallet: TrackedWallet) -> List[dict]:
        """Check a tracked wallet for recent token purchases."""
        new_buys = []
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getSignaturesForAddress",
                "params": [wallet.address, {"limit": 5}],
            }
            async with self.session.post(self.fast_rpc, json=payload) as resp:
                data = await resp.json()
                sigs = data.get("result", [])

            for sig_info in sigs:
                block_time = sig_info.get("blockTime", 0)
                # Only recent transactions (last 90 seconds)
                if block_time and time.time() - block_time > 90:
                    continue
                sig = sig_info.get("signature", "")
                if not sig:
                    continue

                tx_payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                }
                async with self.session.post(self.fast_rpc, json=tx_payload) as resp:
                    tx_data = await resp.json()
                    tx = tx_data.get("result")
                if not tx:
                    continue

                # Detect token buys via post/pre token balance changes
                post_tokens = tx.get("meta", {}).get("postTokenBalances", [])
                pre_tokens = tx.get("meta", {}).get("preTokenBalances", [])

                for post in post_tokens:
                    if post.get("owner") != wallet.address:
                        continue
                    mint = post.get("mint", "")
                    if mint in (SOL_MINT, USDC_MINT):
                        continue
                    post_amt = float(post.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
                    pre_amt = 0.0
                    for pre in pre_tokens:
                        if pre.get("mint") == mint and pre.get("owner") == wallet.address:
                            pre_amt = float(pre.get("uiTokenAmount", {}).get("uiAmount", 0) or 0)
                            break
                    if post_amt > pre_amt:
                        # Estimate SOL spent from balance changes
                        pre_bals = tx.get("meta", {}).get("preBalances", [])
                        post_bals = tx.get("meta", {}).get("postBalances", [])
                        acc_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
                        sol_spent = 0
                        for i, (pb, ab) in enumerate(zip(pre_bals, post_bals)):
                            ak = acc_keys[i] if i < len(acc_keys) else {}
                            aa = ak.get("pubkey", "") if isinstance(ak, dict) else str(ak)
                            if aa == wallet.address:
                                sol_spent = (pb - ab) / LAMPORTS_PER_SOL
                                break
                        if sol_spent > 0.01:
                            new_buys.append({
                                "mint": mint, "wallet": wallet.address,
                                "wallet_label": wallet.label, "sol_spent": sol_spent,
                                "timestamp": block_time, "tx": sig,
                            })
                            logger.info(f"COPY SIGNAL: {wallet.label} bought {mint[:12]}... for {sol_spent:.2f} SOL")
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.debug(f"Wallet check error for {wallet.address[:12]}: {e}")
        return new_buys

    def get_copy_signals(self) -> List[dict]:
        signals = []
        while not self.copy_queue.empty():
            try:
                signals.append(self.copy_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return signals


# ============================================================
# X/TWITTER SENTINEL
# ============================================================
class XSentinelMonitor:
    """Monitor X/Twitter for cabal movements and trending memecoins via Grok."""

    def __init__(self):
        self._running = False
        self._task = None
        self.trending_tokens: Dict[str, dict] = {}
        self.cabal_alerts: deque = deque(maxlen=50)
        self._scan_interval = 45

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._scan_loop())
        logger.info("X Sentinel monitor started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def _scan_loop(self):
        while self._running:
            try:
                await self._scan_x_sentiment()
                await asyncio.sleep(self._scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"X sentinel error: {e}")
                await asyncio.sleep(30)

    async def _scan_x_sentiment(self):
        """Use Grok shadow agent to scan X for memecoin signals."""
        try:
            from farnsworth.core.collective.persistent_agent import call_shadow_agent
        except ImportError:
            return

        prompt = (
            "Scan X/Twitter RIGHT NOW for Solana memecoin activity in the last 30 minutes. "
            "Find: 1) Tokens being shilled by multiple accounts simultaneously (cabal activity), "
            "2) Tokens with sudden spike in mentions, "
            "3) KOL (key opinion leader) calls on new Solana tokens, "
            "4) Dev wallet movements or insider alerts. "
            "Reply in EXACT format, one per line:\n"
            "TOKEN:$SYMBOL|ADDRESS_IF_KNOWN|SIGNAL_TYPE|STRENGTH(1-10)|BRIEF_REASON\n"
            "SIGNAL_TYPE must be one of: CABAL_SHILL, KOL_CALL, TRENDING, INSIDER_ALERT, DEV_MOVEMENT\n"
            "Only include tokens with REAL activity RIGHT NOW. Max 5 tokens."
        )

        try:
            result = await asyncio.wait_for(
                call_shadow_agent("grok", prompt, max_tokens=300, timeout=15.0),
                timeout=18.0,
            )
            if result:
                _, response = result
                self._parse_x_signals(response)
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"X sentinel Grok timeout: {e}")

    def _parse_x_signals(self, response: str):
        if not response:
            return
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line.startswith("TOKEN:"):
                continue
            try:
                parts = line[6:].split("|")
                if len(parts) < 4:
                    continue
                symbol = parts[0].strip().lstrip("$")
                address = parts[1].strip() if len(parts[1].strip()) > 20 else ""
                signal_type = parts[2].strip()
                strength = min(10, max(1, int(parts[3].strip())))
                reason = parts[4].strip() if len(parts) > 4 else ""

                key = address if address else symbol.upper()
                self.trending_tokens[key] = {
                    "symbol": symbol, "address": address,
                    "signal_type": signal_type, "strength": strength,
                    "reason": reason, "timestamp": time.time(),
                }
                if signal_type in ("CABAL_SHILL", "INSIDER_ALERT", "DEV_MOVEMENT"):
                    self.cabal_alerts.append({
                        "symbol": symbol, "type": signal_type,
                        "strength": strength, "reason": reason, "time": time.time(),
                    })
                logger.info(f"X SIGNAL: ${symbol} | {signal_type} | strength={strength} | {reason}")
            except (ValueError, IndexError):
                continue
        # Cleanup old signals (>10 min)
        cutoff = time.time() - 600
        self.trending_tokens = {k: v for k, v in self.trending_tokens.items() if v.get("timestamp", 0) > cutoff}

    def get_hot_tokens(self, min_strength: int = 5) -> List[dict]:
        return [v for v in self.trending_tokens.values() if v.get("strength", 0) >= min_strength]

    def get_token_x_boost(self, token_address: str, symbol: str) -> int:
        """Get X sentiment score bonus for a token. Returns 0-20."""
        signal = self.trending_tokens.get(token_address) or self.trending_tokens.get(symbol.upper())
        if not signal:
            return 0
        strength = signal.get("strength", 0)
        if strength >= 8:
            return 20
        elif strength >= 6:
            return 15
        elif strength >= 4:
            return 10
        return 5


# ============================================================
# QUANTUM TRADE ORACLE (enhanced)
# ============================================================
class QuantumTradeOracle:
    """Enhanced quantum analysis: IBM Quantum QAOA, quantum random timing, FarsightProtocol."""

    def __init__(self):
        self._farsight = None
        self._quantum_proof = None

    async def analyze(self, token: TokenInfo, session: aiohttp.ClientSession,
                      historical_patterns: List[dict] = None) -> Dict:
        """Full quantum analysis pipeline. Returns rug_probability, quantum_edge, timing_jitter."""
        result = {
            "rug_probability": 0.5, "quantum_edge": 0,
            "timing_jitter_ms": 0, "pattern_match": None, "confidence": 0.0,
        }

        # 1. Statistical rug probability (fast, always runs)
        result["rug_probability"] = self._statistical_rug_score(token)

        # 2. FarsightProtocol quantum-enhanced analysis
        farsight_result = await self._farsight_analyze(token, session)
        if farsight_result:
            q_rug = farsight_result.get("rug_probability", 0.5)
            result["rug_probability"] = result["rug_probability"] * 0.4 + q_rug * 0.6
            result["confidence"] = farsight_result.get("confidence", 0)

        # 3. Quantum random entry timing jitter
        result["timing_jitter_ms"] = await self._quantum_timing()

        # 4. Pattern matching against historical trades
        if historical_patterns:
            match = self._match_historical_pattern(token, historical_patterns)
            if match:
                result["pattern_match"] = match
                result["quantum_edge"] = match.get("edge", 0)

        # 5. QAOA-inspired position hint via Bell state measurement
        qaoa_hint = await self._qaoa_position_hint()
        if qaoa_hint:
            result["quantum_edge"] += qaoa_hint

        return result

    def _statistical_rug_score(self, token: TokenInfo) -> float:
        signals = []
        if token.fdv > 0:
            liq_ratio = token.liquidity_usd / token.fdv
            signals.append(0.7 if liq_ratio < 0.05 else 0.4 if liq_ratio < 0.1 else 0.1)
        else:
            signals.append(0.5)
        if token.age_minutes < 5:
            signals.append(0.8)
        elif token.age_minutes < 15:
            signals.append(0.5)
        elif token.age_minutes < 60:
            signals.append(0.3)
        else:
            signals.append(0.15)
        total_txns = token.buy_count_5m + token.sell_count_5m
        if total_txns > 0:
            signals.append(min(1.0, (token.sell_count_5m / total_txns) * 1.2))
        else:
            signals.append(0.5)
        if token.price_change_5m < -30:
            signals.append(0.9)
        elif token.price_change_5m < -15:
            signals.append(0.6)
        else:
            signals.append(0.1)
        return sum(signals) / len(signals) if signals else 0.5

    async def _farsight_analyze(self, token: TokenInfo, session: aiohttp.ClientSession) -> Optional[Dict]:
        try:
            farsight = await self._get_farsight()
            if not farsight:
                return None
            # Try analyze_token first (purpose-built for crypto)
            try:
                result = await asyncio.wait_for(farsight.analyze_token(token.address), timeout=10.0)
                if result:
                    return {
                        "rug_probability": result.get("rug_probability", 0.5),
                        "confidence": result.get("confidence", 0),
                    }
            except (asyncio.TimeoutError, Exception):
                pass
            # Fallback to predict() with quantum enabled
            prediction = await asyncio.wait_for(
                farsight.predict(
                    f"Token ${token.symbol}: liq ${token.liquidity_usd:.0f}, age {token.age_minutes:.0f}m, "
                    f"FDV ${token.fdv:.0f}, B/S(5m) {token.buy_count_5m}/{token.sell_count_5m}. "
                    f"Probability of rug pull in next 1 hour?",
                    category="crypto", include_visual=False, include_quantum=True,
                ),
                timeout=12.0,
            )
            if prediction and prediction.simulation_outcomes:
                rug_prob = prediction.simulation_outcomes.get("rug_pull", 0)
                return {
                    "rug_probability": rug_prob if rug_prob > 0 else 0.5,
                    "confidence": prediction.farsight_confidence,
                }
        except asyncio.TimeoutError:
            logger.debug("FarsightProtocol timeout")
        except Exception as e:
            logger.debug(f"FarsightProtocol error: {e}")
        return None

    async def _quantum_timing(self) -> int:
        """Quantum-random timing jitter to avoid predictable entry patterns."""
        try:
            qp = await self._get_quantum_proof()
            if qp:
                job = await asyncio.wait_for(qp.run_quantum_random(num_bits=8, shots=1), timeout=5.0)
                if job and job.results:
                    bits = list(job.results.keys())[0] if job.results else "00000000"
                    return int(bits, 2) * 8  # 0-2040ms
        except (asyncio.TimeoutError, Exception):
            pass
        return int.from_bytes(os.urandom(2), "big") % 2000

    def _match_historical_pattern(self, token: TokenInfo, patterns: List[dict]) -> Optional[dict]:
        """Find closest matching historical trade pattern."""
        best_match = None
        best_sim = 0.0
        for p in patterns:
            sim = 0.0
            if p.get("source") == token.source:
                sim += 0.2
            p_liq = p.get("liquidity_at_entry", 0)
            if p_liq > 0 and token.liquidity_usd > 0:
                sim += min(p_liq, token.liquidity_usd) / max(p_liq, token.liquidity_usd) * 0.25
            p_age = p.get("age_at_entry", 0)
            if p_age > 0 and token.age_minutes > 0:
                sim += min(p_age, token.age_minutes) / max(p_age, token.age_minutes) * 0.2
            p_score = p.get("entry_score", 0)
            if p_score > 0 and token.score > 0:
                sim += min(p_score, token.score) / max(p_score, token.score) * 0.15
            if p.get("cabal_score", 0) > 50 and token.cabal_score > 50:
                sim += 0.1
            if p.get("outcome") == "win":
                sim += 0.1
            if sim > best_sim:
                best_sim = sim
                edge = 0
                if p.get("outcome") == "win":
                    edge = min(20, int(p.get("pnl_multiple", 1) * 5))
                elif p.get("outcome") == "loss":
                    edge = -10
                elif p.get("outcome") == "rug":
                    edge = -15
                best_match = {
                    "similarity": round(sim, 2), "outcome": p.get("outcome", "unknown"),
                    "pnl_multiple": p.get("pnl_multiple", 1),
                    "edge": max(-20, min(20, edge)), "pattern_symbol": p.get("symbol", "?"),
                }
        return best_match if best_match and best_sim > 0.5 else None

    async def _qaoa_position_hint(self) -> int:
        """QAOA-inspired hint via Bell state measurement."""
        try:
            qp = await self._get_quantum_proof()
            if not qp:
                return 0
            job = await asyncio.wait_for(qp.run_bell_state(shots=20), timeout=5.0)
            if not job or not job.results:
                return 0
            counts = job.results
            total = sum(counts.values())
            if total == 0:
                return 0
            correlated = counts.get("00", 0) + counts.get("11", 0)
            ratio = correlated / total
            if ratio > 0.7:
                return 5
            elif ratio < 0.3:
                return -5
            return 0
        except (asyncio.TimeoutError, Exception):
            return 0

    async def _get_farsight(self):
        if self._farsight is None:
            try:
                from farnsworth.integration.hackathon.farsight_protocol import FarsightProtocol
                self._farsight = FarsightProtocol()
            except ImportError:
                self._farsight = False
        return self._farsight if self._farsight else None

    async def _get_quantum_proof(self):
        if self._quantum_proof is None:
            try:
                from farnsworth.integration.hackathon.quantum_proof import QuantumProof
                self._quantum_proof = QuantumProof()
            except ImportError:
                self._quantum_proof = False
        return self._quantum_proof if self._quantum_proof else None


# ============================================================
# TRADING MEMORY (learns from every trade)
# ============================================================
class TradingMemory:
    """Trading-specific memory layer. Stores outcomes, learns patterns, improves scoring."""

    def __init__(self):
        self._memory = None
        self._initialized = False
        self._trade_patterns: List[dict] = []
        self._score_adjustments: Dict[str, float] = {}

    async def initialize(self):
        try:
            from farnsworth.memory import MemorySystem
            self._memory = MemorySystem()
            await self._memory.initialize()
            self._initialized = True
            await self._load_trade_patterns()
            logger.info(f"TradingMemory initialized, {len(self._trade_patterns)} historical patterns loaded")
        except Exception as e:
            logger.warning(f"TradingMemory init failed (running without memory): {e}")
            self._initialized = False

    async def record_trade(self, entry: TradeMemoryEntry):
        """Store a trade outcome in archival memory + knowledge graph."""
        # Always cache locally
        pattern = {
            "symbol": entry.symbol, "entry_score": entry.entry_score,
            "rug_probability": entry.rug_probability, "cabal_score": entry.cabal_score,
            "source": entry.source, "outcome": entry.outcome,
            "pnl_multiple": entry.pnl_multiple, "hold_minutes": entry.hold_minutes,
            "liquidity_at_entry": entry.liquidity_at_entry, "age_at_entry": entry.age_at_entry,
        }
        self._trade_patterns.append(pattern)
        self._update_score_adjustments()

        if not self._initialized:
            return

        try:
            content = (
                f"TRADE {entry.action.upper()} ${entry.symbol} ({entry.token_address[:12]}...): "
                f"Score={entry.entry_score:.0f}, Rug={entry.rug_probability:.0%}, "
                f"Swarm={entry.swarm_sentiment}, Cabal={entry.cabal_score:.0f}, "
                f"Source={entry.source}, Outcome={entry.outcome}, "
                f"PnL={entry.pnl_multiple:.2f}x in {entry.hold_minutes:.0f}m, "
                f"Liq=${entry.liquidity_at_entry:.0f}, Age={entry.age_at_entry:.0f}m"
            )
            tags = [
                "trading", f"outcome_{entry.outcome}", f"source_{entry.source}",
                f"action_{entry.action}", entry.symbol,
            ]
            if entry.cabal_score > 50:
                tags.append("cabal_detected")
            if entry.pnl_multiple >= 2.0:
                tags.append("big_win")
            elif entry.pnl_multiple <= 0.3:
                tags.append("big_loss")

            importance = 0.7 if entry.outcome in ("win", "rug") else 0.5
            valence = 0.8 if entry.outcome == "win" else -0.5 if entry.outcome in ("loss", "rug") else 0.0

            await self._memory.remember(
                content=content, tags=tags, importance=importance,
                metadata={
                    "type": "trade_outcome", "token_address": entry.token_address,
                    "symbol": entry.symbol, "entry_score": entry.entry_score,
                    "rug_probability": entry.rug_probability, "cabal_score": entry.cabal_score,
                    "source": entry.source, "outcome": entry.outcome,
                    "pnl_multiple": entry.pnl_multiple, "hold_minutes": entry.hold_minutes,
                    "liquidity_at_entry": entry.liquidity_at_entry,
                    "age_at_entry": entry.age_at_entry,
                    "timestamp": entry.timestamp or time.time(),
                },
                emotional_valence=valence,
            )

            # Add to knowledge graph
            try:
                kg = self._memory.knowledge_graph
                if kg:
                    await kg.add_entity(
                        name=entry.symbol, entity_type="token",
                        properties={"address": entry.token_address, "last_outcome": entry.outcome, "last_pnl": entry.pnl_multiple},
                    )
                    await kg.add_relationship(
                        source=entry.symbol, target=f"outcome_{entry.outcome}",
                        relation_type="resulted_in", weight=entry.pnl_multiple, evidence=content,
                    )
                    if entry.source:
                        await kg.add_relationship(
                            source=entry.symbol, target=f"source_{entry.source}",
                            relation_type="detected_by", weight=1.0,
                        )
            except Exception:
                pass

            logger.info(f"Trade recorded to memory: ${entry.symbol} {entry.outcome} {entry.pnl_multiple:.2f}x")
        except Exception as e:
            logger.debug(f"TradingMemory record error: {e}")

    async def recall_similar_trades(self, token: TokenInfo) -> List[dict]:
        """Find past trades with similar characteristics."""
        if not self._initialized:
            return self._trade_patterns
        try:
            results = await self._memory.recall(
                query=f"memecoin trade ${token.symbol} liquidity ${token.liquidity_usd:.0f} age {token.age_minutes:.0f}m source {token.source}",
                top_k=20, min_score=0.2,
            )
            patterns = [r.metadata for r in results if (r.metadata or {}).get("type") == "trade_outcome"]
            return patterns if patterns else self._trade_patterns
        except Exception:
            return self._trade_patterns

    def get_learned_adjustment(self, feature: str) -> float:
        return self._score_adjustments.get(feature, 0)

    def get_historical_patterns(self) -> List[dict]:
        return self._trade_patterns

    def _update_score_adjustments(self):
        if len(self._trade_patterns) < 5:
            return
        source_outcomes = defaultdict(list)
        cabal_outcomes = []
        for p in self._trade_patterns:
            source_outcomes[p.get("source", "unknown")].append(p.get("pnl_multiple", 1.0))
            if p.get("cabal_score", 0) > 50:
                cabal_outcomes.append(p.get("pnl_multiple", 1.0))

        for source, pnls in source_outcomes.items():
            avg = sum(pnls) / len(pnls)
            if avg > 1.5:
                self._score_adjustments[f"source_{source}"] = min(15, int((avg - 1) * 10))
            elif avg < 0.5:
                self._score_adjustments[f"source_{source}"] = max(-15, -int((1 - avg) * 10))
        if cabal_outcomes:
            avg_cabal = sum(cabal_outcomes) / len(cabal_outcomes)
            self._score_adjustments["cabal_detected"] = 10 if avg_cabal > 1.3 else -10 if avg_cabal < 0.6 else 0

    async def _load_trade_patterns(self):
        if not self._initialized:
            return
        try:
            results = await self._memory.recall(query="memecoin trade outcome win loss rug", top_k=100, min_score=0.1)
            for r in results:
                meta = r.metadata or {}
                if meta.get("type") == "trade_outcome":
                    self._trade_patterns.append(meta)
            self._update_score_adjustments()
        except Exception as e:
            logger.debug(f"Trade pattern load error: {e}")

    async def get_win_rate_by_source(self) -> Dict[str, float]:
        source_stats = defaultdict(lambda: {"wins": 0, "total": 0})
        for p in self._trade_patterns:
            source = p.get("source", "unknown")
            source_stats[source]["total"] += 1
            if p.get("outcome") == "win":
                source_stats[source]["wins"] += 1
        return {
            source: round(s["wins"] / s["total"] * 100, 1) if s["total"] > 0 else 0
            for source, s in source_stats.items()
        }


# ============================================================
# DEGEN TRADER (MAIN)
# ============================================================
class DegenTrader:
    """High-frequency Solana memecoin trader powered by collective intelligence."""

    def __init__(self, config: Optional[TraderConfig] = None, wallet_name: str = "degen_trader"):
        self.config = config or TraderConfig()
        self.wallet_name = wallet_name
        self.keypair = None
        self.pubkey = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.seen_tokens: set = set()
        self.running = False
        self.total_pnl_sol = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.start_balance = 0.0
        self._scan_count = 0

        # Intelligence layers
        self.pump_monitor: Optional[PumpFunMonitor] = None
        self.wallet_analyzer: Optional[WalletAnalyzer] = None
        self.quantum_oracle: Optional[QuantumTradeOracle] = None
        self.swarm_intel: Optional[SwarmTradeIntelligence] = None
        # v3 layers
        self.copy_engine: Optional[CopyTradeEngine] = None
        self.x_sentinel: Optional[XSentinelMonitor] = None
        self.trading_memory: Optional[TradingMemory] = None

    async def initialize(self):
        """Load wallet, start session, initialize intelligence layers."""
        wallet_path = WALLET_DIR / f"{self.wallet_name}.json"
        if not wallet_path.exists():
            pubkey, _ = create_wallet(self.wallet_name)
            logger.info(f"New wallet generated: {pubkey}")
            logger.info(f"Fund this wallet with SOL before starting trades")

        self.keypair = load_wallet(self.wallet_name)
        self.pubkey = str(self.keypair.pubkey())

        # Resolve Alchemy RPC if available
        if not self.config.fast_rpc_url:
            alchemy_key = os.environ.get("ALCHEMY_API_KEY", "")
            if alchemy_key:
                self.config.fast_rpc_url = f"https://solana-mainnet.g.alchemy.com/v2/{alchemy_key}"
                logger.info("Using Alchemy RPC for fast on-chain reads")
            else:
                self.config.fast_rpc_url = self.config.rpc_url

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "FarnsworthDegenTrader/2.0"}
        )

        balance = await self.get_sol_balance()
        self.start_balance = balance
        logger.info(f"Wallet: {self.pubkey}")
        logger.info(f"Balance: {balance:.4f} SOL")

        # Initialize intelligence layers
        if self.config.use_pumpfun:
            self.pump_monitor = PumpFunMonitor()
            await self.pump_monitor.start()

        if self.config.use_wallet_analysis:
            self.wallet_analyzer = WalletAnalyzer(self.config.rpc_url, self.config.fast_rpc_url)
            await self.wallet_analyzer.init_session(self.session)

        if self.config.use_quantum:
            self.quantum_oracle = QuantumTradeOracle()

        self.swarm_intel = SwarmTradeIntelligence(self.session)

        # v3: Copy trading engine
        if self.config.use_copy_trading:
            self.copy_engine = CopyTradeEngine(self.session, self.config.fast_rpc_url)
            await self.copy_engine.start()

        # v3: X sentinel
        if self.config.use_x_sentinel:
            self.x_sentinel = XSentinelMonitor()
            await self.x_sentinel.start()

        # v3: Trading memory
        if self.config.use_trading_memory:
            self.trading_memory = TradingMemory()
            await self.trading_memory.initialize()

        self._load_state()
        return self.pubkey

    async def shutdown(self):
        """Clean shutdown of all systems."""
        self.running = False
        self._save_state()
        if self.pump_monitor:
            await self.pump_monitor.stop()
        if self.copy_engine:
            await self.copy_engine.stop()
        if self.x_sentinel:
            await self.x_sentinel.stop()
        if self.session:
            await self.session.close()
        logger.info("Trader shut down cleanly")

    # ----------------------------------------------------------
    # BALANCE & RPC
    # ----------------------------------------------------------
    async def get_sol_balance(self) -> float:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [self.pubkey]}
        async with self.session.post(self.config.rpc_url, json=payload) as resp:
            data = await resp.json()
            return data.get("result", {}).get("value", 0) / LAMPORTS_PER_SOL

    async def get_token_accounts(self) -> Dict[str, float]:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [self.pubkey, {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}, {"encoding": "jsonParsed"}]
        }
        try:
            async with self.session.post(self.config.rpc_url, json=payload) as resp:
                data = await resp.json()
                holdings = {}
                for acc in data.get("result", {}).get("value", []):
                    info = acc.get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                    mint = info.get("mint", "")
                    amount = float(info.get("tokenAmount", {}).get("uiAmount", 0) or 0)
                    if amount > 0:
                        holdings[mint] = amount
                return holdings
        except Exception as e:
            logger.error(f"Token accounts error: {e}")
            return {}

    # ----------------------------------------------------------
    # TOKEN SCANNING (DexScreener + Pump.fun)
    # ----------------------------------------------------------
    async def scan_new_tokens(self) -> List[TokenInfo]:
        """Scan all sources for promising new Solana launches."""
        tokens = []

        # Source 1: Pump.fun real-time (fastest)
        if self.pump_monitor:
            pf_tokens = await self._drain_pumpfun_queue()
            tokens.extend(pf_tokens)

        # Source 2: DexScreener boosted + profiles
        dx_tokens = await self._scan_dexscreener()
        tokens.extend(dx_tokens)

        # Deduplicate by address
        seen = set()
        unique = []
        for t in tokens:
            if t.address not in seen and t.address not in self.seen_tokens:
                seen.add(t.address)
                unique.append(t)

        self._scan_count += 1
        return unique

    async def _drain_pumpfun_queue(self) -> List[TokenInfo]:
        """Get new tokens from pump.fun monitor."""
        tokens = []
        while not self.pump_monitor.new_tokens.empty():
            try:
                pf = self.pump_monitor.new_tokens.get_nowait()
                mint = pf.get("mint", "")
                if not mint or mint in self.seen_tokens:
                    continue

                # Wait a bit for DexScreener to index it
                age_seconds = time.time() - pf.get("timestamp", time.time())
                if age_seconds < 60:
                    # Too fresh for DexScreener, check pump.fun hot stats
                    hot = self.pump_monitor.hot_tokens.get(mint, {})
                    if hot.get("buys", 0) >= 5:  # at least 5 buys
                        token_data = await self._fetch_token_data(mint)
                        if token_data:
                            token_data.source = "pumpfun"
                            token_data.creator_wallet = pf.get("creator", "")
                            tokens.append(token_data)
                else:
                    token_data = await self._fetch_token_data(mint)
                    if token_data:
                        token_data.source = "pumpfun"
                        token_data.creator_wallet = pf.get("creator", "")
                        tokens.append(token_data)
            except asyncio.QueueEmpty:
                break
        return tokens

    async def _scan_dexscreener(self) -> List[TokenInfo]:
        """Scan DexScreener for trending tokens."""
        tokens = []
        try:
            for url in [DEXSCREENER_BOOSTS, DEXSCREENER_PROFILES]:
                async with self.session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    items = data if isinstance(data, list) else []
                    for item in items[:25]:
                        if item.get("chainId") == "solana":
                            addr = item.get("tokenAddress", "")
                            if addr and addr not in self.seen_tokens:
                                token = await self._fetch_token_data(addr)
                                if token:
                                    tokens.append(token)
        except Exception as e:
            logger.error(f"DexScreener scan error: {e}")
        return tokens

    async def _fetch_token_data(self, address: str) -> Optional[TokenInfo]:
        """Fetch detailed token data from DexScreener."""
        try:
            url = f"{DEXSCREENER_TOKENS}/{address}"
            async with self.session.get(url) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                pairs = data.get("pairs")
                if not pairs:
                    return None

                pair = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0))
                created_at = pair.get("pairCreatedAt", 0)
                age_min = max(0, (time.time() * 1000 - created_at) / 60000) if created_at else 999
                txns_5m = pair.get("txns", {}).get("m5", {})
                base = pair.get("baseToken", {})

                return TokenInfo(
                    address=address,
                    symbol=base.get("symbol", "???"),
                    name=base.get("name", "Unknown"),
                    pair_address=pair.get("pairAddress", ""),
                    price_usd=float(pair.get("priceUsd", 0) or 0),
                    liquidity_usd=float(pair.get("liquidity", {}).get("usd", 0) or 0),
                    volume_24h=float(pair.get("volume", {}).get("h24", 0) or 0),
                    price_change_5m=float(pair.get("priceChange", {}).get("m5", 0) or 0),
                    price_change_1h=float(pair.get("priceChange", {}).get("h1", 0) or 0),
                    age_minutes=age_min,
                    fdv=float(pair.get("fdv", 0) or 0),
                    buy_count_5m=int(txns_5m.get("buys", 0) or 0),
                    sell_count_5m=int(txns_5m.get("sells", 0) or 0),
                )
        except Exception as e:
            logger.debug(f"Fetch token error {address}: {e}")
            return None

    # ----------------------------------------------------------
    # TOKEN SCORING (enhanced)
    # ----------------------------------------------------------
    def score_token(self, token: TokenInfo) -> float:
        """Score a token 0-100 based on multiple degen signals."""
        score = 0.0

        # Liquidity sweet spot
        if token.liquidity_usd < self.config.min_liquidity:
            return 0
        if token.liquidity_usd > self.config.max_liquidity:
            return 0

        if 10000 <= token.liquidity_usd <= 100000:
            score += 20
        elif 5000 <= token.liquidity_usd < 10000:
            score += 10
        else:
            score += 5

        # Age (newer = more upside)
        if token.age_minutes < self.config.min_age_minutes:
            return 0
        if token.age_minutes <= 15:
            score += 25
        elif token.age_minutes <= 30:
            score += 20
        elif token.age_minutes <= self.config.max_age_minutes:
            score += 10
        else:
            score += 5

        # Buy/sell pressure
        total_txns = token.buy_count_5m + token.sell_count_5m
        if total_txns > 0:
            buy_ratio = token.buy_count_5m / total_txns
            if buy_ratio > 0.7:
                score += 20
            elif buy_ratio > 0.55:
                score += 10
            elif buy_ratio < 0.3:
                return 0

        # Volume/liquidity ratio
        if token.liquidity_usd > 0:
            vol_ratio = token.volume_24h / token.liquidity_usd
            if vol_ratio > 5:
                score += 15
            elif vol_ratio > 2:
                score += 10
            elif vol_ratio > 0.5:
                score += 5

        # 5-minute momentum
        if token.price_change_5m > 10:
            score += 10
        elif token.price_change_5m > 5:
            score += 5
        elif token.price_change_5m < -20:
            score -= 10

        # FDV (micro cap = huge upside)
        if 0 < token.fdv < 100000:
            score += 10
        elif token.fdv < 1000000:
            score += 5

        # Pump.fun source bonus (earliest detection)
        if token.source == "pumpfun":
            score += 10

        # Cabal bonus/penalty
        if token.top_holders_connected and self.config.cabal_is_bullish:
            score += 15  # organized money backing it

        # v3: X sentinel boost
        if self.x_sentinel:
            x_boost = self.x_sentinel.get_token_x_boost(token.address, token.symbol)
            if x_boost > 0:
                score += x_boost
                logger.debug(f"X boost for {token.symbol}: +{x_boost}")

        # v3: Learned adjustments from trading memory
        if self.trading_memory:
            src_adj = self.trading_memory.get_learned_adjustment(f"source_{token.source}")
            if src_adj != 0:
                score += src_adj
            if token.cabal_score > 50:
                cabal_adj = self.trading_memory.get_learned_adjustment("cabal_detected")
                if cabal_adj != 0:
                    score += cabal_adj

        token.score = max(0, min(100, score))
        return token.score

    # ----------------------------------------------------------
    # DEEP ANALYSIS PIPELINE
    # ----------------------------------------------------------
    async def deep_analyze(self, token: TokenInfo) -> bool:
        """Run full analysis pipeline: wallet graph + quantum + swarm.

        Returns True if token passes all checks.
        """
        # 1. Wallet analysis (detect cabals, check concentration)
        if self.wallet_analyzer and self.config.use_wallet_analysis:
            try:
                analysis = await asyncio.wait_for(
                    self.wallet_analyzer.analyze_token_holders(token.address),
                    timeout=10.0,
                )
                token.cabal_score = analysis.get("cabal_score", 0)
                token.top_holders_connected = len(analysis.get("connected_groups", [])) > 0

                # Check for whale buys (parallel with above)
                whale_buys = await asyncio.wait_for(
                    self.wallet_analyzer.check_whale_buys(token.address),
                    timeout=8.0,
                )
                if whale_buys:
                    total_whale_sol = sum(w.get("sol_spent", 0) for w in whale_buys)
                    logger.info(f"Whale activity on {token.symbol}: {len(whale_buys)} buys, {total_whale_sol:.1f} SOL")
                    token.score = min(100, token.score + 15)  # whale boost

                if analysis.get("concentration", 0) > 0.8:
                    logger.info(f"SKIP {token.symbol}: top holders own >80%")
                    return False

            except asyncio.TimeoutError:
                logger.debug(f"Wallet analysis timeout for {token.symbol}")

        # 2. Quantum oracle (rug probability + pattern matching + QAOA hint)
        if self.quantum_oracle and self.config.use_quantum:
            try:
                patterns = self.trading_memory.get_historical_patterns() if self.trading_memory else []
                q_result = await asyncio.wait_for(
                    self.quantum_oracle.analyze(token, self.session, patterns),
                    timeout=15.0,
                )
                token.rug_probability = q_result.get("rug_probability", 0.5)
                q_edge = q_result.get("quantum_edge", 0)
                if q_edge != 0:
                    token.score = max(0, min(100, token.score + q_edge))
                    logger.info(f"Quantum edge for {token.symbol}: {q_edge:+d}")
                if q_result.get("pattern_match"):
                    pm = q_result["pattern_match"]
                    logger.info(f"Pattern match for {token.symbol}: ~{pm['pattern_symbol']} ({pm['outcome']}, {pm['similarity']:.0%} similar)")
                if token.rug_probability > self.config.max_rug_probability:
                    logger.info(f"SKIP {token.symbol}: rug probability {token.rug_probability:.1%}")
                    return False
                logger.info(f"Rug probability for {token.symbol}: {token.rug_probability:.1%}")
                # Apply timing jitter for less predictable entry
                jitter = q_result.get("timing_jitter_ms", 0)
                if jitter > 0:
                    await asyncio.sleep(jitter / 1000.0)
            except asyncio.TimeoutError:
                logger.debug(f"Quantum analysis timeout for {token.symbol}")

        # 3. Swarm multi-agent analysis
        if self.swarm_intel and self.config.use_swarm:
            try:
                verdict = await asyncio.wait_for(
                    self.swarm_intel.multi_agent_analysis(token),
                    timeout=15.0,
                )
                token.swarm_sentiment = verdict.get("verdict", "SKIP")
                if verdict["verdict"] == "SKIP":
                    logger.info(f"Swarm SKIP on {token.symbol}: {verdict.get('reasons', [])}")
                    return False
                if verdict["verdict"] == "STRONG_BUY":
                    token.score = min(100, token.score + 20)
            except asyncio.TimeoutError:
                logger.debug(f"Swarm analysis timeout for {token.symbol}")

        return True

    # ----------------------------------------------------------
    # TRADE EXECUTION (Jupiter)
    # ----------------------------------------------------------
    async def execute_buy(self, token: TokenInfo, amount_sol: float) -> Optional[Trade]:
        amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)
        logger.info(f"BUY {token.symbol} | {amount_sol:.4f} SOL | score={token.score:.0f} | rug={token.rug_probability:.0%} | swarm={token.swarm_sentiment}")

        tx_sig = await self._jupiter_swap(SOL_MINT, token.address, amount_lamports)
        if tx_sig:
            self.positions[token.address] = Position(
                token_address=token.address, symbol=token.symbol,
                entry_price=token.price_usd, amount_tokens=0,
                amount_sol_spent=amount_sol, entry_time=time.time(),
                source=token.source,
            )
            self.seen_tokens.add(token.address)
            trade = Trade(
                timestamp=time.time(), action="buy", token_address=token.address,
                symbol=token.symbol, amount_sol=amount_sol, price_usd=token.price_usd,
                tx_signature=tx_sig,
                reason=f"score={token.score:.0f} rug={token.rug_probability:.0%} swarm={token.swarm_sentiment} src={token.source}",
            )
            self.trades.append(trade)
            self.total_trades += 1
            self._save_state()
            logger.info(f"BUY OK: {token.symbol} tx={tx_sig[:20]}...")

            # v3: Record buy to trading memory
            if self.trading_memory:
                await self.trading_memory.record_trade(TradeMemoryEntry(
                    token_address=token.address, symbol=token.symbol, action="buy",
                    entry_score=token.score, rug_probability=token.rug_probability,
                    swarm_sentiment=token.swarm_sentiment, cabal_score=token.cabal_score,
                    source=token.source, outcome="pending", pnl_multiple=1.0,
                    hold_minutes=0, liquidity_at_entry=token.liquidity_usd,
                    age_at_entry=token.age_minutes, timestamp=time.time(),
                ))
            return trade

        logger.warning(f"BUY FAILED: {token.symbol}")
        return None

    async def execute_sell(self, token_address: str, reason: str = "manual") -> Optional[Trade]:
        pos = self.positions.get(token_address)
        if not pos:
            return None

        raw_amount = await self._get_raw_token_balance(token_address)
        if raw_amount <= 0:
            self.positions.pop(token_address, None)
            return None

        token_info = await self._fetch_token_data(token_address)
        current_price = token_info.price_usd if token_info else pos.entry_price

        logger.info(f"SELL {pos.symbol} | reason={reason}")
        tx_sig = await self._jupiter_swap(token_address, SOL_MINT, raw_amount)

        if tx_sig:
            if pos.entry_price > 0:
                price_mult = current_price / pos.entry_price
                if price_mult > 1:
                    self.winning_trades += 1
                    self.total_pnl_sol += pos.amount_sol_spent * (price_mult - 1)
                else:
                    self.total_pnl_sol -= pos.amount_sol_spent * (1 - price_mult)

            trade = Trade(
                timestamp=time.time(), action="sell", token_address=token_address,
                symbol=pos.symbol, amount_sol=pos.amount_sol_spent, price_usd=current_price,
                tx_signature=tx_sig, reason=reason,
            )
            self.trades.append(trade)
            self.total_trades += 1
            self.positions.pop(token_address, None)
            self._save_state()
            logger.info(f"SELL OK: {pos.symbol} tx={tx_sig[:20]}...")

            # v3: Record sell outcome to trading memory
            if self.trading_memory:
                hold_min = (time.time() - pos.entry_time) / 60
                price_mult_mem = current_price / pos.entry_price if pos.entry_price > 0 else 1.0
                outcome = "win" if price_mult_mem > 1.0 else "loss"
                if "rug" in reason or "liquidity" in reason:
                    outcome = "rug"
                elif "time" in reason:
                    outcome = "timeout"
                await self.trading_memory.record_trade(TradeMemoryEntry(
                    token_address=token_address, symbol=pos.symbol, action="sell",
                    entry_score=0, rug_probability=0, swarm_sentiment="",
                    cabal_score=0, source=pos.source, outcome=outcome,
                    pnl_multiple=round(price_mult_mem, 3), hold_minutes=round(hold_min, 1),
                    liquidity_at_entry=0, age_at_entry=0, timestamp=time.time(),
                ))
            return trade

        logger.warning(f"SELL FAILED: {pos.symbol}")
        return None

    async def _get_raw_token_balance(self, mint: str) -> int:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [self.pubkey, {"mint": mint}, {"encoding": "jsonParsed"}]
        }
        try:
            async with self.session.post(self.config.rpc_url, json=payload) as resp:
                data = await resp.json()
                accounts = data.get("result", {}).get("value", [])
                if accounts:
                    return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
        except Exception as e:
            logger.error(f"Raw balance error: {e}")
        return 0

    async def _jupiter_swap(self, input_mint: str, output_mint: str, amount: int) -> Optional[str]:
        try:
            from solders.transaction import VersionedTransaction

            # Quote
            params = {
                "inputMint": input_mint, "outputMint": output_mint,
                "amount": str(amount), "slippageBps": str(self.config.slippage_bps),
            }
            async with self.session.get(JUPITER_QUOTE_URL, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Jupiter quote error: {await resp.text()}")
                    return None
                quote = await resp.json()

            if "error" in quote:
                logger.error(f"Jupiter quote: {quote['error']}")
                return None

            # Swap transaction
            swap_body = {
                "quoteResponse": quote, "userPublicKey": self.pubkey,
                "wrapAndUnwrapSol": True,
                "prioritizationFeeLamports": self.config.priority_fee_lamports,
            }
            async with self.session.post(JUPITER_SWAP_URL, json=swap_body) as resp:
                if resp.status != 200:
                    logger.error(f"Jupiter swap error: {await resp.text()}")
                    return None
                swap_data = await resp.json()

            swap_tx_b64 = swap_data.get("swapTransaction")
            if not swap_tx_b64:
                return None

            # Sign and send
            raw_tx = base64.b64decode(swap_tx_b64)
            tx = VersionedTransaction.from_bytes(raw_tx)
            signed_tx = VersionedTransaction(tx.message, [self.keypair])
            signed_bytes = base64.b64encode(bytes(signed_tx)).decode("ascii")

            send_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [signed_bytes, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}]
            }
            # Use fast RPC for sending (lower latency)
            rpc = self.config.fast_rpc_url or self.config.rpc_url
            async with self.session.post(rpc, json=send_payload) as resp:
                result = await resp.json()

            if "error" in result:
                logger.error(f"TX send error: {result['error']}")
                return None

            tx_sig = result.get("result", "")
            if tx_sig:
                await self._confirm_transaction(tx_sig)
            return tx_sig

        except ImportError:
            logger.error("pip install solders")
            return None
        except Exception as e:
            logger.error(f"Jupiter swap error: {e}")
            return None

    async def _confirm_transaction(self, signature: str, timeout: int = 30):
        start = time.time()
        while time.time() - start < timeout:
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getSignatureStatuses",
                "params": [[signature], {"searchTransactionHistory": True}]
            }
            try:
                async with self.session.post(self.config.fast_rpc_url or self.config.rpc_url, json=payload) as resp:
                    data = await resp.json()
                    statuses = data.get("result", {}).get("value", [])
                    if statuses and statuses[0]:
                        s = statuses[0]
                        if s.get("confirmationStatus") in ("confirmed", "finalized"):
                            return s.get("err") is None
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return False

    # ----------------------------------------------------------
    # POSITION MANAGEMENT
    # ----------------------------------------------------------
    async def manage_positions(self):
        if not self.positions:
            return

        for addr in list(self.positions.keys()):
            pos = self.positions.get(addr)
            if not pos:
                continue

            token = await self._fetch_token_data(addr)
            if not token:
                if time.time() - pos.entry_time > 600:
                    await self.execute_sell(addr, reason="data_unavailable")
                continue

            if pos.entry_price <= 0:
                continue

            price_mult = token.price_usd / pos.entry_price
            hold_min = (time.time() - pos.entry_time) / 60

            # Stop loss
            if price_mult <= pos.stop_loss:
                await self.execute_sell(addr, reason=f"stop_loss_{price_mult:.2f}x")
                continue

            # Take profits
            if price_mult >= 10.0 and pos.partial_sells < 3:
                await self.execute_sell(addr, reason=f"tp_10x_{price_mult:.1f}x")
                continue
            elif price_mult >= 5.0 and pos.partial_sells < 2:
                await self.execute_sell(addr, reason=f"tp_5x_{price_mult:.1f}x")
                continue
            elif price_mult >= 2.0 and pos.partial_sells < 1:
                await self.execute_sell(addr, reason=f"tp_2x_{price_mult:.1f}x")
                continue

            # Time exit
            if hold_min > 120 and price_mult < 1.5:
                await self.execute_sell(addr, reason=f"time_exit_{hold_min:.0f}m")
                continue

            # Rug detection: liquidity vanishing
            if token.liquidity_usd < self.config.min_liquidity * 0.3:
                await self.execute_sell(addr, reason="liquidity_rug")
                continue

            # Heavy sells
            total = token.buy_count_5m + token.sell_count_5m
            if total > 5 and token.sell_count_5m / total > 0.8:
                await self.execute_sell(addr, reason="sell_pressure")
                continue

    # ----------------------------------------------------------
    # MAIN TRADING LOOP
    # ----------------------------------------------------------
    async def run(self):
        await self.initialize()
        self.running = True

        balance = await self.get_sol_balance()
        logger.info("=" * 60)
        logger.info("FARNSWORTH DEGEN TRADER v3 - COLLECTIVE INTELLIGENCE")
        logger.info("=" * 60)
        logger.info(f"Wallet:     {self.pubkey}")
        logger.info(f"Balance:    {balance:.4f} SOL")
        logger.info(f"RPC:        {self.config.rpc_url[:40]}...")
        logger.info(f"Fast RPC:   {self.config.fast_rpc_url[:40]}...")
        logger.info(f"Pump.fun:   {'ON' if self.pump_monitor else 'OFF'}")
        logger.info(f"Wallets:    {'ON' if self.wallet_analyzer else 'OFF'}")
        logger.info(f"Quantum:    {'ON' if self.quantum_oracle else 'OFF'}")
        logger.info(f"Swarm:      {'ON' if self.config.use_swarm else 'OFF'}")
        logger.info(f"CopyTrade:  {'ON' if self.copy_engine else 'OFF'}")
        logger.info(f"X Sentinel: {'ON' if self.x_sentinel else 'OFF'}")
        logger.info(f"Memory:     {'ON' if self.trading_memory and self.trading_memory._initialized else 'OFF'}")
        logger.info(f"Max trade:  {self.config.max_position_sol} SOL")
        logger.info(f"Max pos:    {self.config.max_positions}")
        logger.info("=" * 60)

        if balance < self.config.reserve_sol + 0.01:
            logger.error(f"Insufficient balance ({balance:.4f} SOL). Fund the wallet first.")
            self.running = False
            return

        cycle = 0
        while self.running:
            try:
                cycle += 1
                logger.info(f"--- Cycle {cycle} | Pos: {len(self.positions)}/{self.config.max_positions} | PnL: {self.total_pnl_sol:+.4f} SOL | Trades: {self.total_trades} ---")

                # Manage existing positions
                await self.manage_positions()

                # Check capacity
                balance = await self.get_sol_balance()
                available = balance - self.config.reserve_sol
                can_trade = (
                    len(self.positions) < self.config.max_positions
                    and available >= self.config.max_position_sol
                )

                if can_trade:
                    # v3: Process copy trade signals first (fastest alpha)
                    if self.copy_engine:
                        copy_signals = self.copy_engine.get_copy_signals()
                        for signal in copy_signals[:2]:
                            mint = signal.get("mint", "")
                            if mint in self.positions or mint in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.copy_trade_max_sol:
                                break
                            token = await self._fetch_token_data(mint)
                            if token and token.liquidity_usd >= self.config.min_liquidity:
                                token.source = f"copy_{signal.get('wallet_label', 'unknown')}"
                                self.score_token(token)
                                token.score = min(100, token.score + 20)  # copy trade bonus
                                logger.info(f"COPY TRADE: ${token.symbol} from {signal.get('wallet_label', '?')} ({signal.get('sol_spent', 0):.2f} SOL)")
                                await self.execute_buy(token, self.config.copy_trade_max_sol)
                                await asyncio.sleep(0.5)

                    # v3: Check X sentinel for hot tokens
                    if self.x_sentinel:
                        hot = self.x_sentinel.get_hot_tokens(min_strength=7)
                        for signal in hot[:2]:
                            addr = signal.get("address", "")
                            if not addr or addr in self.positions or addr in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            token = await self._fetch_token_data(addr)
                            if token and token.liquidity_usd >= self.config.min_liquidity:
                                token.source = f"x_{signal.get('signal_type', 'trending')}"
                                self.score_token(token)
                                approved = await self.deep_analyze(token)
                                if approved:
                                    await self.execute_buy(token, self.config.max_position_sol)
                                    await asyncio.sleep(0.5)
                                else:
                                    self.seen_tokens.add(addr)

                    # Scan all standard sources
                    tokens = await self.scan_new_tokens()
                    logger.info(f"Found {len(tokens)} new tokens")

                    # Score
                    scored = []
                    for t in tokens:
                        s = self.score_token(t)
                        if s >= self.config.min_score:
                            scored.append(t)
                    scored.sort(key=lambda t: t.score, reverse=True)

                    # Deep analyze and trade top picks
                    for token in scored[:3]:
                        if token.address in self.positions:
                            continue
                        if len(self.positions) >= self.config.max_positions:
                            break

                        balance = await self.get_sol_balance()
                        if balance - self.config.reserve_sol < self.config.max_position_sol:
                            break

                        # Full analysis pipeline
                        approved = await self.deep_analyze(token)
                        if not approved:
                            self.seen_tokens.add(token.address)
                            continue

                        # Re-score after analysis enhancements
                        self.score_token(token)

                        await self.execute_buy(token, self.config.max_position_sol)
                        await asyncio.sleep(1)

                await asyncio.sleep(self.config.scan_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}", exc_info=True)
                await asyncio.sleep(5)

        await self.shutdown()

    # ----------------------------------------------------------
    # STATE
    # ----------------------------------------------------------
    def _save_state(self):
        state = {
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "trades": [asdict(t) for t in self.trades[-200:]],
            "seen_tokens": list(self.seen_tokens)[-5000:],
            "total_pnl_sol": self.total_pnl_sol,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "start_balance": self.start_balance,
            "saved_at": time.time(),
        }
        try:
            STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
        except Exception as e:
            logger.error(f"State save error: {e}")

    def _load_state(self):
        if not STATE_FILE.exists():
            return
        try:
            state = json.loads(STATE_FILE.read_text())
            self.total_pnl_sol = state.get("total_pnl_sol", 0)
            self.total_trades = state.get("total_trades", 0)
            self.winning_trades = state.get("winning_trades", 0)
            self.start_balance = state.get("start_balance", 0)
            self.seen_tokens = set(state.get("seen_tokens", []))
            for addr, pdata in state.get("positions", {}).items():
                self.positions[addr] = Position(**pdata)
            for tdata in state.get("trades", []):
                self.trades.append(Trade(**tdata))
            logger.info(f"Loaded: {self.total_trades} trades, {len(self.positions)} positions")
        except Exception as e:
            logger.error(f"State load error: {e}")

    # ----------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------
    def status(self) -> Dict:
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        return {
            "running": self.running,
            "wallet": self.pubkey,
            "total_pnl_sol": round(self.total_pnl_sol, 4),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "win_rate": round(win_rate, 1),
            "open_positions": len(self.positions),
            "positions": {
                addr: {
                    "symbol": p.symbol, "entry_price": p.entry_price,
                    "sol_spent": p.amount_sol_spent, "source": p.source,
                    "hold_minutes": round((time.time() - p.entry_time) / 60, 1),
                }
                for addr, p in self.positions.items()
            },
            "recent_trades": [
                {"action": t.action, "symbol": t.symbol, "sol": t.amount_sol,
                 "reason": t.reason, "time": datetime.fromtimestamp(t.timestamp).isoformat()}
                for t in self.trades[-10:]
            ],
            "scan_count": self._scan_count,
            "intelligence": {
                "pumpfun": self.pump_monitor is not None and self.pump_monitor.running,
                "wallet_analysis": self.wallet_analyzer is not None,
                "quantum_oracle": self.quantum_oracle is not None,
                "swarm": self.config.use_swarm,
                "copy_trading": self.copy_engine is not None,
                "x_sentinel": self.x_sentinel is not None,
                "trading_memory": self.trading_memory is not None and self.trading_memory._initialized,
                "tracked_wallets": len(self.copy_engine.tracked_wallets) if self.copy_engine else 0,
                "x_signals_active": len(self.x_sentinel.trending_tokens) if self.x_sentinel else 0,
                "learned_patterns": len(self.trading_memory.get_historical_patterns()) if self.trading_memory else 0,
                "fast_rpc": bool(self.config.fast_rpc_url and self.config.fast_rpc_url != self.config.rpc_url),
            },
        }


# ============================================================
# ENTRY POINTS
# ============================================================
async def start_trader(
    rpc_url: str = DEFAULT_RPC,
    wallet_name: str = "degen_trader",
    max_position_sol: float = 0.1,
    max_positions: int = 10,
    scan_interval: int = 8,
    use_swarm: bool = True,
    use_quantum: bool = True,
    use_pumpfun: bool = True,
    use_copy_trading: bool = True,
    use_x_sentinel: bool = True,
    use_trading_memory: bool = True,
):
    config = TraderConfig(
        rpc_url=rpc_url,
        max_position_sol=max_position_sol,
        max_positions=max_positions,
        scan_interval=scan_interval,
        use_swarm=use_swarm,
        use_quantum=use_quantum,
        use_pumpfun=use_pumpfun,
        use_copy_trading=use_copy_trading,
        use_x_sentinel=use_x_sentinel,
        use_trading_memory=use_trading_memory,
    )
    trader = DegenTrader(config=config, wallet_name=wallet_name)
    await trader.run()
    return trader


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Farnsworth Degen Trader v3")
    parser.add_argument("--rpc", default=os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC))
    parser.add_argument("--wallet", default="degen_trader")
    parser.add_argument("--max-sol", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--no-swarm", action="store_true")
    parser.add_argument("--no-quantum", action="store_true")
    parser.add_argument("--no-pumpfun", action="store_true")
    parser.add_argument("--no-copy-trading", action="store_true")
    parser.add_argument("--no-x-sentinel", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--create-wallet", action="store_true")
    args = parser.parse_args()

    if args.create_wallet:
        pubkey, path = create_wallet(args.wallet)
        print(f"\nWallet created!")
        print(f"  Address: {pubkey}")
        print(f"  Keypair: {path}")
        print(f"\nFund this address, then start trading:")
        print(f"  python -m farnsworth.trading.degen_trader --max-sol 0.1")
    else:
        asyncio.run(start_trader(
            rpc_url=args.rpc, wallet_name=args.wallet,
            max_position_sol=args.max_sol, max_positions=args.max_positions,
            scan_interval=args.interval,
            use_swarm=not args.no_swarm, use_quantum=not args.no_quantum,
            use_pumpfun=not args.no_pumpfun,
            use_copy_trading=not args.no_copy_trading,
            use_x_sentinel=not args.no_x_sentinel,
            use_trading_memory=not args.no_memory,
        ))
