"""
Farnsworth Degen Trader v3.7 - Instant Snipe + Quantum Analysis Edition

High-frequency Solana memecoin trader powered by the Farnsworth swarm.
- DIRECT Pump.fun bonding curve buys (no Jupiter needed pre-graduation)
- PumpPortal local transaction API for speed-critical buys
- Pre-bonding curve sniping: buy seconds after launch, sell before/after graduation
- Pump.fun WebSocket for instant new launch detection + trade velocity tracking
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
import struct
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
PUMPPORTAL_LOCAL_API = "https://pumpportal.fun/api/trade-local"
RAYDIUM_AMM_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"

# Multi-launchpad program IDs
BONK_LAUNCHLAB_PROGRAM = "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj"
BONK_PLATFORM_CONFIG = "FfYek5vEz23cMkWsdJwG2oa6EphsvXSHrGpdALN4g6W1"
BAGS_DBC_PROGRAM = "dbcij3LWUppWqq96dh6gJWwBifmcGfLSB5D4DuSMaqN"
BAGS_CREATOR_PROGRAM = "BAGSB9TpGrZxQbEsrEznv5jXXdwyP6AXerN8aVRiAmcv"
BAGS_API_URL = "https://public-api-v2.bags.fm/api/v1"

# Platform identifiers for multi-launchpad support
PLATFORM_PUMP = "pump"        # pump.fun
PLATFORM_BONK = "bonk"        # letsbonk.fun (via Raydium LaunchLab)
PLATFORM_BAGS = "bags"        # bags.fm (via Meteora DBC)

# Pump.fun Program Addresses (bonding curve direct trading)
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_GLOBAL_ACCOUNT = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
PUMP_FEE_RECIPIENTS = [
    "62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV",
    "7VtfL8fvgNfhz17qKRMjzQEXgbdpnHHHQRh54R9jP2RJ",
    "7hTckgnGnLQR6sdH7YkqFTAA7VwTfYFaZ6EhEsU3saCX",
    "9rPYyANsfQZw3DnDmKE3YCQF5E8oD89UXoHn9JFEhJUz",
    "AVmoTthdrX6tKt4nDjco2D775W2YK3sDhxPcMmzUAmTY",
    "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV7AbicfhtW4xC9iM",
    "FWsW1xNtWscwNmKv6wVsU1iTzRN6wmmk3MjxRP5tT7hz",
    "G5UZAVbAf46s7cKWoyKu8kYTip9DGTpbLZ2qa9Aq69dP",
]
PUMP_BUY_DISCRIMINATOR = bytes([102, 6, 61, 18, 1, 218, 235, 234])
PUMP_SELL_DISCRIMINATOR = bytes([51, 230, 133, 164, 1, 127, 131, 173])
PUMP_GRADUATION_SOL = 85.0  # ~85 SOL triggers graduation to PumpSwap
PUMP_INITIAL_VIRTUAL_TOKEN = 1_073_000_000_000_000  # 6 decimals
PUMP_INITIAL_VIRTUAL_SOL = 30_000_000_000  # 30 SOL in lamports
PUMP_INITIAL_REAL_TOKEN = 793_100_000_000_000
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ASSOC_TOKEN_PROGRAM = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
RENT_SYSVAR = "SysvarRent111111111111111111111111111111111"

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
    source: str = "dexscreener"    # dexscreener/pumpfun/raydium/bonding_curve
    creator_wallet: str = ""
    top_holders_connected: bool = False
    # v3.5: Bonding curve fields
    platform: str = ""                   # pump/bonk/bags - which launchpad
    on_bonding_curve: bool = False       # still on bonding curve (any platform)
    curve_progress: float = 0.0          # 0-100% towards graduation
    curve_sol_raised: float = 0.0        # SOL raised so far
    buy_velocity_per_min: float = 0.0    # buys per minute (momentum)
    dev_bought_more: bool = False        # dev buying their own token = bullish
    initial_buy_sol: float = 0.0         # creator's initial buy size


@dataclass
class BondingCurveState:
    """Parsed state of a pump.fun bonding curve account."""
    virtual_token_reserves: int = 0
    virtual_sol_reserves: int = 0
    real_token_reserves: int = 0
    real_sol_reserves: int = 0
    token_total_supply: int = 0
    complete: bool = False

    @property
    def price_sol(self) -> float:
        """Current price per token in SOL."""
        if self.virtual_token_reserves == 0:
            return 0
        return (self.virtual_sol_reserves / 1e9) / (self.virtual_token_reserves / 1e6)

    @property
    def sol_raised(self) -> float:
        """Total SOL raised from bonding curve sales."""
        return self.real_sol_reserves / 1e9

    @property
    def progress_pct(self) -> float:
        """Percent progress towards graduation (0-100)."""
        if PUMP_GRADUATION_SOL <= 0:
            return 0
        return min(100.0, (self.sol_raised / PUMP_GRADUATION_SOL) * 100)

    @property
    def tokens_remaining(self) -> float:
        """Tokens still available on the curve."""
        return self.real_token_reserves / 1e6

    def calc_tokens_for_sol(self, sol_amount: float) -> int:
        """Calculate how many raw tokens you get for X SOL."""
        sol_lamports = int(sol_amount * 1e9)
        fee = int(sol_lamports * 0.01)  # 1% pump.fun fee
        sol_after_fee = sol_lamports - fee
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_sol = self.virtual_sol_reserves + sol_after_fee
        if new_virtual_sol == 0:
            return 0
        new_virtual_tokens = k // new_virtual_sol
        tokens_out = self.virtual_token_reserves - new_virtual_tokens
        return max(0, tokens_out)

    def calc_sol_for_tokens(self, token_amount: int) -> int:
        """Calculate how much SOL lamports you get for selling X raw tokens."""
        k = self.virtual_token_reserves * self.virtual_sol_reserves
        new_virtual_tokens = self.virtual_token_reserves + token_amount
        if new_virtual_tokens == 0:
            return 0
        new_virtual_sol = k // new_virtual_tokens
        sol_out = self.virtual_sol_reserves - new_virtual_sol
        fee = int(sol_out * 0.01)
        return max(0, sol_out - fee)


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
    entry_velocity: float = 0.0    # buys/min at time of entry
    peak_velocity: float = 0.0     # highest velocity seen since entry


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
    min_liquidity: float = 1000.0
    max_liquidity: float = 200000.0
    max_fdv: float = 500000.0       # low cap only - skip anything above 500k FDV
    min_age_minutes: float = 0.5
    max_age_minutes: float = 15.0   # FRESH LAUNCHES ONLY - under 15 minutes
    min_score: float = 60.0
    scan_interval: int = 5          # faster scanning for fresh launches
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
    # v3.5: Bonding curve sniper
    use_bonding_curve: bool = True     # direct pump.fun bonding curve buys
    bonding_curve_max_sol: float = 0.08  # max SOL per bonding curve buy
    bonding_curve_min_buys: int = 3    # min buy count before we ape in
    bonding_curve_max_progress: float = 50.0  # max % curve progress (get in early)
    bonding_curve_min_velocity: float = 2.0  # min buys/min momentum
    use_pumpportal: bool = True        # use PumpPortal API for faster execution
    graduation_sell_pct: float = 0.5   # sell 50% at graduation for guaranteed profit
    sniper_mode: bool = True           # ultra-fast path: skip deep analysis for hot launches
    # v3.6: Cabal coordination tracking
    use_cabal_follow: bool = True       # follow connected wallets into low-cap tokens
    cabal_follow_max_fdv: float = 100000.0  # only follow cabal into sub-100k FDV
    cabal_follow_min_wallets: int = 2   # min connected wallets buying same token
    cabal_follow_max_sol: float = 0.08  # max SOL per cabal-follow buy
    velocity_drop_sell_pct: float = 0.4 # sell when velocity drops to 40% of peak
    # v3.7: Instant snipe on big dev buy at pool creation
    instant_snipe: bool = True             # snipe pool the moment it launches if dev buy is big
    instant_snipe_min_dev_sol: float = 0.3 # minimum dev buy SOL to trigger instant snipe
    instant_snipe_max_sol: float = 0.06    # max SOL to spend on instant snipe


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
# PUMP.FUN MONITOR (v3.5 - Enhanced for bonding curve sniping)
# ============================================================
class PumpFunMonitor:
    """Real-time multi-launchpad monitoring via PumpPortal WebSocket.

    Monitors Pump.fun, Bonk (LetsBonk.fun), and BAGS (bags.fm) for
    new token launches with buy velocity tracking, unique buyer counting,
    creator activity monitoring, and sniper signal generation.

    PumpPortal WS delivers events from both pump.fun and bonk natively.
    BAGS is monitored via periodic API polling.
    """

    def __init__(self):
        self.ws = None
        self.new_tokens: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.hot_tokens: Dict[str, dict] = {}  # mint -> detailed trade stats
        self.sniper_signals: asyncio.Queue = asyncio.Queue(maxsize=100)  # high-priority buys
        self.cabal_signals: asyncio.Queue = asyncio.Queue(maxsize=100)   # cabal coordination buys
        self.running = False
        self._task = None
        self._bags_task = None
        self._tracked_creators: Dict[str, List[str]] = {}  # creator -> [mints they made]
        # v3.6: Track buyer wallets per token for cabal coordination detection
        self._wallet_token_buys: Dict[str, Set[str]] = {}  # wallet -> set of mints they bought
        self._token_buyer_wallets: Dict[str, Set[str]] = {}  # mint -> set of buyer wallets
        self._cabal_signaled: Set[str] = set()  # mints already signaled as cabal buys
        # v3.7: Instant snipe signals for big dev buys at launch
        self.instant_snipe_signals: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._reconnect_count = 0
        # Platform stats
        self.platform_counts = {PLATFORM_PUMP: 0, PLATFORM_BONK: 0, PLATFORM_BAGS: 0}
        self.sniper_history: List[dict] = []  # last N sniper signals for dashboard

    async def start(self):
        """Connect to PumpPortal WebSocket and start multi-platform monitoring."""
        self.running = True
        self._task = asyncio.create_task(self._listen())
        self._bags_task = asyncio.create_task(self._poll_bags())
        logger.info("Multi-launchpad monitor started (Pump.fun + Bonk + BAGS)")

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
        if self._bags_task:
            self._bags_task.cancel()
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
                async with websockets.connect(
                    PUMPFUN_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self.ws = ws
                    self._reconnect_count += 1
                    # Subscribe to new token creates + token trades
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    logger.info(f"PumpPortal WS connected (attempt #{self._reconnect_count}), subscribed to new tokens + trades")

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
                logger.warning(f"PumpFun WS error: {e}, reconnecting in 2s... (reconnects: {self._reconnect_count})")
                await asyncio.sleep(2)

    def _detect_platform(self, data: dict) -> str:
        """Detect which launchpad a PumpPortal event came from."""
        # PumpPortal includes pool field in newer events
        pool = data.get("pool", "")
        if pool == "bonk" or pool == "launchlab":
            return PLATFORM_BONK
        if pool == "pump":
            return PLATFORM_PUMP
        # Check program ID for fallback detection
        program = data.get("programId", "")
        if program == BONK_LAUNCHLAB_PROGRAM:
            return PLATFORM_BONK
        # Default to pump.fun
        return PLATFORM_PUMP

    async def _handle_message(self, data: dict):
        """Process PumpPortal WebSocket messages (Pump.fun + Bonk)."""
        if data.get("txType") == "create":
            # New token created on launchpad
            platform = self._detect_platform(data)
            creator = data.get("traderPublicKey", "")
            mint = data.get("mint", "")
            initial_sol = data.get("solAmount", 0) / LAMPORTS_PER_SOL if data.get("solAmount") else 0
            platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"
            token_data = {
                "mint": mint,
                "name": data.get("name", ""),
                "symbol": data.get("symbol", ""),
                "creator": creator,
                "initial_buy_sol": initial_sol,
                "timestamp": time.time(),
                "source": "bonding_curve",
                "on_bonding_curve": True,
                "platform": platform,
            }
            if mint:
                self.platform_counts[platform] = self.platform_counts.get(platform, 0) + 1
                # Track this token's trade stats from birth
                self.hot_tokens[mint] = {
                    "buys": 1 if initial_sol > 0 else 0,
                    "sells": 0,
                    "volume_sol": initial_sol,
                    "first_seen": time.time(),
                    "unique_buyers": {creator} if initial_sol > 0 else set(),
                    "creator": creator,
                    "creator_bought": initial_sol > 0,
                    "creator_sol": initial_sol,
                    "buy_timestamps": [time.time()] if initial_sol > 0 else [],
                    "symbol": data.get("symbol", ""),
                    "name": data.get("name", ""),
                    "largest_buy_sol": initial_sol,
                    "platform": platform,
                }
                # Track creator history
                if creator:
                    if creator not in self._tracked_creators:
                        self._tracked_creators[creator] = []
                    self._tracked_creators[creator].append(mint)

                try:
                    self.new_tokens.put_nowait(token_data)
                except asyncio.QueueFull:
                    self.new_tokens.get_nowait()
                    self.new_tokens.put_nowait(token_data)
                logger.info(f"[{platform_label}] NEW: ${token_data['symbol']} by {creator[:8]}... (dev buy: {initial_sol:.3f} SOL)")

                # v3.7: Instant snipe on big dev buy at pool creation
                # If dev launches with significant SOL, snipe immediately before others pile in
                if initial_sol >= 0.3:  # threshold checked against config in main loop
                    instant_signal = {
                        "mint": mint,
                        "symbol": data.get("symbol", ""),
                        "name": data.get("name", ""),
                        "buys": 1,
                        "sells": 0,
                        "unique_buyers": 1,
                        "velocity": 0,
                        "volume_sol": initial_sol,
                        "age_seconds": 0,
                        "creator": creator,
                        "creator_bought": True,
                        "creator_sol": initial_sol,
                        "largest_buy_sol": initial_sol,
                        "timestamp": time.time(),
                        "platform": platform,
                        "instant_snipe": True,
                        "dev_buy_sol": initial_sol,
                    }
                    try:
                        self.instant_snipe_signals.put_nowait(instant_signal)
                    except asyncio.QueueFull:
                        self.instant_snipe_signals.get_nowait()
                        self.instant_snipe_signals.put_nowait(instant_signal)
                    logger.info(f"[{platform_label}] INSTANT SNIPE SIGNAL: ${data.get('symbol', '?')} dev buy {initial_sol:.3f} SOL — sniping before others!")

        elif data.get("txType") in ("buy", "sell"):
            mint = data.get("mint", "")
            trader = data.get("traderPublicKey", "")
            if mint:
                if mint not in self.hot_tokens:
                    platform = self._detect_platform(data)
                    self.hot_tokens[mint] = {
                        "buys": 0, "sells": 0, "volume_sol": 0,
                        "first_seen": time.time(), "unique_buyers": set(),
                        "creator": "", "creator_bought": False, "creator_sol": 0,
                        "buy_timestamps": [], "symbol": "", "name": "",
                        "largest_buy_sol": 0, "platform": platform,
                    }
                stats = self.hot_tokens[mint]
                sol_amount = data.get("solAmount", 0) / LAMPORTS_PER_SOL if data.get("solAmount") else 0

                if data["txType"] == "buy":
                    stats["buys"] += 1
                    stats["unique_buyers"].add(trader)
                    stats["buy_timestamps"].append(time.time())
                    if sol_amount > stats.get("largest_buy_sol", 0):
                        stats["largest_buy_sol"] = sol_amount
                    # Check if creator is buying more (bullish signal)
                    if trader == stats.get("creator") and stats["buys"] > 1:
                        stats["creator_bought"] = True
                        stats["creator_sol"] += sol_amount
                    # v3.6: Track wallet→token buys for cabal coordination detection
                    if trader:
                        if trader not in self._wallet_token_buys:
                            self._wallet_token_buys[trader] = set()
                        self._wallet_token_buys[trader].add(mint)
                        if mint not in self._token_buyer_wallets:
                            self._token_buyer_wallets[mint] = set()
                        self._token_buyer_wallets[mint].add(trader)
                else:
                    stats["sells"] += 1
                    # Creator selling = bearish, could be rug
                    if trader == stats.get("creator"):
                        stats["creator_sold"] = True
                stats["volume_sol"] += sol_amount

                # Check sniper signal: fast buys from multiple unique wallets
                self._check_sniper_signal(mint, stats)
                # v3.6: Check cabal coordination signal
                if data["txType"] == "buy":
                    self._check_cabal_coordination(mint, stats)

                # v3.7: Instant snipe — big buy on a very fresh token (< 15 seconds old)
                # Catches dev buys that come as separate buy txns after create
                if data["txType"] == "buy" and sol_amount >= 0.3 and mint not in self._cabal_signaled:
                    age_s = time.time() - stats.get("first_seen", time.time())
                    if age_s < 15 and stats["buys"] <= 3:
                        platform = stats.get("platform", PLATFORM_PUMP)
                        platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"
                        instant_signal = {
                            "mint": mint,
                            "symbol": stats.get("symbol", ""),
                            "name": stats.get("name", ""),
                            "buys": stats["buys"],
                            "sells": stats["sells"],
                            "unique_buyers": len(stats.get("unique_buyers", set())),
                            "velocity": 0,
                            "volume_sol": stats["volume_sol"],
                            "age_seconds": age_s,
                            "creator": stats.get("creator", ""),
                            "creator_bought": trader == stats.get("creator", ""),
                            "creator_sol": sol_amount,
                            "largest_buy_sol": sol_amount,
                            "timestamp": time.time(),
                            "platform": platform,
                            "instant_snipe": True,
                            "dev_buy_sol": sol_amount,
                        }
                        try:
                            self.instant_snipe_signals.put_nowait(instant_signal)
                        except asyncio.QueueFull:
                            self.instant_snipe_signals.get_nowait()
                            self.instant_snipe_signals.put_nowait(instant_signal)
                        logger.info(
                            f"[{platform_label}] INSTANT SNIPE SIGNAL: ${stats.get('symbol', '?')} | "
                            f"big buy {sol_amount:.3f} SOL at {age_s:.0f}s age — sniping NOW"
                        )

        # Cleanup old hot tokens (older than 30 min)
        cutoff = time.time() - 1800
        expired_mints = {k for k, v in self.hot_tokens.items() if v.get("first_seen", 0) <= cutoff}
        self.hot_tokens = {k: v for k, v in self.hot_tokens.items() if k not in expired_mints}
        # Cleanup wallet tracking for expired tokens
        if expired_mints:
            for mint in expired_mints:
                self._token_buyer_wallets.pop(mint, None)
                self._cabal_signaled.discard(mint)
            for wallet in list(self._wallet_token_buys):
                self._wallet_token_buys[wallet] -= expired_mints
                if not self._wallet_token_buys[wallet]:
                    del self._wallet_token_buys[wallet]
        # Cleanup old creator tracking (keep last 100)
        if len(self._tracked_creators) > 500:
            oldest = sorted(self._tracked_creators.keys(), key=lambda c: len(self._tracked_creators[c]))[:250]
            for c in oldest:
                del self._tracked_creators[c]

    def _check_sniper_signal(self, mint: str, stats: dict):
        """Emit a sniper signal if token shows strong early momentum."""
        age_seconds = time.time() - stats.get("first_seen", time.time())
        if age_seconds < 5 or age_seconds > 300:  # 5s-5min window
            return

        buys = stats["buys"]
        sells = stats["sells"]
        unique = len(stats.get("unique_buyers", set()))
        velocity = (buys / (age_seconds / 60)) if age_seconds > 0 else 0
        creator_sold = stats.get("creator_sold", False)

        # Sniper criteria: multiple unique buyers, good velocity, no creator dump
        platform = stats.get("platform", PLATFORM_PUMP)
        if (buys >= 3 and unique >= 3 and velocity >= 2.0
                and sells <= buys * 0.3 and not creator_sold):
            platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"
            signal = {
                "mint": mint,
                "symbol": stats.get("symbol", ""),
                "name": stats.get("name", ""),
                "buys": buys,
                "sells": sells,
                "unique_buyers": unique,
                "velocity": velocity,
                "volume_sol": stats["volume_sol"],
                "age_seconds": age_seconds,
                "creator": stats.get("creator", ""),
                "creator_bought": stats.get("creator_bought", False),
                "creator_sol": stats.get("creator_sol", 0),
                "largest_buy_sol": stats.get("largest_buy_sol", 0),
                "timestamp": time.time(),
                "platform": platform,
            }
            try:
                self.sniper_signals.put_nowait(signal)
            except asyncio.QueueFull:
                self.sniper_signals.get_nowait()
                self.sniper_signals.put_nowait(signal)
            # Keep last 50 signals for dashboard
            self.sniper_history.append(signal)
            if len(self.sniper_history) > 50:
                self.sniper_history = self.sniper_history[-50:]
            logger.info(
                f"[{platform_label}] SNIPER: ${signal['symbol']} | {buys} buys ({unique} unique) | "
                f"{velocity:.1f}/min | {stats['volume_sol']:.2f} SOL vol | age {age_seconds:.0f}s"
            )

    def _check_cabal_coordination(self, mint: str, stats: dict):
        """Detect when connected wallets (sharing buy history) converge on one token.

        If multiple wallets that have been buying the same OTHER tokens now buy THIS token,
        it's likely a coordinated cabal play. Emit a cabal signal for low-cap tokens.
        """
        if mint in self._cabal_signaled:
            return

        buyers = self._token_buyer_wallets.get(mint, set())
        if len(buyers) < 2:
            return

        # Check how many buyer wallets share purchases in OTHER tokens
        # (wallets buying same tokens = likely connected/coordinated)
        buyer_list = list(buyers)
        connected_pairs = 0
        connected_wallets = set()

        for i in range(len(buyer_list)):
            w1_tokens = self._wallet_token_buys.get(buyer_list[i], set())
            for j in range(i + 1, min(len(buyer_list), i + 10)):  # cap comparisons
                w2_tokens = self._wallet_token_buys.get(buyer_list[j], set())
                # Shared buys in OTHER tokens (excluding this mint)
                shared = (w1_tokens & w2_tokens) - {mint}
                if len(shared) >= 1:  # bought at least 1 other token in common
                    connected_pairs += 1
                    connected_wallets.add(buyer_list[i])
                    connected_wallets.add(buyer_list[j])

        # Signal if enough connected wallets are converging
        if len(connected_wallets) >= 2 and connected_pairs >= 1:
            age_seconds = time.time() - stats.get("first_seen", time.time())
            velocity = stats["buys"] / (age_seconds / 60) if age_seconds > 0 else 0
            platform = stats.get("platform", PLATFORM_PUMP)
            signal = {
                "mint": mint,
                "symbol": stats.get("symbol", ""),
                "name": stats.get("name", ""),
                "connected_wallets": len(connected_wallets),
                "connected_pairs": connected_pairs,
                "total_buyers": len(buyers),
                "buys": stats["buys"],
                "velocity": velocity,
                "volume_sol": stats["volume_sol"],
                "age_seconds": age_seconds,
                "creator": stats.get("creator", ""),
                "platform": platform,
                "timestamp": time.time(),
            }
            try:
                self.cabal_signals.put_nowait(signal)
            except asyncio.QueueFull:
                self.cabal_signals.get_nowait()
                self.cabal_signals.put_nowait(signal)
            self._cabal_signaled.add(mint)
            logger.info(
                f"CABAL SIGNAL: ${signal['symbol']} | {len(connected_wallets)} connected wallets "
                f"({connected_pairs} pairs) | {stats['buys']} buys | {velocity:.1f}/min | "
                f"{stats['volume_sol']:.2f} SOL vol"
            )

    def get_buy_velocity(self, mint: str) -> float:
        """Get current buy velocity (buys per minute) for a token."""
        stats = self.hot_tokens.get(mint)
        if not stats:
            return 0
        age_seconds = time.time() - stats.get("first_seen", time.time())
        if age_seconds <= 0:
            return 0
        return stats["buys"] / (age_seconds / 60)

    def get_token_stats(self, mint: str) -> Optional[dict]:
        """Get detailed stats for a token."""
        stats = self.hot_tokens.get(mint)
        if not stats:
            return None
        age_seconds = time.time() - stats.get("first_seen", time.time())
        return {
            "buys": stats["buys"],
            "sells": stats["sells"],
            "unique_buyers": len(stats.get("unique_buyers", set())),
            "volume_sol": stats["volume_sol"],
            "velocity": stats["buys"] / (age_seconds / 60) if age_seconds > 0 else 0,
            "age_seconds": age_seconds,
            "creator_bought": stats.get("creator_bought", False),
            "creator_sold": stats.get("creator_sold", False),
            "largest_buy_sol": stats.get("largest_buy_sol", 0),
        }

    def is_serial_deployer(self, creator: str) -> bool:
        """Check if creator has deployed multiple tokens recently (rug signal)."""
        mints = self._tracked_creators.get(creator, [])
        return len(mints) > 2  # 3+ tokens in 30min = serial deployer

    async def _poll_bags(self):
        """Poll BAGS (bags.fm) API for new token launches."""
        import aiohttp as _aio
        session = None
        while self.running:
            try:
                if not session:
                    bags_key = os.environ.get("BAGS_API_KEY", "")
                    headers = {"x-api-key": bags_key} if bags_key else {}
                    session = _aio.ClientSession(headers=headers)

                # Poll bags.fm for recently created tokens
                url = f"{BAGS_API_URL}/tokens?sort=created&order=desc&limit=10"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tokens = data if isinstance(data, list) else data.get("data", data.get("tokens", []))
                        for item in tokens[:5]:
                            mint = item.get("mint", item.get("tokenMint", ""))
                            if not mint or mint in self.hot_tokens:
                                continue
                            symbol = item.get("symbol", "")
                            name = item.get("name", "")
                            creator = item.get("creator", item.get("creatorAddress", ""))
                            self.platform_counts[PLATFORM_BAGS] = self.platform_counts.get(PLATFORM_BAGS, 0) + 1
                            token_data = {
                                "mint": mint, "name": name, "symbol": symbol,
                                "creator": creator, "initial_buy_sol": 0,
                                "timestamp": time.time(), "source": "bonding_curve",
                                "on_bonding_curve": True, "platform": PLATFORM_BAGS,
                            }
                            self.hot_tokens[mint] = {
                                "buys": 0, "sells": 0, "volume_sol": 0,
                                "first_seen": time.time(), "unique_buyers": set(),
                                "creator": creator, "creator_bought": False, "creator_sol": 0,
                                "buy_timestamps": [], "symbol": symbol, "name": name,
                                "largest_buy_sol": 0, "platform": PLATFORM_BAGS,
                            }
                            try:
                                self.new_tokens.put_nowait(token_data)
                            except asyncio.QueueFull:
                                self.new_tokens.get_nowait()
                                self.new_tokens.put_nowait(token_data)
                            logger.info(f"[BAGS] NEW: ${symbol} by {creator[:8] if creator else '?'}...")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"BAGS poll error: {e}")
            await asyncio.sleep(30)  # poll every 30s

        if session:
            await session.close()

    def get_sniper_feed(self, max_age_seconds: float = 900) -> List[dict]:
        """Get recent sniper activity for dashboard - fresh launches only."""
        now = time.time()
        fresh = [s for s in self.sniper_history if now - s.get("timestamp", 0) < max_age_seconds]
        return list(reversed(fresh[-20:]))

    def get_cabal_feed(self, max_age_seconds: float = 900) -> List[dict]:
        """Get recent cabal coordination signals for dashboard - fresh only."""
        now = time.time()
        result = []
        for mint in list(self._cabal_signaled):
            stats = self.hot_tokens.get(mint)
            if not stats:
                continue
            age = now - stats.get("first_seen", now)
            if age > max_age_seconds:
                continue
            buyers = self._token_buyer_wallets.get(mint, set())
            velocity = stats["buys"] / (age / 60) if age > 0 else 0
            result.append({
                "mint": mint,
                "symbol": stats.get("symbol", ""),
                "connected_wallets": len(buyers),
                "buys": stats["buys"],
                "sells": stats["sells"],
                "velocity": velocity,
                "volume_sol": stats["volume_sol"],
                "age_seconds": age,
                "platform": stats.get("platform", PLATFORM_PUMP),
            })
        return result[:15]


# ============================================================
# BONDING CURVE ENGINE (v3.5 - Direct pump.fun trading)
# ============================================================
class BondingCurveEngine:
    """Direct trading on pump.fun bonding curves.

    Bypasses Jupiter/Raydium entirely for pre-graduation tokens.
    Uses PumpPortal local transaction API for speed, with direct
    on-chain instruction fallback.
    """

    def __init__(self, rpc_url: str, fast_rpc_url: str = ""):
        self.rpc_url = rpc_url
        self.fast_rpc_url = fast_rpc_url or rpc_url
        self._curve_cache: Dict[str, Tuple[BondingCurveState, float]] = {}  # mint -> (state, timestamp)
        self._fee_idx = 0

    def _next_fee_recipient(self) -> str:
        """Round-robin fee recipient selection."""
        recipient = PUMP_FEE_RECIPIENTS[self._fee_idx % len(PUMP_FEE_RECIPIENTS)]
        self._fee_idx += 1
        return recipient

    async def get_bonding_curve_state(self, mint: str, session: aiohttp.ClientSession) -> Optional[BondingCurveState]:
        """Fetch and parse the bonding curve account for a token."""
        # Check cache (5 second TTL)
        cached = self._curve_cache.get(mint)
        if cached and time.time() - cached[1] < 5:
            return cached[0]

        try:
            # Derive bonding curve PDA
            from solders.pubkey import Pubkey
            mint_pk = Pubkey.from_string(mint)
            program_pk = Pubkey.from_string(PUMP_PROGRAM_ID)
            curve_pk, _ = Pubkey.find_program_address(
                [b"bonding-curve", bytes(mint_pk)], program_pk
            )

            # Fetch account data via RPC
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getAccountInfo",
                "params": [str(curve_pk), {"encoding": "base64"}]
            }
            async with session.post(self.fast_rpc_url, json=payload) as resp:
                result = await resp.json()

            account = result.get("result", {}).get("value")
            if not account or not account.get("data"):
                return None

            data_b64 = account["data"][0]
            data = base64.b64decode(data_b64)

            if len(data) < 49:  # 8 discriminator + 5*8 fields + 1 bool
                return None

            # Parse bonding curve layout
            state = BondingCurveState(
                virtual_token_reserves=struct.unpack_from("<Q", data, 8)[0],
                virtual_sol_reserves=struct.unpack_from("<Q", data, 16)[0],
                real_token_reserves=struct.unpack_from("<Q", data, 24)[0],
                real_sol_reserves=struct.unpack_from("<Q", data, 32)[0],
                token_total_supply=struct.unpack_from("<Q", data, 40)[0],
                complete=bool(data[48]),
            )

            self._curve_cache[mint] = (state, time.time())
            return state

        except ImportError:
            logger.error("pip install solders for bonding curve trading")
            return None
        except Exception as e:
            logger.debug(f"Bonding curve fetch error for {mint}: {e}")
            return None

    async def buy_on_curve_pumpportal(
        self, mint: str, sol_amount: float, pubkey: str, keypair,
        session: aiohttp.ClientSession, slippage: int = 15, priority_fee: float = 0.005,
        pool: str = "pump",
    ) -> Optional[str]:
        """Buy a token on the bonding curve via PumpPortal local API.

        Supports multiple launchpads: pool="pump" (pump.fun), pool="bonk" (letsbonk.fun),
        pool="auto" (auto-detect). PumpPortal handles the transaction building.
        """
        try:
            from solders.transaction import VersionedTransaction

            # Get unsigned transaction from PumpPortal
            payload = {
                "publicKey": pubkey,
                "action": "buy",
                "mint": mint,
                "amount": sol_amount,
                "denominatedInSol": "true",
                "slippage": slippage,
                "priorityFee": priority_fee,
                "pool": pool,
            }
            async with session.post(PUMPPORTAL_LOCAL_API, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"PumpPortal buy error: {err}")
                    return None
                tx_bytes = await resp.read()

            if not tx_bytes or len(tx_bytes) < 10:
                logger.error("PumpPortal returned empty transaction")
                return None

            # Deserialize, sign, and send
            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode("ascii")

            send_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [signed_b64, {"encoding": "base64", "skipPreflight": True, "maxRetries": 2}]
            }
            rpc = self.fast_rpc_url
            async with session.post(rpc, json=send_payload) as resp:
                result = await resp.json()

            if "error" in result:
                logger.error(f"Bonding curve TX error: {result['error']}")
                return None

            tx_sig = result.get("result", "")
            if tx_sig:
                logger.info(f"BONDING CURVE BUY: {mint[:12]}... | {sol_amount:.4f} SOL | tx={tx_sig[:20]}...")
            return tx_sig

        except ImportError:
            logger.error("pip install solders for bonding curve trading")
            return None
        except Exception as e:
            logger.error(f"PumpPortal buy error: {e}")
            return None

    async def sell_on_curve_pumpportal(
        self, mint: str, token_amount_pct: float, pubkey: str, keypair,
        session: aiohttp.ClientSession, slippage: int = 15, priority_fee: float = 0.005,
        pool: str = "pump",
    ) -> Optional[str]:
        """Sell tokens on the bonding curve via PumpPortal.

        Supports pool="pump", pool="bonk", pool="auto".
        token_amount_pct: fraction to sell (1.0 = all, 0.5 = half)
        """
        try:
            from solders.transaction import VersionedTransaction

            # First get our token balance
            balance = await self._get_token_balance(mint, pubkey, session)
            if balance <= 0:
                return None

            sell_amount = int(balance * token_amount_pct)
            if sell_amount <= 0:
                return None

            payload = {
                "publicKey": pubkey,
                "action": "sell",
                "mint": mint,
                "amount": sell_amount,
                "denominatedInSol": "false",
                "slippage": slippage,
                "priorityFee": priority_fee,
                "pool": pool,
            }
            async with session.post(PUMPPORTAL_LOCAL_API, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    logger.error(f"PumpPortal sell error: {err}")
                    return None
                tx_bytes = await resp.read()

            if not tx_bytes or len(tx_bytes) < 10:
                return None

            tx = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])
            signed_b64 = base64.b64encode(bytes(signed_tx)).decode("ascii")

            send_payload = {
                "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                "params": [signed_b64, {"encoding": "base64", "skipPreflight": True, "maxRetries": 2}]
            }
            async with session.post(self.fast_rpc_url, json=send_payload) as resp:
                result = await resp.json()

            if "error" in result:
                logger.error(f"Bonding curve sell TX error: {result['error']}")
                return None

            tx_sig = result.get("result", "")
            if tx_sig:
                logger.info(f"BONDING CURVE SELL: {mint[:12]}... | {token_amount_pct:.0%} | tx={tx_sig[:20]}...")
            return tx_sig

        except Exception as e:
            logger.error(f"PumpPortal sell error: {e}")
            return None

    async def _get_token_balance(self, mint: str, owner: str, session: aiohttp.ClientSession) -> int:
        """Get raw token balance for owner."""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTokenAccountsByOwner",
                "params": [owner, {"mint": mint}, {"encoding": "jsonParsed"}]
            }
            async with session.post(self.fast_rpc_url, json=payload) as resp:
                data = await resp.json()
            accounts = data.get("result", {}).get("value", [])
            if accounts:
                info = accounts[0].get("account", {}).get("data", {}).get("parsed", {}).get("info", {})
                return int(info.get("tokenAmount", {}).get("amount", 0))
        except Exception as e:
            logger.debug(f"Token balance error: {e}")
        return 0

    def is_pre_graduation(self, state: BondingCurveState) -> bool:
        """Check if token is still on bonding curve (hasn't graduated)."""
        return not state.complete and state.real_token_reserves > 0

    def estimate_graduation_time(self, state: BondingCurveState, velocity_sol_per_min: float) -> float:
        """Estimate minutes until graduation based on current buy velocity."""
        remaining_sol = PUMP_GRADUATION_SOL - state.sol_raised
        if remaining_sol <= 0 or velocity_sol_per_min <= 0:
            return 0
        return remaining_sol / velocity_sol_per_min


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

    # Established tokens that will NEVER pass fresh-launch filters — skip them
    ESTABLISHED_BLACKLIST = {
        "WIF", "BONK", "DOGE", "SHIB", "PEPE", "FLOKI", "MYRO", "MEW",
        "POPCAT", "WEN", "BOME", "SLERF", "SAMO", "ORCA", "RAY", "JUP",
        "JTO", "PYTH", "RENDER", "HNT", "SOL", "USDC", "USDT", "BTC",
        "ETH", "TRUMP", "MELANIA", "PNUT", "GOAT", "MOODENG", "SPX",
        "GIGA", "RETARDIO", "PONKE", "MOTHER", "FWOG", "MICHI", "LUCE",
    }

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
                # Skip established tokens — they'll never pass our fresh launch filters
                if symbol.upper() in self.ESTABLISHED_BLACKLIST:
                    logger.debug(f"X sentinel: skipping established token ${symbol}")
                    continue
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
# QUANTUM WALLET PREDICTION ENGINE (v3.7)
# ============================================================
class QuantumWalletPredictor:
    """Predict wallet movements BEFORE they happen using quantum correlation.

    Tracks wallet behavior patterns across the PumpPortal stream:
    - Wallet buy sequences (what tokens a wallet buys in order)
    - Time-between-buys patterns (cadence detection)
    - Cross-wallet correlation (when wallet A buys, wallet B follows)
    - Token affinity scoring (which token characteristics attract which wallets)

    Uses quantum Bell state correlations to weight prediction confidence:
    strongly correlated (|00⟩+|11⟩) = high confidence prediction
    anti-correlated (|01⟩+|10⟩) = inverse/contrarian signal
    """

    def __init__(self):
        # wallet -> list of (mint, timestamp, sol_amount) in order
        self.wallet_sequences: Dict[str, List[Tuple[str, float, float]]] = {}
        # wallet_pair -> correlation score (how often they buy same tokens)
        self.wallet_correlations: Dict[str, float] = {}
        # wallet -> average time between buys in seconds
        self.wallet_cadence: Dict[str, float] = {}
        # Predictions: mint -> {predicted_buyers, confidence, quantum_correlation}
        self.predictions: Dict[str, dict] = {}
        self._quantum_proof = None
        self._max_wallet_history = 20
        self._max_predictions = 30
        self.prediction_hits = 0
        self.prediction_misses = 0

    def record_buy(self, wallet: str, mint: str, sol_amount: float):
        """Record a wallet buy and update correlation patterns."""
        now = time.time()
        if wallet not in self.wallet_sequences:
            self.wallet_sequences[wallet] = []
        seq = self.wallet_sequences[wallet]
        seq.append((mint, now, sol_amount))
        if len(seq) > self._max_wallet_history:
            self.wallet_sequences[wallet] = seq[-self._max_wallet_history:]

        # Update cadence (average time between buys)
        if len(seq) >= 2:
            deltas = [seq[i][1] - seq[i-1][1] for i in range(1, len(seq))]
            self.wallet_cadence[wallet] = sum(deltas) / len(deltas)

        # Check if this was a predicted buy
        pred = self.predictions.get(mint)
        if pred and wallet in pred.get("predicted_buyers", set()):
            self.prediction_hits += 1
            logger.info(f"QUANTUM PREDICTION HIT: {wallet[:8]}... bought ${mint[:8]} as predicted (conf: {pred.get('confidence', 0):.0%})")

        # Update wallet-pair correlations
        for other_wallet, other_seq in self.wallet_sequences.items():
            if other_wallet == wallet:
                continue
            other_mints = {s[0] for s in other_seq}
            my_mints = {s[0] for s in seq}
            shared = my_mints & other_mints
            if len(shared) >= 2:
                pair_key = tuple(sorted([wallet, other_wallet]))
                correlation = len(shared) / max(len(my_mints), len(other_mints))
                self.wallet_correlations[str(pair_key)] = correlation

    async def predict_next_buys(self, hot_tokens: Dict[str, dict],
                                wallet_token_buys: Dict[str, set]) -> List[dict]:
        """Predict which tokens are about to get bought and by whom.

        Uses wallet cadence, correlation patterns, and quantum Bell states
        to predict the next likely buys.
        """
        predictions = []
        now = time.time()

        # Find wallets that are "due" for their next buy (based on cadence)
        active_wallets = []
        for wallet, cadence in self.wallet_cadence.items():
            if cadence <= 0 or cadence > 600:  # skip if > 10 min cadence
                continue
            seq = self.wallet_sequences.get(wallet, [])
            if not seq:
                continue
            last_buy_time = seq[-1][1]
            time_since = now - last_buy_time
            # Wallet is "due" if time since last buy > 80% of their cadence
            if time_since > cadence * 0.8:
                active_wallets.append({
                    "wallet": wallet,
                    "cadence": cadence,
                    "overdue_ratio": time_since / cadence,
                    "last_tokens": [s[0] for s in seq[-5:]],
                    "avg_sol": sum(s[2] for s in seq) / len(seq) if seq else 0,
                })

        if not active_wallets:
            return predictions

        # For each hot token, calculate probability of being bought next
        for mint, stats in hot_tokens.items():
            if stats.get("buys", 0) < 1:
                continue
            age_s = now - stats.get("first_seen", now)
            if age_s > 900:  # only predict for tokens < 15 min
                continue

            predicted_buyers = set()
            total_confidence = 0

            for aw in active_wallets[:20]:  # cap computation
                wallet = aw["wallet"]
                wallet_mints = wallet_token_buys.get(wallet, set())

                # Check if correlated wallets already bought this token
                token_buyers = set()
                for buyer in stats.get("unique_buyers", set()):
                    if buyer != wallet:
                        pair_key = str(tuple(sorted([wallet, buyer])))
                        corr = self.wallet_correlations.get(pair_key, 0)
                        if corr > 0.3:
                            token_buyers.add(buyer)

                if token_buyers:
                    # Correlated wallets bought this → predict this wallet will too
                    overdue = min(2.0, aw["overdue_ratio"])
                    corr_strength = len(token_buyers) / max(1, len(stats.get("unique_buyers", set())))
                    confidence = min(0.95, overdue * 0.3 + corr_strength * 0.5)
                    if confidence > 0.3:
                        predicted_buyers.add(wallet)
                        total_confidence += confidence

            if predicted_buyers and total_confidence > 0.3:
                # Quantum enhancement: use Bell state to weight confidence
                q_factor = await self._quantum_correlation_factor()
                final_confidence = min(0.95, (total_confidence / len(predicted_buyers)) * (0.7 + q_factor * 0.3))

                pred = {
                    "mint": mint,
                    "symbol": stats.get("symbol", ""),
                    "predicted_buyers": predicted_buyers,
                    "predicted_count": len(predicted_buyers),
                    "confidence": final_confidence,
                    "quantum_factor": q_factor,
                    "age_seconds": age_s,
                    "current_buys": stats.get("buys", 0),
                    "timestamp": now,
                }
                predictions.append(pred)
                self.predictions[mint] = pred

        # Cleanup old predictions
        cutoff = now - 300  # 5 min
        self.predictions = {k: v for k, v in self.predictions.items() if v.get("timestamp", 0) > cutoff}

        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:self._max_predictions]

    async def _quantum_correlation_factor(self) -> float:
        """Use Bell state measurement to get quantum correlation factor (0-1)."""
        try:
            qp = await self._get_quantum()
            if not qp:
                return 0.5  # neutral
            job = await asyncio.wait_for(qp.run_bell_state(shots=10), timeout=3.0)
            if not job or not job.results:
                return 0.5
            counts = job.results
            total = sum(counts.values())
            if total == 0:
                return 0.5
            correlated = counts.get("00", 0) + counts.get("11", 0)
            return correlated / total  # 0 = anti-correlated, 1 = strongly correlated
        except (asyncio.TimeoutError, Exception):
            return 0.5

    async def _get_quantum(self):
        if self._quantum_proof is None:
            try:
                from farnsworth.integration.hackathon.quantum_proof import QuantumProof
                self._quantum_proof = QuantumProof()
            except ImportError:
                self._quantum_proof = False
        return self._quantum_proof if self._quantum_proof else None

    def get_prediction_feed(self, max_items: int = 10) -> List[dict]:
        """Get current predictions for dashboard display."""
        now = time.time()
        result = []
        for mint, pred in sorted(self.predictions.items(),
                                  key=lambda x: x[1].get("confidence", 0), reverse=True):
            if now - pred.get("timestamp", 0) > 300:
                continue
            result.append({
                "mint": mint,
                "symbol": pred.get("symbol", ""),
                "predicted_buyers": pred.get("predicted_count", 0),
                "confidence": round(pred.get("confidence", 0), 2),
                "quantum_factor": round(pred.get("quantum_factor", 0.5), 2),
                "age_seconds": round(now - pred.get("timestamp", now)),
                "current_buys": pred.get("current_buys", 0),
            })
        return result[:max_items]

    def cleanup(self):
        """Clean up old data to prevent memory growth."""
        if len(self.wallet_sequences) > 1000:
            # Keep only most active wallets
            sorted_wallets = sorted(self.wallet_sequences.keys(),
                                    key=lambda w: len(self.wallet_sequences[w]), reverse=True)
            keep = set(sorted_wallets[:500])
            self.wallet_sequences = {w: s for w, s in self.wallet_sequences.items() if w in keep}
            self.wallet_cadence = {w: c for w, c in self.wallet_cadence.items() if w in keep}
        if len(self.wallet_correlations) > 5000:
            # Keep top correlations only
            top = sorted(self.wallet_correlations.items(), key=lambda x: x[1], reverse=True)[:2500]
            self.wallet_correlations = dict(top)


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
        # v3.5: Bonding curve direct trading
        self.curve_engine: Optional[BondingCurveEngine] = None
        self._sniper_bought: set = set()  # mints already sniped
        # v3.7: Quantum wallet prediction
        self.wallet_predictor: Optional[QuantumWalletPredictor] = None

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
            headers={"User-Agent": "FarnsworthDegenTrader/3.5"}
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

        # v3.5: Bonding curve engine
        if self.config.use_bonding_curve:
            self.curve_engine = BondingCurveEngine(self.config.rpc_url, self.config.fast_rpc_url)
            logger.info("Bonding curve engine enabled (direct pump.fun trading)")

        # v3.7: Quantum wallet prediction engine
        if self.config.use_quantum:
            self.wallet_predictor = QuantumWalletPredictor()
            logger.info("Quantum wallet prediction engine enabled (Bell state correlations)")

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
        """Get new tokens from pump.fun monitor.

        v3.5: Enhanced to pull bonding curve state for fresh tokens
        instead of waiting for DexScreener indexing.
        """
        tokens = []
        while not self.pump_monitor.new_tokens.empty():
            try:
                pf = self.pump_monitor.new_tokens.get_nowait()
                mint = pf.get("mint", "")
                if not mint or mint in self.seen_tokens or mint in self._sniper_bought:
                    continue

                age_seconds = time.time() - pf.get("timestamp", time.time())
                hot = self.pump_monitor.hot_tokens.get(mint, {})
                buys = hot.get("buys", 0)

                # v3.5: For fresh tokens still on bonding curve, build TokenInfo
                # from pump.fun data + curve state (don't wait for DexScreener)
                if age_seconds < 120 and self.curve_engine:
                    if buys >= self.config.bonding_curve_min_buys:
                        curve_state = await self.curve_engine.get_bonding_curve_state(mint, self.session)
                        if curve_state and not curve_state.complete:
                            velocity = self.pump_monitor.get_buy_velocity(mint)
                            unique = len(hot.get("unique_buyers", set()))
                            token_data = TokenInfo(
                                address=mint,
                                symbol=pf.get("symbol", hot.get("symbol", "???")),
                                name=pf.get("name", hot.get("name", "Unknown")),
                                pair_address="",
                                price_usd=curve_state.price_sol * 150,  # rough SOL->USD
                                liquidity_usd=curve_state.sol_raised * 150,
                                volume_24h=hot.get("volume_sol", 0) * 150,
                                age_minutes=age_seconds / 60,
                                holders=unique,
                                fdv=curve_state.price_sol * 1e9 * 150,  # rough
                                buy_count_5m=buys,
                                sell_count_5m=hot.get("sells", 0),
                                source="bonding_curve",
                                creator_wallet=pf.get("creator", ""),
                                on_bonding_curve=True,
                                curve_progress=curve_state.progress_pct,
                                curve_sol_raised=curve_state.sol_raised,
                                buy_velocity_per_min=velocity,
                                initial_buy_sol=pf.get("initial_buy_sol", 0),
                            )
                            tokens.append(token_data)
                            continue

                # Fallback: try DexScreener for older tokens
                if age_seconds >= 60 or buys >= 5:
                    token_data = await self._fetch_token_data(mint)
                    if token_data:
                        token_data.source = "pumpfun"
                        token_data.creator_wallet = pf.get("creator", "")
                        # Enrich with pump.fun stats
                        pf_stats = self.pump_monitor.get_token_stats(mint)
                        if pf_stats:
                            token_data.buy_velocity_per_min = pf_stats.get("velocity", 0)
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

        # FDV cap - low cap only
        if token.fdv > self.config.max_fdv and token.fdv > 0:
            return 0  # HARD REJECT - not low cap

        # Liquidity sweet spot (skip for bonding curve tokens - they have no traditional liquidity)
        if not token.on_bonding_curve:
            if token.liquidity_usd < self.config.min_liquidity:
                return 0
            if token.liquidity_usd > self.config.max_liquidity:
                return 0

            if 5000 <= token.liquidity_usd <= 50000:
                score += 20  # sweet spot for fresh low caps
            elif 1000 <= token.liquidity_usd < 5000:
                score += 15  # early liquidity
            elif 50000 < token.liquidity_usd <= 100000:
                score += 10
            else:
                score += 5
        else:
            # Bonding curve: score based on SOL raised
            if token.curve_sol_raised >= 10:
                score += 20
            elif token.curve_sol_raised >= 3:
                score += 15
            elif token.curve_sol_raised >= 0.5:
                score += 10
            else:
                score += 5

        # Age - FRESH LAUNCHES ONLY (hard reject anything over max_age_minutes)
        if not token.on_bonding_curve and token.age_minutes < self.config.min_age_minutes:
            return 0
        if token.age_minutes > self.config.max_age_minutes:
            return 0  # HARD REJECT - too old, we only trade fresh launches
        if token.age_minutes <= 3:
            score += 30  # ultra-fresh = maximum alpha
        elif token.age_minutes <= 7:
            score += 25  # still very early
        elif token.age_minutes <= 15:
            score += 15  # acceptable freshness

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

        # FDV (micro/low cap = huge upside on fresh launches)
        if 0 < token.fdv < 50000:
            score += 15  # micro cap, maximum upside
        elif token.fdv < 150000:
            score += 10  # low cap, great upside
        elif token.fdv < 500000:
            score += 5   # still acceptable

        # Pump.fun source bonus (earliest detection)
        if token.source == "pumpfun":
            score += 10

        # v3.5: Bonding curve scoring (pre-graduation plays)
        if token.on_bonding_curve:
            score += 15  # base bonus for being early
            # Buy velocity = momentum
            if token.buy_velocity_per_min >= 5.0:
                score += 20  # extremely hot
            elif token.buy_velocity_per_min >= 3.0:
                score += 15
            elif token.buy_velocity_per_min >= 1.5:
                score += 10
            # Early in curve = more upside
            if token.curve_progress < 10:
                score += 15  # very early
            elif token.curve_progress < 25:
                score += 10
            elif token.curve_progress < 50:
                score += 5
            # Dev bought their own token = skin in the game
            if token.initial_buy_sol >= 0.5:
                score += 10
            elif token.initial_buy_sol >= 0.1:
                score += 5

        # Cabal bonus/penalty
        if token.top_holders_connected and self.config.cabal_is_bullish:
            score += 15  # organized money backing it
            # Extra boost for low-cap cabal plays (under 100k FDV = coordinated pump)
            if token.fdv > 0 and token.fdv < self.config.cabal_follow_max_fdv:
                score += 10  # low-cap + connected wallets = strong signal

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
    # BONDING CURVE SNIPER (v3.5)
    # ----------------------------------------------------------
    async def execute_sniper_buy(self, signal: dict, amount_sol: float) -> Optional[Trade]:
        """Ultra-fast bonding curve buy from sniper signal. Minimal checks."""
        mint = signal.get("mint", "")
        symbol = signal.get("symbol", "???")

        if not self.curve_engine or not self.session:
            return None
        if mint in self._sniper_bought or mint in self.positions:
            return None

        # Quick rug checks (fast, no deep analysis)
        creator = signal.get("creator", "")
        if creator and self.pump_monitor and self.pump_monitor.is_serial_deployer(creator):
            logger.info(f"SNIPER SKIP {symbol}: serial deployer {creator[:8]}...")
            self._sniper_bought.add(mint)
            return None

        # Check bonding curve state
        curve_state = await self.curve_engine.get_bonding_curve_state(mint, self.session)
        if not curve_state:
            logger.debug(f"SNIPER SKIP {symbol}: can't read curve state")
            return None
        if curve_state.complete:
            logger.debug(f"SNIPER SKIP {symbol}: already graduated")
            return None
        if curve_state.progress_pct > self.config.bonding_curve_max_progress:
            logger.info(f"SNIPER SKIP {symbol}: curve {curve_state.progress_pct:.1f}% (max {self.config.bonding_curve_max_progress}%)")
            return None

        # Determine PumpPortal pool based on platform
        platform = signal.get("platform", PLATFORM_PUMP)
        pool = platform if platform in (PLATFORM_PUMP, PLATFORM_BONK) else "auto"
        platform_label = "PUMP" if platform == PLATFORM_PUMP else "BONK" if platform == PLATFORM_BONK else "BAGS"

        # Execute via PumpPortal (fastest path)
        logger.info(
            f"[{platform_label}] SNIPER BUY ${symbol} | {amount_sol:.4f} SOL | "
            f"curve {curve_state.progress_pct:.1f}% | {signal.get('buys', 0)} buys | "
            f"{signal.get('velocity', 0):.1f}/min | {signal.get('unique_buyers', 0)} unique"
        )

        tx_sig = None
        if self.config.use_pumpportal:
            tx_sig = await self.curve_engine.buy_on_curve_pumpportal(
                mint, amount_sol, self.pubkey, self.keypair, self.session,
                pool=pool,
            )

        if not tx_sig:
            # Fallback to Jupiter (token might have just graduated)
            tx_sig = await self._jupiter_swap(SOL_MINT, mint, int(amount_sol * LAMPORTS_PER_SOL))

        if tx_sig:
            self._sniper_bought.add(mint)
            entry_vel = signal.get("velocity", 0)
            self.positions[mint] = Position(
                token_address=mint, symbol=symbol,
                entry_price=curve_state.price_sol * 1e9,  # approx USD
                amount_tokens=0, amount_sol_spent=amount_sol,
                entry_time=time.time(),
                take_profit_levels=[3.0, 7.0, 15.0],  # higher targets for early entries
                stop_loss=0.4,  # tighter stop for sniper plays
                source="bonding_curve",
                entry_velocity=entry_vel,
                peak_velocity=entry_vel,
            )
            self.seen_tokens.add(mint)
            trade = Trade(
                timestamp=time.time(), action="buy", token_address=mint,
                symbol=symbol, amount_sol=amount_sol,
                price_usd=curve_state.price_sol * 1e9,
                tx_signature=tx_sig,
                reason=f"SNIPER curve={curve_state.progress_pct:.0f}% buys={signal.get('buys', 0)} vel={signal.get('velocity', 0):.1f}/min",
            )
            self.trades.append(trade)
            self.total_trades += 1
            self._save_state()
            logger.info(f"SNIPER BUY OK: ${symbol} tx={tx_sig[:20]}...")

            # Record to memory
            if self.trading_memory:
                await self.trading_memory.record_trade(TradeMemoryEntry(
                    token_address=mint, symbol=symbol, action="buy",
                    entry_score=80, rug_probability=0.1,
                    swarm_sentiment="SNIPER", cabal_score=0,
                    source="bonding_curve", outcome="pending", pnl_multiple=1.0,
                    hold_minutes=0, liquidity_at_entry=curve_state.sol_raised * 150,  # rough USD
                    age_at_entry=signal.get("age_seconds", 0) / 60,
                    timestamp=time.time(),
                ))
            return trade

        logger.warning(f"SNIPER BUY FAILED: ${symbol}")
        self._sniper_bought.add(mint)  # don't retry
        return None

    async def check_graduation_sells(self):
        """Check if any bonding curve positions have graduated and sell partial."""
        if not self.curve_engine or not self.session:
            return

        for addr in list(self.positions.keys()):
            pos = self.positions.get(addr)
            if not pos or pos.source != "bonding_curve":
                continue

            curve_state = await self.curve_engine.get_bonding_curve_state(addr, self.session)
            if not curve_state:
                continue

            # Token graduated! Sell configured percentage for guaranteed profit
            if curve_state.complete and pos.partial_sells == 0:
                sell_pct = self.config.graduation_sell_pct
                logger.info(f"GRADUATION DETECTED: ${pos.symbol} | Selling {sell_pct:.0%}")
                tx_sig = await self.curve_engine.sell_on_curve_pumpportal(
                    addr, sell_pct, self.pubkey, self.keypair, self.session,
                )
                if tx_sig:
                    pos.partial_sells += 1
                    trade = Trade(
                        timestamp=time.time(), action="sell", token_address=addr,
                        symbol=pos.symbol, amount_sol=pos.amount_sol_spent * sell_pct,
                        price_usd=0, tx_signature=tx_sig,
                        reason=f"GRADUATION_SELL {sell_pct:.0%}",
                    )
                    self.trades.append(trade)
                    self.total_trades += 1
                    self._save_state()

    # ----------------------------------------------------------
    # TRADE EXECUTION (Jupiter)
    # ----------------------------------------------------------
    async def execute_buy(self, token: TokenInfo, amount_sol: float) -> Optional[Trade]:
        amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)
        logger.info(f"BUY {token.symbol} | {amount_sol:.4f} SOL | score={token.score:.0f} | rug={token.rug_probability:.0%} | swarm={token.swarm_sentiment}")

        # v3.5: Use bonding curve for pre-graduation tokens
        tx_sig = None
        if token.on_bonding_curve and self.curve_engine and self.config.use_pumpportal:
            tx_sig = await self.curve_engine.buy_on_curve_pumpportal(
                token.address, amount_sol, self.pubkey, self.keypair, self.session,
            )
        if not tx_sig:
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

            # v3.6: Velocity-drop sell - exit when buy momentum dies
            if self.pump_monitor and pos.entry_velocity > 0:
                current_velocity = self.pump_monitor.get_buy_velocity(addr)
                # Track peak velocity
                if current_velocity > pos.peak_velocity:
                    pos.peak_velocity = current_velocity
                # Sell when velocity drops below threshold of peak (momentum dying)
                if pos.peak_velocity > 0 and hold_min >= 2.0:  # give at least 2 min before checking
                    velocity_ratio = current_velocity / pos.peak_velocity
                    if velocity_ratio <= self.config.velocity_drop_sell_pct:
                        logger.info(
                            f"VELOCITY DROP: ${pos.symbol} | vel {current_velocity:.1f}/min "
                            f"(peak {pos.peak_velocity:.1f}/min, now {velocity_ratio:.0%}) | "
                            f"PnL {price_mult:.2f}x | selling"
                        )
                        await self.execute_sell(addr, reason=f"velocity_drop_{velocity_ratio:.0%}_of_peak")
                        continue

    # ----------------------------------------------------------
    # MAIN TRADING LOOP
    # ----------------------------------------------------------
    async def run(self):
        await self.initialize()
        self.running = True

        balance = await self.get_sol_balance()
        logger.info("=" * 60)
        logger.info("FARNSWORTH DEGEN TRADER v3.7 - INSTANT SNIPE + QUANTUM ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Wallet:     {self.pubkey}")
        logger.info(f"Balance:    {balance:.4f} SOL")
        logger.info(f"RPC:        {self.config.rpc_url[:40]}...")
        logger.info(f"Fast RPC:   {self.config.fast_rpc_url[:40]}...")
        logger.info(f"Pump.fun:   {'ON' if self.pump_monitor else 'OFF'}")
        logger.info(f"BondCurve:  {'ON' if self.curve_engine else 'OFF'} (PumpPortal: {'ON' if self.config.use_pumpportal else 'OFF'})")
        logger.info(f"InstSnipe:  {'ON' if self.config.instant_snipe else 'OFF'} (dev buy >= {self.config.instant_snipe_min_dev_sol} SOL → instant {self.config.instant_snipe_max_sol} SOL)")
        logger.info(f"Sniper:     {'ON' if self.config.sniper_mode else 'OFF'} (max {self.config.bonding_curve_max_sol} SOL, <{self.config.bonding_curve_max_progress}% curve)")
        logger.info(f"FreshOnly:  <{self.config.max_age_minutes}min | FDV cap: ${self.config.max_fdv:,.0f} | Liq: ${self.config.min_liquidity:,.0f}-${self.config.max_liquidity:,.0f}")
        logger.info(f"CabalFollow: {'ON' if self.config.use_cabal_follow else 'OFF'} (FDV<${self.config.cabal_follow_max_fdv:,.0f}, {self.config.cabal_follow_min_wallets}+ wallets, vel-drop sell at {self.config.velocity_drop_sell_pct:.0%})")
        logger.info(f"Wallets:    {'ON' if self.wallet_analyzer else 'OFF'}")
        logger.info(f"Quantum:    {'ON' if self.quantum_oracle else 'OFF'}")
        logger.info(f"QPredict:   {'ON' if self.wallet_predictor else 'OFF'} (Bell state wallet correlation → pre-buy before crowd)")
        logger.info(f"Swarm:      {'ON' if self.config.use_swarm else 'OFF'}")
        logger.info(f"CopyTrade:  {'ON' if self.copy_engine else 'OFF'}")
        logger.info(f"X Sentinel: {'ON' if self.x_sentinel else 'OFF'}")
        logger.info(f"Memory:     {'ON' if self.trading_memory and self.trading_memory._initialized else 'OFF'}")
        logger.info(f"Max trade:  {self.config.max_position_sol} SOL")
        logger.info(f"Max pos:    {self.config.max_positions}")
        logger.info(f"Grad sell:  {self.config.graduation_sell_pct:.0%} at graduation")
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
                    # v3.7: INSTANT SNIPE — big dev buy at pool creation, buy BEFORE others
                    if (self.pump_monitor and self.curve_engine
                            and self.config.instant_snipe and self.config.sniper_mode):
                        instant_signals = []
                        while not self.pump_monitor.instant_snipe_signals.empty():
                            try:
                                sig = self.pump_monitor.instant_snipe_signals.get_nowait()
                                instant_signals.append(sig)
                            except asyncio.QueueEmpty:
                                break
                        for signal in instant_signals[:2]:  # max 2 instant snipes per cycle
                            mint = signal.get("mint", "")
                            dev_sol = signal.get("dev_buy_sol", 0)
                            if mint in self.positions or mint in self._sniper_bought:
                                continue
                            if dev_sol < self.config.instant_snipe_min_dev_sol:
                                continue  # below threshold
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.instant_snipe_max_sol:
                                break
                            logger.info(
                                f"INSTANT SNIPE: ${signal.get('symbol', '?')} | dev buy {dev_sol:.3f} SOL | "
                                f"buying {self.config.instant_snipe_max_sol} SOL BEFORE the crowd"
                            )
                            await self.execute_sniper_buy(signal, self.config.instant_snipe_max_sol)
                            await asyncio.sleep(0.2)  # minimal delay — speed is everything

                    # v3.5: Process sniper signals (momentum-based, bonding curve)
                    if self.pump_monitor and self.curve_engine and self.config.sniper_mode:
                        sniper_signals = []
                        while not self.pump_monitor.sniper_signals.empty():
                            try:
                                sig = self.pump_monitor.sniper_signals.get_nowait()
                                sniper_signals.append(sig)
                            except asyncio.QueueEmpty:
                                break
                        for signal in sniper_signals[:3]:  # max 3 sniper buys per cycle
                            mint = signal.get("mint", "")
                            if mint in self.positions or mint in self._sniper_bought:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.bonding_curve_max_sol:
                                break
                            await self.execute_sniper_buy(signal, self.config.bonding_curve_max_sol)
                            await asyncio.sleep(0.3)  # small delay between buys

                    # v3.5: Check graduation sells for bonding curve positions
                    if self.curve_engine:
                        await self.check_graduation_sells()

                    # v3.6: Process cabal coordination signals (connected wallets converging)
                    if self.pump_monitor and self.config.use_cabal_follow:
                        cabal_sigs = []
                        while not self.pump_monitor.cabal_signals.empty():
                            try:
                                sig = self.pump_monitor.cabal_signals.get_nowait()
                                cabal_sigs.append(sig)
                            except asyncio.QueueEmpty:
                                break
                        for signal in cabal_sigs[:3]:
                            mint = signal.get("mint", "")
                            if not mint or mint in self.positions or mint in self.seen_tokens or mint in self._sniper_bought:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.cabal_follow_max_sol:
                                break
                            # Fetch token data and enforce low-cap + fresh filter
                            token = await self._fetch_token_data(mint)
                            if not token:
                                continue
                            if token.age_minutes > self.config.max_age_minutes:
                                logger.debug(f"SKIP cabal {token.symbol}: too old ({token.age_minutes:.0f}m)")
                                continue
                            if token.fdv > self.config.cabal_follow_max_fdv and token.fdv > 0:
                                logger.debug(f"SKIP cabal {token.symbol}: FDV ${token.fdv:.0f} > ${self.config.cabal_follow_max_fdv:.0f}")
                                continue
                            # Tag and buy
                            token.source = "cabal_follow"
                            token.cabal_score = min(100, signal.get("connected_wallets", 2) * 30)
                            token.top_holders_connected = True
                            self.score_token(token)
                            token.score = min(100, token.score + 25)  # cabal coordination bonus
                            velocity = signal.get("velocity", 0)
                            logger.info(
                                f"CABAL FOLLOW: ${token.symbol} | {signal.get('connected_wallets', 0)} connected wallets | "
                                f"FDV ${token.fdv:.0f} | age {token.age_minutes:.0f}m | vel {velocity:.1f}/min"
                            )
                            buy_result = await self.execute_buy(token, self.config.cabal_follow_max_sol)
                            if buy_result and mint in self.positions:
                                # Record entry velocity for velocity-drop sell
                                self.positions[mint].entry_velocity = velocity
                                self.positions[mint].peak_velocity = velocity
                            await asyncio.sleep(0.3)

                    # v3: Process copy trade signals (second fastest alpha)
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
                                # Enforce fresh launch filter on copy trades too
                                if token.age_minutes > self.config.max_age_minutes:
                                    logger.debug(f"SKIP copy {token.symbol}: too old ({token.age_minutes:.0f}m > {self.config.max_age_minutes}m)")
                                    continue
                                token.source = f"copy_{signal.get('wallet_label', 'unknown')}"
                                self.score_token(token)
                                token.score = min(100, token.score + 20)  # copy trade bonus
                                logger.info(f"COPY TRADE: ${token.symbol} from {signal.get('wallet_label', '?')} ({signal.get('sol_spent', 0):.2f} SOL) age:{token.age_minutes:.0f}m")
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
                                # Enforce fresh launch filter on X sentinel too
                                if token.age_minutes > self.config.max_age_minutes:
                                    logger.debug(f"SKIP X signal {token.symbol}: too old ({token.age_minutes:.0f}m > {self.config.max_age_minutes}m)")
                                    self.seen_tokens.add(addr)
                                    continue
                                token.source = f"x_{signal.get('signal_type', 'trending')}"
                                self.score_token(token)
                                approved = await self.deep_analyze(token)
                                if approved:
                                    await self.execute_buy(token, self.config.max_position_sol)
                                    await asyncio.sleep(0.5)
                                else:
                                    self.seen_tokens.add(addr)

                    # v3.7: Feed wallet buys to quantum predictor + run predictions
                    if self.wallet_predictor and self.pump_monitor:
                        # Feed recent buy data from PumpPortal stream
                        for mint, stats in self.pump_monitor.hot_tokens.items():
                            for buyer in stats.get("unique_buyers", set()):
                                self.wallet_predictor.record_buy(buyer, mint, stats.get("largest_buy_sol", 0))
                        # Run quantum predictions
                        predictions = await self.wallet_predictor.predict_next_buys(
                            self.pump_monitor.hot_tokens,
                            self.pump_monitor._wallet_token_buys,
                        )
                        # Act on high-confidence predictions (pre-buy before wallets move)
                        for pred in predictions[:2]:
                            mint = pred.get("mint", "")
                            conf = pred.get("confidence", 0)
                            if conf < 0.5 or mint in self.positions or mint in self._sniper_bought or mint in self.seen_tokens:
                                continue
                            if len(self.positions) >= self.config.max_positions:
                                break
                            balance = await self.get_sol_balance()
                            if balance - self.config.reserve_sol < self.config.instant_snipe_max_sol:
                                break
                            logger.info(
                                f"QUANTUM PREDICTION BUY: ${pred.get('symbol', '?')} | "
                                f"{pred.get('predicted_count', 0)} wallets predicted to buy | "
                                f"conf: {conf:.0%} | q_factor: {pred.get('quantum_factor', 0.5):.2f} | "
                                f"already {pred.get('current_buys', 0)} buys"
                            )
                            # Build a sniper-compatible signal and execute
                            signal = {
                                "mint": mint,
                                "symbol": pred.get("symbol", ""),
                                "buys": pred.get("current_buys", 0),
                                "unique_buyers": pred.get("predicted_count", 0),
                                "velocity": 0,
                                "creator": "",
                                "platform": "pump",
                                "instant_snipe": True,
                                "dev_buy_sol": 0,
                            }
                            await self.execute_sniper_buy(signal, self.config.instant_snipe_max_sol)
                            await asyncio.sleep(0.3)
                        # Cleanup old predictor data periodically
                        if self._scan_count % 10 == 0:
                            self.wallet_predictor.cleanup()

                    # Scan all standard sources
                    tokens = await self.scan_new_tokens()
                    # Pre-filter: fresh launches only
                    fresh = [t for t in tokens if t.age_minutes <= self.config.max_age_minutes]
                    stale = len(tokens) - len(fresh)
                    if stale > 0:
                        logger.info(f"Found {len(tokens)} tokens, filtered {stale} older than {self.config.max_age_minutes}m → {len(fresh)} fresh launches")
                    else:
                        logger.info(f"Found {len(fresh)} fresh tokens (all under {self.config.max_age_minutes}m)")

                    # Score
                    scored = []
                    for t in fresh:
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
    def _get_scan_feed(self) -> List[dict]:
        """Get top active tokens being scanned — live data for dashboard feeds."""
        if not self.pump_monitor:
            return []
        now = time.time()
        result = []
        for mint, stats in self.pump_monitor.hot_tokens.items():
            age_s = now - stats.get("first_seen", now)
            if age_s > self.config.max_age_minutes * 60:
                continue
            buys = stats.get("buys", 0)
            if buys < 1:
                continue
            unique = len(stats.get("unique_buyers", set()))
            velocity = buys / (age_s / 60) if age_s > 0 else 0
            result.append({
                "mint": mint,
                "symbol": stats.get("symbol", "???"),
                "buys": buys,
                "sells": stats.get("sells", 0),
                "unique_buyers": unique,
                "velocity": round(velocity, 1),
                "volume_sol": round(stats.get("volume_sol", 0), 3),
                "age_seconds": round(age_s),
                "largest_buy_sol": round(stats.get("largest_buy_sol", 0), 3),
                "creator_bought": stats.get("creator_bought", False),
                "platform": stats.get("platform", PLATFORM_PUMP),
            })
        result.sort(key=lambda x: x["velocity"], reverse=True)
        return result[:15]

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
                    "entry_velocity": round(p.entry_velocity, 1),
                    "peak_velocity": round(p.peak_velocity, 1),
                    "current_velocity": round(self.pump_monitor.get_buy_velocity(addr), 1) if self.pump_monitor else 0,
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
                "bonding_curve": self.curve_engine is not None,
                "sniper_mode": self.config.sniper_mode,
                "instant_snipe": self.config.instant_snipe,
                "instant_snipe_min_dev_sol": self.config.instant_snipe_min_dev_sol,
                "ws_reconnects": self.pump_monitor._reconnect_count if self.pump_monitor else 0,
                "tracked_wallets": len(self.copy_engine.tracked_wallets) if self.copy_engine else 0,
                "x_signals_active": len(self.x_sentinel.trending_tokens) if self.x_sentinel else 0,
                "learned_patterns": len(self.trading_memory.get_historical_patterns()) if self.trading_memory else 0,
                "fast_rpc": bool(self.config.fast_rpc_url and self.config.fast_rpc_url != self.config.rpc_url),
                "sniper_buys": len(self._sniper_bought),
                "cabal_follow": self.config.use_cabal_follow,
                "cabal_follow_max_fdv": self.config.cabal_follow_max_fdv,
                "cabal_signals_seen": len(self.pump_monitor._cabal_signaled) if self.pump_monitor else 0,
                "velocity_drop_sell_pct": self.config.velocity_drop_sell_pct,
                "hot_tokens_tracked": len(self.pump_monitor.hot_tokens) if self.pump_monitor else 0,
                "wallets_tracked": len(self.pump_monitor._wallet_token_buys) if self.pump_monitor else 0,
                "platform_counts": self.pump_monitor.platform_counts if self.pump_monitor else {},
            },
            "sniper_feed": self.pump_monitor.get_sniper_feed(max_age_seconds=self.config.max_age_minutes * 60) if self.pump_monitor else [],
            "cabal_feed": self.pump_monitor.get_cabal_feed(max_age_seconds=self.config.max_age_minutes * 60) if self.pump_monitor else [],
            "scan_feed": self._get_scan_feed() if self.pump_monitor else [],
            "prediction_feed": self.wallet_predictor.get_prediction_feed() if self.wallet_predictor else [],
            "prediction_stats": {
                "hits": self.wallet_predictor.prediction_hits if self.wallet_predictor else 0,
                "misses": self.wallet_predictor.prediction_misses if self.wallet_predictor else 0,
                "wallets_modeled": len(self.wallet_predictor.wallet_sequences) if self.wallet_predictor else 0,
                "correlations": len(self.wallet_predictor.wallet_correlations) if self.wallet_predictor else 0,
                "active_predictions": len(self.wallet_predictor.predictions) if self.wallet_predictor else 0,
            },
            "x_feed": [
                {"symbol": v.get("symbol", ""), "signal_type": v.get("signal_type", ""),
                 "strength": v.get("strength", 0), "reason": v.get("reason", ""),
                 "address": v.get("address", ""), "timestamp": v.get("timestamp", 0)}
                for v in (self.x_sentinel.trending_tokens.values() if self.x_sentinel else [])
            ][:10],
            "config": {
                "max_age_minutes": self.config.max_age_minutes,
                "max_fdv": self.config.max_fdv,
                "cabal_follow_max_fdv": self.config.cabal_follow_max_fdv,
                "velocity_drop_sell_pct": self.config.velocity_drop_sell_pct,
                "instant_snipe": self.config.instant_snipe,
                "instant_snipe_min_dev_sol": self.config.instant_snipe_min_dev_sol,
                "instant_snipe_max_sol": self.config.instant_snipe_max_sol,
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
    use_bonding_curve: bool = True,
    sniper_mode: bool = True,
    bonding_curve_max_sol: float = 0.08,
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
        use_bonding_curve=use_bonding_curve,
        sniper_mode=sniper_mode,
        bonding_curve_max_sol=bonding_curve_max_sol,
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

    parser = argparse.ArgumentParser(description="Farnsworth Degen Trader v3.5 - Bonding Curve Sniper")
    parser.add_argument("--rpc", default=os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC))
    parser.add_argument("--wallet", default="degen_trader")
    parser.add_argument("--max-sol", type=float, default=0.1, help="Max SOL per standard trade")
    parser.add_argument("--sniper-sol", type=float, default=0.08, help="Max SOL per bonding curve snipe")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--no-swarm", action="store_true")
    parser.add_argument("--no-quantum", action="store_true")
    parser.add_argument("--no-pumpfun", action="store_true")
    parser.add_argument("--no-copy-trading", action="store_true")
    parser.add_argument("--no-x-sentinel", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-sniper", action="store_true", help="Disable bonding curve sniper mode")
    parser.add_argument("--no-bonding-curve", action="store_true", help="Disable direct bonding curve trading")
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
            use_bonding_curve=not args.no_bonding_curve,
            sniper_mode=not args.no_sniper,
            bonding_curve_max_sol=args.sniper_sol,
        ))
