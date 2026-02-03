"""
Farnsworth DexScreener Integration.

"Finding the next moonshot so you don't have to."

Full-featured DexScreener API integration:
- Token pair data
- Trending tokens
- Boosted tokens
- New pairs tracking
- Token profiles
- Multi-chain support
- Price alerts
"""

import aiohttp
import asyncio
from loguru import logger
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class TokenData:
    """Structured token data from DexScreener."""
    address: str
    name: str = ""
    symbol: str = ""
    price_usd: float = 0.0
    price_native: float = 0.0
    change_5m: float = 0.0
    change_1h: float = 0.0
    change_6h: float = 0.0
    change_24h: float = 0.0
    volume_24h: float = 0.0
    liquidity_usd: float = 0.0
    market_cap: float = 0.0
    fdv: float = 0.0
    pair_address: str = ""
    chain_id: str = ""
    dex_id: str = ""
    txns_buys_24h: int = 0
    txns_sells_24h: int = 0
    created_at: Optional[datetime] = None
    url: str = ""

    @property
    def buy_sell_ratio(self) -> float:
        if self.txns_sells_24h == 0:
            return float('inf') if self.txns_buys_24h > 0 else 0
        return self.txns_buys_24h / self.txns_sells_24h

    @property
    def is_honeypot_risk(self) -> bool:
        """Basic honeypot detection based on buy/sell ratio."""
        return self.buy_sell_ratio > 10 or self.buy_sell_ratio < 0.1

    def to_dict(self) -> Dict:
        return {
            "address": self.address,
            "name": self.name,
            "symbol": self.symbol,
            "price": self.price_usd,
            "change_24h": self.change_24h,
            "volume_24h": self.volume_24h,
            "liquidity": self.liquidity_usd,
            "market_cap": self.market_cap,
            "fdv": self.fdv,
            "chain": self.chain_id,
            "dex": self.dex_id,
            "buys_24h": self.txns_buys_24h,
            "sells_24h": self.txns_sells_24h,
            "buy_sell_ratio": self.buy_sell_ratio,
            "url": self.url
        }


class DexScreenerAPI:
    """
    Full DexScreener API client.

    Endpoints:
    - /dex/tokens/{tokenAddress} - Get pairs by token address
    - /dex/pairs/{chainId}/{pairAddress} - Get pair by chain and address
    - /dex/search?q={query} - Search pairs
    - /token-profiles/latest/v1 - Latest token profiles
    - /token-boosts/latest/v1 - Latest boosted tokens
    - /orders/v1/{chainId}/{pairAddress} - Paid orders for a pair
    """

    BASE_URL = "https://api.dexscreener.com"

    # Supported chains
    CHAINS = {
        "solana": "solana",
        "ethereum": "ethereum",
        "bsc": "bsc",
        "polygon": "polygon",
        "arbitrum": "arbitrum",
        "base": "base",
        "avalanche": "avalanche",
        "optimism": "optimism",
        "fantom": "fantom",
        "cronos": "cronos",
        "sui": "sui",
        "ton": "ton"
    }

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
        self._cache_ttl = 30  # seconds

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"}
            )
        return self._session

    async def _request(self, endpoint: str, use_cache: bool = True) -> Optional[Dict]:
        """Make an API request with caching."""
        cache_key = endpoint

        # Check cache
        if use_cache and cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}{endpoint}"

            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._cache[cache_key] = (data, datetime.now().timestamp())
                    return data
                elif resp.status == 429:
                    logger.warning("DexScreener rate limited")
                    await asyncio.sleep(1)
                    return await self._request(endpoint, use_cache=False)
                else:
                    logger.error(f"DexScreener API error: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"DexScreener request error: {e}")
            return None

    def _parse_pair(self, pair: Dict) -> TokenData:
        """Parse a pair response into TokenData."""
        base_token = pair.get("baseToken", {})
        txns = pair.get("txns", {}).get("h24", {})

        created_at = None
        if pair.get("pairCreatedAt"):
            try:
                created_at = datetime.fromtimestamp(pair["pairCreatedAt"] / 1000)
            except:
                pass

        return TokenData(
            address=base_token.get("address", ""),
            name=base_token.get("name", ""),
            symbol=base_token.get("symbol", ""),
            price_usd=float(pair.get("priceUsd", 0) or 0),
            price_native=float(pair.get("priceNative", 0) or 0),
            change_5m=float(pair.get("priceChange", {}).get("m5", 0) or 0),
            change_1h=float(pair.get("priceChange", {}).get("h1", 0) or 0),
            change_6h=float(pair.get("priceChange", {}).get("h6", 0) or 0),
            change_24h=float(pair.get("priceChange", {}).get("h24", 0) or 0),
            volume_24h=float(pair.get("volume", {}).get("h24", 0) or 0),
            liquidity_usd=float(pair.get("liquidity", {}).get("usd", 0) or 0),
            market_cap=float(pair.get("marketCap", 0) or 0),
            fdv=float(pair.get("fdv", 0) or 0),
            pair_address=pair.get("pairAddress", ""),
            chain_id=pair.get("chainId", ""),
            dex_id=pair.get("dexId", ""),
            txns_buys_24h=int(txns.get("buys", 0) or 0),
            txns_sells_24h=int(txns.get("sells", 0) or 0),
            created_at=created_at,
            url=pair.get("url", "")
        )

    # ==================== TOKEN DATA ====================

    async def get_token(self, token_address: str) -> Optional[TokenData]:
        """Get token data by address (auto-detects chain)."""
        data = await self._request(f"/latest/dex/tokens/{token_address}")
        if data and data.get("pairs"):
            # Return the pair with highest liquidity
            pairs = sorted(
                data["pairs"],
                key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0),
                reverse=True
            )
            return self._parse_pair(pairs[0])
        return None

    async def get_token_pairs(self, token_address: str) -> List[TokenData]:
        """Get all pairs for a token address."""
        data = await self._request(f"/latest/dex/tokens/{token_address}")
        if data and data.get("pairs"):
            return [self._parse_pair(p) for p in data["pairs"]]
        return []

    async def get_pair(self, chain_id: str, pair_address: str) -> Optional[TokenData]:
        """Get specific pair data by chain and pair address."""
        data = await self._request(f"/latest/dex/pairs/{chain_id}/{pair_address}")
        if data and data.get("pairs"):
            return self._parse_pair(data["pairs"][0])
        return None

    async def search(self, query: str, limit: int = 10) -> List[TokenData]:
        """Search for pairs by token name, symbol, or address."""
        data = await self._request(f"/latest/dex/search?q={query}")
        if data and data.get("pairs"):
            pairs = [self._parse_pair(p) for p in data["pairs"][:limit]]
            return pairs
        return []

    # ==================== TRENDING & BOOSTED ====================

    async def get_token_profiles(self, limit: int = 20) -> List[Dict]:
        """Get latest token profiles (paid promotions)."""
        data = await self._request("/token-profiles/latest/v1")
        if data:
            return data[:limit] if isinstance(data, list) else []
        return []

    async def get_boosted_tokens(self, limit: int = 20) -> List[Dict]:
        """Get latest boosted tokens."""
        data = await self._request("/token-boosts/latest/v1")
        if data:
            return data[:limit] if isinstance(data, list) else []
        return []

    async def get_top_boosted(self) -> List[Dict]:
        """Get tokens with most active boosts."""
        data = await self._request("/token-boosts/top/v1")
        if data:
            return data if isinstance(data, list) else []
        return []

    # ==================== NEW PAIRS ====================

    async def get_new_pairs(self, chain: str = None, min_liquidity: float = 1000) -> List[TokenData]:
        """
        Get recently created pairs.

        Note: DexScreener doesn't have a direct "new pairs" endpoint,
        so we search for common launch patterns.
        """
        # Search for common new token patterns
        queries = ["launch", "fair", "stealth", "new"]
        all_pairs = []

        for q in queries:
            pairs = await self.search(q, limit=50)
            for pair in pairs:
                if pair.liquidity_usd >= min_liquidity:
                    if pair.created_at and pair.created_at > datetime.now() - timedelta(hours=24):
                        if chain is None or pair.chain_id == chain:
                            all_pairs.append(pair)

        # Deduplicate by pair address
        seen = set()
        unique_pairs = []
        for p in all_pairs:
            if p.pair_address not in seen:
                seen.add(p.pair_address)
                unique_pairs.append(p)

        # Sort by creation time (newest first)
        unique_pairs.sort(key=lambda p: p.created_at or datetime.min, reverse=True)
        return unique_pairs[:20]

    # ==================== ANALYSIS ====================

    async def analyze_token(self, token_address: str) -> Dict:
        """
        Comprehensive token analysis.

        Returns risk assessment, liquidity analysis, and trading metrics.
        """
        token = await self.get_token(token_address)
        if not token:
            return {"error": "Token not found"}

        all_pairs = await self.get_token_pairs(token_address)

        # Calculate total liquidity across all pairs
        total_liquidity = sum(p.liquidity_usd for p in all_pairs)
        total_volume = sum(p.volume_24h for p in all_pairs)

        # Risk assessment
        risks = []
        risk_score = 0

        if token.liquidity_usd < 10000:
            risks.append("LOW LIQUIDITY")
            risk_score += 30

        if token.buy_sell_ratio > 5:
            risks.append("HIGH BUY/SELL RATIO (possible honeypot)")
            risk_score += 40
        elif token.buy_sell_ratio < 0.2:
            risks.append("LOW BUY/SELL RATIO (heavy selling)")
            risk_score += 20

        if token.created_at and token.created_at > datetime.now() - timedelta(hours=24):
            risks.append("NEWLY CREATED (<24h)")
            risk_score += 15

        if token.change_24h < -50:
            risks.append("MASSIVE DROP (>50% in 24h)")
            risk_score += 25

        if len(all_pairs) == 1:
            risks.append("SINGLE PAIR (low distribution)")
            risk_score += 10

        # Determine risk level
        if risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "token": token.to_dict(),
            "analysis": {
                "total_liquidity": total_liquidity,
                "total_volume_24h": total_volume,
                "pair_count": len(all_pairs),
                "chains": list(set(p.chain_id for p in all_pairs)),
                "dexes": list(set(p.dex_id for p in all_pairs)),
            },
            "risk": {
                "score": risk_score,
                "level": risk_level,
                "flags": risks
            }
        }

    async def compare_tokens(self, addresses: List[str]) -> List[Dict]:
        """Compare multiple tokens side by side."""
        results = []
        for addr in addresses[:5]:  # Limit to 5
            analysis = await self.analyze_token(addr)
            results.append(analysis)
        return results

    # ==================== PRICE MONITORING ====================

    async def check_price_change(
        self,
        token_address: str,
        threshold_percent: float = 10.0,
        timeframe: str = "1h"
    ) -> Optional[Dict]:
        """
        Check if token price has changed beyond threshold.

        Args:
            token_address: Token contract address
            threshold_percent: Alert threshold (default 10%)
            timeframe: "5m", "1h", "6h", or "24h"

        Returns:
            Alert data if threshold exceeded, None otherwise
        """
        token = await self.get_token(token_address)
        if not token:
            return None

        change_map = {
            "5m": token.change_5m,
            "1h": token.change_1h,
            "6h": token.change_6h,
            "24h": token.change_24h
        }

        change = change_map.get(timeframe, token.change_1h)

        if abs(change) >= threshold_percent:
            direction = "up" if change > 0 else "down"
            return {
                "token": token.symbol,
                "address": token.address,
                "change": change,
                "direction": direction,
                "timeframe": timeframe,
                "price": token.price_usd,
                "alert": f"{token.symbol} is {direction} {abs(change):.1f}% in {timeframe}"
            }

        return None

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ==================== SKILL INTERFACE ====================

class DexScreenerSkill:
    """
    Simplified skill interface for agent integration.

    Compatible with the tool router and agent system.
    """

    def __init__(self):
        self.api = DexScreenerAPI()

    async def get_token_pairs(self, chain_id: str, token_address: str) -> List[Dict]:
        """Fetch pairs for a specific token address on a chain."""
        pairs = await self.api.get_token_pairs(token_address)
        return [p.to_dict() for p in pairs]

    async def search_pairs(self, query: str) -> List[Dict]:
        """Search for pairs by token name, symbol, or address."""
        pairs = await self.api.search(query)
        return [p.to_dict() for p in pairs]

    async def get_pair(self, chain_id: str, pair_address: str) -> Dict:
        """Get data for a specific pair."""
        pair = await self.api.get_pair(chain_id, pair_address)
        return pair.to_dict() if pair else {}

    async def get_token_info(self, token_address: str) -> Dict:
        """Get comprehensive token info."""
        return await self.api.analyze_token(token_address)

    async def get_trending(self) -> List[Dict]:
        """Get trending/boosted tokens."""
        return await self.api.get_boosted_tokens()

    async def get_new_tokens(self, chain: str = None) -> List[Dict]:
        """Get newly created tokens."""
        pairs = await self.api.get_new_pairs(chain)
        return [p.to_dict() for p in pairs]


# Global instances
dex_screener_api = DexScreenerAPI()
dex_screener = DexScreenerSkill()
