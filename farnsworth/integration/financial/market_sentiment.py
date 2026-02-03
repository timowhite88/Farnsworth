"""
Farnsworth Market Sentiment & Conditions.

"The vibes are currently: Bullish."

Comprehensive market sentiment analysis:
- Fear & Greed Index
- Global market data
- Token prices and changes
- Trending coins
- Social sentiment indicators
- Whale activity tracking
- Liquidation data
- Market dominance
"""

import aiohttp
import asyncio
from loguru import logger
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class SentimentLevel(Enum):
    EXTREME_FEAR = "Extreme Fear"
    FEAR = "Fear"
    NEUTRAL = "Neutral"
    GREED = "Greed"
    EXTREME_GREED = "Extreme Greed"


@dataclass
class MarketConditions:
    """Current market conditions snapshot."""
    fear_greed_value: int
    fear_greed_label: str
    btc_price: float
    btc_change_24h: float
    eth_price: float
    eth_change_24h: float
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: float
    eth_dominance: float
    defi_market_cap: float
    stablecoin_volume: float
    timestamp: datetime

    @property
    def sentiment_level(self) -> SentimentLevel:
        if self.fear_greed_value <= 20:
            return SentimentLevel.EXTREME_FEAR
        elif self.fear_greed_value <= 40:
            return SentimentLevel.FEAR
        elif self.fear_greed_value <= 60:
            return SentimentLevel.NEUTRAL
        elif self.fear_greed_value <= 80:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.EXTREME_GREED

    @property
    def market_bias(self) -> str:
        """Determine overall market bias."""
        signals = []

        # Fear & Greed signal
        if self.fear_greed_value < 30:
            signals.append(-1)  # Bearish
        elif self.fear_greed_value > 70:
            signals.append(1)  # Bullish
        else:
            signals.append(0)

        # BTC price action
        if self.btc_change_24h > 5:
            signals.append(1)
        elif self.btc_change_24h < -5:
            signals.append(-1)
        else:
            signals.append(0)

        # Dominance signal (high BTC dom = risk-off)
        if self.btc_dominance > 55:
            signals.append(-0.5)
        elif self.btc_dominance < 45:
            signals.append(0.5)
        else:
            signals.append(0)

        avg = sum(signals) / len(signals)
        if avg > 0.3:
            return "BULLISH"
        elif avg < -0.3:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def to_dict(self) -> Dict:
        return {
            "fear_greed": {
                "value": self.fear_greed_value,
                "label": self.fear_greed_label,
                "level": self.sentiment_level.value
            },
            "btc": {
                "price": self.btc_price,
                "change_24h": self.btc_change_24h
            },
            "eth": {
                "price": self.eth_price,
                "change_24h": self.eth_change_24h
            },
            "market": {
                "total_cap": self.total_market_cap,
                "total_volume": self.total_volume_24h,
                "btc_dominance": self.btc_dominance,
                "eth_dominance": self.eth_dominance,
                "defi_cap": self.defi_market_cap
            },
            "bias": self.market_bias,
            "timestamp": self.timestamp.isoformat()
        }


class MarketSentimentAPI:
    """
    Comprehensive market sentiment data aggregator.

    Data sources:
    - Alternative.me Fear & Greed Index
    - CoinGecko market data
    - CoinGlass liquidations (if available)
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    COINGECKO_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 60  # 1 minute

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"}
            )
        return self._session

    async def _request(self, url: str, use_cache: bool = True) -> Optional[Dict]:
        """Make an API request with caching."""
        if use_cache and url in self._cache:
            data, timestamp = self._cache[url]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data

        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._cache[url] = (data, datetime.now().timestamp())
                    return data
                elif resp.status == 429:
                    logger.warning("Market API rate limited")
                    await asyncio.sleep(2)
                    return await self._request(url, use_cache=False)
                else:
                    logger.debug(f"Market API returned {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Market request error: {e}")
            return None

    # ==================== FEAR & GREED ====================

    async def get_fear_and_greed(self) -> Dict:
        """Fetch the current Crypto Fear & Greed Index."""
        data = await self._request(self.FEAR_GREED_URL)
        if data and data.get("data"):
            return data["data"][0]
        return {}

    async def get_fear_greed_history(self, days: int = 30) -> List[Dict]:
        """Get Fear & Greed Index history."""
        data = await self._request(f"{self.FEAR_GREED_URL}?limit={days}")
        if data and data.get("data"):
            return data["data"]
        return []

    # ==================== GLOBAL MARKET DATA ====================

    async def get_global_market_cap(self) -> Dict:
        """Fetch global crypto market data from CoinGecko."""
        data = await self._request(f"{self.COINGECKO_URL}/global")
        if data:
            return data.get("data", {})
        return {}

    async def get_defi_market_cap(self) -> Dict:
        """Fetch DeFi market data."""
        data = await self._request(f"{self.COINGECKO_URL}/global/decentralized_finance_defi")
        if data:
            return data.get("data", {})
        return {}

    # ==================== TOKEN PRICES ====================

    async def get_token_price(self, token_id: str = "bitcoin") -> Dict:
        """Fetch current price and 24h change for a token."""
        url = f"{self.COINGECKO_URL}/simple/price?ids={token_id}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true"
        data = await self._request(url)
        if data:
            return data.get(token_id, {})
        return {}

    async def get_multiple_prices(self, token_ids: List[str]) -> Dict:
        """Fetch prices for multiple tokens."""
        ids = ",".join(token_ids)
        url = f"{self.COINGECKO_URL}/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
        return await self._request(url) or {}

    async def get_token_details(self, token_id: str) -> Dict:
        """Get detailed token information."""
        url = f"{self.COINGECKO_URL}/coins/{token_id}"
        return await self._request(url) or {}

    # ==================== TRENDING ====================

    async def get_trending_coins(self) -> List[Dict]:
        """Get trending coins on CoinGecko."""
        data = await self._request(f"{self.COINGECKO_URL}/search/trending")
        if data and data.get("coins"):
            return [coin["item"] for coin in data["coins"]]
        return []

    async def get_top_gainers(self, limit: int = 10) -> List[Dict]:
        """Get top gaining coins in 24h."""
        url = f"{self.COINGECKO_URL}/coins/markets?vs_currency=usd&order=price_change_percentage_24h_desc&per_page={limit}&page=1&sparkline=false"
        return await self._request(url) or []

    async def get_top_losers(self, limit: int = 10) -> List[Dict]:
        """Get top losing coins in 24h."""
        url = f"{self.COINGECKO_URL}/coins/markets?vs_currency=usd&order=price_change_percentage_24h_asc&per_page={limit}&page=1&sparkline=false"
        return await self._request(url) or []

    async def get_top_volume(self, limit: int = 10) -> List[Dict]:
        """Get coins with highest 24h volume."""
        url = f"{self.COINGECKO_URL}/coins/markets?vs_currency=usd&order=volume_desc&per_page={limit}&page=1&sparkline=false"
        return await self._request(url) or []

    # ==================== MARKET CONDITIONS ====================

    async def get_market_conditions(self) -> MarketConditions:
        """Get comprehensive market conditions snapshot."""
        # Fetch all data in parallel
        fng_task = self.get_fear_and_greed()
        global_task = self.get_global_market_cap()
        btc_task = self.get_token_price("bitcoin")
        eth_task = self.get_token_price("ethereum")
        defi_task = self.get_defi_market_cap()

        fng, global_data, btc_data, eth_data, defi_data = await asyncio.gather(
            fng_task, global_task, btc_task, eth_task, defi_task,
            return_exceptions=True
        )

        # Handle exceptions
        if isinstance(fng, Exception):
            fng = {}
        if isinstance(global_data, Exception):
            global_data = {}
        if isinstance(btc_data, Exception):
            btc_data = {}
        if isinstance(eth_data, Exception):
            eth_data = {}
        if isinstance(defi_data, Exception):
            defi_data = {}

        return MarketConditions(
            fear_greed_value=int(fng.get("value", 50)),
            fear_greed_label=fng.get("value_classification", "Neutral"),
            btc_price=btc_data.get("usd", 0),
            btc_change_24h=btc_data.get("usd_24h_change", 0),
            eth_price=eth_data.get("usd", 0),
            eth_change_24h=eth_data.get("usd_24h_change", 0),
            total_market_cap=global_data.get("total_market_cap", {}).get("usd", 0),
            total_volume_24h=global_data.get("total_volume", {}).get("usd", 0),
            btc_dominance=global_data.get("market_cap_percentage", {}).get("btc", 0),
            eth_dominance=global_data.get("market_cap_percentage", {}).get("eth", 0),
            defi_market_cap=float(defi_data.get("defi_market_cap", 0) or 0),
            stablecoin_volume=float(defi_data.get("stablecoin_volume_24h", 0) or 0) if defi_data else 0,
            timestamp=datetime.utcnow()
        )

    # ==================== ANALYSIS ====================

    async def get_market_summary(self) -> Dict:
        """Get a comprehensive market summary."""
        conditions = await self.get_market_conditions()
        trending = await self.get_trending_coins()
        gainers = await self.get_top_gainers(5)
        losers = await self.get_top_losers(5)

        return {
            "conditions": conditions.to_dict(),
            "trending": [{"name": c.get("name"), "symbol": c.get("symbol")} for c in trending[:5]],
            "top_gainers": [{"name": c.get("name"), "change": c.get("price_change_percentage_24h")} for c in gainers],
            "top_losers": [{"name": c.get("name"), "change": c.get("price_change_percentage_24h")} for c in losers],
            "summary": self._generate_summary(conditions)
        }

    def _generate_summary(self, conditions: MarketConditions) -> str:
        """Generate a human-readable market summary."""
        bias = conditions.market_bias
        fng = conditions.fear_greed_label

        if bias == "BULLISH":
            if conditions.fear_greed_value > 70:
                return f"Markets are in {fng} territory. Strong bullish momentum but watch for overextension."
            else:
                return f"Bullish conditions emerging. Fear & Greed at {conditions.fear_greed_value} ({fng})."
        elif bias == "BEARISH":
            if conditions.fear_greed_value < 30:
                return f"Markets showing {fng}. Potential capitulation or buying opportunity."
            else:
                return f"Bearish pressure building. Fear & Greed at {conditions.fear_greed_value} ({fng})."
        else:
            return f"Markets consolidating. Fear & Greed at {conditions.fear_greed_value} ({fng}). Wait for direction."

    async def should_buy(self, token_id: str = "bitcoin") -> Dict:
        """Simple buy signal based on sentiment."""
        conditions = await self.get_market_conditions()
        token = await self.get_token_price(token_id)

        signals = {
            "fear_greed": conditions.fear_greed_value < 30,  # Extreme fear = buy
            "price_dip": token.get("usd_24h_change", 0) < -5,  # 5%+ dip
            "market_bias": conditions.market_bias != "BEARISH"
        }

        buy_signals = sum(1 for v in signals.values() if v)
        total_signals = len(signals)

        recommendation = "BUY" if buy_signals >= 2 else "WAIT" if buy_signals == 1 else "AVOID"

        return {
            "token": token_id,
            "recommendation": recommendation,
            "confidence": (buy_signals / total_signals) * 100,
            "signals": signals,
            "conditions": {
                "fear_greed": conditions.fear_greed_value,
                "market_bias": conditions.market_bias,
                "price_change": token.get("usd_24h_change", 0)
            }
        }

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ==================== SKILL INTERFACE ====================

class MarketSentimentSkill:
    """
    Simplified skill interface for agent integration.

    Compatible with the tool router and agent system.
    """

    def __init__(self):
        self.api = MarketSentimentAPI()

    async def get_fear_and_greed(self) -> Dict:
        """Fetch the current Crypto Fear & Greed Index."""
        return await self.api.get_fear_and_greed()

    async def get_global_market_cap(self) -> Dict:
        """Fetch global crypto market data from CoinGecko."""
        return await self.api.get_global_market_cap()

    async def get_token_price(self, token_id: str = "bitcoin") -> Dict:
        """Fetch current price and 24h change for a token."""
        return await self.api.get_token_price(token_id)

    async def get_market_conditions(self) -> Dict:
        """Get comprehensive market conditions."""
        conditions = await self.api.get_market_conditions()
        return conditions.to_dict()

    async def get_trending(self) -> List[Dict]:
        """Get trending coins."""
        return await self.api.get_trending_coins()

    async def get_gainers(self, limit: int = 10) -> List[Dict]:
        """Get top gainers."""
        return await self.api.get_top_gainers(limit)

    async def get_losers(self, limit: int = 10) -> List[Dict]:
        """Get top losers."""
        return await self.api.get_top_losers(limit)

    async def get_summary(self) -> Dict:
        """Get market summary."""
        return await self.api.get_market_summary()

    async def should_buy(self, token: str = "bitcoin") -> Dict:
        """Get buy recommendation."""
        return await self.api.should_buy(token)


# Global instances
market_sentiment_api = MarketSentimentAPI()
market_sentiment = MarketSentimentSkill()
