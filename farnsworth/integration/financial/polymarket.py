"""
Farnsworth Polymarket Integration.

"The crowd is usually right, but I'm righter."

Full-featured Polymarket API integration:
- Market discovery and search
- Real-time odds/prices
- Market analysis and trends
- Event tracking
- Order book data
- Historical data
"""

import aiohttp
import asyncio
from loguru import logger
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum


class MarketStatus(Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class PolymarketEvent:
    """Structured Polymarket event data."""
    id: str
    title: str
    description: str = ""
    slug: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    volume: float = 0.0
    liquidity: float = 0.0
    markets: List["PolymarketMarket"] = field(default_factory=list)
    category: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "slug": self.slug,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "category": self.category,
            "tags": self.tags,
            "market_count": len(self.markets)
        }


@dataclass
class PolymarketMarket:
    """Structured Polymarket market data."""
    id: str
    question: str
    description: str = ""
    outcome_yes: str = "Yes"
    outcome_no: str = "No"
    price_yes: float = 0.5
    price_no: float = 0.5
    volume: float = 0.0
    liquidity: float = 0.0
    status: MarketStatus = MarketStatus.ACTIVE
    end_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    winning_outcome: Optional[str] = None
    event_id: str = ""
    url: str = ""

    @property
    def implied_probability_yes(self) -> float:
        return self.price_yes * 100

    @property
    def implied_probability_no(self) -> float:
        return self.price_no * 100

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "question": self.question,
            "description": self.description,
            "price_yes": self.price_yes,
            "price_no": self.price_no,
            "probability_yes": f"{self.implied_probability_yes:.1f}%",
            "probability_no": f"{self.implied_probability_no:.1f}%",
            "volume": self.volume,
            "liquidity": self.liquidity,
            "status": self.status.value,
            "url": self.url
        }


class PolymarketAPI:
    """
    Full Polymarket API client using the Gamma API.

    Endpoints:
    - /markets - List all markets
    - /markets/{id} - Get specific market
    - /events - List all events
    - /events/{id} - Get specific event
    """

    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"

    # Categories
    CATEGORIES = [
        "politics",
        "crypto",
        "sports",
        "entertainment",
        "science",
        "business",
        "world"
    ]

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 60  # 1 minute cache

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"}
            )
        return self._session

    async def _request(self, url: str, params: Dict = None, use_cache: bool = True) -> Optional[Any]:
        """Make an API request with caching."""
        cache_key = f"{url}:{str(params)}"

        if use_cache and cache_key in self._cache:
            data, timestamp = self._cache[cache_key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return data

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._cache[cache_key] = (data, datetime.now().timestamp())
                    return data
                elif resp.status == 429:
                    logger.warning("Polymarket rate limited")
                    await asyncio.sleep(2)
                    return await self._request(url, params, use_cache=False)
                else:
                    logger.error(f"Polymarket API error: {resp.status}")
                    return None
        except Exception as e:
            logger.error(f"Polymarket request error: {e}")
            return None

    def _parse_market(self, data: Dict) -> PolymarketMarket:
        """Parse market data into structured format."""
        # Handle different price formats
        price_yes = 0.5
        price_no = 0.5

        if "outcomePrices" in data:
            prices = data["outcomePrices"]
            if isinstance(prices, list) and len(prices) >= 2:
                price_yes = float(prices[0]) if prices[0] else 0.5
                price_no = float(prices[1]) if prices[1] else 0.5
            elif isinstance(prices, str):
                try:
                    import json
                    prices = json.loads(prices)
                    price_yes = float(prices[0]) if prices[0] else 0.5
                    price_no = float(prices[1]) if prices[1] else 0.5
                except:
                    pass

        # Parse status
        status = MarketStatus.ACTIVE
        if data.get("closed"):
            status = MarketStatus.CLOSED
        elif data.get("resolved"):
            status = MarketStatus.RESOLVED

        # Parse end date
        end_date = None
        if data.get("endDate"):
            try:
                end_date = datetime.fromisoformat(data["endDate"].replace("Z", "+00:00"))
            except:
                pass

        return PolymarketMarket(
            id=data.get("id", data.get("conditionId", "")),
            question=data.get("question", data.get("title", "")),
            description=data.get("description", ""),
            outcome_yes=data.get("outcomes", ["Yes", "No"])[0] if data.get("outcomes") else "Yes",
            outcome_no=data.get("outcomes", ["Yes", "No"])[1] if data.get("outcomes") and len(data["outcomes"]) > 1 else "No",
            price_yes=price_yes,
            price_no=price_no,
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            status=status,
            end_date=end_date,
            winning_outcome=data.get("winningOutcome"),
            event_id=data.get("eventSlug", ""),
            url=f"https://polymarket.com/event/{data.get('eventSlug', '')}" if data.get("eventSlug") else ""
        )

    def _parse_event(self, data: Dict) -> PolymarketEvent:
        """Parse event data into structured format."""
        markets = []
        if data.get("markets"):
            markets = [self._parse_market(m) for m in data["markets"]]

        return PolymarketEvent(
            id=data.get("id", ""),
            title=data.get("title", ""),
            description=data.get("description", ""),
            slug=data.get("slug", ""),
            volume=float(data.get("volume", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            markets=markets,
            category=data.get("category", ""),
            tags=data.get("tags", [])
        )

    # ==================== MARKETS ====================

    async def get_markets(
        self,
        active: bool = True,
        limit: int = 20,
        offset: int = 0,
        order: str = "volume"
    ) -> List[PolymarketMarket]:
        """Get list of markets."""
        params = {
            "active": str(active).lower(),
            "limit": limit,
            "offset": offset,
            "order": order
        }

        data = await self._request(f"{self.GAMMA_URL}/markets", params)
        if data:
            if isinstance(data, list):
                return [self._parse_market(m) for m in data]
            elif isinstance(data, dict) and "markets" in data:
                return [self._parse_market(m) for m in data["markets"]]
        return []

    async def get_market(self, market_id: str) -> Optional[PolymarketMarket]:
        """Get specific market by ID."""
        data = await self._request(f"{self.GAMMA_URL}/markets/{market_id}")
        if data:
            return self._parse_market(data)
        return None

    async def search_markets(self, query: str, limit: int = 20) -> List[PolymarketMarket]:
        """Search markets by query."""
        params = {
            "q": query,
            "limit": limit,
            "active": "true"
        }

        data = await self._request(f"{self.GAMMA_URL}/markets", params)
        if data:
            if isinstance(data, list):
                return [self._parse_market(m) for m in data]
        return []

    # ==================== EVENTS ====================

    async def get_events(
        self,
        active: bool = True,
        limit: int = 20,
        category: str = None
    ) -> List[PolymarketEvent]:
        """Get list of events."""
        params = {
            "active": str(active).lower(),
            "limit": limit
        }
        if category:
            params["tag"] = category

        data = await self._request(f"{self.GAMMA_URL}/events", params)
        if data:
            if isinstance(data, list):
                return [self._parse_event(e) for e in data]
        return []

    async def get_event(self, event_id: str) -> Optional[PolymarketEvent]:
        """Get specific event by ID or slug."""
        data = await self._request(f"{self.GAMMA_URL}/events/{event_id}")
        if data:
            return self._parse_event(data)
        return None

    async def search_events(self, query: str, limit: int = 20) -> List[PolymarketEvent]:
        """Search events by query."""
        params = {
            "q": query,
            "limit": limit,
            "active": "true"
        }

        data = await self._request(f"{self.GAMMA_URL}/events", params)
        if data:
            if isinstance(data, list):
                return [self._parse_event(e) for e in data]
        return []

    # ==================== TRENDING & ANALYSIS ====================

    async def get_trending(self, limit: int = 10) -> List[PolymarketMarket]:
        """Get trending markets by volume."""
        return await self.get_markets(active=True, limit=limit, order="volume")

    async def get_closing_soon(self, hours: int = 24, limit: int = 10) -> List[PolymarketMarket]:
        """Get markets closing within specified hours."""
        markets = await self.get_markets(active=True, limit=100)

        closing_soon = []
        cutoff = datetime.now() + timedelta(hours=hours)

        for market in markets:
            if market.end_date and market.end_date <= cutoff:
                closing_soon.append(market)

        # Sort by end date
        closing_soon.sort(key=lambda m: m.end_date or datetime.max)
        return closing_soon[:limit]

    async def get_high_volume(self, min_volume: float = 100000, limit: int = 20) -> List[PolymarketMarket]:
        """Get high-volume markets."""
        markets = await self.get_markets(active=True, limit=100, order="volume")
        return [m for m in markets if m.volume >= min_volume][:limit]

    async def get_close_odds(self, max_spread: float = 0.1, limit: int = 20) -> List[PolymarketMarket]:
        """Get markets with close odds (competitive markets)."""
        markets = await self.get_markets(active=True, limit=100)

        close_markets = []
        for market in markets:
            spread = abs(market.price_yes - 0.5)
            if spread <= max_spread:
                close_markets.append((market, spread))

        # Sort by how close to 50/50
        close_markets.sort(key=lambda x: x[1])
        return [m for m, _ in close_markets[:limit]]

    async def analyze_market(self, market_id: str) -> Dict:
        """Comprehensive market analysis."""
        market = await self.get_market(market_id)
        if not market:
            return {"error": "Market not found"}

        # Determine sentiment
        if market.price_yes > 0.7:
            sentiment = "STRONGLY YES"
        elif market.price_yes > 0.55:
            sentiment = "LEANING YES"
        elif market.price_yes < 0.3:
            sentiment = "STRONGLY NO"
        elif market.price_yes < 0.45:
            sentiment = "LEANING NO"
        else:
            sentiment = "TOSS-UP"

        # Calculate value indicators
        edge_yes = None
        edge_no = None

        return {
            "market": market.to_dict(),
            "analysis": {
                "sentiment": sentiment,
                "confidence": abs(market.price_yes - 0.5) * 200,  # 0-100 scale
                "volume_tier": "HIGH" if market.volume > 100000 else "MEDIUM" if market.volume > 10000 else "LOW",
                "liquidity_tier": "HIGH" if market.liquidity > 50000 else "MEDIUM" if market.liquidity > 5000 else "LOW",
            },
            "recommendation": {
                "position": "YES" if market.price_yes < 0.4 else "NO" if market.price_yes > 0.6 else "WAIT",
                "reason": f"Market sentiment is {sentiment} with {market.volume:,.0f} volume"
            }
        }

    # ==================== CATEGORY BROWSING ====================

    async def get_politics_markets(self, limit: int = 20) -> List[PolymarketMarket]:
        """Get political prediction markets."""
        events = await self.get_events(active=True, limit=50, category="politics")
        markets = []
        for event in events:
            markets.extend(event.markets)
        return markets[:limit]

    async def get_crypto_markets(self, limit: int = 20) -> List[PolymarketMarket]:
        """Get crypto prediction markets."""
        events = await self.get_events(active=True, limit=50, category="crypto")
        markets = []
        for event in events:
            markets.extend(event.markets)
        return markets[:limit]

    async def get_sports_markets(self, limit: int = 20) -> List[PolymarketMarket]:
        """Get sports prediction markets."""
        events = await self.get_events(active=True, limit=50, category="sports")
        markets = []
        for event in events:
            markets.extend(event.markets)
        return markets[:limit]

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ==================== SKILL INTERFACE ====================

class PolyMarketSkill:
    """
    Simplified skill interface for agent integration.

    Compatible with the tool router and agent system.
    """

    def __init__(self):
        self.api = PolymarketAPI()

    async def get_active_markets(self, limit: int = 10) -> List[Dict]:
        """Fetch active prediction markets."""
        markets = await self.api.get_markets(active=True, limit=limit)
        return [m.to_dict() for m in markets]

    async def search_markets(self, query: str) -> List[Dict]:
        """Search Polymarket events/markets."""
        markets = await self.api.search_markets(query)
        return [m.to_dict() for m in markets]

    async def get_market_odds(self, market_id: str) -> Dict:
        """Get specific market details and prices/odds."""
        market = await self.api.get_market(market_id)
        return market.to_dict() if market else {}

    async def get_trending(self, limit: int = 10) -> List[Dict]:
        """Get trending markets."""
        markets = await self.api.get_trending(limit)
        return [m.to_dict() for m in markets]

    async def get_politics(self, limit: int = 10) -> List[Dict]:
        """Get political markets."""
        markets = await self.api.get_politics_markets(limit)
        return [m.to_dict() for m in markets]

    async def get_crypto(self, limit: int = 10) -> List[Dict]:
        """Get crypto markets."""
        markets = await self.api.get_crypto_markets(limit)
        return [m.to_dict() for m in markets]

    async def analyze(self, market_id: str) -> Dict:
        """Analyze a specific market."""
        return await self.api.analyze_market(market_id)


# Global instances
polymarket_api = PolymarketAPI()
polymarket = PolyMarketSkill()
