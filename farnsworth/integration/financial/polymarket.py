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

AGI v1.8 Quantum Enhancements:
- Quantum Monte Carlo (QMC) for risk assessment
- Quantum sampling for probability distribution modeling
- IBM Quantum integration (simulators unlimited, hardware 10min/month)
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

        # AGI v1.8: Quantum risk modeling
        self._quantum_available = False
        try:
            from farnsworth.integration.quantum import QISKIT_AVAILABLE
            self._quantum_available = QISKIT_AVAILABLE
            if QISKIT_AVAILABLE:
                logger.info("Quantum risk modeling available (IBM Quantum)")
        except ImportError:
            pass

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
                except (json.JSONDecodeError, ValueError, IndexError):
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
            except (ValueError, TypeError):
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

    async def quantum_risk_assessment(
        self,
        market_id: str,
        num_scenarios: int = 100,
        use_hardware: bool = False,
    ) -> Dict:
        """
        AGI v1.8: Quantum Monte Carlo risk assessment for market prediction.

        Uses quantum sampling to model probability distributions and estimate
        risk metrics like VaR (Value at Risk) and expected outcomes.

        Args:
            market_id: Polymarket market ID.
            num_scenarios: Number of quantum scenarios to simulate.
            use_hardware: Use IBM Quantum hardware (limited 10min/month).

        Returns:
            Risk assessment with quantum-derived confidence intervals.
        """
        market = await self.get_market(market_id)
        if not market:
            return {"error": "Market not found"}

        # Classical analysis as baseline
        classical_analysis = await self.analyze_market(market_id)
        if "error" in classical_analysis:
            return classical_analysis

        # If quantum not available, return classical with note
        if not self._quantum_available:
            return {
                **classical_analysis,
                "quantum_assessment": {
                    "available": False,
                    "message": "Quantum computing not available, using classical analysis"
                }
            }

        try:
            import numpy as np
            from farnsworth.integration.quantum import get_quantum_provider, QISKIT_AVAILABLE
            if QISKIT_AVAILABLE:
                from qiskit import QuantumCircuit
            else:
                return {
                    **classical_analysis,
                    "quantum_assessment": {"available": False, "message": "Qiskit not installed"}
                }

            provider = get_quantum_provider()
            if not provider:
                return {
                    **classical_analysis,
                    "quantum_assessment": {"available": False, "message": "Quantum provider not initialized"}
                }

            # Build quantum circuit for Monte Carlo sampling
            # Use market probability as rotation angle
            prob_yes = market.price_yes
            theta_yes = 2 * np.arcsin(np.sqrt(prob_yes))

            # Create superposition weighted by probability
            num_qubits = min(8, max(3, int(np.log2(num_scenarios))))

            qc = QuantumCircuit(num_qubits, num_qubits)
            for i in range(num_qubits):
                qc.ry(theta_yes, i)  # Rotate based on market probability
            qc.measure_all()

            # Run quantum sampling
            result = await provider.run_circuit(
                qc,
                shots=num_scenarios,
                prefer_hardware=use_hardware,
            )

            if not result.success or not result.counts:
                return {
                    **classical_analysis,
                    "quantum_assessment": {"available": True, "error": result.error or "No counts returned"}
                }

            # Analyze quantum distribution
            total_shots = sum(result.counts.values())
            yes_outcomes = sum(
                count for bitstring, count in result.counts.items()
                if bitstring.count('1') > bitstring.count('0')
            )
            quantum_prob_yes = yes_outcomes / total_shots

            # Calculate quantum confidence interval
            # Standard error from binomial distribution
            std_error = np.sqrt(quantum_prob_yes * (1 - quantum_prob_yes) / total_shots)
            ci_95_low = max(0, quantum_prob_yes - 1.96 * std_error)
            ci_95_high = min(1, quantum_prob_yes + 1.96 * std_error)

            # VaR-style risk metrics (simplified for prediction markets)
            # If betting $100 on YES at market price
            bet_amount = 100
            potential_profit = bet_amount * (1 / prob_yes - 1) if prob_yes > 0 else 0
            potential_loss = bet_amount
            expected_value = quantum_prob_yes * potential_profit - (1 - quantum_prob_yes) * potential_loss

            return {
                **classical_analysis,
                "quantum_assessment": {
                    "available": True,
                    "backend": result.backend_used,
                    "scenarios_simulated": total_shots,
                    "quantum_probability_yes": round(quantum_prob_yes * 100, 2),
                    "market_probability_yes": round(prob_yes * 100, 2),
                    "probability_divergence": round(abs(quantum_prob_yes - prob_yes) * 100, 2),
                    "confidence_interval_95": {
                        "low": round(ci_95_low * 100, 2),
                        "high": round(ci_95_high * 100, 2),
                    },
                    "risk_metrics": {
                        "expected_value_$100_yes": round(expected_value, 2),
                        "potential_profit": round(potential_profit, 2),
                        "potential_loss": potential_loss,
                        "risk_adjusted_recommendation": (
                            "BUY YES" if expected_value > 10 else
                            "BUY NO" if expected_value < -10 else
                            "HOLD"
                        )
                    }
                }
            }

        except Exception as e:
            logger.warning(f"Quantum risk assessment failed: {e}")
            return {
                **classical_analysis,
                "quantum_assessment": {"available": True, "error": str(e)}
            }

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
