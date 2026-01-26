"""
Farnsworth Memecoin Tracker - Pump.fun & Bags.fm.

"Tracking the degens, one bonding curve at a time."
"""

import aiohttp
from loguru import logger
from typing import Dict, Any, List, Optional

class MemecoinSkill:
    def __init__(self):
        self.pump_base_url = "https://frontend-api.pump.fun" # Public frontend API
        self.bags_base_url = "https://api.bags.fm/v1" 
        self.bags_api_key = None

    def set_bags_api_key(self, api_key: str):
        self.bags_api_key = api_key

    async def get_pump_token(self, mint_address: str) -> Dict:
        """Fetch token details and bonding curve progress from Pump.fun."""
        url = f"{self.pump_base_url}/coins/{mint_address}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Calculate bonding curve progress (simplified)
                    # Real progress uses bonding_curve account data, but API often provides it
                    return data
                else:
                    logger.error(f"Pump.fun API error: {resp.status}")
                    return {"error": f"Token {mint_address} not found on pump.fun."}

    async def get_pump_new_tokens(self, limit: int = 10) -> List[Dict]:
        """Fetch recently created tokens on Pump.fun."""
        url = f"{self.pump_base_url}/coins?offset=0&limit={limit}&sort=created_timestamp&order=DESC"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []

    async def get_bags_token(self, token_address: str) -> Dict:
        """Fetch token info from Bags.fm."""
        if not self.bags_api_key:
            return {"error": "Bags.fm API Key not configured."}
            
        url = f"{self.bags_base_url}/tokens/{token_address}"
        headers = {"x-api-key": self.bags_api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"error": f"Bags.fm error: {resp.status}"}

    async def get_bags_trending(self) -> List[Dict]:
        """Fetch trending tokens on Bags.fm."""
        if not self.bags_api_key:
            return []
            
        url = f"{self.bags_base_url}/analytics/trending"
        headers = {"x-api-key": self.bags_api_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []

memecoin_tracker = MemecoinSkill()
