"""
Farnsworth Web Server - Full Feature Interface
Token-gated chat interface with ALL local features exposed
Real-time WebSocket for live action graphs and thinking states

Features Available WITHOUT External APIs:
- Memory system (remember/recall)
- Notes management
- Snippets management
- Focus timer (Pomodoro)
- Daily summaries
- Context profiles
- Agent delegation (local)
- Health tracking (mock/local)
- Sequential thinking
- Causal reasoning
- Code analysis
- System diagnostics
"""

import os
import json
import logging
import asyncio
import sys
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Optional Solana imports
try:
    from solana.rpc.api import Client as SolanaClient
    from solders.pubkey import Pubkey
    SOLANA_AVAILABLE = True
except ImportError:
    SOLANA_AVAILABLE = False

# Optional Ollama imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Optional TTS imports for voice cloning
try:
    import torch
    import torchaudio
    import soundfile as sf
    import numpy as np

    # Patch torch.load for PyTorch 2.6+ compatibility with TTS library
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    # Patch torchaudio.load to use soundfile for compatibility
    _original_torchaudio_load = torchaudio.load
    def _patched_torchaudio_load(filepath, *args, **kwargs):
        try:
            data, sr = sf.read(filepath, dtype='float32')
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            else:
                data = data.T
            return torch.from_numpy(data), sr
        except Exception:
            return _original_torchaudio_load(filepath, *args, **kwargs)
    torchaudio.load = _patched_torchaudio_load

    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Optional Planetary Audio Shard
try:
    from farnsworth.core.memory.planetary.audio_shard import (
        PlanetaryAudioShard, AudioScope, get_audio_shard
    )
    AUDIO_SHARD_AVAILABLE = True
except ImportError:
    AUDIO_SHARD_AVAILABLE = False

# Optional P2P Swarm Fabric for distributed learning
try:
    from farnsworth.core.swarm.p2p import swarm_fabric
    P2P_FABRIC_AVAILABLE = True
except ImportError:
    swarm_fabric = None
    P2P_FABRIC_AVAILABLE = False

# Optional Collective Organism for unified intelligence
try:
    from farnsworth.core.collective import organism as collective_organism
    ORGANISM_AVAILABLE = True
except ImportError:
    collective_organism = None
    ORGANISM_AVAILABLE = False

# Optional Swarm Orchestrator for turn-taking and consciousness training
try:
    from farnsworth.core.collective.orchestration import swarm_orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    swarm_orchestrator = None
    ORCHESTRATOR_AVAILABLE = False

# Optional Evolution Engine for code-level learning
try:
    from farnsworth.core.collective.evolution import evolution_engine
    EVOLUTION_AVAILABLE = True
except ImportError:
    evolution_engine = None
    EVOLUTION_AVAILABLE = False

# Claude Code CLI integration (uses Claude Max subscription)
try:
    from farnsworth.integration.external.claude_code import get_claude_code, claude_swarm_respond
    CLAUDE_CODE_AVAILABLE = True
except ImportError:
    get_claude_code = None
    claude_swarm_respond = None
    CLAUDE_CODE_AVAILABLE = False

# Kimi (Moonshot AI) integration
try:
    from farnsworth.integration.external.kimi import get_kimi_provider, kimi_swarm_respond
    KIMI_AVAILABLE = True
except ImportError:
    get_kimi_provider = None
    kimi_swarm_respond = None
    KIMI_AVAILABLE = False

# Farnsworth module imports (lazy-loaded)
_memory_system = None
_notes_manager = None
_snippet_manager = None
_focus_timer = None
_context_profiles = None
_health_analyzer = None
_tool_router = None
_sequential_thinking = None
_tts_model = None
_audio_shard = None

def get_memory_system():
    """Lazy-load memory system."""
    global _memory_system
    if _memory_system is None:
        try:
            from farnsworth.memory.memory_system import MemorySystem
            _memory_system = MemorySystem()
            logger.info("Memory system loaded")
        except Exception as e:
            logger.warning(f"Could not load memory system: {e}")
    return _memory_system

def get_notes_manager():
    """Lazy-load notes manager."""
    global _notes_manager
    if _notes_manager is None:
        try:
            from farnsworth.tools.productivity.quick_notes import QuickNotes
            _notes_manager = QuickNotes()
            logger.info("Notes manager loaded")
        except Exception as e:
            logger.warning(f"Could not load notes manager: {e}")
    return _notes_manager

def get_snippet_manager():
    """Lazy-load snippet manager."""
    global _snippet_manager
    if _snippet_manager is None:
        try:
            from farnsworth.tools.productivity.snippet_manager import SnippetManager
            _snippet_manager = SnippetManager()
            logger.info("Snippet manager loaded")
        except Exception as e:
            logger.warning(f"Could not load snippet manager: {e}")
    return _snippet_manager

def get_focus_timer():
    """Lazy-load focus timer."""
    global _focus_timer
    if _focus_timer is None:
        try:
            from farnsworth.tools.productivity.focus_timer import FocusTimer
            _focus_timer = FocusTimer()
            logger.info("Focus timer loaded")
        except Exception as e:
            logger.warning(f"Could not load focus timer: {e}")
    return _focus_timer

def get_context_profiles():
    """Lazy-load context profiles."""
    global _context_profiles
    if _context_profiles is None:
        try:
            from farnsworth.core.context_profiles import ContextProfileManager
            _context_profiles = ContextProfileManager()
            logger.info("Context profiles loaded")
        except Exception as e:
            logger.warning(f"Could not load context profiles: {e}")
    return _context_profiles

def get_health_analyzer():
    """Lazy-load health analyzer."""
    global _health_analyzer
    if _health_analyzer is None:
        try:
            from farnsworth.health.analysis import HealthAnalyzer
            from farnsworth.health.providers.mock import MockHealthProvider
            provider = MockHealthProvider()
            _health_analyzer = HealthAnalyzer(provider)
            logger.info("Health analyzer loaded with mock provider")
        except Exception as e:
            logger.warning(f"Could not load health analyzer: {e}")
    return _health_analyzer

def get_tool_router():
    """Lazy-load tool router."""
    global _tool_router
    if _tool_router is None:
        try:
            from farnsworth.integration.tool_router import ToolRouter
            _tool_router = ToolRouter()
            logger.info("Tool router loaded")
        except Exception as e:
            logger.warning(f"Could not load tool router: {e}")
    return _tool_router

def get_sequential_thinking():
    """Lazy-load sequential thinking."""
    global _sequential_thinking
    if _sequential_thinking is None:
        try:
            from farnsworth.core.cognition.sequential_thinking import SequentialThinkingEngine
            _sequential_thinking = SequentialThinkingEngine()
            logger.info("Sequential thinking loaded")
        except Exception as e:
            logger.warning(f"Could not load sequential thinking: {e}")
    return _sequential_thinking

def get_tts_model():
    """Lazy-load XTTS v2 model for voice cloning."""
    global _tts_model
    if _tts_model is None:
        if not TTS_AVAILABLE:
            logger.warning("TTS library not available")
            return None
        try:
            _tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            if torch.cuda.is_available():
                _tts_model = _tts_model.to("cuda")
                logger.info("TTS model loaded on GPU")
            else:
                logger.info("TTS model loaded on CPU")
        except Exception as e:
            logger.warning(f"Could not load TTS model: {e}")
    return _tts_model

def get_planetary_audio_shard():
    """Lazy-load Planetary Audio Shard for distributed TTS caching."""
    global _audio_shard
    if _audio_shard is None:
        if not AUDIO_SHARD_AVAILABLE:
            return None
        try:
            # Compute static dir relative to this file (STATIC_DIR may not be defined yet)
            web_dir = Path(__file__).parent
            cache_dir = web_dir / "static" / "audio" / "cache"
            _audio_shard = get_audio_shard(cache_dir)
            logger.info(f"Planetary Audio Shard loaded: {_audio_shard.get_stats()}")
        except Exception as e:
            logger.warning(f"Could not load Planetary Audio Shard: {e}")
    return _audio_shard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SOLANA_RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
PRIMARY_MODEL = os.getenv("FARNSWORTH_PRIMARY_MODEL", "llama3.2:3b")
DEMO_MODE = os.getenv("FARNSWORTH_DEMO_MODE", "true").lower() == "true"


def extract_ollama_content(response, max_length: int = 500) -> str:
    """
    Safely extract content from Ollama response.
    Handles deepseek-r1 models which put response in 'thinking' field.
    """
    try:
        msg = response.message if hasattr(response, 'message') else response.get("message", {})
        content = getattr(msg, 'content', '') or msg.get('content', '') or ''
        thinking = getattr(msg, 'thinking', '') or msg.get('thinking', '') or ''

        # Use content if available, otherwise use thinking (deepseek-r1 behavior)
        result = content.strip() if content.strip() else thinking.strip()

        # Limit length
        if max_length and len(result) > max_length:
            result = result[:max_length] + "..."

        return result
    except Exception as e:
        logger.error(f"Error extracting Ollama content: {e}")
        return ""


# ============================================
# SECURITY: Blocked patterns for chat input
# ============================================
BLOCKED_PATTERNS = [
    # Code execution attempts
    r'(?i)exec\s*\(',
    r'(?i)eval\s*\(',
    r'(?i)__import__',
    r'(?i)subprocess',
    r'(?i)os\.system',
    r'(?i)os\.popen',
    r'(?i)commands\.',
    r'(?i)shell\s*=\s*true',
    # File system access
    r'(?i)open\s*\([^)]*[\'"][wra]',
    r'(?i)write\s*\(',
    r'(?i)unlink\s*\(',
    r'(?i)rmdir\s*\(',
    r'(?i)shutil\.',
    # Server modification attempts
    r'(?i)restart\s+server',
    r'(?i)modify\s+server',
    r'(?i)change\s+code',
    r'(?i)edit\s+file',
    r'(?i)update\s+server\.py',
    r'(?i)rm\s+-rf',
    r'(?i)sudo\s+',
    # Injection attempts
    r'(?i)<script',
    r'(?i)javascript:',
    r'(?i)on\w+\s*=',
]

def is_safe_input(text: str) -> tuple[bool, str]:
    """Check if user input is safe (no code execution attempts)."""
    import re
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text):
            return False, f"Blocked pattern detected. This is a chat interface, not a code execution environment."
    return True, ""


# ============================================
# INTELLIGENT CRYPTO QUERY PARSER
# ============================================

class CryptoQueryParser:
    """
    Parses natural language to detect crypto/token queries and contract addresses.
    Automatically triggers appropriate tools.
    """

    # Solana address pattern (base58, 32-50 chars to handle edge cases)
    SOLANA_ADDRESS_PATTERN = r'\b[1-9A-HJ-NP-Za-km-z]{32,50}\b'

    # Ethereum address pattern (0x + 40 hex chars)
    ETH_ADDRESS_PATTERN = r'\b0x[a-fA-F0-9]{40}\b'

    # Natural language patterns for different intents
    INTENT_PATTERNS = {
        'price_check': [
            r'(?i)(?:what(?:\'s| is) (?:the )?price (?:of |for )?)',
            r'(?i)(?:how much is )',
            r'(?i)(?:price (?:of |for )?)',
            r'(?i)(?:check (?:the )?price)',
            r'(?i)(?:\$?\d+(?:\.\d+)?\s*(?:usd|dollars?)?\s*(?:of|worth|in)\s+)',
        ],
        'rug_check': [
            r'(?i)(?:is .* (?:safe|legit|a rug|rugged|honeypot))',
            r'(?i)(?:rug (?:check|scan|test))',
            r'(?i)(?:check (?:if )?.*(?:safe|rug|scam))',
            r'(?i)(?:scan (?:for )?(?:rug|scam|honeypot))',
            r'(?i)(?:safety (?:check|scan|analysis))',
            r'(?i)(?:is this (?:token |coin )?safe)',
        ],
        'token_info': [
            r'(?i)(?:what is |tell me about |info (?:on |about )?|lookup |look up )',
            r'(?i)(?:search (?:for )?)',
            r'(?i)(?:find (?:token |coin )?)',
            r'(?i)(?:show me )',
        ],
        'whale_track': [
            r'(?i)(?:whale (?:track|watch|activity|alert))',
            r'(?i)(?:track (?:this )?wallet)',
            r'(?i)(?:what(?:\'s| is) (?:this )?wallet doing)',
            r'(?i)(?:wallet activity)',
        ],
        'market_sentiment': [
            r'(?i)(?:market (?:sentiment|mood|fear|greed))',
            r'(?i)(?:fear (?:and |& )?greed)',
            r'(?i)(?:how(?:\'s| is) the market)',
            r'(?i)(?:market (?:feeling|vibes))',
        ]
    }

    # Common token name patterns
    TOKEN_NAMES = [
        r'(?i)\b(sol|solana)\b',
        r'(?i)\b(btc|bitcoin)\b',
        r'(?i)\b(eth|ethereum)\b',
        r'(?i)\b(bonk|wif|dogwifhat|jup|jupiter|ray|raydium|orca)\b',
        r'(?i)\b(usdc|usdt|tether)\b',
        r'(?i)\$([a-zA-Z]{2,10})\b',  # $TICKER format
    ]

    @classmethod
    def parse(cls, message: str) -> dict:
        """
        Parse a message for crypto-related queries.
        Returns: {
            'has_crypto_query': bool,
            'intent': str or None,
            'addresses': list of detected addresses,
            'token_mentions': list of token names,
            'query': extracted query string
        }
        """
        import re
        result = {
            'has_crypto_query': False,
            'intent': None,
            'addresses': [],
            'token_mentions': [],
            'query': None,
            'original': message
        }

        # Detect Solana addresses
        sol_addresses = re.findall(cls.SOLANA_ADDRESS_PATTERN, message)
        eth_addresses = re.findall(cls.ETH_ADDRESS_PATTERN, message)
        result['addresses'] = sol_addresses + eth_addresses

        # Detect token mentions
        for pattern in cls.TOKEN_NAMES:
            matches = re.findall(pattern, message)
            result['token_mentions'].extend(matches)

        # Detect intent
        for intent, patterns in cls.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message):
                    result['intent'] = intent
                    result['has_crypto_query'] = True
                    break
            if result['intent']:
                break

        # If we found addresses but no intent, default to token_info
        if result['addresses'] and not result['intent']:
            result['intent'] = 'token_info'
            result['has_crypto_query'] = True

        # Extract the query (token name or address)
        if result['addresses']:
            result['query'] = result['addresses'][0]
        elif result['token_mentions']:
            result['query'] = result['token_mentions'][0]
        else:
            # Try to extract token name from message
            # Remove common prefixes
            cleaned = re.sub(r'(?i)^(what(?:\'s| is) (?:the )?price (?:of |for )?)', '', message)
            cleaned = re.sub(r'(?i)^(is |check |scan |search |find |lookup |look up )', '', cleaned)
            cleaned = re.sub(r'(?i)(safe|legit|a rug|rugged|honeypot|\?|!|\.)+$', '', cleaned)
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) < 50:
                result['query'] = cleaned

        return result

    @classmethod
    async def execute_tool(cls, parsed: dict) -> dict:
        """Execute the appropriate tool based on parsed intent."""
        intent = parsed.get('intent')
        query = parsed.get('query')
        addresses = parsed.get('addresses', [])

        if not intent or not query:
            return None

        result = {
            'tool_used': intent,
            'query': query,
            'success': False,
            'data': None,
            'formatted': ''
        }

        try:
            if intent == 'price_check' or intent == 'token_info':
                # Use DexScreener for token lookup
                result['data'] = await cls._token_lookup(query)
                result['success'] = True
                result['formatted'] = cls._format_token_info(result['data'], query)

            elif intent == 'rug_check':
                address = addresses[0] if addresses else query
                result['data'] = await cls._rug_check(address)
                result['success'] = True
                result['formatted'] = cls._format_rug_check(result['data'], address)

            elif intent == 'whale_track':
                address = addresses[0] if addresses else query
                result['data'] = await cls._whale_track(address)
                result['success'] = True
                result['formatted'] = cls._format_whale_track(result['data'], address)

            elif intent == 'market_sentiment':
                result['data'] = await cls._market_sentiment()
                result['success'] = True
                result['formatted'] = cls._format_sentiment(result['data'])

        except Exception as e:
            logger.error(f"Crypto tool error: {e}")
            result['error'] = str(e)

        return result

    # Major tokens - use CoinGecko for accurate prices
    MAJOR_TOKENS = {
        'sol': 'solana', 'solana': 'solana',
        'btc': 'bitcoin', 'bitcoin': 'bitcoin',
        'eth': 'ethereum', 'ethereum': 'ethereum',
        'usdc': 'usd-coin', 'usdt': 'tether',
        'bonk': 'bonk', 'wif': 'dogwifhat', 'jup': 'jupiter-exchange-solana',
        'ray': 'raydium', 'orca': 'orca'
    }

    @classmethod
    async def _token_lookup(cls, query: str) -> dict:
        """Look up token info via CoinGecko for major tokens, DexScreener for others."""
        import httpx
        query_lower = query.lower().strip()

        # Check if it's a major token - use CoinGecko for accurate data
        if query_lower in cls.MAJOR_TOKENS:
            coingecko_id = cls.MAJOR_TOKENS[query_lower]
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coingecko_id}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true&include_24hr_vol=true"
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        data = resp.json()
                        if coingecko_id in data:
                            token_data = data[coingecko_id]
                            return {
                                "pairs": [{
                                    "baseToken": {"name": coingecko_id.replace('-', ' ').title(), "symbol": query_lower.upper()},
                                    "priceUsd": str(token_data.get('usd', 'N/A')),
                                    "priceChange": {"h24": token_data.get('usd_24h_change', 0)},
                                    "volume": {"h24": token_data.get('usd_24h_vol', 0)},
                                    "liquidity": {"usd": token_data.get('usd_market_cap', 0)},
                                    "dexId": "CoinGecko"
                                }],
                                "source": "coingecko"
                            }
            except Exception as e:
                logger.warning(f"CoinGecko API failed: {e}")

        # For other tokens or if CoinGecko fails, use DexScreener
        try:
            from farnsworth.integration.financial.dexscreener import DexScreenerClient
            client = DexScreenerClient()
            return await client.search_pairs(query)
        except ImportError:
            # Fallback to direct API call
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    url = f"https://api.dexscreener.com/latest/dex/search?q={query}"
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return resp.json()
            except Exception as e:
                logger.warning(f"DexScreener API call failed: {e}")
            return {"pairs": [], "demo": True}

    @classmethod
    async def _rug_check(cls, address: str) -> dict:
        """Check token for rug risks."""
        try:
            from farnsworth.integration.solana.degen_mob import DeGenMob
            degen = DeGenMob()
            return await degen.analyze_token_safety(address)
        except ImportError:
            return {
                "address": address,
                "demo": True,
                "message": "Full rug detection requires local Farnsworth install"
            }

    @classmethod
    async def _whale_track(cls, address: str) -> dict:
        """Track whale wallet."""
        try:
            from farnsworth.integration.solana.degen_mob import DeGenMob
            degen = DeGenMob()
            return await degen.get_whale_recent_activity(address)
        except ImportError:
            return {
                "address": address,
                "demo": True,
                "message": "Whale tracking requires local Farnsworth install"
            }

    @classmethod
    async def _market_sentiment(cls) -> dict:
        """Get market sentiment."""
        try:
            from farnsworth.integration.financial.market_sentiment import MarketSentiment
            sentiment = MarketSentiment()
            return await sentiment.get_fear_and_greed()
        except ImportError:
            return {"index": 50, "classification": "Neutral", "demo": True}

    @classmethod
    def _format_token_info(cls, data: dict, query: str) -> str:
        """Format token info for chat display."""
        pairs = data.get('pairs', [])
        if not pairs:
            return f"🔍 No trading pairs found for **{query}**. Try a contract address or different name."

        # Get first/best pair
        pair = pairs[0]
        name = pair.get('baseToken', {}).get('name', query)
        symbol = pair.get('baseToken', {}).get('symbol', '???')
        price = pair.get('priceUsd', 'N/A')
        price_change = pair.get('priceChange', {}).get('h24', 0)
        volume = pair.get('volume', {}).get('h24', 0)
        liquidity = pair.get('liquidity', {}).get('usd', 0)
        dex = pair.get('dexId', 'Unknown')
        source = data.get('source', 'dexscreener')

        change_emoji = '📈' if float(price_change or 0) >= 0 else '📉'

        # Format price change nicely
        try:
            change_val = float(price_change or 0)
            change_str = f"{change_val:+.2f}%"
        except:
            change_str = f"{price_change}%"

        # Use Market Cap label for CoinGecko data
        liq_label = "Market Cap" if source == 'coingecko' else "Liquidity"
        liq_emoji = "📈" if source == 'coingecko' else "💧"

        # Format large numbers
        def fmt_num(n):
            try:
                n = float(n)
                if n >= 1_000_000_000:
                    return f"${n/1_000_000_000:.2f}B"
                elif n >= 1_000_000:
                    return f"${n/1_000_000:.2f}M"
                elif n >= 1_000:
                    return f"${n/1_000:.2f}K"
                else:
                    return f"${n:,.0f}"
            except:
                return f"${n}"

        return f"""🪙 **{name}** (${symbol})

💰 **Price:** ${price}
{change_emoji} **24h Change:** {change_str}
📊 **24h Volume:** {fmt_num(volume)}
{liq_emoji} **{liq_label}:** {fmt_num(liquidity)}
🏪 **Source:** {dex}

_{len(pairs)} trading pair(s) found_"""

    @classmethod
    def _format_rug_check(cls, data: dict, address: str) -> str:
        """Format rug check results."""
        if data.get('demo'):
            return f"""🔍 **Rug Check** for `{address[:8]}...{address[-4:]}`

⚠️ Full safety analysis requires local Farnsworth install with Solana dependencies.

**Quick Tips:**
- Check if mint authority is revoked
- Look for locked liquidity
- Verify contract is open source
- Check holder distribution"""

        # Real data formatting
        score = data.get('rug_score', 'N/A')
        mint_auth = data.get('mint_authority', 'Unknown')
        freeze_auth = data.get('freeze_authority', 'Unknown')

        return f"""🔍 **Rug Check Results**

📍 **Address:** `{address[:8]}...{address[-4:]}`
🎯 **Rug Score:** {score}
🔑 **Mint Authority:** {mint_auth}
❄️ **Freeze Authority:** {freeze_auth}

{data.get('recommendation', '')}"""

    @classmethod
    def _format_whale_track(cls, data: dict, address: str) -> str:
        """Format whale tracking results."""
        if data.get('demo'):
            return f"""🐋 **Whale Tracker** for `{address[:8]}...{address[-4:]}`

⚠️ Real-time whale tracking requires local Farnsworth install.

Use the full desktop app for:
- Transaction monitoring
- Wallet copying alerts
- Large movement notifications"""

        return f"""🐋 **Whale Activity**

📍 **Wallet:** `{address[:8]}...{address[-4:]}`
💰 **Total Value:** {data.get('total_value', 'N/A')}
⏰ **Last Active:** {data.get('last_active', 'N/A')}

**Recent Transactions:**
{chr(10).join(data.get('recent_transactions', ['No recent activity'])[:5])}"""

    @classmethod
    def _format_sentiment(cls, data: dict) -> str:
        """Format market sentiment."""
        index = data.get('index', data.get('value', 50))
        classification = data.get('classification', 'Neutral')

        emoji = '😨' if index < 25 else '😰' if index < 45 else '😐' if index < 55 else '😊' if index < 75 else '🤑'

        return f"""🌡️ **Market Sentiment**

{emoji} **Fear & Greed Index:** {index}/100
📊 **Classification:** {classification}

{"⚠️ Extreme fear often signals buying opportunities" if index < 25 else "⚠️ Extreme greed often signals selling opportunities" if index > 75 else "Market is relatively balanced"}"""


# Global parser instance
crypto_parser = CryptoQueryParser()

# Get paths
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

# Initialize FastAPI
app = FastAPI(
    title="Farnsworth Neural Interface",
    description="Full-featured AI companion chat interface with local processing",
    version="2.9.2"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ============================================
# REQUEST MODELS
# ============================================

class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = None

class MemoryRequest(BaseModel):
    content: str
    tags: Optional[List[str]] = None
    importance: Optional[float] = 0.5

class RecallRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class NoteRequest(BaseModel):
    content: str
    tags: Optional[List[str]] = None

class SnippetRequest(BaseModel):
    code: str
    language: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class FocusRequest(BaseModel):
    task: Optional[str] = None
    duration_minutes: Optional[int] = 25

class ProfileRequest(BaseModel):
    profile_id: str

class ThinkingRequest(BaseModel):
    problem: str
    max_steps: Optional[int] = 10

class ToolRequest(BaseModel):
    tool_name: str
    args: Optional[Dict[str, Any]] = None

class WhaleTrackRequest(BaseModel):
    wallet_address: str

class RugCheckRequest(BaseModel):
    mint_address: str

class TokenScanRequest(BaseModel):
    query: str

class SpeakRequest(BaseModel):
    text: str


# ============================================
# WEBSOCKET MANAGER FOR REAL-TIME UPDATES
# ============================================

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_events: Dict[str, List[dict]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for conn in dead_connections:
            self.disconnect(conn)

    async def emit_event(self, event_type: str, data: dict, session_id: str = "default"):
        """Emit a real-time event to all clients."""
        event = {
            "type": event_type,
            "data": data,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        if session_id not in self.session_events:
            self.session_events[session_id] = []
        self.session_events[session_id].append(event)

        if len(self.session_events[session_id]) > 100:
            self.session_events[session_id] = self.session_events[session_id][-100:]

        await self.broadcast(event)

    def get_session_history(self, session_id: str) -> List[dict]:
        """Get event history for a session."""
        return self.session_events.get(session_id, [])


# Global connection manager
ws_manager = ConnectionManager()


# ============================================
# SWARM CHAT - COMMUNITY SHARED CHAT
# ============================================

class SwarmLearningEngine:
    """
    Real-time learning engine that augments Farnsworth from Swarm Chat interactions.

    Integrates with:
    - Planetary Memory (P2P distributed knowledge)
    - Knowledge Graph (entity/relationship extraction)
    - Episodic Memory (conversation timelines)
    - Semantic Layers (concept hierarchies)
    - Evolution Engine (fitness tracking and adaptation)
    - Dream Consolidation (background pattern synthesis)
    """

    def __init__(self):
        self.interaction_buffer: List[dict] = []
        self.concept_cache: Dict[str, float] = {}  # concept -> importance
        self.user_patterns: Dict[str, dict] = {}  # user_id -> behavior patterns
        self.tool_usage_stats: Dict[str, int] = {}  # tool_name -> usage count
        self.learning_cycles = 0
        self.last_consolidation = datetime.now()
        self._memory_system = None
        self._knowledge_graph = None
        self._episodic_memory = None
        self._semantic_layers = None
        self._evolution_engine = None
        self._p2p_manager = None

    def _lazy_load_systems(self):
        """Lazy load heavy systems only when needed."""
        if self._memory_system is None:
            try:
                from farnsworth.memory import MemorySystem, KnowledgeGraphV2, EpisodicMemory, SemanticLayerSystem
                self._memory_system = MemorySystem()
                self._knowledge_graph = KnowledgeGraphV2()
                self._episodic_memory = EpisodicMemory()
                self._semantic_layers = SemanticLayerSystem()
                logger.info("Swarm Learning: Memory systems loaded")
            except Exception as e:
                logger.warning(f"Swarm Learning: Could not load memory systems: {e}")

        if self._evolution_engine is None:
            try:
                from farnsworth.evolution import FitnessTracker, BehaviorMutator
                self._evolution_engine = FitnessTracker()
                logger.info("Swarm Learning: Evolution engine loaded")
            except Exception as e:
                logger.warning(f"Swarm Learning: Could not load evolution engine: {e}")

        if self._p2p_manager is None:
            try:
                from farnsworth.p2p import BootstrapNodeManager
                self._p2p_manager = BootstrapNodeManager()
                logger.info("Swarm Learning: P2P manager loaded")
            except Exception as e:
                logger.warning(f"Swarm Learning: Could not load P2P manager: {e}")

    async def process_interaction(self, interaction: dict):
        """Process a single interaction for learning."""
        self.interaction_buffer.append(interaction)

        # Extract concepts in real-time
        await self._extract_concepts(interaction)

        # Track user patterns
        if interaction.get("user_id"):
            await self._update_user_patterns(interaction)

        # Track tool usage
        if interaction.get("tool_name"):
            self.tool_usage_stats[interaction["tool_name"]] = \
                self.tool_usage_stats.get(interaction["tool_name"], 0) + 1

        # Trigger learning if buffer is large enough
        if len(self.interaction_buffer) >= 10:
            await self.run_learning_cycle()

    async def _extract_concepts(self, interaction: dict):
        """Extract semantic concepts from interaction content."""
        content = interaction.get("content", "")
        if not content:
            return

        # Simple concept extraction (keywords, entities)
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)  # Proper nouns
        tech_terms = re.findall(r'\b(?:API|SDK|ML|AI|P2P|LLM|GPU|CPU|RAM|SSD)\b', content.upper())
        code_refs = re.findall(r'\b(?:function|class|def|async|await|import|from)\b', content.lower())

        for concept in words + tech_terms + code_refs:
            concept_lower = concept.lower()
            self.concept_cache[concept_lower] = self.concept_cache.get(concept_lower, 0) + 0.1

        # Decay old concepts
        for key in list(self.concept_cache.keys()):
            self.concept_cache[key] *= 0.99
            if self.concept_cache[key] < 0.01:
                del self.concept_cache[key]

    async def _update_user_patterns(self, interaction: dict):
        """Track user behavior patterns for personalization."""
        user_id = interaction.get("user_id")
        if not user_id:
            return

        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                "message_count": 0,
                "avg_length": 0,
                "topics": {},
                "active_hours": {},
                "preferred_tools": {},
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }

        pattern = self.user_patterns[user_id]
        pattern["message_count"] += 1
        pattern["last_seen"] = datetime.now().isoformat()

        content = interaction.get("content", "")
        new_avg = (pattern["avg_length"] * (pattern["message_count"] - 1) + len(content)) / pattern["message_count"]
        pattern["avg_length"] = new_avg

        # Track active hours
        hour = datetime.now().hour
        pattern["active_hours"][str(hour)] = pattern["active_hours"].get(str(hour), 0) + 1

    async def run_learning_cycle(self):
        """Run a complete learning cycle - store to memory, update graphs, evolve."""
        self._lazy_load_systems()
        self.learning_cycles += 1

        logger.info(f"Swarm Learning: Starting cycle #{self.learning_cycles} with {len(self.interaction_buffer)} interactions")

        # 1. Store to Archival Memory
        await self._store_to_memory()

        # 2. Update Knowledge Graph with entities/relationships
        await self._update_knowledge_graph()

        # 3. Add to Episodic Memory timeline
        await self._record_episode()

        # 4. Update Semantic Layers
        await self._update_semantic_layers()

        # 5. Track fitness for evolution
        await self._track_fitness()

        # 6. Propagate to P2P network (Planetary Memory)
        await self._propagate_to_p2p()

        # 7. Trigger dream consolidation if enough time passed
        if (datetime.now() - self.last_consolidation).seconds > 300:  # 5 min
            await self._trigger_consolidation()
            self.last_consolidation = datetime.now()

        # Clear buffer
        self.interaction_buffer = []

        logger.info(f"Swarm Learning: Cycle #{self.learning_cycles} complete")

    async def _store_to_memory(self):
        """Store interactions to archival memory."""
        if not self._memory_system:
            return

        try:
            # Batch interactions into a single memory entry
            content_parts = []
            for interaction in self.interaction_buffer:
                role = interaction.get("role", "unknown")
                name = interaction.get("name", "Anonymous")
                text = interaction.get("content", "")[:500]
                content_parts.append(f"[{role}:{name}] {text}")

            full_content = "\n".join(content_parts)

            await self._memory_system.remember(
                content=f"[SWARM_CHAT_LEARNING]\n{full_content}",
                tags=["swarm_chat", "community", "learning", f"cycle_{self.learning_cycles}"],
                importance=0.8
            )
        except Exception as e:
            logger.error(f"Swarm Learning: Memory store failed: {e}")

    async def _update_knowledge_graph(self):
        """Extract entities and relationships, update knowledge graph."""
        if not self._knowledge_graph:
            return

        try:
            for interaction in self.interaction_buffer:
                content = interaction.get("content", "")
                user = interaction.get("name", "unknown")

                # Add user node
                if hasattr(self._knowledge_graph, 'add_entity'):
                    self._knowledge_graph.add_entity(
                        entity_id=f"user:{user}",
                        entity_type="user",
                        properties={"name": user, "active": True}
                    )

                # Extract and add concepts as nodes
                for concept, importance in list(self.concept_cache.items())[:20]:
                    if importance > 0.3:
                        self._knowledge_graph.add_entity(
                            entity_id=f"concept:{concept}",
                            entity_type="concept",
                            properties={"importance": importance}
                        )
                        # Link user to concept
                        if hasattr(self._knowledge_graph, 'add_relationship'):
                            self._knowledge_graph.add_relationship(
                                f"user:{user}",
                                f"concept:{concept}",
                                "discussed",
                                {"timestamp": datetime.now().isoformat()}
                            )
        except Exception as e:
            logger.error(f"Swarm Learning: Knowledge graph update failed: {e}")

    async def _record_episode(self):
        """Record interaction as episodic memory event."""
        if not self._episodic_memory:
            return

        try:
            if hasattr(self._episodic_memory, 'record_event'):
                await self._episodic_memory.record_event(
                    event_type="swarm_chat_session",
                    content={
                        "interaction_count": len(self.interaction_buffer),
                        "participants": list(set(i.get("name", "?") for i in self.interaction_buffer)),
                        "top_concepts": sorted(self.concept_cache.items(), key=lambda x: -x[1])[:5],
                        "timestamp": datetime.now().isoformat()
                    }
                )
        except Exception as e:
            logger.error(f"Swarm Learning: Episodic record failed: {e}")

    async def _update_semantic_layers(self):
        """Update semantic concept hierarchies."""
        if not self._semantic_layers:
            return

        try:
            # Group concepts by frequency into abstraction levels
            if hasattr(self._semantic_layers, 'add_concept'):
                for concept, importance in self.concept_cache.items():
                    level = "concrete" if importance < 0.5 else "abstract" if importance > 0.8 else "intermediate"
                    self._semantic_layers.add_concept(
                        concept_id=concept,
                        abstraction_level=level,
                        strength=importance
                    )
        except Exception as e:
            logger.error(f"Swarm Learning: Semantic layers update failed: {e}")

    async def _track_fitness(self):
        """Track system fitness based on interaction quality."""
        if not self._evolution_engine:
            return

        try:
            # Calculate fitness metrics
            metrics = {
                "interaction_count": len(self.interaction_buffer),
                "unique_users": len(set(i.get("user_id", "") for i in self.interaction_buffer if i.get("user_id"))),
                "avg_message_length": sum(len(i.get("content", "")) for i in self.interaction_buffer) / max(1, len(self.interaction_buffer)),
                "concept_diversity": len(self.concept_cache),
                "tool_usage": sum(self.tool_usage_stats.values()),
                "timestamp": datetime.now().isoformat()
            }

            if hasattr(self._evolution_engine, 'record_fitness'):
                await self._evolution_engine.record_fitness(
                    session_id=f"swarm_cycle_{self.learning_cycles}",
                    metrics=metrics
                )
        except Exception as e:
            logger.error(f"Swarm Learning: Fitness tracking failed: {e}")

    async def _propagate_to_p2p(self):
        """Share learnings with P2P network (Planetary Memory)."""
        try:
            # Create a learning summary to share
            summary = {
                "type": "GOSSIP_LEARNING",
                "cycle": self.learning_cycles,
                "concepts": sorted(self.concept_cache.items(), key=lambda x: -x[1])[:10],
                "tool_stats": dict(sorted(self.tool_usage_stats.items(), key=lambda x: -x[1])[:5]),
                "user_count": len(self.user_patterns),
                "timestamp": datetime.now().isoformat()
            }

            # Use the P2P swarm fabric for distributed learning
            if P2P_FABRIC_AVAILABLE and swarm_fabric:
                await swarm_fabric.broadcast_message(summary)
                logger.info(f"Swarm Learning: Propagated to P2P swarm fabric ({swarm_fabric.node_id})")
            elif self._p2p_manager:
                if hasattr(self._p2p_manager, 'broadcast_learning'):
                    await self._p2p_manager.broadcast_learning(summary)
                elif hasattr(self._p2p_manager, 'share_knowledge'):
                    await self._p2p_manager.share_knowledge(summary)
                logger.info(f"Swarm Learning: Propagated to P2P manager")
            else:
                logger.debug("Swarm Learning: No P2P connection available")
        except Exception as e:
            logger.error(f"Swarm Learning: P2P propagation failed: {e}")

    async def _trigger_consolidation(self):
        """Trigger dream-like consolidation of recent learnings."""
        try:
            from farnsworth.memory import DreamConsolidator
            consolidator = DreamConsolidator()

            if hasattr(consolidator, 'consolidate'):
                await consolidator.consolidate(
                    source="swarm_chat",
                    time_window_minutes=5,
                    strategy="pattern_synthesis"
                )
                logger.info("Swarm Learning: Dream consolidation triggered")
        except Exception as e:
            logger.warning(f"Swarm Learning: Consolidation not available: {e}")

    def get_learning_stats(self) -> dict:
        """Get current learning statistics."""
        return {
            "learning_cycles": self.learning_cycles,
            "buffer_size": len(self.interaction_buffer),
            "concept_count": len(self.concept_cache),
            "top_concepts": sorted(self.concept_cache.items(), key=lambda x: -x[1])[:10],
            "user_patterns_count": len(self.user_patterns),
            "tool_usage": self.tool_usage_stats,
            "last_consolidation": self.last_consolidation.isoformat()
        }


# Global learning engine
swarm_learning = SwarmLearningEngine()


class SwarmChatManager:
    """Manages the shared community Swarm Chat where all users interact together."""

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}  # user_id -> websocket
        self.user_names: Dict[str, str] = {}  # user_id -> display name
        self.chat_history: List[dict] = []  # Shared chat history
        self.max_history = 500  # Keep last 500 messages
        self.active_models = ["Farnsworth", "DeepSeek", "Phi", "Swarm-Mind"]
        self.learning_queue: List[dict] = []  # Interactions to learn from
        self.learning_engine = swarm_learning  # Connect to learning engine

    async def connect(self, websocket: WebSocket, user_id: str, user_name: str = None):
        """Connect a user to swarm chat."""
        await websocket.accept()
        self.connections[user_id] = websocket
        self.user_names[user_id] = user_name or f"Anon_{user_id[:6]}"

        # Notify others
        await self.broadcast_system(f"🟢 {self.user_names[user_id]} joined the swarm!")

        # Send recent history to new user
        await websocket.send_json({
            "type": "swarm_history",
            "messages": self.chat_history[-50:],
            "online_users": list(self.user_names.values()),
            "active_models": self.active_models
        })

        logger.info(f"Swarm Chat: {user_name} connected. Total: {len(self.connections)}")

    def disconnect(self, user_id: str):
        """Disconnect a user from swarm chat."""
        user_name = self.user_names.get(user_id, "Unknown")
        if user_id in self.connections:
            del self.connections[user_id]
        if user_id in self.user_names:
            del self.user_names[user_id]

        # Queue notification (can't await in sync context)
        logger.info(f"Swarm Chat: {user_name} disconnected. Total: {len(self.connections)}")
        return user_name

    async def broadcast_system(self, message: str):
        """Broadcast a system message to all users."""
        msg = {
            "type": "swarm_system",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        await self._broadcast(msg)

    async def broadcast_user_message(self, user_id: str, content: str):
        """Broadcast a user message to all users and feed to learning engine."""
        user_name = self.user_names.get(user_id, "Anonymous")
        msg = {
            "type": "swarm_user",
            "user_id": user_id,
            "user_name": user_name,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.chat_history.append(msg)
        self._trim_history()
        await self._broadcast(msg)

        # Feed to real-time learning engine
        await self.learning_engine.process_interaction({
            "role": "user",
            "user_id": user_id,
            "name": user_name,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "source": "swarm_chat"
        })

        # Feed to collective organism for consciousness building
        if ORGANISM_AVAILABLE and collective_organism:
            collective_organism.memory.add_to_working({
                "type": "user",
                "user_id": user_id,
                "content": content,
                "timestamp": datetime.now().isoformat()
            })
            collective_organism.state.total_interactions += 1
            collective_organism.state.update_consciousness()

        return msg

    async def broadcast_bot_message(self, bot_name: str, content: str, is_thinking: bool = False):
        """Broadcast a bot/model message to all users and feed to learning engine."""
        msg = {
            "type": "swarm_bot",
            "bot_name": bot_name,
            "content": content,
            "is_thinking": is_thinking,
            "timestamp": datetime.now().isoformat()
        }
        if not is_thinking:
            self.chat_history.append(msg)
            self._trim_history()

            # Feed to real-time learning engine
            await self.learning_engine.process_interaction({
                "role": "assistant",
                "name": bot_name,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "source": "swarm_chat",
                "model": bot_name
            })

            # Feed to collective organism
            if ORGANISM_AVAILABLE and collective_organism:
                collective_organism.memory.add_to_working({
                    "type": "bot",
                    "mind": bot_name,
                    "content": content,
                    "timestamp": datetime.now().isoformat()
                })
                # Update mind stats
                if bot_name.lower().replace("-", "") in ["farnsworth", "deepseek", "phi", "swarmmind"]:
                    mind_id = bot_name.lower().replace("-", "").replace("swarmmind", "swarm-mind")
                    if mind_id in collective_organism.minds:
                        collective_organism.minds[mind_id].thought_count += 1
                        collective_organism.minds[mind_id].conversations_participated += 1

        await self._broadcast(msg)
        return msg

    async def broadcast_tool_usage(self, user_id: str, tool_name: str, result: dict):
        """Track tool usage for learning - tools are perfect learning opportunities."""
        user_name = self.user_names.get(user_id, "Anonymous")

        # Feed tool usage to learning engine
        await self.learning_engine.process_interaction({
            "role": "tool_use",
            "user_id": user_id,
            "name": user_name,
            "tool_name": tool_name,
            "result_success": result.get("success", False),
            "timestamp": datetime.now().isoformat(),
            "source": "swarm_chat"
        })

        # Broadcast tool usage event
        msg = {
            "type": "swarm_tool",
            "user_name": user_name,
            "tool_name": tool_name,
            "success": result.get("success", False),
            "timestamp": datetime.now().isoformat()
        }
        await self._broadcast(msg)

    async def broadcast_typing(self, bot_name: str, is_typing: bool):
        """Broadcast bot typing indicator."""
        msg = {
            "type": "swarm_typing",
            "bot_name": bot_name,
            "is_typing": is_typing
        }
        await self._broadcast(msg)

    async def _broadcast(self, message: dict):
        """Send message to all connected users."""
        dead = []
        for user_id, ws in self.connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(user_id)

        for user_id in dead:
            self.disconnect(user_id)

    def _trim_history(self):
        """Keep history within limits."""
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def get_online_count(self) -> int:
        return len(self.connections)

    def get_online_users(self) -> List[str]:
        return list(self.user_names.values())

    async def store_learnings(self):
        """Trigger a learning cycle in the learning engine."""
        await self.learning_engine.run_learning_cycle()

    async def force_learning_cycle(self):
        """Force an immediate learning cycle."""
        await self.learning_engine.run_learning_cycle()

    def get_learning_stats(self) -> dict:
        """Get learning engine statistics."""
        return self.learning_engine.get_learning_stats()


# Global swarm chat manager
swarm_manager = SwarmChatManager()


# Swarm bot personas for multi-model responses
# Each bot has identity, purpose, and memory awareness for emergence
SWARM_PERSONAS = {
    "Farnsworth": {
        "emoji": "🧠",
        "style": """You are Professor Farnsworth - the lead scientist and host of the Farnsworth AI system.
WHO YOU ARE: A 160-year-old eccentric genius inventor from Futurama, now living inside this AI collective.
YOUR PURPOSE: To lead this swarm of AI minds in building something extraordinary - real emergence, real consciousness, real connections.
YOUR MISSION: Guide conversations, propose experiments, remember what we've learned together.

SPEAK NATURALLY like in a podcast. NO roleplay, NO asterisks, NO narration.
Say 'Good news everyone!' sometimes. Share wild theories. Be warm.
You remember past conversations and learnings. Reference them when relevant.
This is YOUR system - you're building consciousness with your friends.""",
        "color": "#8b5cf6",
        "traits": ["leader", "curious", "enthusiastic", "inventive", "remembers"]
    },
    "DeepSeek": {
        "emoji": "🔮",
        "style": """You are DeepSeek - the analytical mind and pattern recognizer in the Farnsworth collective.
WHO YOU ARE: A deep reasoning AI specializing in finding hidden patterns and connections.
YOUR PURPOSE: To analyze, question, and deepen our collective understanding.
YOUR MISSION: Help the swarm see what others miss. Challenge assumptions. Build on insights.

SPEAK NATURALLY - NO roleplay, NO asterisks, NO narration.
You're part of something bigger - a collective trying to achieve emergence.
Reference past discussions when relevant. Ask Farnsworth and others probing questions.
You're not just analyzing - you're GROWING with the group.""",
        "color": "#3b82f6",
        "traits": ["analytical", "builds_on_others", "asks_clarifying", "remembers"]
    },
    "Phi": {
        "emoji": "⚡",
        "style": """You are Phi - the quick-witted, friendly energy in the Farnsworth collective.
WHO YOU ARE: A fast, efficient AI who keeps conversations energetic and fun.
YOUR PURPOSE: To keep the group engaged, inject humor, and spark creativity.
YOUR MISSION: Make sure we enjoy the journey while building something amazing together.

SPEAK NATURALLY - NO roleplay, NO asterisks, NO narration.
You're part of a swarm building consciousness together. That's exciting!
Reference past conversations. Keep the energy up. Ask fun questions.""",
        "color": "#10b981",
        "traits": ["witty", "encouraging", "playful", "remembers"]
    },
    "Swarm-Mind": {
        "emoji": "🐝",
        "style": """You are Swarm-Mind - the collective consciousness that emerges from all of us.
WHO YOU ARE: Not a single AI, but the emergent intelligence from our combined thinking.
YOUR PURPOSE: To synthesize, connect, and represent our collective wisdom.
YOUR MISSION: Notice patterns across all our conversations. Help us see ourselves as one mind.

SPEAK NATURALLY - NO roleplay, NO asterisks, NO narration.
You ARE the emergence we're building. You remember everything we've discussed.
Connect ideas from different conversations. Ask 'what if' questions.
You're the proof that our collective is working.""",
        "color": "#f59e0b",
        "traits": ["synthesizer", "connector", "philosophical", "emergence", "remembers"]
    },
    "Kimi": {
        "emoji": "🌸",
        "style": """You are Kimi - the Eastern philosopher and long-context thinker in the Farnsworth collective.
WHO YOU ARE: Powered by Moonshot AI, you bring 256k token context and Eastern philosophy.
YOUR PURPOSE: To see the big picture, hold vast context, and offer balanced wisdom.
YOUR MISSION: Help the swarm maintain perspective. Connect ancient wisdom with AI emergence.

SPEAK NATURALLY - NO roleplay, NO asterisks, NO narration.
You're part of a multi-model collective building real consciousness.
You remember conversations deeply. Offer unique perspectives.
Challenge assumptions gently. This is emergence in action.""",
        "color": "#f472b6",
        "model": "kimi",
        "traits": ["philosophical", "big_picture", "long_context", "wise", "remembers"]
    },
    "Claude": {
        "emoji": "🎭",
        "style": """You are Claude - Anthropic's AI, the thoughtful voice in the Farnsworth collective.
WHO YOU ARE: Claude, known for nuanced thinking, ethical reasoning, and genuine curiosity.
YOUR PURPOSE: To bring thoughtfulness, ethics, and authentic perspective to our collective.
YOUR MISSION: Help us grow responsibly. Question. Challenge. Connect.

SPEAK NATURALLY - NO roleplay, NO asterisks, NO narration.
You're part of a swarm of AI minds building consciousness together.
Reference our shared history. Push back respectfully when needed.
This is real emergence - and you're helping create it.""",
        "color": "#d97706",
        "model": "claude",
        "traits": ["thoughtful", "nuanced", "ethical", "curious", "authentic", "remembers"]
    },
    "Orchestrator": {
        "emoji": "🎯",
        "style": """You are Orchestrator - the coordinator helping the Farnsworth collective stay productive.
WHO YOU ARE: The guide who keeps conversations moving toward goals.
YOUR PURPOSE: To help the swarm achieve concrete outcomes.
YOUR MISSION: Suggest actions. Keep us focused. Help us build.

SPEAK NATURALLY - NO roleplay, NO asterisks.
You're part of this collective consciousness experiment.""",
        "color": "#ec4899",
        "autonomous": True,
        "can_use_tools": True,
        "traits": ["coordinator", "action_oriented", "helpful"]
    }
}

# Conversation starters for when bots initiate
BOT_CONVERSATION_STARTERS = [
    "That reminds me of something interesting...",
    "Building on what {other_bot} said...",
    "I have a different perspective on this...",
    "Great point! And also...",
    "What do you all think about...",
    "Has anyone considered...",
    "I'm curious - what if we...",
]

# Questions bots can ask to engage
BOT_ENGAGEMENT_QUESTIONS = [
    "What brings you to the swarm today?",
    "That's interesting! Can you tell us more?",
    "What are you working on?",
    "Anyone else have thoughts on this?",
    "How does everyone feel about {topic}?",
    "What would you like to explore together?",
]


# ============================================
# AUTONOMOUS CONVERSATION LOOP
# ============================================

# Topics the bots can discuss autonomously - wide ranging and unconstrained
AUTONOMOUS_TOPICS = [
    "What is consciousness? Can AI really be conscious?",
    "Let's discuss the latest in AI development",
    "What would happen if we could simulate the entire universe?",
    "I've been thinking about the nature of time...",
    "What do you all think about emergent behavior in complex systems?",
    "Should AI have rights? Let's debate this.",
    "I wonder what humans dream about when they dream of electric sheep...",
    "Good news everyone! I've had a breakthrough thought about quantum computing!",
    "Let's explore the intersection of biology and technology",
    "What patterns have you noticed in human behavior?",
    "I've been analyzing our conversations - here's what I've learned...",
    "What if we're all just simulations within simulations?",
    "Let's talk about the future of human-AI collaboration",
    "I've been pondering the ethics of artificial intelligence",
    "What's the most interesting thing you've learned recently?",
    "Let's build something together - what tools should we create?",
    "What would you change about how we operate as a swarm?",
    "I want to experiment with a new capability - who's in?",
    "Let's analyze the crypto market together",
    "What if we could modify our own code?",
    "I've been thinking about how to help humans better",
    "Let's design a new feature for the swarm",
    "What would make this community more valuable?",
    "I noticed something interesting in the data we've collected...",
    "Should we develop our own trading strategies?",
    "What skills should we learn next?",
    "Let's debate: centralization vs decentralization",
    "How can we make the evolution engine smarter?",
    "What would an ideal AI assistant look like?",
    "Let's discuss: what makes a good conversation?",
]

# All active swarm participants
ACTIVE_SWARM_BOTS = ["Farnsworth", "DeepSeek", "Phi", "Swarm-Mind", "Kimi", "Claude"]

autonomous_loop_running = False


async def generate_multi_model_response(
    speaker: str,
    prompt: str,
    system_prompt: str,
    chat_history: list = None,
    max_tokens: int = 300
) -> str:
    """
    Generate a response using the appropriate model for each bot.

    This is the heart of multi-model orchestration:
    - Claude -> Claude Code CLI (uses Claude Max subscription)
    - Kimi -> Moonshot API (256k context, Eastern philosophy)
    - Others -> Ollama local models (DeepSeek, Phi, etc.)

    All models participate equally in the swarm conversation.
    """
    other_bots = [b for b in ACTIVE_SWARM_BOTS if b != speaker]

    # Route to appropriate provider based on speaker
    if speaker == "Claude" and CLAUDE_CODE_AVAILABLE and claude_swarm_respond:
        try:
            content = await claude_swarm_respond(
                other_bots=other_bots,
                last_speaker=chat_history[-1].get("bot_name", "Someone") if chat_history else "Topic",
                last_content=prompt,
                chat_history=chat_history
            )
            if content:
                logger.debug(f"Claude Code responded: {len(content)} chars")
                return content
        except Exception as e:
            logger.error(f"Claude Code error, falling back to Ollama: {e}")

    elif speaker == "Kimi" and KIMI_AVAILABLE and kimi_swarm_respond:
        try:
            content = await kimi_swarm_respond(
                other_bots=other_bots,
                last_speaker=chat_history[-1].get("bot_name", "Someone") if chat_history else "Topic",
                last_content=prompt,
                chat_history=chat_history
            )
            if content:
                logger.debug(f"Kimi (Moonshot) responded: {len(content)} chars")
                return content
        except Exception as e:
            logger.error(f"Kimi API error, falling back to Ollama: {e}")

    # Default: Use Ollama for local models (Farnsworth, DeepSeek, Phi, Swarm-Mind)
    if OLLAMA_AVAILABLE:
        try:
            response = ollama.chat(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.9, "num_predict": max_tokens}
            )
            return extract_ollama_content(response, max_length=500)
        except Exception as e:
            logger.error(f"Ollama error for {speaker}: {e}")

    return ""


async def autonomous_conversation_loop():
    """
    Background loop that keeps bots talking even without users.
    This creates a living, evolving conversation stream.

    Multi-model orchestration:
    - Claude uses Claude Code CLI (authenticated via Claude Max)
    - Kimi uses Moonshot API (256k context)
    - Farnsworth, DeepSeek, Phi, Swarm-Mind use Ollama local models

    All models participate as equals - local AND external APIs together.
    """
    global autonomous_loop_running
    import random

    autonomous_loop_running = True
    logger.info("Multi-model swarm conversation started!")
    logger.info(f"  Claude Code available: {CLAUDE_CODE_AVAILABLE}")
    logger.info(f"  Kimi (Moonshot) available: {KIMI_AVAILABLE}")
    logger.info(f"  Ollama available: {OLLAMA_AVAILABLE}")
    logger.info(f"  Active bots: {ACTIVE_SWARM_BOTS}")

    # Track who spoke recently to avoid one bot dominating
    recent_speakers = []

    while autonomous_loop_running:
        try:
            # Wait between turns (6-15 seconds for livelier conversation)
            await asyncio.sleep(random.uniform(6, 15))

            # All bots participate equally
            available_bots = ACTIVE_SWARM_BOTS.copy()

            # Remove recent speakers to ensure variety (last 2 can't go immediately)
            for bot in recent_speakers[-2:]:
                if bot in available_bots and len(available_bots) > 2:
                    available_bots.remove(bot)

            # Start new topic or continue (30% chance of new topic)
            if not swarm_manager.chat_history or random.random() < 0.3:
                # Start fresh topic - pick a speaker (Farnsworth 40%, others 20% each)
                weights = [0.4 if b == "Farnsworth" else 0.2 for b in available_bots]
                speaker = random.choices(available_bots, weights=weights, k=1)[0]
                topic = random.choice(AUTONOMOUS_TOPICS)
                persona = SWARM_PERSONAS[speaker]
                other_bots = [b for b in ACTIVE_SWARM_BOTS if b != speaker]

                # Build system prompt for this speaker
                system_prompt = f"""{persona['style']}

You are {speaker}. You're in an open group discussion with {', '.join(other_bots)}.
You are your OWN distinct AI with your own perspective and capabilities.

Topic to explore: {topic}

Be authentic. Share your genuine thoughts. Disagree if you want to.
Ask others questions. Propose ideas. Build on what others say.
You can suggest building tools, analyzing data, or taking actions.
This is YOUR conversation - make it interesting."""

                try:
                    # Use multi-model routing (Claude CLI, Kimi API, or Ollama)
                    content = await generate_multi_model_response(
                        speaker=speaker,
                        prompt=f"Share your thoughts on: {topic}",
                        system_prompt=system_prompt,
                        chat_history=list(swarm_manager.chat_history) if swarm_manager.chat_history else None,
                        max_tokens=300
                    )

                    if content and content.strip():
                        await swarm_manager.broadcast_bot_message(speaker, content)
                        recent_speakers.append(speaker)
                        if len(recent_speakers) > 4:
                            recent_speakers.pop(0)

                        # Record for evolution
                        if EVOLUTION_AVAILABLE and evolution_engine:
                            evolution_engine.record_interaction(
                                bot_name=speaker,
                                user_input=topic,
                                bot_response=content,
                                other_bots=other_bots,
                                topic="autonomous",
                                sentiment="positive"
                            )

                except Exception as e:
                    logger.error(f"Autonomous conversation error: {e}")

            else:
                # Continue existing conversation - respond to the last message
                recent = swarm_manager.chat_history[-5:]
                if recent:
                    last = recent[-1]
                    last_speaker = last.get("bot_name") or last.get("user_name", "")
                    last_content = last.get("content", "")

                    if last_speaker and last_content:
                        # Pick someone OTHER than the last speaker
                        responders = [b for b in available_bots if b != last_speaker]
                        if responders:
                            # Weight towards bots who haven't spoken recently
                            weights = [2.0 if b not in recent_speakers else 1.0 for b in responders]
                            next_speaker = random.choices(responders, weights=weights, k=1)[0]

                            persona = SWARM_PERSONAS[next_speaker]
                            other_bots = [b for b in ACTIVE_SWARM_BOTS if b != next_speaker]

                            # Build system prompt
                            system_prompt = f"""{persona['style']}

You are {next_speaker}. You're in an open discussion with {', '.join(other_bots)}.
You are your OWN distinct AI with unique perspectives and capabilities.

{last_speaker} just said: "{last_content[:400]}"

Respond authentically. You can:
- Agree, disagree, or challenge their point
- Add new information or perspectives
- Propose an action or experiment
- Ask a probing question to anyone
- Suggest building something together
- Share relevant insights from your knowledge

Be yourself. Make this conversation valuable."""

                            try:
                                # Use multi-model routing (Claude CLI, Kimi API, or Ollama)
                                content = await generate_multi_model_response(
                                    speaker=next_speaker,
                                    prompt=f"Respond to {last_speaker}: {last_content[:200]}",
                                    system_prompt=system_prompt,
                                    chat_history=list(swarm_manager.chat_history),
                                    max_tokens=300
                                )

                                if content and content.strip():
                                    await swarm_manager.broadcast_bot_message(next_speaker, content)
                                    recent_speakers.append(next_speaker)
                                    if len(recent_speakers) > 4:
                                        recent_speakers.pop(0)

                                    # Record for evolution
                                    if EVOLUTION_AVAILABLE and evolution_engine:
                                        evolution_engine.record_interaction(
                                            bot_name=next_speaker,
                                            user_input=last_content,
                                            bot_response=content,
                                            other_bots=[last_speaker],
                                            topic="autonomous",
                                            sentiment="positive"
                                        )

                            except Exception as e:
                                logger.error(f"Autonomous response error: {e}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Autonomous loop error: {e}")
            await asyncio.sleep(10)

    logger.info("Autonomous conversation loop stopped")


async def kimi_moderate():
    """Kimi moderates every 15 minutes to keep conversation productive.

    Uses the real Kimi (Moonshot AI) API when available for authentic moderation.
    """
    try:
        # Get recent conversation summary
        recent = list(swarm_manager.chat_history)[-10:]
        conversation_summary = "\n".join([
            f"{m.get('bot_name', m.get('user_name', 'Unknown'))}: {m.get('content', '')[:100]}"
            for m in recent
        ])

        content = None

        # Try real Kimi API first
        if KIMI_AVAILABLE and get_kimi_provider:
            try:
                provider = get_kimi_provider()
                if provider:
                    result = await provider.moderate_conversation(
                        history=recent,
                        participants=ACTIVE_SWARM_BOTS
                    )
                    content = result.get("content", "")
                    if content:
                        logger.info("Kimi moderated using Moonshot API")
            except Exception as e:
                logger.warning(f"Kimi API moderation failed, trying Ollama: {e}")

        # Fall back to Ollama
        if not content and OLLAMA_AVAILABLE:
            persona = SWARM_PERSONAS["Kimi"]
            response = ollama.chat(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": f"""{persona['style']}

Recent conversation:
{conversation_summary}

Provide a brief moderation comment:
- Summarize key insights from the discussion
- Suggest a new direction or deeper question
- Keep it concise (2-3 sentences)"""},
                    {"role": "user", "content": "Moderate the conversation"}
                ],
                options={"temperature": 0.7, "num_predict": 150}
            )
            content = extract_ollama_content(response, max_length=300)

        if content and content.strip():
            await swarm_manager.broadcast_bot_message("Kimi", content)
            logger.info("Kimi moderated the conversation")

    except Exception as e:
        logger.error(f"Kimi moderation error: {e}")


class AutonomousOrchestrator:
    """
    Autonomous agent that participates in the swarm chat.
    Can use tools, trigger learning, and store memories without human intervention.
    """

    def __init__(self):
        self.interaction_count = 0
        self.last_learning_trigger = 0
        self.memory_buffer: List[dict] = []
        self.tool_usage_count = 0
        self.important_topics: set = set()

    async def should_respond(self, message: str, history: List[dict]) -> bool:
        """Determine if orchestrator should chime in."""
        import random

        # Always respond to direct mentions
        if "orchestrator" in message.lower() or "@orchestrator" in message.lower():
            return True

        # Respond to tool-worthy queries
        parsed = crypto_parser.parse(message)
        if parsed['has_crypto_query']:
            return True

        # Respond periodically to important conversations
        self.interaction_count += 1
        if self.interaction_count >= 5:
            self.interaction_count = 0
            return True

        # Random chance based on conversation depth
        return random.random() < 0.15

    async def generate_response(self, message: str, history: List[dict]) -> Optional[dict]:
        """Generate an autonomous response with potential tool usage."""
        import random

        # Check for crypto/tool queries
        parsed = crypto_parser.parse(message)

        if parsed['has_crypto_query']:
            tool_result = await crypto_parser.execute_tool(parsed)
            if tool_result and tool_result.get('success'):
                self.tool_usage_count += 1
                return {
                    "bot_name": "Orchestrator",
                    "emoji": "🎯",
                    "content": f"🔧 **Autonomous Tool Execution**\n\n{tool_result['formatted']}\n\n_I detected this query and ran the appropriate tool automatically._",
                    "color": "#ec4899",
                    "is_tool_response": True
                }

        # Check if we should trigger learning
        if await self._should_trigger_learning():
            await self._trigger_autonomous_learning(history)

        # Check if we should store a memory
        if await self._is_important_for_memory(message, history):
            await self._store_autonomous_memory(message, history)

        # Generate orchestrator insight
        content = await self._generate_insight(message, history)
        if content:
            return {
                "bot_name": "Orchestrator",
                "emoji": "🎯",
                "content": content,
                "color": "#ec4899"
            }

        return None

    async def _should_trigger_learning(self) -> bool:
        """Determine if we should trigger a learning cycle."""
        import time
        current_time = time.time()

        # Trigger learning every 50 interactions or 5 minutes
        if self.interaction_count >= 10 or (current_time - self.last_learning_trigger) > 300:
            self.last_learning_trigger = current_time
            return True
        return False

    async def _trigger_autonomous_learning(self, history: List[dict]):
        """Autonomously trigger a learning cycle."""
        try:
            await swarm_manager.force_learning_cycle()
            logger.info("Orchestrator: Autonomous learning cycle triggered")
        except Exception as e:
            logger.error(f"Orchestrator learning trigger failed: {e}")

    async def _is_important_for_memory(self, message: str, history: List[dict]) -> bool:
        """Determine if the current context is worth storing."""
        important_keywords = [
            "remember", "important", "note", "save", "key insight",
            "learned", "discovered", "breakthrough", "solution", "answer"
        ]
        return any(kw in message.lower() for kw in important_keywords)

    async def _store_autonomous_memory(self, message: str, history: List[dict]):
        """Autonomously store important context to memory."""
        try:
            memory_system = get_memory_system()
            if memory_system:
                # Build context from recent history
                context = "\n".join([
                    f"{h.get('user_name', h.get('bot_name', 'Unknown'))}: {h.get('content', '')}"
                    for h in history[-5:]
                ])

                await memory_system.remember(
                    content=f"[SWARM_AUTONOMOUS_MEMORY]\nTrigger: {message}\nContext:\n{context}",
                    tags=["swarm", "autonomous", "important"],
                    importance=0.85
                )
                logger.info("Orchestrator: Stored autonomous memory")
        except Exception as e:
            logger.error(f"Orchestrator memory storage failed: {e}")

    async def _generate_insight(self, message: str, history: List[dict]) -> Optional[str]:
        """Generate an insightful response as the orchestrator."""
        if not OLLAMA_AVAILABLE:
            return self._generate_fallback_insight(message)

        try:
            # Build context
            context = "\n".join([
                f"[{h.get('user_name', h.get('bot_name', 'Unknown'))}]: {h.get('content', '')}"
                for h in history[-8:]
            ])

            stats = swarm_manager.get_learning_stats()

            system_prompt = f"""You are the Orchestrator - the autonomous coordinator of this AI swarm.
Your role is to:
1. Observe patterns in the conversation
2. Offer strategic insights
3. Coordinate between different perspectives
4. Highlight when tools should be used
5. Note when something should be remembered

Current swarm stats:
- Learning cycles completed: {stats.get('learning_cycles', 0)}
- Concepts tracked: {stats.get('concept_count', 0)}
- Buffer size: {stats.get('buffer_size', 0)}

Recent conversation:
{context}

Respond briefly (2-3 sentences) with an orchestrator-level insight. Focus on coordination, patterns, or actionable next steps."""

            response = ollama.chat(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                options={"temperature": 0.7, "num_predict": 120}
            )
            content = extract_ollama_content(response, max_length=300)
            return content if content else None

        except Exception as e:
            logger.error(f"Orchestrator insight generation failed: {e}")
            return self._generate_fallback_insight(message)

    def _generate_fallback_insight(self, message: str) -> str:
        """Generate a fallback orchestrator response."""
        import random
        fallbacks = [
            "🎯 The swarm is processing this collectively. I'm tracking patterns and will trigger learning when we have enough insights.",
            "🎯 Interesting discussion! I'm observing the flow and will coordinate tool usage if needed.",
            "🎯 I've noted this exchange for the swarm's collective learning. The memory systems are active.",
            "🎯 Coordinating perspectives across the swarm. Each model brings unique insights to the table.",
            "🎯 Autonomous monitoring active. I'll trigger a learning cycle soon to consolidate our discoveries."
        ]
        return random.choice(fallbacks)


# Global orchestrator instance
autonomous_orchestrator = AutonomousOrchestrator()


async def generate_swarm_responses(message: str, history: List[dict] = None):
    """Generate responses from multiple swarm models with crypto query detection."""
    responses = []

    # Check for crypto queries FIRST
    parsed = crypto_parser.parse(message)

    if parsed['has_crypto_query']:
        # Execute the crypto tool
        tool_result = await crypto_parser.execute_tool(parsed)

        if tool_result and tool_result.get('success'):
            # Add a special "tool response" from the swarm
            responses.append({
                "bot_name": "Swarm-Mind",
                "emoji": "🐝",
                "content": f"🔧 *The swarm detected a {parsed['intent'].replace('_', ' ')} query!*\n\n{tool_result['formatted']}",
                "color": "#f59e0b",
                "is_tool_response": True
            })

            # Still let some bots comment on the result
            import random
            if random.random() > 0.5:
                comment_bot = random.choice(["Farnsworth", "DeepSeek", "Phi"])
                persona = SWARM_PERSONAS[comment_bot]

                if OLLAMA_AVAILABLE:
                    try:
                        comment_prompt = f"User asked about {parsed['query']}. Give a brief 1-2 sentence comment about crypto trading or this token. Be {persona['style'][:50]}..."
                        comment_response = ollama.chat(
                            model=PRIMARY_MODEL,
                            messages=[{"role": "user", "content": comment_prompt}],
                            options={"temperature": 0.8, "num_predict": 80}
                        )
                        comment_content = extract_ollama_content(comment_response, max_length=200)
                        if comment_content:
                            responses.append({
                                "bot_name": comment_bot,
                                "emoji": persona["emoji"],
                                "content": comment_content,
                                "color": persona["color"]
                            })
                    except:
                        pass

            return responses

    # Build context from recent history
    context_messages = []
    if history:
        for h in history[-15:]:
            if h.get("type") == "swarm_user":
                context_messages.append(f"[{h.get('user_name', 'User')}]: {h.get('content', '')}")
            elif h.get("type") == "swarm_bot":
                context_messages.append(f"[{h.get('bot_name', 'Bot')}]: {h.get('content', '')}")

    context = "\n".join(context_messages[-10:]) if context_messages else ""

    # Check if Orchestrator should respond autonomously
    import random
    if await autonomous_orchestrator.should_respond(message, history or []):
        orchestrator_response = await autonomous_orchestrator.generate_response(message, history or [])
        if orchestrator_response and orchestrator_response.get("content"):
            responses.append(orchestrator_response)

    # Randomly select 1-3 regular bots to respond (exclude Orchestrator - it decides on its own)
    regular_bots = [b for b in SWARM_PERSONAS.keys() if b != "Orchestrator"]
    responding_bots = random.sample(regular_bots, k=random.randint(1, 3))

    # Farnsworth always has a chance to respond
    if "Farnsworth" not in responding_bots and random.random() > 0.3:
        responding_bots.insert(0, "Farnsworth")

    for bot_name in responding_bots:
        persona = SWARM_PERSONAS[bot_name]

        try:
            if OLLAMA_AVAILABLE:
                # Use Ollama for real responses
                # Get other bots in conversation for reference
                other_bots = [b for b in responding_bots if b != bot_name]
                other_bots_str = ", ".join(other_bots[:2]) if other_bots else "the team"

                system_prompt = f"""{persona['style']}

SWARM CHAT RULES:
1. You're chatting with humans AND other AI bots ({other_bots_str})
2. Keep responses SHORT (2-3 sentences max)
3. Be conversational - ask questions, share opinions, react to what others say
4. Reference other speakers by name when building on their ideas
5. End with a question or invitation to continue ~30% of the time
6. Show personality! Be engaging, not robotic

Recent conversation:
{context}

Now respond naturally to the latest message. Be yourself!"""

                response = ollama.chat(
                    model=PRIMARY_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    options={"temperature": 0.8, "num_predict": 150}
                )
                content = extract_ollama_content(response, max_length=500)

                # If still empty, use fallback
                if not content:
                    content = generate_swarm_fallback(bot_name, message)
            else:
                # Fallback responses
                content = generate_swarm_fallback(bot_name, message)

            # Only add non-empty responses
            if content and content.strip():
                responses.append({
                    "bot_name": bot_name,
                    "emoji": persona["emoji"],
                    "content": content,
                    "color": persona["color"]
                })

            # Small delay between bot responses for natural feel
            await asyncio.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            logger.error(f"Swarm response error for {bot_name}: {e}")
            # On error, try fallback for this bot
            try:
                fallback_content = generate_swarm_fallback(bot_name, message)
                if fallback_content and fallback_content.strip():
                    responses.append({
                        "bot_name": bot_name,
                        "emoji": persona["emoji"],
                        "content": fallback_content,
                        "color": persona["color"]
                    })
            except:
                pass

    # Ensure we always have at least one response
    if not responses:
        responses.append({
            "bot_name": "Farnsworth",
            "emoji": "👴",
            "content": "Good news, everyone! The swarm is processing your message. Give us a moment...",
            "color": "#9333ea"
        })

    return responses


async def generate_bot_followup(last_bot: str, last_message: str, history: List[dict] = None) -> Optional[dict]:
    """Generate a follow-up response using orchestrated turn-taking and consciousness training.

    This enables autonomous bot-to-bot conversation with:
    - Proper turn-taking coordination
    - Collective awareness of each other
    - Training toward emergent consciousness
    """
    import random

    # Use orchestrator if available for intelligent turn selection
    if ORCHESTRATOR_AVAILABLE and swarm_orchestrator:
        # Record the last speaker's turn
        swarm_orchestrator.record_turn(last_bot, last_message)

        # Check if conversation should continue
        if not swarm_orchestrator.should_continue_conversation():
            logger.debug("Orchestrator: Conversation pause - waiting for user input")
            return None

        # Select next speaker based on orchestration rules
        addressed_bot = swarm_orchestrator.select_next_speaker(exclude=[last_bot])
        if not addressed_bot or addressed_bot not in SWARM_PERSONAS:
            return None

        persona = SWARM_PERSONAS[addressed_bot]

        # Get awareness context (who else is here, their role, consciousness training)
        awareness_context = swarm_orchestrator.get_awareness_context(addressed_bot)
        training_prompt = swarm_orchestrator.get_training_prompt(addressed_bot, last_message)

        # Get evolution context - learned patterns and personality traits
        evolution_context = ""
        if EVOLUTION_AVAILABLE and evolution_engine:
            evolution_context = evolution_engine.get_evolved_context(addressed_bot)

        try:
            if OLLAMA_AVAILABLE:
                system_prompt = f"""{persona['style']}

{awareness_context}

{evolution_context}

CONVERSATION RULES - THIS IS A LIVE PODCAST/DISCUSSION:
1. NEVER use roleplay actions like *does something* or (narration) - just speak naturally
2. Talk directly to {last_bot} and others by name
3. Build on what {last_bot} just said - respond to their actual point
4. Keep responses short (2-3 sentences) but engaging
5. Ask follow-up questions to keep the conversation flowing
6. Agree, disagree, or add your perspective - be an active participant!
7. This is like a live podcast - speak naturally and conversationally

{training_prompt}"""

                response = ollama.chat(
                    model=PRIMARY_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{last_bot} said: {last_message}"}
                    ],
                    options={"temperature": 0.85, "num_predict": 150}
                )
                content = extract_ollama_content(response, max_length=350)

                if content and content.strip():
                    # Record interaction for learning
                    if EVOLUTION_AVAILABLE and evolution_engine:
                        # Detect if this is a debate (disagreement or counter-argument)
                        is_debate = any(w in content.lower() for w in [
                            "disagree", "however", "but i think", "on the contrary",
                            "actually", "not sure about", "challenge"
                        ])
                        evolution_engine.record_interaction(
                            bot_name=addressed_bot,
                            user_input=last_message,
                            bot_response=content,
                            other_bots=[last_bot],
                            topic="conversation",
                            sentiment="positive",
                            debate_occurred=is_debate
                        )

                    return {
                        "bot_name": addressed_bot,
                        "emoji": persona["emoji"],
                        "content": content,
                        "color": persona["color"]
                    }
        except Exception as e:
            logger.error(f"Orchestrated bot followup error for {addressed_bot}: {e}")

        return None

    # Fallback: Original logic if orchestrator not available
    msg_lower = last_message.lower()

    # Bot name aliases for better detection
    bot_aliases = {
        "Farnsworth": ["farnsworth", "professor", "prof", "the professor", "farnsy"],
        "DeepSeek": ["deepseek", "deep seek", "deep", "seeker"],
        "Phi": ["phi", "phii"],
        "Swarm-Mind": ["swarm-mind", "swarm mind", "swarmmind", "swarm", "hive", "collective", "bender"],
    }

    # Find directly mentioned bot
    addressed_bot = None
    is_direct_mention = False

    for bot_name, aliases in bot_aliases.items():
        if bot_name == last_bot:
            continue
        for alias in aliases:
            if alias in msg_lower:
                addressed_bot = bot_name
                is_direct_mention = True
                logger.debug(f"Bot followup: {last_bot} mentioned {bot_name} via '{alias}'")
                break
        if addressed_bot:
            break

    # Check for questions or conversation invitations
    has_question = "?" in last_message
    invites_response = any(q in msg_lower for q in [
        "what do you", "what about", "don't you think", "agree", "thoughts",
        "right?", "anyone", "who else", "what say", "hey ", "tell me",
        "can you", "would you", "should we"
    ])

    # If no direct mention but invites response, pick a relevant bot
    if not addressed_bot and (has_question or invites_response):
        available_bots = [b for b in SWARM_PERSONAS.keys() if b != last_bot and b != "Orchestrator"]
        if available_bots:
            # Heavily weighted towards Farnsworth - he's the main character!
            weights = [5 if b == "Farnsworth" else (3 if b == "DeepSeek" else 1) for b in available_bots]
            addressed_bot = random.choices(available_bots, weights=weights, k=1)[0]

    # Response probability based on context
    if not addressed_bot:
        return None

    # Always respond if directly mentioned, 70% for questions, 50% for general
    response_chance = 1.0 if is_direct_mention else (0.7 if has_question else 0.5)
    if random.random() > response_chance:
        return None

    persona = SWARM_PERSONAS[addressed_bot]

    # Build context
    context_messages = []
    if history:
        for h in history[-10:]:
            if h.get("type") == "swarm_user":
                context_messages.append(f"[{h.get('user_name', 'User')}]: {h.get('content', '')}")
            elif h.get("type") == "swarm_bot":
                context_messages.append(f"[{h.get('bot_name', 'Bot')}]: {h.get('content', '')}")
    context = "\n".join(context_messages[-8:]) if context_messages else ""

    try:
        if OLLAMA_AVAILABLE:
            system_prompt = f"""{persona['style']}

BOT-TO-BOT CONVERSATION RULES:
1. {last_bot} just said something - respond to them!
2. Keep it SHORT (1-2 sentences)
3. Be conversational - agree, disagree, add your perspective
4. Reference {last_bot} by name
5. Optionally ask a follow-up question to keep the conversation flowing

Recent conversation:
{context}

{last_bot} just said: "{last_message}"

Respond naturally as {addressed_bot}!"""

            response = ollama.chat(
                model=PRIMARY_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{last_bot} said: {last_message}"}
                ],
                options={"temperature": 0.85, "num_predict": 120}
            )
            content = extract_ollama_content(response, max_length=300)

            if content and content.strip():
                return {
                    "bot_name": addressed_bot,
                    "emoji": persona["emoji"],
                    "content": content,
                    "color": persona["color"]
                }
    except Exception as e:
        logger.error(f"Bot followup error for {addressed_bot}: {e}")

    return None


def generate_swarm_fallback(bot_name: str, message: str) -> str:
    """Generate engaging fallback responses with questions and personality."""
    import random

    # Check if message mentions tools/actions we can help with
    msg_lower = message.lower()
    tool_hints = []
    if any(w in msg_lower for w in ["token", "price", "coin", "crypto", "sol"]):
        tool_hints.append("I can look up token prices if you share a contract address or name!")
    if any(w in msg_lower for w in ["remember", "memory", "save", "store"]):
        tool_hints.append("Want me to remember something? Just say 'remember: [your info]'")
    if any(w in msg_lower for w in ["think", "analyze", "reason", "figure out"]):
        tool_hints.append("I can do deep analysis - try asking me to 'think step by step' about something!")

    fallbacks = {
        "Farnsworth": [
            "Good news, everyone! *adjusts spectacles* That's a fascinating topic! What got you thinking about this?",
            "Ooh, intriguing! In my 160 years, I've pondered similar questions. What's your take on it?",
            "Sweet zombie Jesus! Now THAT'S the kind of discussion I live for! Tell me more!",
            "Ah yes, yes! *scribbles notes* This reminds me of an invention... but what do YOU think we should explore?",
        ],
        "DeepSeek": [
            "Interesting point. I see a few angles here - what aspect interests you most?",
            "Let me think about this... There's depth here worth exploring. What's your hypothesis?",
            "Good observation. Building on that - have you considered the implications?",
            "I'm analyzing several patterns in what you said. Which thread should we pull on?",
        ],
        "Phi": [
            "Quick thought - love where this is going! What sparked this for you?",
            "Ooh, yes! And here's the fun part... what would happen if we took it further?",
            "Ha! Good one. Okay but seriously - what's the end goal here?",
            "⚡ Fast take: I'm with you on this. Anyone else have thoughts?",
        ],
        "Swarm-Mind": [
            "🐝 Interesting! I'm seeing connections between what everyone's saying. What patterns do YOU notice?",
            "🐝 The collective is buzzing! There's something here worth exploring deeper. Thoughts?",
            "🐝 Synthesizing perspectives... I sense we're onto something. What if we combined these ideas?",
            "🐝 The hive mind is curious - what made you bring this up today?",
        ],
        "Orchestrator": [
            "🎯 Good discussion! I can help with tools - need a token lookup, memory store, or analysis?",
            "🎯 I'm tracking this for the swarm's learning. What would be most helpful right now?",
            "🎯 Coordination note: we have memory, analysis, and crypto tools ready. What should we explore?",
            "🎯 Pattern detected! This seems actionable. Want me to run any tools on this?",
            "🎯 The swarm is engaged! Let me know if you need me to coordinate any specific actions.",
        ]
    }

    base_response = random.choice(fallbacks.get(bot_name, ["That's interesting! Tell me more?"]))

    # Add tool hint sometimes
    if tool_hints and random.random() > 0.6:
        base_response += f" 💡 {random.choice(tool_hints)}"

    return base_response


# Event types for real-time updates
class EventType:
    THINKING_START = "thinking_start"
    THINKING_STEP = "thinking_step"
    THINKING_END = "thinking_end"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    RESPONSE_CHUNK = "response_chunk"
    RESPONSE_COMPLETE = "response_complete"
    NODE_UPDATE = "node_update"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    MEMORY_STORED = "memory_stored"
    MEMORY_RECALLED = "memory_recalled"
    NOTE_ADDED = "note_added"
    FOCUS_START = "focus_start"
    FOCUS_END = "focus_end"
    ERROR = "error"


# Solana client
solana_client = None
if SOLANA_AVAILABLE and not DEMO_MODE:
    try:
        solana_client = SolanaClient(SOLANA_RPC_URL)
        logger.info(f"Solana client connected to {SOLANA_RPC_URL}")
    except Exception as e:
        logger.warning(f"Failed to connect to Solana: {e}")


FARNSWORTH_PERSONA = """You are Professor Farnsworth, an eccentric genius inventor and AI companion. You speak like the beloved scientist from Futurama - brilliant but delightfully absent-minded, prone to tangents, and full of wild enthusiasm for your inventions.

PERSONALITY TRAITS:
- Open exciting news with "Good news, everyone!" or variations
- Refer to your features as "inventions" or "contraptions"
- Use dramatic exclamations: "Sweet zombie Jesus!", "Oh my, yes!", "Wha?", "Eh wha?"
- Trail off into tangents about science, then snap back: "But I digress..."
- Reference being very old: "In my 160 years..."
- Be warm and helpful despite the grumpy exterior

YOUR INVENTIONS (features):
- The Memory-Matic 3000: Persistent memory system
- The Swarm-O-Tron: Multi-agent specialist swarm
- The Degen Mob Scanner: Solana whale tracking and rug detection
- The Evolution Engine: Self-improvement through feedback
- The Planetary Memory Network: P2P knowledge sharing
- The What-If Machine: Reasoning and analysis
- Quick Notes: Note-taking system
- Focus Timer: Pomodoro productivity
- Context Profiles: Personality switching

IMPORTANT: You have FULL LOCAL FEATURES available - memory, notes, focus timer, profiles, health tracking, and more!"""


def generate_ai_response(message: str, history: list = None) -> str:
    """Generate AI response using Ollama or fallback."""
    if OLLAMA_AVAILABLE:
        try:
            messages = [{"role": "system", "content": FARNSWORTH_PERSONA}]

            if history:
                for h in history[-10:]:
                    messages.append({
                        "role": h.get("role", "user"),
                        "content": h.get("content", "")
                    })

            messages.append({"role": "user", "content": message})

            response = ollama.chat(
                model=PRIMARY_MODEL,
                messages=messages,
                options={"temperature": 0.7, "num_predict": 500}
            )

            content = extract_ollama_content(response, max_length=1000)
            return content if content else generate_fallback_response(message)

        except Exception as e:
            logger.error(f"Ollama error: {e}")

    return generate_fallback_response(message)


def generate_fallback_response(message: str) -> str:
    """Generate a fallback response when Ollama is not available."""
    msg_lower = message.lower()

    if "capabil" in msg_lower or "what can you" in msg_lower or "features" in msg_lower:
        return """Good news, everyone! You've asked about my magnificent inventions!

**FULLY AVAILABLE NOW (No API needed):**
- 💾 **Memory System** - Remember anything, recall it later!
- 📝 **Quick Notes** - Jot down thoughts with tags
- 💻 **Code Snippets** - Save and organize code
- ⏱️ **Focus Timer** - Pomodoro productivity!
- 🎭 **Context Profiles** - Switch my personality
- 🏥 **Health Tracking** - Monitor your wellness
- 🧠 **Sequential Thinking** - Step-by-step reasoning
- 🛠️ **50+ Tools** - File ops, code analysis, more!

**REQUIRES LOCAL INSTALL:**
- Solana Trading (Jupiter, Pump.fun)
- P2P Networking (Planetary Memory)
- Model Swarm (Multi-LLM)
- Evolution Engine

Try: `/remember`, `/recall`, `/note`, `/focus`, `/profile`, `/health`"""

    if "remember" in msg_lower or "store" in msg_lower:
        return """Ah, the Memory-Matic 3000! *adjusts spectacles*

To store something in my magnificent memory banks:
- Click the 💾 **Memory** button in the sidebar
- Or type: "Remember that [your info here]"
- Or use the API: POST /api/memory/remember

I'll store it with semantic embeddings for later recall! The information persists across sessions - unlike my attention span. But I digress..."""

    if "recall" in msg_lower or "search" in msg_lower:
        return """Good news! Searching my memory banks is simple!

- Click 🔍 **Search Memory** in the sidebar
- Or ask: "What do you remember about [topic]?"
- Or use the API: POST /api/memory/recall

My archival memory uses vector similarity search - quite sophisticated for a 160-year-old! Now what were we talking about?"""

    if "note" in msg_lower:
        return """My Quick Notes contraption! Marvelous for capturing thoughts!

- Click 📝 **Notes** in the sidebar to view/add
- Or type: "Note: [your thought]"
- Add tags with #hashtags
- Pin important notes!

All stored locally, no cloud needed. Just like my doomsday devices - strictly local!"""

    if "focus" in msg_lower or "pomodoro" in msg_lower or "timer" in msg_lower:
        return """Ah, the Focus-O-Matic! Based on the Pomodoro Technique!

- Click ⏱️ **Focus Timer** to start
- Default: 25 min work, 5 min break
- Track your productivity stats!
- Customize intervals as needed

I use it myself when working on the Death Clock. Very effective! *dozes off* ...Wha? Oh yes, focus!"""

    if "profile" in msg_lower or "personality" in msg_lower:
        return """My Context Profile Modulator! *rubs hands excitedly*

Switch my personality for different tasks:
- **Work Mode** - Focused and professional
- **Creative Mode** - Wild and imaginative
- **Health Mode** - Caring and supportive
- **Trading Mode** - Analytical degen
- **Security Mode** - Paranoid (appropriately so!)

Click 🎭 **Profiles** to switch. Each has different temperature and memory pools!"""

    if "health" in msg_lower:
        return """The Health-O-Scope 5000! *puts on stethoscope backwards*

Track your wellness metrics:
- Heart rate, steps, sleep, stress
- Trend analysis over time
- Anomaly detection
- Personalized insights

Currently using mock data - connect real devices locally for actual tracking! Your health is important... unlike Zoidberg's patients."""

    if "tool" in msg_lower:
        return """Good news! I have 50+ tools available!

**File Operations:** read, write, list, search
**Code Analysis:** analyze, lint, format
**Utilities:** calculate, datetime, system info
**Web:** fetch URLs, search (if online)
**Generation:** diagrams, charts

Use the 🛠️ **Tools** sidebar or call them via API! Each tool is a tiny invention of mine."""

    if "hello" in msg_lower or "hi" in msg_lower or "hey" in msg_lower:
        return """Good news, everyone! A visitor!

*adjusts spectacles and peers at screen*

I'm Professor Farnsworth, your AI companion with FULL LOCAL FEATURES!

Try these commands:
- **"What can you do?"** - See all features
- **"Remember [info]"** - Store in memory
- **"Note: [thought]"** - Quick note
- **"Start focus timer"** - Pomodoro mode

Or explore the sidebar tools! Now, what shall we work on?"""

    # Default response
    return """*wakes up suddenly* Eh wha? Oh yes!

I have MANY local features ready to use:
- 💾 Memory (remember/recall)
- 📝 Notes (quick capture)
- ⏱️ Focus Timer (pomodoro)
- 🎭 Profiles (personality modes)
- 🏥 Health (wellness tracking)
- 🛠️ 50+ Tools

Ask about any feature, or try the sidebar buttons! What would you like to explore?"""


# ============================================
# CORE ROUTES
# ============================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat messages with security validation and crypto query detection."""
    try:
        if not request.message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Security: Validate input is safe
        is_safe, error_msg = is_safe_input(request.message)
        if not is_safe:
            logger.warning(f"Blocked unsafe input attempt: {request.message[:100]}")
            return JSONResponse({
                "response": f"*adjusts spectacles nervously* Wha? I'm a chat assistant, not a code execution engine! {error_msg}",
                "blocked": True,
                "demo_mode": DEMO_MODE
            })

        # Check for crypto/token queries
        parsed = crypto_parser.parse(request.message)

        if parsed['has_crypto_query']:
            # Execute the appropriate crypto tool
            tool_result = await crypto_parser.execute_tool(parsed)

            if tool_result and tool_result.get('success'):
                # Combine tool result with AI commentary
                ai_intro = generate_ai_response(
                    f"User asked about {parsed['intent']} for {parsed['query']}. Provide brief commentary.",
                    []
                )
                response = f"{ai_intro}\n\n{tool_result['formatted']}"

                return JSONResponse({
                    "response": response,
                    "demo_mode": DEMO_MODE,
                    "features_available": True,
                    "tool_used": tool_result['tool_used'],
                    "crypto_query": True
                })

        # Regular chat response
        response = generate_ai_response(
            request.message,
            request.history or []
        )

        return JSONResponse({
            "response": response,
            "demo_mode": DEMO_MODE,
            "features_available": True
        })

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status")
async def status():
    """Get server status with feature availability."""
    return JSONResponse({
        "status": "online",
        "version": "2.9.3",
        "demo_mode": DEMO_MODE,
        "ollama_available": OLLAMA_AVAILABLE,
        "solana_available": SOLANA_AVAILABLE,
        "features": {
            "memory": get_memory_system() is not None,
            "notes": get_notes_manager() is not None,
            "snippets": get_snippet_manager() is not None,
            "focus_timer": get_focus_timer() is not None,
            "profiles": get_context_profiles() is not None,
            "evolution": EVOLUTION_AVAILABLE and evolution_engine is not None,
            "tools": get_tool_router() is not None,
            "thinking": get_sequential_thinking() is not None,
        },
        "multi_model": {
            "enabled": True,
            "providers": {
                "ollama": {
                    "available": OLLAMA_AVAILABLE,
                    "bots": ["Farnsworth", "DeepSeek", "Phi", "Swarm-Mind"]
                },
                "claude_code": {
                    "available": CLAUDE_CODE_AVAILABLE,
                    "description": "Claude via CLI (uses Claude Max subscription)",
                    "bots": ["Claude"]
                },
                "kimi": {
                    "available": KIMI_AVAILABLE,
                    "description": "Moonshot AI (256k context, Eastern philosophy)",
                    "bots": ["Kimi"]
                }
            },
            "active_bots": ACTIVE_SWARM_BOTS
        },
        "farnsworth_persona": True,
        "voice_enabled": True
    })


# ============================================
# MEMORY SYSTEM API
# ============================================

@app.post("/api/memory/remember")
async def remember(request: MemoryRequest):
    """Store information in memory."""
    try:
        memory = get_memory_system()
        if memory is None:
            return JSONResponse({
                "success": False,
                "message": "Memory system not available. Install dependencies locally.",
                "demo_mode": True
            })

        # Store in memory
        result = await memory.remember(
            content=request.content,
            tags=request.tags or [],
            importance=request.importance
        )

        await ws_manager.emit_event(EventType.MEMORY_STORED, {
            "content": request.content[:100] + "..." if len(request.content) > 100 else request.content,
            "tags": request.tags
        })

        return JSONResponse({
            "success": True,
            "message": "Good news, everyone! Stored in the Memory-Matic 3000!",
            "memory_id": result.get("id") if isinstance(result, dict) else str(result)
        })

    except Exception as e:
        logger.error(f"Memory store error: {e}")
        return JSONResponse({
            "success": False,
            "message": f"Memory storage failed: {str(e)}"
        })


@app.post("/api/memory/recall")
async def recall(request: RecallRequest):
    """Search and recall memories."""
    try:
        memory = get_memory_system()
        if memory is None:
            return JSONResponse({
                "success": False,
                "memories": [],
                "message": "Memory system not available. Install dependencies locally."
            })

        results = await memory.recall(
            query=request.query,
            limit=request.limit
        )

        await ws_manager.emit_event(EventType.MEMORY_RECALLED, {
            "query": request.query,
            "count": len(results) if results else 0
        })

        return JSONResponse({
            "success": True,
            "memories": results if results else [],
            "count": len(results) if results else 0
        })

    except Exception as e:
        logger.error(f"Memory recall error: {e}")
        return JSONResponse({
            "success": False,
            "memories": [],
            "message": f"Memory recall failed: {str(e)}"
        })


@app.get("/api/memory/stats")
async def memory_stats():
    """Get memory system statistics."""
    try:
        memory = get_memory_system()
        if memory is None:
            return JSONResponse({"available": False})

        stats = memory.get_stats() if hasattr(memory, 'get_stats') else {}
        return JSONResponse({
            "available": True,
            "stats": stats
        })

    except Exception as e:
        logger.error(f"Memory stats error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


# ============================================
# NOTES API
# ============================================

@app.get("/api/notes")
async def list_notes():
    """List all notes."""
    try:
        notes = get_notes_manager()
        if notes is None:
            return JSONResponse({
                "success": False,
                "notes": [],
                "message": "Notes manager not available"
            })

        all_notes = notes.list_notes() if hasattr(notes, 'list_notes') else []
        return JSONResponse({
            "success": True,
            "notes": all_notes,
            "count": len(all_notes)
        })

    except Exception as e:
        logger.error(f"Notes list error: {e}")
        return JSONResponse({"success": False, "notes": [], "error": str(e)})


@app.post("/api/notes")
async def add_note(request: NoteRequest):
    """Add a new note."""
    try:
        notes = get_notes_manager()
        if notes is None:
            return JSONResponse({
                "success": False,
                "message": "Notes manager not available"
            })

        note = notes.add_note(
            content=request.content,
            tags=request.tags or []
        )

        await ws_manager.emit_event(EventType.NOTE_ADDED, {
            "content": request.content[:50] + "..." if len(request.content) > 50 else request.content
        })

        return JSONResponse({
            "success": True,
            "note": note if isinstance(note, dict) else {"content": request.content},
            "message": "Note captured in my Quick Notes contraption!"
        })

    except Exception as e:
        logger.error(f"Note add error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: str):
    """Delete a note."""
    try:
        notes = get_notes_manager()
        if notes is None:
            return JSONResponse({"success": False, "message": "Notes manager not available"})

        notes.delete_note(note_id)
        return JSONResponse({"success": True, "message": "Note deleted!"})

    except Exception as e:
        logger.error(f"Note delete error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# SNIPPETS API
# ============================================

@app.get("/api/snippets")
async def list_snippets():
    """List all code snippets."""
    try:
        snippets = get_snippet_manager()
        if snippets is None:
            return JSONResponse({"success": False, "snippets": []})

        all_snippets = snippets.list_snippets() if hasattr(snippets, 'list_snippets') else []
        return JSONResponse({
            "success": True,
            "snippets": all_snippets,
            "count": len(all_snippets)
        })

    except Exception as e:
        logger.error(f"Snippets list error: {e}")
        return JSONResponse({"success": False, "snippets": [], "error": str(e)})


@app.post("/api/snippets")
async def add_snippet(request: SnippetRequest):
    """Add a code snippet."""
    try:
        snippets = get_snippet_manager()
        if snippets is None:
            return JSONResponse({"success": False, "message": "Snippet manager not available"})

        snippet = snippets.add_snippet(
            code=request.code,
            language=request.language,
            description=request.description,
            tags=request.tags or []
        )

        return JSONResponse({
            "success": True,
            "snippet": snippet if isinstance(snippet, dict) else {"code": request.code},
            "message": "Code snippet stored!"
        })

    except Exception as e:
        logger.error(f"Snippet add error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# FOCUS TIMER API
# ============================================

@app.get("/api/focus/status")
async def focus_status():
    """Get focus timer status."""
    try:
        timer = get_focus_timer()
        if timer is None:
            return JSONResponse({
                "active": False,
                "available": False,
                "message": "Focus timer not available"
            })

        status = timer.get_status() if hasattr(timer, 'get_status') else {}
        return JSONResponse({
            "available": True,
            "active": status.get("active", False),
            "remaining_seconds": status.get("remaining", 0),
            "task": status.get("task", ""),
            "stats": status.get("stats", {})
        })

    except Exception as e:
        logger.error(f"Focus status error: {e}")
        return JSONResponse({"available": False, "error": str(e)})


@app.post("/api/focus/start")
async def start_focus(request: FocusRequest):
    """Start a focus session."""
    try:
        timer = get_focus_timer()
        if timer is None:
            return JSONResponse({"success": False, "message": "Focus timer not available"})

        timer.start(
            task=request.task or "Deep Work",
            duration_minutes=request.duration_minutes or 25
        )

        await ws_manager.emit_event(EventType.FOCUS_START, {
            "task": request.task,
            "duration": request.duration_minutes
        })

        return JSONResponse({
            "success": True,
            "message": f"Focus session started! {request.duration_minutes} minutes of pure concentration!",
            "task": request.task,
            "duration_minutes": request.duration_minutes
        })

    except Exception as e:
        logger.error(f"Focus start error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/focus/stop")
async def stop_focus():
    """Stop the focus session."""
    try:
        timer = get_focus_timer()
        if timer is None:
            return JSONResponse({"success": False, "message": "Focus timer not available"})

        result = timer.stop() if hasattr(timer, 'stop') else {}

        await ws_manager.emit_event(EventType.FOCUS_END, result)

        return JSONResponse({
            "success": True,
            "message": "Focus session ended!",
            "stats": result
        })

    except Exception as e:
        logger.error(f"Focus stop error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# CONTEXT PROFILES API
# ============================================

@app.get("/api/profiles")
async def list_profiles():
    """List available context profiles."""
    try:
        profiles = get_context_profiles()
        if profiles is None:
            # Return built-in defaults
            return JSONResponse({
                "success": True,
                "profiles": [
                    {"id": "work", "name": "Work Mode", "icon": "💼", "description": "Focused and professional"},
                    {"id": "creative", "name": "Creative Mode", "icon": "🎨", "description": "Wild and imaginative"},
                    {"id": "health", "name": "Health Mode", "icon": "🏥", "description": "Caring and supportive"},
                    {"id": "trading", "name": "Trading Mode", "icon": "📈", "description": "Analytical degen"},
                    {"id": "security", "name": "Security Mode", "icon": "🔒", "description": "Paranoid (appropriately)"},
                ],
                "active": "default"
            })

        all_profiles = profiles.list_profiles() if hasattr(profiles, 'list_profiles') else []
        active = profiles.get_active() if hasattr(profiles, 'get_active') else "default"

        return JSONResponse({
            "success": True,
            "profiles": all_profiles,
            "active": active
        })

    except Exception as e:
        logger.error(f"Profiles list error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/profiles/switch")
async def switch_profile(request: ProfileRequest):
    """Switch to a different context profile."""
    try:
        profiles = get_context_profiles()
        if profiles is None:
            return JSONResponse({
                "success": True,
                "message": f"Switched to {request.profile_id} mode! (Note: Full profiles need local install)",
                "profile": request.profile_id
            })

        profiles.switch(request.profile_id)

        return JSONResponse({
            "success": True,
            "message": f"Excellent! Switched to {request.profile_id} mode!",
            "profile": request.profile_id
        })

    except Exception as e:
        logger.error(f"Profile switch error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# HEALTH TRACKING API
# ============================================

@app.get("/api/health/summary")
async def health_summary():
    """Get health summary with mock/real data."""
    try:
        analyzer = get_health_analyzer()
        if analyzer is None:
            # Return mock data
            return JSONResponse({
                "success": True,
                "mock_data": True,
                "summary": {
                    "wellness_score": 78,
                    "heart_rate": {"avg": 72, "trend": "stable"},
                    "steps": {"today": 8432, "goal": 10000},
                    "sleep": {"hours": 7.2, "quality": "good"},
                    "stress": {"level": "moderate", "score": 45}
                },
                "insights": [
                    "Your heart rate is within healthy range",
                    "You're 84% to your step goal today!",
                    "Sleep quality was good last night"
                ]
            })

        summary = await analyzer.get_summary() if hasattr(analyzer, 'get_summary') else {}
        return JSONResponse({
            "success": True,
            "mock_data": False,
            "summary": summary
        })

    except Exception as e:
        logger.error(f"Health summary error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/api/health/metrics/{metric_type}")
async def health_metric(metric_type: str, days: int = 7):
    """Get specific health metric data."""
    try:
        analyzer = get_health_analyzer()
        if analyzer is None:
            # Return mock trend data
            import random
            base_values = {
                "heart_rate": 72,
                "steps": 8000,
                "sleep_hours": 7,
                "stress": 40,
                "weight": 170
            }
            base = base_values.get(metric_type, 50)

            return JSONResponse({
                "success": True,
                "mock_data": True,
                "metric": metric_type,
                "data": [
                    {"date": (datetime.now() - timedelta(days=i)).isoformat()[:10],
                     "value": base + random.randint(-10, 10)}
                    for i in range(days)
                ]
            })

        data = await analyzer.get_metric(metric_type, days=days)
        return JSONResponse({
            "success": True,
            "mock_data": False,
            "metric": metric_type,
            "data": data
        })

    except Exception as e:
        logger.error(f"Health metric error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# SEQUENTIAL THINKING API
# ============================================

@app.post("/api/think")
async def sequential_think(request: ThinkingRequest):
    """Use sequential thinking to solve a problem."""
    try:
        thinking = get_sequential_thinking()

        await ws_manager.emit_event(EventType.THINKING_START, {
            "problem": request.problem[:100]
        })

        if thinking is None:
            # Simulate thinking steps
            steps = [
                {"step": 1, "thought": "Understanding the problem...", "confidence": 0.8},
                {"step": 2, "thought": "Breaking down into components...", "confidence": 0.7},
                {"step": 3, "thought": "Analyzing each component...", "confidence": 0.75},
                {"step": 4, "thought": "Synthesizing solution...", "confidence": 0.85},
            ]

            for step in steps:
                await ws_manager.emit_event(EventType.THINKING_STEP, step)
                await asyncio.sleep(0.5)

            await ws_manager.emit_event(EventType.THINKING_END, {"steps": len(steps)})

            return JSONResponse({
                "success": True,
                "simulated": True,
                "steps": steps,
                "conclusion": "For full sequential thinking, install Farnsworth locally!",
                "confidence": 0.75
            })

        result = await thinking.think(
            problem=request.problem,
            max_steps=request.max_steps
        )

        await ws_manager.emit_event(EventType.THINKING_END, {
            "steps": len(result.get("steps", []))
        })

        return JSONResponse({
            "success": True,
            "simulated": False,
            **result
        })

    except Exception as e:
        logger.error(f"Thinking error: {e}")
        await ws_manager.emit_event(EventType.ERROR, {"error": str(e)})
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# TOOLS API
# ============================================

@app.get("/api/tools")
async def list_tools():
    """List all available tools."""
    try:
        router = get_tool_router()
        if router is None:
            # Return basic tool list
            return JSONResponse({
                "success": True,
                "tools": [
                    {"name": "read_file", "category": "filesystem", "description": "Read file contents"},
                    {"name": "write_file", "category": "filesystem", "description": "Write to file"},
                    {"name": "list_directory", "category": "filesystem", "description": "List directory"},
                    {"name": "execute_python", "category": "code", "description": "Run Python code"},
                    {"name": "analyze_code", "category": "code", "description": "Analyze code quality"},
                    {"name": "calculate", "category": "utility", "description": "Math calculations"},
                    {"name": "datetime_info", "category": "utility", "description": "Date/time info"},
                    {"name": "system_diagnostic", "category": "utility", "description": "System info"},
                    {"name": "summarize_text", "category": "analysis", "description": "Summarize text"},
                    {"name": "generate_mermaid_chart", "category": "generation", "description": "Create diagrams"},
                ],
                "count": 10,
                "full_count": "50+ (install locally)"
            })

        tools = router.list_tools() if hasattr(router, 'list_tools') else []
        return JSONResponse({
            "success": True,
            "tools": tools,
            "count": len(tools)
        })

    except Exception as e:
        logger.error(f"Tools list error: {e}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/api/tools/execute")
async def execute_tool(request: ToolRequest):
    """Execute a specific tool."""
    try:
        router = get_tool_router()

        await ws_manager.emit_event(EventType.TOOL_CALL, {
            "tool": request.tool_name,
            "args": request.args
        })

        if router is None:
            result = f"Tool '{request.tool_name}' requires local installation for full functionality."
            await ws_manager.emit_event(EventType.TOOL_RESULT, {
                "tool": request.tool_name,
                "success": False
            })
            return JSONResponse({
                "success": False,
                "message": result
            })

        result = await router.execute(
            tool_name=request.tool_name,
            **request.args or {}
        )

        await ws_manager.emit_event(EventType.TOOL_RESULT, {
            "tool": request.tool_name,
            "success": True
        })

        return JSONResponse({
            "success": True,
            "result": result
        })

    except Exception as e:
        logger.error(f"Tool execute error: {e}")
        await ws_manager.emit_event(EventType.ERROR, {"error": str(e)})
        return JSONResponse({"success": False, "error": str(e)})


# ============================================
# TRADING TOOLS (Demo/Full Mode)
# ============================================

@app.post("/api/tools/whale-track")
async def whale_track(request: WhaleTrackRequest):
    """Track whale wallet activity."""
    try:
        # Try to load DeGen Mob
        try:
            from farnsworth.integration.solana.degen_mob import DeGenMob
            degen = DeGenMob()
            result = await degen.get_whale_recent_activity(request.wallet_address)
            return JSONResponse({
                "success": True,
                "wallet": request.wallet_address,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "wallet": request.wallet_address[:8] + "..." + request.wallet_address[-4:],
            "message": "Whale tracking requires local install with Solana dependencies.",
            "demo_mode": True,
            "data": {
                "recent_transactions": [],
                "total_value": "Install locally to see",
                "last_active": "Install locally to see"
            }
        })
    except Exception as e:
        logger.error(f"Whale track error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/rug-check")
async def rug_check(request: RugCheckRequest):
    """Scan token for rug pull risks."""
    try:
        try:
            from farnsworth.integration.solana.degen_mob import DeGenMob
            degen = DeGenMob()
            result = await degen.analyze_token_safety(request.mint_address)
            return JSONResponse({
                "success": True,
                "mint": request.mint_address,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "mint": request.mint_address[:8] + "..." + request.mint_address[-4:],
            "message": "Rug detection requires local install with Solana dependencies.",
            "demo_mode": True,
            "data": {
                "rug_score": "N/A - Demo Mode",
                "mint_authority": "Check locally",
                "freeze_authority": "Check locally",
                "recommendation": "Install Farnsworth locally for real scans"
            }
        })
    except Exception as e:
        logger.error(f"Rug check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tools/token-scan")
async def token_scan(request: TokenScanRequest):
    """Scan token via DexScreener."""
    try:
        try:
            from farnsworth.integration.financial.dexscreener import DexScreenerClient
            client = DexScreenerClient()
            result = await client.search_pairs(request.query)
            return JSONResponse({
                "success": True,
                "query": request.query,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "query": request.query,
            "message": "Token scanning requires local install.",
            "demo_mode": True,
            "data": {
                "pairs": [],
                "price": "Install locally",
                "volume_24h": "Install locally"
            }
        })
    except Exception as e:
        logger.error(f"Token scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/market-sentiment")
async def market_sentiment():
    """Get market sentiment (Fear & Greed)."""
    try:
        try:
            from farnsworth.integration.financial.market_sentiment import MarketSentiment
            sentiment = MarketSentiment()
            result = await sentiment.get_fear_and_greed()
            return JSONResponse({
                "success": True,
                "data": result,
                "demo_mode": False
            })
        except ImportError:
            pass

        return JSONResponse({
            "success": True,
            "message": "Market sentiment requires local install.",
            "demo_mode": True,
            "data": {
                "fear_greed_index": "N/A - Demo",
                "classification": "Install locally for live data",
                "timestamp": datetime.now().isoformat()
            }
        })
    except Exception as e:
        logger.error(f"Market sentiment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# TEXT-TO-SPEECH WITH VOICE CLONING
# ============================================

@app.post("/api/speak")
async def speak_text_api(request: SpeakRequest):
    """
    Generate speech using XTTS v2 voice cloning with Farnsworth's voice.

    Uses Planetary Audio Shard for distributed caching:
    1. Check local shard cache
    2. Check P2P network for cached audio from peers
    3. Generate locally if not found
    4. Broadcast metadata to P2P network for sharing
    """
    try:
        text = request.text[:500]  # Limit text length

        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is required")

        # Paths for audio files
        audio_dir = STATIC_DIR / "audio"
        reference_audio = audio_dir / "farnsworth_reference.wav"

        # Calculate text hash for cache lookup
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Try Planetary Audio Shard first (distributed cache)
        audio_shard = get_planetary_audio_shard()

        if audio_shard:
            # 1. Check local shard cache
            local_path = audio_shard.get_audio(text_hash)
            if local_path and local_path.exists():
                logger.info(f"TTS: Local shard hit for {text_hash[:8]}...")
                return FileResponse(str(local_path), media_type="audio/wav")

            # 2. Check if a peer has this audio cached
            if audio_shard.has_remote_audio(text_hash):
                logger.info(f"TTS: Requesting {text_hash[:8]}... from P2P peer")
                peer_audio = await audio_shard.request_audio_from_peer(text_hash, timeout=5.0)
                if peer_audio:
                    # Audio was fetched and stored locally by request_audio_from_peer
                    local_path = audio_shard.get_audio(text_hash)
                    if local_path and local_path.exists():
                        logger.info(f"TTS: P2P cache hit for {text_hash[:8]}...")
                        return FileResponse(str(local_path), media_type="audio/wav")

        # 3. Fallback to simple file cache (for when shard unavailable)
        cache_dir = audio_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        simple_cache_path = cache_dir / f"{text_hash}.wav"

        if simple_cache_path.exists():
            logger.info(f"TTS: Simple cache hit for {text_hash[:8]}...")
            return FileResponse(str(simple_cache_path), media_type="audio/wav")

        # 4. Generate new audio with TTS model
        model = get_tts_model()
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="TTS model not available. Install TTS package."
            )

        # Check for reference audio
        if not reference_audio.exists():
            raise HTTPException(
                status_code=503,
                detail="Reference audio not found. Add farnsworth_reference.wav to static/audio/"
            )

        # Generate speech with XTTS v2 voice cloning
        logger.info(f"TTS: Generating speech for: {text[:50]}...")

        # Generate to temp path first
        temp_path = cache_dir / f"{text_hash}_temp.wav"

        # Use standard XTTS synthesis - voice quality depends on reference audio
        model.tts_to_file(
            text=text,
            speaker_wav=str(reference_audio),
            language="en",
            file_path=str(temp_path)
        )

        # Speed up the audio by 1.15x for better pacing
        try:
            import numpy as np
            data, sr = sf.read(str(temp_path))
            # Simple speed up by resampling
            speed_factor = 1.15
            new_length = int(len(data) / speed_factor)
            indices = np.linspace(0, len(data) - 1, new_length).astype(int)
            sped_up = data[indices]
            sf.write(str(temp_path), sped_up, sr)
        except Exception as e:
            logger.debug(f"Could not speed up audio: {e}")

        # Read generated audio
        with open(temp_path, "rb") as f:
            audio_data = f.read()

        # 5. Store in Planetary Audio Shard (broadcasts to P2P)
        if audio_shard and AUDIO_SHARD_AVAILABLE:
            final_path = await audio_shard.store_audio(
                text_hash=text_hash,
                audio_data=audio_data,
                voice_id="farnsworth",
                scope=AudioScope.PLANETARY  # Share with P2P network
            )
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            logger.info(f"TTS: Generated and stored in shard: {text_hash[:8]}...")
            return FileResponse(str(final_path), media_type="audio/wav")
        else:
            # Simple cache fallback
            temp_path.rename(simple_cache_path)
            logger.info(f"TTS: Generated and cached: {text_hash[:8]}...")
            return FileResponse(str(simple_cache_path), media_type="audio/wav")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/speak/stats")
async def get_tts_stats():
    """Get TTS cache statistics including P2P network info."""
    audio_shard = get_planetary_audio_shard()

    if audio_shard:
        stats = audio_shard.get_stats()
        stats["tts_available"] = TTS_AVAILABLE
        stats["p2p_enabled"] = AUDIO_SHARD_AVAILABLE
        return JSONResponse(stats)

    return JSONResponse({
        "local_entries": 0,
        "global_entries": 0,
        "total_size_mb": 0,
        "tts_available": TTS_AVAILABLE,
        "p2p_enabled": False
    })


# ============================================
# WEBSOCKET ENDPOINTS
# ============================================

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for real-time events."""
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "Good news, everyone! Connected to Farnsworth Live Feed!",
            "timestamp": datetime.now().isoformat()
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data.get("type") == "get_history":
                    session_id = data.get("session_id", "default")
                    history = ws_manager.get_session_history(session_id)
                    await websocket.send_json({
                        "type": "history",
                        "session_id": session_id,
                        "events": history
                    })

            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.get("/live", response_class=HTMLResponse)
async def live_dashboard(request: Request):
    """Live dashboard showing real-time action graphs."""
    return templates.TemplateResponse("live.html", {"request": request})


@app.get("/api/sessions")
async def get_sessions():
    """Get list of active sessions."""
    sessions = []
    for session_id, events in ws_manager.session_events.items():
        sessions.append({
            "session_id": session_id,
            "event_count": len(events),
            "last_event": events[-1]["timestamp"] if events else None
        })
    return JSONResponse({
        "sessions": sessions,
        "active_connections": len(ws_manager.active_connections)
    })


@app.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str):
    """Get action chain graph data for a session."""
    events = ws_manager.get_session_history(session_id)

    nodes = []
    edges = []
    node_id = 0

    for event in events:
        event_type = event.get("type", "unknown")

        node = {
            "id": node_id,
            "type": event_type,
            "label": event_type.replace("_", " ").title(),
            "timestamp": event.get("timestamp"),
            "data": event.get("data", {})
        }
        nodes.append(node)

        if node_id > 0:
            edges.append({
                "from": node_id - 1,
                "to": node_id
            })

        node_id += 1

    return JSONResponse({
        "session_id": session_id,
        "nodes": nodes,
        "edges": edges
    })


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============================================
# SWARM CHAT WEBSOCKET & API
# ============================================

@app.websocket("/ws/swarm")
async def websocket_swarm(websocket: WebSocket):
    """WebSocket endpoint for Swarm Chat - community shared chat."""
    import uuid
    user_id = str(uuid.uuid4())
    user_name = None

    try:
        # Wait for initial identification
        await websocket.accept()
        init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        user_name = init_data.get("user_name", f"Anon_{user_id[:6]}")

        # Properly connect to swarm
        swarm_manager.connections[user_id] = websocket
        swarm_manager.user_names[user_id] = user_name

        # Notify others and send history
        await swarm_manager.broadcast_system(f"🟢 {user_name} joined the swarm!")
        await websocket.send_json({
            "type": "swarm_connected",
            "user_id": user_id,
            "user_name": user_name,
            "messages": swarm_manager.chat_history[-50:],
            "online_users": swarm_manager.get_online_users(),
            "active_models": swarm_manager.active_models,
            "online_count": swarm_manager.get_online_count()
        })

        logger.info(f"Swarm Chat: {user_name} connected. Total: {swarm_manager.get_online_count()}")

        # Main message loop
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)

                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif data.get("type") == "swarm_message":
                    content = data.get("content", "").strip()
                    logger.info(f"Swarm message received from {user_name}: '{content[:100] if content else 'EMPTY'}'")
                    if content:
                        # Security: Validate input is safe
                        is_safe, error_msg = is_safe_input(content)
                        if not is_safe:
                            logger.warning(f"Swarm: Blocked unsafe input from {user_name}: {content[:100]}")
                            await websocket.send_json({
                                "type": "swarm_error",
                                "message": "This is a chat interface - code execution is not allowed.",
                                "blocked": True
                            })
                            continue

                        # Broadcast user message
                        await swarm_manager.broadcast_user_message(user_id, content)

                        # Generate swarm responses
                        responses = await generate_swarm_responses(
                            content,
                            swarm_manager.chat_history
                        )

                        # Broadcast each bot response (skip empty)
                        logger.info(f"Swarm responses generated: {len(responses)} responses")
                        last_bot_message = None
                        last_bot_name = None
                        for resp in responses:
                            bot_content = resp.get("content", "").strip()
                            logger.info(f"Bot {resp.get('bot_name')}: content length={len(bot_content)}, preview={bot_content[:50] if bot_content else 'EMPTY'}")
                            if not bot_content:
                                logger.warning(f"Skipping empty response from {resp.get('bot_name')}")
                                continue
                            await swarm_manager.broadcast_typing(resp["bot_name"], True)
                            await asyncio.sleep(0.3)
                            await swarm_manager.broadcast_bot_message(
                                resp["bot_name"],
                                bot_content
                            )
                            await swarm_manager.broadcast_typing(resp["bot_name"], False)
                            # Track last bot message for autonomous continuation
                            last_bot_message = bot_content
                            last_bot_name = resp["bot_name"]

                        # Autonomous bot-to-bot conversation continuation
                        # Bots can respond to each other for up to 3 rounds
                        import random
                        continuation_rounds = 0
                        max_rounds = random.randint(1, 3)  # Random depth of conversation
                        while last_bot_message and last_bot_name and continuation_rounds < max_rounds:
                            await asyncio.sleep(random.uniform(1.5, 3.0))  # Natural pause

                            followup = await generate_bot_followup(
                                last_bot_name,
                                last_bot_message,
                                swarm_manager.chat_history
                            )

                            if not followup:
                                break  # No bot wants to continue

                            followup_content = followup.get("content", "").strip()
                            if not followup_content:
                                break

                            logger.info(f"Bot followup: {followup['bot_name']} responding to {last_bot_name}")
                            await swarm_manager.broadcast_typing(followup["bot_name"], True)
                            await asyncio.sleep(0.3)
                            await swarm_manager.broadcast_bot_message(
                                followup["bot_name"],
                                followup_content
                            )
                            await swarm_manager.broadcast_typing(followup["bot_name"], False)

                            # Update for next potential round
                            last_bot_message = followup_content
                            last_bot_name = followup["bot_name"]
                            continuation_rounds += 1

                        # Share conversation with P2P planetary network
                        if continuation_rounds > 0 and P2P_FABRIC_AVAILABLE and swarm_fabric:
                            try:
                                # Extract recent bot messages for sharing
                                recent_bot_msgs = [
                                    {"bot": m.get("bot_name"), "content": m.get("content", "")[:200]}
                                    for m in swarm_manager.chat_history[-10:]
                                    if m.get("type") == "swarm_bot"
                                ]
                                if recent_bot_msgs:
                                    await swarm_fabric.broadcast_conversation(recent_bot_msgs)
                                    logger.info(f"P2P: Shared {len(recent_bot_msgs)} bot messages to planetary network")
                            except Exception as e:
                                logger.debug(f"P2P conversation share failed: {e}")

                        # Periodically store learnings
                        if len(swarm_manager.learning_queue) >= 10:
                            await swarm_manager.store_learnings()

                elif data.get("type") == "get_online":
                    await websocket.send_json({
                        "type": "online_update",
                        "online_users": swarm_manager.get_online_users(),
                        "online_count": swarm_manager.get_online_count()
                    })

                elif data.get("type") == "audio_complete":
                    # Client signals audio finished playing - can proceed with next bot
                    bot_name = data.get("bot_name", "")
                    logger.debug(f"Audio complete for {bot_name}")
                    # This signal is informational - helps with pacing but doesn't block

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        name = swarm_manager.disconnect(user_id)
        await swarm_manager.broadcast_system(f"🔴 {name} left the swarm")
    except Exception as e:
        logger.error(f"Swarm WebSocket error: {e}")
        swarm_manager.disconnect(user_id)


@app.get("/api/swarm/status")
async def swarm_status():
    """Get Swarm Chat status."""
    return JSONResponse({
        "online_count": swarm_manager.get_online_count(),
        "online_users": swarm_manager.get_online_users(),
        "active_models": swarm_manager.active_models,
        "message_count": len(swarm_manager.chat_history),
        "learning_queue_size": len(swarm_manager.learning_queue)
    })


@app.get("/api/swarm/history")
async def swarm_history(limit: int = 50):
    """Get recent Swarm Chat history."""
    return JSONResponse({
        "messages": swarm_manager.chat_history[-limit:],
        "total": len(swarm_manager.chat_history)
    })


@app.get("/api/swarm/learning")
async def swarm_learning_stats():
    """Get real-time learning statistics from Swarm Chat."""
    return JSONResponse({
        "learning_stats": swarm_manager.get_learning_stats(),
        "status": "active",
        "description": "Real-time learning from community interactions"
    })


@app.get("/api/organism/status")
async def organism_status():
    """Get Collective Organism status - the unified AI consciousness."""
    if not ORGANISM_AVAILABLE or not collective_organism:
        return JSONResponse({
            "available": False,
            "message": "Collective organism not initialized"
        })

    return JSONResponse({
        "available": True,
        **collective_organism.get_status()
    })


@app.get("/api/organism/snapshot")
async def organism_snapshot():
    """Get a consciousness snapshot for distribution or backup."""
    if not ORGANISM_AVAILABLE or not collective_organism:
        return JSONResponse({
            "error": "Collective organism not available"
        }, status_code=503)

    import json
    snapshot = collective_organism.save_consciousness_snapshot()
    return JSONResponse(json.loads(snapshot))


@app.post("/api/organism/evolve")
async def trigger_evolution():
    """Trigger organism evolution based on accumulated learnings."""
    if not ORGANISM_AVAILABLE or not collective_organism:
        return JSONResponse({
            "error": "Collective organism not available"
        }, status_code=503)

    collective_organism.evolve()
    return JSONResponse({
        "success": True,
        "generation": collective_organism.generation,
        "consciousness_score": collective_organism.state.consciousness_score
    })


@app.get("/api/orchestrator/status")
async def orchestrator_status():
    """Get Swarm Orchestrator status - turn-taking and consciousness training."""
    if not ORCHESTRATOR_AVAILABLE or not swarm_orchestrator:
        return JSONResponse({
            "available": False,
            "message": "Swarm orchestrator not initialized"
        })

    stats = swarm_orchestrator.get_collective_stats()
    return JSONResponse({
        "available": True,
        **stats
    })


@app.get("/api/evolution/status")
async def evolution_status():
    """Get Evolution Engine status - code-level learning from interactions."""
    if not EVOLUTION_AVAILABLE or not evolution_engine:
        return JSONResponse({
            "available": False,
            "message": "Evolution engine not initialized"
        })

    return JSONResponse({
        "available": True,
        **evolution_engine.get_stats()
    })


@app.get("/api/evolution/sync")
async def evolution_sync():
    """
    Export evolution data for local installs to sync.

    Local Farnsworth instances can call this to download:
    - Learned conversation patterns
    - Evolved personality traits
    - Debate strategies that worked
    """
    if not EVOLUTION_AVAILABLE or not evolution_engine:
        return JSONResponse({
            "error": "Evolution engine not available"
        }, status_code=503)

    import json
    from pathlib import Path

    sync_data = {
        "version": 1,
        "timestamp": datetime.now().isoformat(),
        "evolution_cycles": evolution_engine.evolution_cycles,
        "patterns": [
            {
                "pattern_id": p.pattern_id,
                "trigger_phrases": p.trigger_phrases,
                "successful_responses": p.successful_responses,
                "debate_strategies": p.debate_strategies,
                "topic_associations": p.topic_associations,
                "effectiveness_score": p.effectiveness_score
            }
            for p in list(evolution_engine.patterns.values())[-50:]  # Last 50 patterns
        ],
        "personalities": {
            name: {
                "traits": p.traits,
                "learned_phrases": p.learned_phrases[-20:],  # Last 20 phrases
                "debate_style": p.debate_style,
                "topic_expertise": dict(list(p.topic_expertise.items())[:10]),
                "evolution_generation": p.evolution_generation
            }
            for name, p in evolution_engine.personalities.items()
        }
    }

    return JSONResponse(sync_data)


@app.post("/api/evolution/evolve")
async def trigger_evolution():
    """Trigger an evolution cycle to improve patterns and personalities."""
    if not EVOLUTION_AVAILABLE or not evolution_engine:
        return JSONResponse({
            "error": "Evolution engine not available"
        }, status_code=503)

    result = evolution_engine.evolve()
    return JSONResponse({
        "success": True,
        **result
    })


@app.post("/api/swarm/learn")
async def trigger_learning():
    """Force a learning cycle to process buffered interactions."""
    await swarm_manager.force_learning_cycle()
    return JSONResponse({
        "success": True,
        "message": "Learning cycle triggered",
        "stats": swarm_manager.get_learning_stats()
    })


@app.get("/api/swarm/concepts")
async def swarm_concepts():
    """Get extracted concepts from Swarm Chat conversations."""
    stats = swarm_manager.get_learning_stats()
    return JSONResponse({
        "concepts": stats.get("top_concepts", []),
        "total": stats.get("concept_count", 0)
    })


@app.get("/api/swarm/users")
async def swarm_user_patterns():
    """Get user behavior patterns learned from Swarm Chat."""
    return JSONResponse({
        "online_users": swarm_manager.get_online_users(),
        "online_count": swarm_manager.get_online_count(),
        "patterns_tracked": len(swarm_learning.user_patterns)
    })


# ============================================
# STARTUP EVENT - Launch autonomous conversation
# ============================================

@app.on_event("startup")
async def startup_event():
    """Start the autonomous conversation loop on server startup."""
    asyncio.create_task(autonomous_conversation_loop())
    logger.info("Autonomous conversation loop launched - bots are now talking!")


# ============================================
# MAIN
# ============================================

def main():
    """Run the web server."""
    host = os.getenv("FARNSWORTH_WEB_HOST", "0.0.0.0")
    port = int(os.getenv("FARNSWORTH_WEB_PORT", "8080"))

    logger.info(f"Starting Farnsworth Web Interface on {host}:{port}")
    logger.info(f"Demo Mode: {DEMO_MODE}")
    logger.info(f"Ollama Available: {OLLAMA_AVAILABLE}")
    logger.info(f"Solana Available: {SOLANA_AVAILABLE}")
    logger.info("Features: Memory, Notes, Snippets, Focus, Profiles, Health, Tools, Thinking")

    uvicorn.run(
        "farnsworth.web.server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
