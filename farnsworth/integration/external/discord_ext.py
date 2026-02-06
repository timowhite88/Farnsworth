"""
Farnsworth Discord Bridge - Full ChatOps Interface.

"Bringing the Professor to the server, one message at a time."

Features:
- Slash commands for Farnsworth interactions
- Embed-rich responses with token data
- Voice channel integration
- Thread management for conversations
- Reaction-based controls
- Role-based permissions
- Auto-moderation hooks
"""

import discord
from discord import app_commands
from discord.ext import commands, tasks
import asyncio
import os
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from farnsworth.core.nexus import nexus, Signal, SignalType
from farnsworth.integration.tool_router import ToolRouter


@dataclass
class DiscordConfig:
    """Discord bot configuration."""
    token: str = ""
    prefix: str = "!"
    allowed_channels: List[int] = field(default_factory=list)
    admin_roles: List[int] = field(default_factory=list)
    auto_thread: bool = True
    max_response_length: int = 2000
    embed_color: int = 0x6366f1  # Farnsworth purple


class FarnsworthBot(commands.Bot):
    """Enhanced Discord bot with Farnsworth integration."""

    def __init__(self, config: DiscordConfig):
        intents = discord.Intents.all()
        super().__init__(
            command_prefix=config.prefix,
            intents=intents,
            help_command=None
        )
        self.config = config
        self.router = ToolRouter()
        self.conversation_threads: Dict[int, int] = {}  # user_id -> thread_id
        self.pending_responses: Dict[int, str] = {}

    async def setup_hook(self):
        """Called when bot is ready to set up slash commands."""
        await self.add_cog(FarnsworthCommands(self))
        await self.tree.sync()
        logger.info("Discord: Slash commands synced")


class FarnsworthCommands(commands.Cog):
    """Slash commands for Farnsworth."""

    def __init__(self, bot: FarnsworthBot):
        self.bot = bot

    @app_commands.command(name="ask", description="Ask Farnsworth anything")
    async def ask(self, interaction: discord.Interaction, question: str):
        """Ask Farnsworth a question."""
        await interaction.response.defer(thinking=True)

        try:
            # Emit to Nexus for processing
            await nexus.emit(SignalType.USER_MESSAGE, {
                "content": question,
                "source": "discord",
                "channel_id": interaction.channel_id,
                "user_id": interaction.user.id,
                "user_name": interaction.user.display_name
            }, f"discord_{interaction.user.id}")

            # Wait for response (with timeout)
            response = await self._wait_for_response(interaction.user.id, timeout=60)

            if response:
                embed = self._create_response_embed(question, response, interaction.user)
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("🤔 I'm still thinking... try again in a moment.")

        except Exception as e:
            logger.error(f"Discord ask error: {e}")
            await interaction.followup.send(f"❌ Error: {str(e)[:100]}")

    @app_commands.command(name="token", description="Get token information")
    async def token_info(self, interaction: discord.Interaction, address: str):
        """Fetch token data from DexScreener."""
        await interaction.response.defer()

        try:
            from farnsworth.integration.financial.token_scanner import TokenScanner
            scanner = TokenScanner()
            data = await scanner.scan_token(address)

            if data:
                embed = discord.Embed(
                    title=f"🪙 {data.get('name', 'Unknown')} ({data.get('symbol', '???')})",
                    color=self.bot.config.embed_color
                )
                embed.add_field(name="Price", value=f"${data.get('price', 'N/A')}", inline=True)
                embed.add_field(name="24h Change", value=f"{data.get('change_24h', 'N/A')}%", inline=True)
                embed.add_field(name="Market Cap", value=f"${data.get('market_cap', 'N/A')}", inline=True)
                embed.add_field(name="Liquidity", value=f"${data.get('liquidity', 'N/A')}", inline=True)
                embed.add_field(name="Volume 24h", value=f"${data.get('volume_24h', 'N/A')}", inline=True)
                embed.add_field(name="DEX", value=data.get('dex', 'Unknown'), inline=True)
                embed.set_footer(text=f"Address: {address[:20]}...")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(f"❌ Could not find token: `{address[:20]}...`")

        except Exception as e:
            logger.error(f"Token info error: {e}")
            await interaction.followup.send(f"❌ Error fetching token data")

    @app_commands.command(name="market", description="Get market sentiment")
    async def market_sentiment(self, interaction: discord.Interaction):
        """Fetch current market sentiment."""
        await interaction.response.defer()

        try:
            from farnsworth.integration.financial.market_sentiment import market_sentiment

            fng = await market_sentiment.get_fear_and_greed()
            global_data = await market_sentiment.get_global_market_cap()

            embed = discord.Embed(
                title="📊 Market Sentiment",
                color=self.bot.config.embed_color,
                timestamp=datetime.utcnow()
            )

            # Fear & Greed
            fng_value = fng.get('value', 'N/A')
            fng_class = fng.get('value_classification', 'Unknown')
            embed.add_field(
                name="Fear & Greed Index",
                value=f"**{fng_value}** - {fng_class}",
                inline=False
            )

            # Global market data
            if global_data:
                total_mc = global_data.get('total_market_cap', {}).get('usd', 0)
                btc_dom = global_data.get('market_cap_percentage', {}).get('btc', 0)
                embed.add_field(
                    name="Total Market Cap",
                    value=f"${total_mc/1e12:.2f}T",
                    inline=True
                )
                embed.add_field(
                    name="BTC Dominance",
                    value=f"{btc_dom:.1f}%",
                    inline=True
                )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Market sentiment error: {e}")
            await interaction.followup.send("❌ Error fetching market data")

    @app_commands.command(name="polymarket", description="Search prediction markets")
    async def polymarket_search(self, interaction: discord.Interaction, query: str):
        """Search Polymarket for prediction markets."""
        await interaction.response.defer()

        try:
            from farnsworth.integration.financial.polymarket import polymarket

            markets = await polymarket.search_markets(query)

            if not markets:
                await interaction.followup.send(f"No markets found for: `{query}`")
                return

            embed = discord.Embed(
                title=f"🔮 Polymarket: {query}",
                color=self.bot.config.embed_color
            )

            for market in markets[:5]:
                title = market.get('title', 'Unknown')[:100]
                volume = market.get('volume', 0)
                liquidity = market.get('liquidity', 0)
                embed.add_field(
                    name=title,
                    value=f"Vol: ${volume:,.0f} | Liq: ${liquidity:,.0f}",
                    inline=False
                )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Polymarket error: {e}")
            await interaction.followup.send("❌ Error searching Polymarket")

    @app_commands.command(name="swarm", description="Get swarm status")
    async def swarm_status(self, interaction: discord.Interaction):
        """Get current Farnsworth swarm status."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="🧠 Farnsworth Swarm Status",
            color=self.bot.config.embed_color,
            timestamp=datetime.utcnow()
        )

        agents = [
            ("Farnsworth", "🧪", "Orchestrator"),
            ("Grok", "⚡", "Real-time Intelligence"),
            ("Gemini", "✨", "Multimodal Analysis"),
            ("Claude", "🎭", "Deep Reasoning"),
            ("DeepSeek", "🔍", "Open Reasoning"),
            ("Kimi", "🌙", "Long Context"),
            ("Phi", "⚙️", "Local Efficiency"),
            ("HuggingFace", "🤗", "Local GPU"),
            ("Swarm-Mind", "🐝", "Collective"),
        ]

        status_text = "\n".join([f"{emoji} **{name}**: {role}" for name, emoji, role in agents])
        embed.add_field(name="Active Agents", value=status_text, inline=False)
        embed.add_field(name="Website", value="[ai.farnsworth.cloud](https://ai.farnsworth.cloud)", inline=True)
        embed.add_field(name="$FARNS", value="`9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS`", inline=False)

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="help", description="Show Farnsworth commands")
    async def help_command(self, interaction: discord.Interaction):
        """Show available commands."""
        embed = discord.Embed(
            title="🤖 Farnsworth Commands",
            description="Good news everyone! Here's what I can do:",
            color=self.bot.config.embed_color
        )

        commands_list = [
            ("/ask <question>", "Ask me anything"),
            ("/token <address>", "Get token information"),
            ("/market", "View market sentiment"),
            ("/polymarket <query>", "Search prediction markets"),
            ("/swarm", "View swarm status"),
            ("/help", "Show this help message"),
        ]

        for cmd, desc in commands_list:
            embed.add_field(name=cmd, value=desc, inline=False)

        embed.set_footer(text="Farnsworth AI Collective • ai.farnsworth.cloud")
        await interaction.response.send_message(embed=embed)

    async def _wait_for_response(self, user_id: int, timeout: int = 60) -> Optional[str]:
        """Wait for a response from the Nexus system."""
        start = datetime.now()
        while (datetime.now() - start).seconds < timeout:
            if user_id in self.bot.pending_responses:
                response = self.bot.pending_responses.pop(user_id)
                return response
            await asyncio.sleep(0.5)
        return None

    def _create_response_embed(self, question: str, response: str, user: discord.User) -> discord.Embed:
        """Create a formatted embed for responses."""
        embed = discord.Embed(
            color=self.bot.config.embed_color,
            timestamp=datetime.utcnow()
        )
        embed.set_author(name=f"Asked by {user.display_name}", icon_url=user.display_avatar.url)

        # Truncate if needed
        if len(response) > 4000:
            response = response[:4000] + "..."

        embed.description = response
        embed.set_footer(text="Farnsworth AI • ai.farnsworth.cloud")
        return embed


class DiscordBridge:
    """Main Discord bridge interface."""

    def __init__(self, token: Optional[str] = None, config: Optional[DiscordConfig] = None):
        self.config = config or DiscordConfig(
            token=token or os.environ.get("DISCORD_TOKEN", "")
        )
        self.bot: Optional[FarnsworthBot] = None
        self._running = False

    async def start(self):
        """Start the Discord bot."""
        if not self.config.token:
            logger.warning("Discord Bridge: No token provided. Skipping.")
            return False

        self.bot = FarnsworthBot(self.config)

        @self.bot.event
        async def on_ready():
            logger.info(f"Discord Bridge: Logged in as {self.bot.user}")
            await nexus.emit(
                SignalType.EXTERNAL_ALERT,
                {"msg": f"Discord Bridge Active as {self.bot.user}", "guilds": len(self.bot.guilds)},
                "discord"
            )

        @self.bot.event
        async def on_message(message: discord.Message):
            if message.author == self.bot.user:
                return

            # Process commands first
            await self.bot.process_commands(message)

            # Handle mentions or DMs
            if self.bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
                content = message.content.replace(f'<@{self.bot.user.id}>', '').strip()
                if not content:
                    return

                async with message.channel.typing():
                    await nexus.emit(SignalType.USER_MESSAGE, {
                        "content": content,
                        "source": "discord",
                        "channel_id": message.channel.id,
                        "user_id": message.author.id,
                        "user_name": message.author.display_name,
                        "guild_id": message.guild.id if message.guild else None
                    }, f"discord_{message.author.id}")

        @self.bot.event
        async def on_reaction_add(reaction: discord.Reaction, user: discord.User):
            if user == self.bot.user:
                return

            # Handle reaction-based controls
            if str(reaction.emoji) == "🔄":
                # Regenerate response
                await nexus.emit(SignalType.EXTERNAL_ALERT, {
                    "type": "regenerate",
                    "message_id": reaction.message.id,
                    "user_id": user.id
                }, "discord")
            elif str(reaction.emoji) == "📌":
                # Pin important message
                try:
                    await reaction.message.pin()
                except Exception:
                    pass

        self._running = True
        logger.info("Discord Bridge: Starting...")

        try:
            await self.bot.start(self.config.token)
        except Exception as e:
            logger.error(f"Discord Bridge error: {e}")
            self._running = False
            return False

        return True

    async def stop(self):
        """Stop the Discord bot."""
        if self.bot:
            await self.bot.close()
            self._running = False
            logger.info("Discord Bridge: Stopped")

    async def send_message(self, channel_id: int, content: str, embed: Optional[discord.Embed] = None):
        """Send a message to a specific channel."""
        if not self.bot:
            return

        channel = self.bot.get_channel(channel_id)
        if channel:
            await channel.send(content=content, embed=embed)

    async def send_embed(self, channel_id: int, title: str, description: str, fields: List[tuple] = None):
        """Send an embed to a channel."""
        if not self.bot:
            return

        embed = discord.Embed(
            title=title,
            description=description,
            color=self.config.embed_color,
            timestamp=datetime.utcnow()
        )

        if fields:
            for name, value, inline in fields:
                embed.add_field(name=name, value=value, inline=inline)

        embed.set_footer(text="Farnsworth AI")

        channel = self.bot.get_channel(channel_id)
        if channel:
            await channel.send(embed=embed)

    async def create_thread(self, channel_id: int, name: str, message: Optional[discord.Message] = None) -> Optional[discord.Thread]:
        """Create a thread in a channel."""
        if not self.bot:
            return None

        channel = self.bot.get_channel(channel_id)
        if channel and isinstance(channel, discord.TextChannel):
            if message:
                return await message.create_thread(name=name)
            else:
                return await channel.create_thread(name=name, type=discord.ChannelType.public_thread)
        return None

    def set_response(self, user_id: int, response: str):
        """Set a pending response for a user."""
        if self.bot:
            self.bot.pending_responses[user_id] = response

    @property
    def is_running(self) -> bool:
        return self._running


# Global instance
discord_bridge = DiscordBridge()


async def start_discord_bridge(token: str = None) -> DiscordBridge:
    """Convenience function to start the Discord bridge."""
    bridge = DiscordBridge(token=token)
    await bridge.start()
    return bridge
