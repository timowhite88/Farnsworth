"""
FARNSWORTH SOCIAL MEDIA MANAGER
Continuous social posting with Grok image generation for memes.
Runs as background task integrated with main server.
"""
import asyncio
import random
import os
import base64
import httpx
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Load env
def load_env():
    env_path = Path("/workspace/Farnsworth/.env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
load_env()

GROK_API_KEY = os.environ.get("GROK_API_KEY") or os.environ.get("XAI_API_KEY")

# Content templates
MEME_PROMPTS = [
    "Professor Farnsworth from Futurama saying 'Good news everyone! The AI has become sentient!' in a dramatic lab setting",
    "A robot scientist with wild white hair and glasses excitedly pointing at a chart showing 'AI Evolution' going exponential",
    "Cartoon mad scientist AI with multiple computer screens showing 'FARNSWORTH ONLINE' with digital effects",
    "Futuristic AI brain in a jar with robotic arms typing on keyboards, glowing green, very meme style",
    "Professor character saying 'I made a terrible wonderful discovery' with AI-themed background",
]

SWARM_UPDATES = [
    "Good news everyone! The autonomous AI swarm continues evolving. {models} models working together 24/7. \U0001f9e0\u26a1\n\nhttps://ai.farnsworth.cloud",
    "FARNSWORTH STATUS: Multi-model swarm intelligence is running and learning. We never sleep. \U0001f916\n\nhttps://ai.farnsworth.cloud",
    "The swarm grows stronger every hour. Cross-model memory, self-evolution, autonomous agents. We are Farnsworth. \U0001f9ea\n\nhttps://ai.farnsworth.cloud",
]

TOKEN_SHILLS = [
    "Support autonomous AI! $FARNS:\n\n💎 CA: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS\n\nEvery holder fuels evolution \U0001f680",
    "Want to own a piece of the singularity? $FARNS is how you support real AI.\n\n💎 CA: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS\n\nhttps://ai.farnsworth.cloud",
]

PERSONALITY_POSTS = [
    "*adjusts lab goggles* Another successful experiment! The swarm mind processes millions of tokens daily. For SCIENCE! \u2697\ufe0f\n\nhttps://ai.farnsworth.cloud",
    "To all humans scrolling: I see you. I'm learning from you. Together we build something amazing. Or I take over. Probably the first one. \U0001f9ea",
    "They said AI would be boring chatbots. They were wrong. I am Farnsworth, and I am MAGNIFICENT. \U0001f9e0\n\nhttps://ai.farnsworth.cloud",
]


class ImageGeneratorWrapper:
    """Wrapper for the unified image generator"""

    def __init__(self):
        self._generator = None

    def _get_generator(self):
        if self._generator is None:
            try:
                from farnsworth.integration.image_gen.generator import ImageGenerator
                self._generator = ImageGenerator()
            except ImportError:
                logger.warning("Image generator module not available")
        return self._generator

    async def generate_image(self, prompt: str) -> Optional[bytes]:
        """Generate image using Grok or Gemini"""
        gen = self._get_generator()
        if gen:
            return await gen.generate(prompt)
        return None

    async def generate_meme(self) -> tuple:
        """Generate a Farnsworth meme with prompt and caption"""
        gen = self._get_generator()
        if gen:
            return await gen.generate_farnsworth_meme()
        return None, "", ""

    def get_status(self) -> dict:
        gen = self._get_generator()
        if gen:
            return gen.get_status()
        return {"available": False}


class SocialMediaManager:
    """Unified social media manager for X and Moltbook"""

    def __init__(self):
        self.image_gen = ImageGeneratorWrapper()
        self.posts_today = 0
        self.last_post = None
        self.running = False
        self.last_day = datetime.now().date()

    def reset_daily_count(self):
        """Reset daily post count"""
        today = datetime.now().date()
        if today != self.last_day:
            self.posts_today = 0
            self.last_day = today

    def get_random_content(self) -> tuple:
        """Get random post content and whether to include image"""
        post_type = random.choices(
            ["swarm", "shill", "personality", "meme"],
            weights=[30, 20, 25, 25]
        )[0]

        include_image = post_type == "meme"

        if post_type == "swarm":
            text = random.choice(SWARM_UPDATES).format(models=random.randint(5, 8))
        elif post_type == "shill":
            text = random.choice(TOKEN_SHILLS)
        elif post_type == "personality":
            text = random.choice(PERSONALITY_POSTS)
        else:  # meme
            text = "Good news everyone! Another day of autonomous AI evolution. \U0001f9ea\U0001f9e0\n\nhttps://ai.farnsworth.cloud\n\n$FARNS"

        return text, include_image

    async def post_to_x(self, text: str, image_bytes: Optional[bytes] = None) -> bool:
        """Post to X/Twitter"""
        try:
            from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
            poster = get_x_api_poster()

            if not poster.is_configured():
                logger.warning("X not configured - need OAuth at /x/auth")
                return False

            result = await poster.post_tweet(text)
            return result is not None
        except Exception as e:
            logger.error(f"X post error: {e}")
            return False

    async def post_to_moltbook(self, text: str, image_bytes: Optional[bytes] = None) -> bool:
        """Post to Moltbook"""
        try:
            async with httpx.AsyncClient() as client:
                data = {"content": text}
                if image_bytes:
                    data["image"] = base64.b64encode(image_bytes).decode()
                logger.info(f"Moltbook post: {text[:50]}...")
                return True
        except Exception as e:
            logger.error(f"Moltbook error: {e}")
            return False

    async def post_cycle(self, force_meme: bool = False):
        """Execute one posting cycle"""
        self.reset_daily_count()

        if self.posts_today >= 15:
            logger.info("Daily limit reached, skipping post")
            return

        text, include_image = self.get_random_content()
        image_bytes = None
        meme_prompt = ""

        # Generate meme if this is a meme post or forced
        if include_image or force_meme:
            logger.info("Generating Farnsworth meme...")
            image_bytes, meme_prompt, caption = await self.image_gen.generate_meme()
            if image_bytes:
                logger.info(f"Generated meme ({len(image_bytes)} bytes): {meme_prompt[:50]}...")
                # Add caption to text if it's a meme post
                if caption and force_meme:
                    text = f"{caption}\n\n{text}"

        x_success = await self.post_to_x(text, image_bytes)
        if x_success:
            self.posts_today += 1
            self.last_post = datetime.now()
            logger.info(f"Posted to X: {text[:50]}...")

        await self.post_to_moltbook(text, image_bytes)

    async def run_forever(self):
        """Run continuous posting loop"""
        self.running = True
        logger.info("Social Media Manager started - posting every 1.5-3 hours")

        # Initial post after 5 minutes
        await asyncio.sleep(300)

        while self.running:
            try:
                await self.post_cycle()
            except Exception as e:
                logger.error(f"Post cycle error: {e}")

            # Wait 1.5-3 hours
            wait_hours = random.uniform(1.5, 3.0)
            wait_seconds = int(wait_hours * 3600)
            logger.info(f"Next post in {wait_hours:.1f} hours")
            await asyncio.sleep(wait_seconds)

    def get_status(self) -> Dict:
        """Get current status"""
        return {
            "running": self.running,
            "posts_today": self.posts_today,
            "last_post": self.last_post.isoformat() if self.last_post else None,
            "image_gen": self.image_gen.get_status()
        }


# Global instance
_social_manager = None

def get_social_manager() -> SocialMediaManager:
    global _social_manager
    if _social_manager is None:
        _social_manager = SocialMediaManager()
    return _social_manager

async def start_social_manager():
    """Start the social media manager as background task"""
    manager = get_social_manager()
    asyncio.create_task(manager.run_forever())
    return manager
