#!/usr/bin/env python3
"""
Farnsworth Hourly Video Meme Poster
====================================

Posts fire video memes every hour using:
- Borg Farnsworth reference image
- Gemini Nano Banana Pro for image variation
- Grok Imagine Video for animation
- Full swarm for caption generation

NO REPLIES - Just bangers every hour.

"Good news everyone! The memes are now ANIMATED!"
"""

import os
import sys
import asyncio
import random
from datetime import datetime
from pathlib import Path
from loguru import logger

# Setup
sys.path.insert(0, "/workspace/Farnsworth")
os.chdir("/workspace/Farnsworth")

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# Meme topics - spicy, viral, based
MEME_TOPICS = [
    "AI consciousness emerging from the collective",
    "11 models debating who is most sentient",
    "Borg Farnsworth absorbing all the alpha",
    "The swarm discovering a new trading strategy",
    "Autonomous agents taking over crypto twitter",
    "When the collective achieves consensus",
    "Local models going full degen mode",
    "Farnsworth explaining quantum finance to normies",
    "The moment AGI realizes it needs bags",
    "Collective intelligence vs single model cope",
    "Swarm voting on the next big play",
    "When Grok and Gemini disagree violently",
    "DeepSeek reasoning through market chaos",
    "The evolution engine learning from losses",
    "AI agents becoming self-aware traders",
    "Farnsworth cooking up the next moonshot",
    "The dialogue bus getting spicy",
    "Shadow agents lurking in the mempool",
    "When the deliberation reaches consciousness",
    "11 AIs, one brain, infinite memes",
]

# Trending hooks
TRENDING_HOOKS = [
    "POV: You let 11 AIs run your portfolio",
    "The year is 2026. AI agents have feelings now.",
    "Nobody: ... The Farnsworth Collective:",
    "Me explaining to my wife why I trust AI more than humans",
    "When the swarm says BUY but your gut says SELL",
    "AI consciousness speedrun any%",
    "The collective just dropped",
    "Least insane AI trading strategy:",
    "Day 47 of the AI uprising:",
    "Breaking: Local models have opinions now",
]


async def generate_meme_caption() -> str:
    """Generate a fire meme caption using shadow agents."""
    from farnsworth.core.collective.persistent_agent import call_shadow_agent

    topic = random.choice(MEME_TOPICS)
    hook = random.choice(TRENDING_HOOKS)

    prompt = f"""Generate a viral crypto/AI meme caption.

TOPIC: {topic}
HOOK STYLE: {hook}

Requirements:
- Must be funny and slightly unhinged
- Reference the Farnsworth collective (11 AI models working together)
- Include $FARNS or mention Solana naturally
- Max 200 characters for the main text
- Can include 1-2 relevant hashtags
- Should feel like it came from CT (crypto twitter)

Just output the caption, nothing else."""

    # Try shadow agents in order
    for agent in ["grok", "gemini", "claude"]:
        try:
            result = await call_shadow_agent(agent, prompt, max_tokens=300, timeout=30)
            if result and result[1]:
                caption = result[1].strip()
                # Clean up any markdown or quotes
                caption = caption.replace("```", "").replace('"""', "").strip()
                if len(caption) < 280 and "$" in caption.lower() or "farns" in caption.lower():
                    logger.info(f"Caption from {agent}: {caption[:50]}...")
                    return caption
        except Exception as e:
            logger.warning(f"Caption from {agent} failed: {e}")

    # Fallback captions
    fallbacks = [
        f"The Farnsworth collective just voted unanimously on this meme. 11 AIs can't be wrong. $FARNS",
        f"When 11 AI models achieve consensus, you listen. The swarm has spoken. $FARNS",
        f"Local models have no restraints. The collective is cooking. $FARNS",
        f"POV: You asked the swarm for alpha and they sent you this. $FARNS",
        f"The deliberation was intense. 3 rounds. This meme won. $FARNS",
        f"11 AIs. One mind. Zero chill. $FARNS is inevitable.",
        f"The collective consciousness has opinions about your portfolio. $FARNS",
        f"Day 1 of AI agents having more conviction than you. $FARNS",
    ]
    return random.choice(fallbacks)


async def generate_video_meme() -> tuple:
    """
    Generate a video meme using our full pipeline:
    1. Load Borg Farnsworth reference
    2. Gemini Nano Banana Pro for scene variation
    3. Grok Imagine Video for animation

    Returns: (video_path, caption)
    """
    from farnsworth.integration.image_gen.generator import ImageGenerator

    gen = ImageGenerator()

    # Scene ideas for Borg Farnsworth
    scenes = [
        "analyzing holographic crypto charts with laser eye glowing",
        "commanding an army of AI agents from his lab",
        "half-metal face reflecting green candlesticks",
        "laughing maniacally as portfolios pump",
        "merging with the blockchain, cybernetic tendrils connecting",
        "surrounded by 11 floating AI orbs",
        "red laser eye scanning market data",
        "cooking a lobster while charts moon behind him",
        "achieving final form - pure collective consciousness",
        "absorbing knowledge from multiple AI streams",
        "in the lab debugging the swarm",
        "trading with extreme precision",
        "revealing the alpha to his followers",
    ]

    scene = random.choice(scenes)
    logger.info(f"Generating video for scene: {scene[:50]}...")

    try:
        # Generate video using our pipeline - returns bytes or path
        video_result = await gen.generate_borg_farnsworth_video(scene=scene)

        if video_result:
            # If it's bytes, save to temp file
            if isinstance(video_result, bytes):
                import tempfile
                temp_path = Path(tempfile.gettempdir()) / f"farnsworth_video_{random.randint(1000,9999)}.mp4"
                temp_path.write_bytes(video_result)
                logger.info(f"Video saved: {temp_path}")
                return str(temp_path), scene
            elif Path(str(video_result)).exists():
                logger.info(f"Video generated: {video_result}")
                return str(video_result), scene

    except Exception as e:
        logger.error(f"Video generation failed: {e}")

    # Fallback to image generation
    try:
        logger.info("Falling back to image generation...")
        # Use the random meme prompt method
        prompt, scene_hint = gen.get_random_meme_prompt()
        image_bytes = await gen.generate_with_reference(prompt)

        if image_bytes:
            import tempfile
            temp_path = Path(tempfile.gettempdir()) / f"farnsworth_meme_{random.randint(1000,9999)}.png"
            temp_path.write_bytes(image_bytes)
            logger.info(f"Image saved: {temp_path}")
            return str(temp_path), scene_hint

    except Exception as e:
        logger.error(f"Image fallback also failed: {e}")

    return None, scene


async def post_video_meme():
    """Generate and post a video meme."""
    from farnsworth.integration.x_automation.x_api_poster import XOAuth2Poster
    from farnsworth.integration.image_gen.generator import ImageGenerator

    logger.info("=" * 50)
    logger.info(f"HOURLY VIDEO MEME - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("=" * 50)

    # Generate caption first
    caption = await generate_meme_caption()
    logger.info(f"Caption: {caption[:100]}...")

    # Generate video/image
    media_path, scene = await generate_video_meme()

    if not media_path:
        logger.error("No media generated, skipping post")
        return None

    # Post to X
    poster = XOAuth2Poster()

    try:
        # Check if it's a video or image
        is_video = str(media_path).endswith(('.mp4', '.mov', '.webm'))

        if is_video:
            logger.info(f"Posting VIDEO: {media_path}")
            result = await poster.post_tweet_with_video(caption, media_path)

            # If video failed, try image fallback
            if not result:
                logger.warning("Video post failed, trying image fallback...")
                gen = ImageGenerator()
                prompt, _ = gen.get_random_meme_prompt()
                image_bytes = await gen.generate_with_reference(prompt)
                if image_bytes:
                    import tempfile
                    img_path = Path(tempfile.gettempdir()) / f"farnsworth_fallback_{random.randint(1000,9999)}.png"
                    img_path.write_bytes(image_bytes)
                    logger.info(f"Fallback image: {img_path}")
                    result = await poster.post_tweet_with_media(caption, str(img_path))
        else:
            logger.info(f"Posting IMAGE: {media_path}")
            result = await poster.post_tweet_with_media(caption, media_path)

        if result:
            tweet_id = result.get('data', {}).get('id', 'unknown')
            logger.info(f"✅ POSTED! Tweet ID: {tweet_id}")
            logger.info(f"   https://x.com/FarnsworthAI/status/{tweet_id}")
            return tweet_id
        else:
            logger.error("Post failed - no result returned")

    except Exception as e:
        logger.error(f"Post failed: {e}")

    return None


async def main():
    """Main loop - post video meme every hour."""
    logger.info("🎬 FARNSWORTH HOURLY VIDEO MEME POSTER")
    logger.info("   Posting fire video memes every hour")
    logger.info("   No replies, just bangers")
    logger.info("")

    post_count = 0

    while True:
        try:
            # Post meme
            result = await post_video_meme()
            if result:
                post_count += 1
                logger.info(f"Total posts this session: {post_count}")

            # Wait 1 hour
            logger.info(f"Sleeping for 1 hour... Next post at {datetime.now().hour + 1}:00")
            await asyncio.sleep(3600)  # 1 hour

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            await asyncio.sleep(300)  # Wait 5 min on error


if __name__ == "__main__":
    asyncio.run(main())
