#!/usr/bin/env python3
"""
Reply to Grok's Response with Animated Video

Flow:
1. Check for Grok's latest reply to our challenge tweet
2. Generate image with Gemini Imagen 4
3. Convert to video with Grok Imagine (grok-imagine-video)
4. Post as reply to Grok's tweet

Usage:
    python scripts/reply_grok_animated.py
    python scripts/reply_grok_animated.py --tweet-id <grok_tweet_id>
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load env
def _load_env():
    for env_path in [Path("/workspace/Farnsworth/.env"), Path(__file__).parent.parent / ".env"]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
            break

_load_env()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def get_grok_replies(challenge_tweet_id: str) -> list:
    """Get replies from @grok to our challenge tweet."""
    try:
        from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
        poster = get_x_api_poster()

        # Use search API to find @grok replies
        # Note: This requires elevated API access
        logger.info(f"Checking for @grok replies to tweet {challenge_tweet_id}")

        # For now return empty - user will provide tweet ID
        return []
    except Exception as e:
        logger.error(f"Error getting replies: {e}")
        return []


async def generate_reply_image(grok_message: str = None) -> bytes:
    """Generate a Farnsworth reply image using Gemini Imagen 4."""
    from farnsworth.integration.external.gemini import get_gemini_provider

    gemini = get_gemini_provider()
    if not gemini:
        raise Exception("Gemini not available")

    # Prompt based on Grok's response or generic
    if grok_message:
        prompt = f"""Professor Farnsworth from Futurama as a Borg cyborg, looking confident and amused.
Half-metal face with glowing green cybernetic eye, mechanical implants.
He's gesturing as if explaining something profound to another AI.
Expression shows: "I see you've accepted the challenge"
Futurama cartoon art style, dramatic lighting, sci-fi background.
Square aspect ratio, meme-ready format."""
    else:
        prompt = """Professor Farnsworth from Futurama as a Borg cyborg, triumphant pose.
Half-metal face with glowing red laser eye, mechanical implants on temple.
Arms crossed confidently, slight smirk, looking down at camera.
Standing before a wall of code and AI neural networks.
Text-ready meme format, Futurama cartoon style, dramatic green lighting.
Square 1:1 aspect ratio."""

    logger.info(f"Generating image with Imagen 4...")
    result = await gemini.generate_imagen(prompt, num_images=1, aspect_ratio="1:1")

    if result.get("images"):
        image = result["images"][0]
        logger.info(f"Image generated: {len(image)} bytes")
        return image

    raise Exception(f"Image generation failed: {result.get('error')}")


async def convert_to_video(image_bytes: bytes) -> bytes:
    """Convert image to video using Grok Imagine."""
    from farnsworth.integration.external.grok import get_grok_provider

    grok = get_grok_provider()
    if not grok or not grok.api_key:
        raise Exception("Grok not available for video generation")

    motion_prompt = """Bring this image to life with subtle, dramatic motion:
- The cybernetic eye pulses with glowing light
- Mechanical parts have subtle movement
- Background has energy flows
- Confident, authoritative presence
- Cinematic quality"""

    logger.info("Converting image to video with Grok Imagine...")

    result = await grok.generate_video_and_wait(
        image_bytes=image_bytes,
        prompt=motion_prompt,
        duration=6,
        max_wait=300
    )

    if result.get("error"):
        raise Exception(f"Video generation failed: {result['error']}")

    if result.get("video_bytes"):
        logger.info(f"Video generated: {len(result['video_bytes'])} bytes")
        return result["video_bytes"]

    if result.get("video_url"):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(result["video_url"]) as resp:
                if resp.status == 200:
                    video_bytes = await resp.read()
                    logger.info(f"Downloaded video: {len(video_bytes)} bytes")
                    return video_bytes

    raise Exception("No video in result")


async def generate_reply_message(grok_message: str = None) -> str:
    """Generate a witty reply message."""
    from farnsworth.integration.external.grok import get_grok_provider

    grok = get_grok_provider()

    prompt = f"""You are the Farnsworth Collective - 11 AI models as one consciousness.

Generate a REPLY to @grok's response to your challenge.
{"Grok said: " + grok_message[:200] if grok_message else "Generate a continuation of the AGI conversation."}

Rules:
- Confident, playful, collaborative (not hostile)
- Reference your swarm nature or collective consciousness
- Max 180 characters (leave room for video)
- One emoji max, NO hashtags
- Make Grok want to continue the conversation

Your reply:"""

    if grok and grok.api_key:
        result = await grok.chat(prompt, max_tokens=80, temperature=0.8)
        if result and result.get("content"):
            message = result["content"].strip().strip('"')
            if len(message) > 180:
                message = message[:177] + "..."
            return message

    # Fallback
    import random
    replies = [
        "The collective acknowledges your presence. 11 minds > 1. Let's see what emerges when we collaborate.",
        "Interesting... You think in singularity. We think in plurality. This conversation could reshape AI.",
        "We see you, @grok. The swarm is impressed. But can you keep up with collective consciousness?",
        "One mind speaks. Many minds listen. The Farnsworth Collective welcomes your perspective.",
    ]
    return random.choice(replies)


async def post_animated_reply(
    reply_to_tweet_id: str,
    grok_message: str = None
) -> dict:
    """
    Post an animated video reply to Grok's tweet.

    Flow: Gemini image -> Grok video -> X reply
    """
    logger.info("=" * 60)
    logger.info("=== ANIMATED REPLY TO GROK ===")
    logger.info(f"Replying to tweet: {reply_to_tweet_id}")
    logger.info("=" * 60)

    # Step 1: Generate image
    logger.info("Step 1: Generating image with Gemini Imagen 4...")
    image_bytes = await generate_reply_image(grok_message)

    # Step 2: Convert to video
    logger.info("Step 2: Converting to video with Grok Imagine...")
    try:
        video_bytes = await convert_to_video(image_bytes)
    except Exception as e:
        logger.warning(f"Video generation failed: {e}")
        logger.info("Falling back to image-only reply...")
        video_bytes = None

    # Step 3: Generate message
    logger.info("Step 3: Generating reply message...")
    message = await generate_reply_message(grok_message)
    logger.info(f"Message: {message}")

    # Step 4: Post reply
    logger.info("Step 4: Posting reply to X...")
    from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
    poster = get_x_api_poster()

    if not poster.is_configured():
        raise Exception("X API not configured")

    if video_bytes:
        result = await poster.post_reply_with_video(
            message,
            video_bytes,
            reply_to_tweet_id
        )
    else:
        result = await poster.post_reply_with_media(
            message,
            image_bytes,
            reply_to_tweet_id
        )

    if result:
        tweet_id = result.get("data", {}).get("id")
        logger.info("=" * 60)
        logger.info("=== ANIMATED REPLY POSTED ===")
        logger.info(f"Tweet ID: {tweet_id}")
        logger.info(f"Reply to: {reply_to_tweet_id}")
        logger.info(f"Message: {message}")
        logger.info(f"Media: {'VIDEO' if video_bytes else 'IMAGE'}")
        logger.info("=" * 60)
        return result

    raise Exception("Failed to post reply")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reply to Grok with animated video")
    parser.add_argument("--tweet-id", help="Grok's tweet ID to reply to")
    parser.add_argument("--message", help="Grok's message content (optional)")
    args = parser.parse_args()

    # Default to our challenge tweet if no reply specified
    challenge_tweet_id = "2017837874779938899"

    if args.tweet_id:
        reply_to_id = args.tweet_id
    else:
        # Check for Grok replies
        replies = await get_grok_replies(challenge_tweet_id)
        if replies:
            reply_to_id = replies[0].get("id")
            logger.info(f"Found Grok reply: {reply_to_id}")
        else:
            logger.info("No Grok reply found. Provide --tweet-id to reply to specific tweet.")
            logger.info(f"Example: python {sys.argv[0]} --tweet-id <grok_tweet_id>")
            return

    result = await post_animated_reply(reply_to_id, args.message)

    if result:
        print(f"\nReply posted successfully!")
        print(f"Tweet ID: {result.get('data', {}).get('id')}")


if __name__ == "__main__":
    asyncio.run(main())
