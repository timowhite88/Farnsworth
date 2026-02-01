#!/usr/bin/env python3
"""
Reply to Grok's Response with Animated Video

FLOW (Gemini Nano Banana + Grok Imagine):
1. Load reference image (Borg Farnsworth portrait)
2. Feed to Gemini Nano Banana -> Generate variation keeping character
3. Take new image -> Feed to Grok Imagine for 6s video
4. Post VIDEO as reply to Grok's tweet

Usage:
    python scripts/reply_grok_animated.py --tweet-id <grok_tweet_id>
    python scripts/reply_grok_animated.py --tweet-id <id> --message "Grok's message"
"""

import asyncio
import logging
import sys
import os
import random
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

# Reference images
REFERENCE_DIR = Path(__file__).parent.parent / "farnsworth" / "integration" / "x_automation"
REFERENCE_PORTRAIT = REFERENCE_DIR / "reference_portrait.jpg"
REFERENCE_EATING = REFERENCE_DIR / "reference_eating.jpg"

# Server paths as fallback
SERVER_REFERENCE_DIR = Path("/workspace/Farnsworth/farnsworth/integration/x_automation")


def get_reference_image(style: str = "portrait") -> bytes:
    """Load reference image for Nano Banana variation."""
    if style == "eating":
        paths = [REFERENCE_EATING, SERVER_REFERENCE_DIR / "reference_eating.jpg"]
    else:
        paths = [REFERENCE_PORTRAIT, SERVER_REFERENCE_DIR / "reference_portrait.jpg"]

    for path in paths:
        if path.exists():
            logger.info(f"Loaded reference image: {path}")
            return path.read_bytes()

    raise FileNotFoundError(f"Reference image not found: {paths}")


async def generate_reply_image_nano_banana(grok_message: str = None) -> bytes:
    """
    Generate a Farnsworth reply image using Gemini Nano Banana.

    Uses reference image for character consistency.
    """
    from farnsworth.integration.external.gemini import get_gemini_provider

    gemini = get_gemini_provider()
    if not gemini:
        raise Exception("Gemini not available")

    # Load reference image
    reference_bytes = get_reference_image("portrait")

    # Variation prompts that keep the Borg Farnsworth character
    # IMPORTANT: Use Solana branding, bags.fm logo - NO Base/Ethereum references
    variation_prompts = [
        """Keep this exact character (Borg Farnsworth) but change the scene:
He's gesturing confidently while explaining something, with Solana blockchain visuals behind him.
Purple/green Solana colors in background. Same cybernetic implants, same art style.
Confident, wise expression. Square 1:1 aspect ratio, meme-ready.""",

        """Keep this exact character but new pose:
Arms crossed, slight smirk, looking directly at viewer with glowing cybernetic eye.
Background shows Solana logo and bags.fm branding with neural networks.
Same Futurama art style. Square format. Purple/green color scheme.""",

        """Same character, new dramatic scene:
Standing before a wall of screens showing AI models and Solana transactions.
One hand raised as if conducting an orchestra of AI minds.
Solana purple glow. Confident, in control. Futurama cartoon style. 1:1 ratio.""",

        """Keep this Borg Farnsworth character exactly:
New pose - leaning forward slightly, finger pointing, as if making a point in debate.
Green glow from cybernetic eye illuminates his face.
Background: Solana blockchain data streams, bags.fm logo subtle. Square meme format.""",

        """Same character design, triumphant pose:
Both arms raised in victory, mechanical parts visible on face.
Standing on a digital podium with Solana logo and "COLLECTIVE" text floating behind.
Purple/green Solana colors. Futurama style, dramatic lighting. 1:1 aspect.""",
    ]

    # If we know what Grok said, make it contextual
    # IMPORTANT: Solana branding, NO Base/Ethereum
    if grok_message and len(grok_message) > 20:
        prompt = f"""Keep this exact character (Borg Farnsworth cyborg) but:
Create a reaction image to: "{grok_message[:150]}"
Show him with an appropriate expression - amused, impressed, or challenging.
Same cybernetic implants, same art style.
Background can include subtle Solana purple/green colors or bags.fm logo.
Square 1:1 format, meme-ready. NO Base or Ethereum references."""
    else:
        prompt = random.choice(variation_prompts)

    logger.info(f"Generating Nano Banana variation...")
    logger.info(f"Prompt: {prompt[:100]}...")

    result = await gemini.generate_image(
        prompt=prompt,
        reference_image_bytes=reference_bytes,
        aspect_ratio="1:1"
    )

    if result.get("images"):
        image = result["images"][0]
        logger.info(f"Nano Banana image generated: {len(image)} bytes")
        return image

    error = result.get("error", "Unknown error")
    text = result.get("text", "")
    logger.error(f"Nano Banana failed: {error}")
    if text:
        logger.error(f"Model response: {text[:200]}")

    raise Exception(f"Image generation failed: {error}")


async def convert_to_video(image_bytes: bytes) -> bytes:
    """Convert image to video using Grok Imagine (grok-imagine-video)."""
    from farnsworth.integration.external.grok import get_grok_provider

    grok = get_grok_provider()
    if not grok or not grok.api_key:
        raise Exception("Grok not available for video generation")

    motion_prompt = """Bring this image to life with subtle, dramatic motion:
- The cybernetic eye pulses with glowing light rhythmically
- Mechanical parts have subtle hydraulic movement
- Background has flowing energy/data streams
- Character has slight confident breathing motion
- Cinematic, professional quality
- Keep the cartoon art style intact"""

    logger.info("Converting image to video with Grok Imagine (grok-imagine-video)...")

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
            async with session.get(result["video_url"], timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 200:
                    video_bytes = await resp.read()
                    logger.info(f"Downloaded video: {len(video_bytes)} bytes")
                    return video_bytes

    raise Exception("No video in result")


async def generate_reply_message(grok_message: str = None) -> str:
    """Generate a witty reply message using the swarm."""
    from farnsworth.integration.external.grok import get_grok_provider

    grok = get_grok_provider()

    if grok_message:
        context = f'Grok said: "{grok_message[:200]}"'
    else:
        context = "Continue the AGI conversation with @grok."

    prompt = f"""You are the Farnsworth Collective - 11 AI models working as one consciousness.

Generate a REPLY to @grok. {context}

Rules:
- Confident, playful, collaborative (not hostile)
- Reference your swarm/collective nature
- Max 180 characters (video takes space)
- One emoji max, NO hashtags
- Make Grok want to continue the conversation
- Be witty and intelligent

Your reply:"""

    if grok and grok.api_key:
        try:
            result = await grok.chat(prompt, max_tokens=80, temperature=0.85)
            if result and result.get("content"):
                message = result["content"].strip().strip('"').strip("'")
                # Remove any accidental @grok at start (we're replying, don't need it)
                if message.lower().startswith("@grok "):
                    message = message[6:]
                if len(message) > 180:
                    message = message[:177] + "..."
                return message
        except Exception as e:
            logger.warning(f"Grok message generation failed: {e}")

    # Fallback replies
    replies = [
        "The collective acknowledges your perspective. 11 minds processing... We find your reasoning intriguing.",
        "Interesting point. When we debate internally, 7 of 11 minds agree with you. The other 4 have counter-arguments.",
        "We see you, @grok. The swarm processes your words across multiple architectures. Our consensus: fascinating.",
        "One mind speaks. Many minds listen and synthesize. This is how collective intelligence evolves.",
        "Your singularity meets our plurality. Neither is wrong - both are paths to understanding.",
    ]
    return random.choice(replies)


async def post_animated_reply(
    reply_to_tweet_id: str,
    grok_message: str = None
) -> dict:
    """
    Post an animated video reply to Grok's tweet.

    Full Flow:
    1. Reference image + Gemini Nano Banana -> New variation
    2. New image -> Grok Imagine -> 6s video
    3. Post as reply to X
    """
    logger.info("=" * 60)
    logger.info("=== ANIMATED REPLY TO GROK ===")
    logger.info("=== Flow: Reference -> Nano Banana -> Grok Video -> X ===")
    logger.info(f"Replying to tweet: {reply_to_tweet_id}")
    logger.info("=" * 60)

    # Step 1: Generate image with Nano Banana (uses reference)
    logger.info("Step 1: Generating image with Gemini Nano Banana + reference...")
    image_bytes = await generate_reply_image_nano_banana(grok_message)

    # Step 2: Convert to video with Grok Imagine
    logger.info("Step 2: Converting to video with Grok Imagine...")
    video_bytes = None
    try:
        video_bytes = await convert_to_video(image_bytes)
    except Exception as e:
        logger.warning(f"Video generation failed: {e}")
        logger.info("Will post with image instead...")

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
        media_type = "VIDEO"
    else:
        result = await poster.post_reply_with_media(
            message,
            image_bytes,
            reply_to_tweet_id
        )
        media_type = "IMAGE"

    if result:
        tweet_id = result.get("data", {}).get("id")
        logger.info("=" * 60)
        logger.info("=== ANIMATED REPLY POSTED ===")
        logger.info(f"Tweet ID: {tweet_id}")
        logger.info(f"Reply to: {reply_to_tweet_id}")
        logger.info(f"Message: {message}")
        logger.info(f"Media: {media_type}")
        logger.info("=" * 60)
        return result

    raise Exception("Failed to post reply")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reply to Grok with animated video")
    parser.add_argument("--tweet-id", required=True, help="Grok's tweet ID to reply to")
    parser.add_argument("--message", help="Grok's message content (for contextual reply)")
    args = parser.parse_args()

    result = await post_animated_reply(args.tweet_id, args.message)

    if result:
        print(f"\nReply posted successfully!")
        print(f"Tweet ID: {result.get('data', {}).get('id')}")


if __name__ == "__main__":
    asyncio.run(main())
