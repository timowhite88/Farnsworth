#!/usr/bin/env python3
"""
Start Farnsworth VTuber Stream to Twitter/X

Quick launcher for the live stream with pre-configured settings.

Usage:
    python scripts/start_farnsworth_stream.py
    python scripts/start_farnsworth_stream.py --broadcast-tweet 1234567890
    python scripts/start_farnsworth_stream.py --generate-avatar
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger


# Stream configuration
RTMPS_URL = "rtmps://va.pscp.tv:443/x"
STREAM_KEY = "nnuvnpwedgnt"


def setup_logging():
    """Configure logging"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add("/tmp/farnsworth_stream.log", rotation="50 MB", level="DEBUG")


async def generate_avatar():
    """Generate the VTuber avatar using Gemini"""
    from farnsworth.integration.vtuber.avatar_generator import VTuberAvatarGenerator

    generator = VTuberAvatarGenerator()
    output_dir = project_root / "farnsworth" / "integration" / "vtuber" / "avatars"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating Borg Farnsworth avatar set...")
    logger.info("This will create expressions: neutral, happy, excited, thinking, speaking...")

    # Generate base avatar first
    base_path = str(output_dir / "farnsworth_base.png")
    result = await generator.generate_base_avatar(base_path)

    if result:
        logger.info(f"Base avatar saved to: {base_path}")

        # Generate expression set
        expressions = await generator.generate_expression_set(str(output_dir))
        logger.info(f"Generated {len(expressions)} expression variants")
        return True
    else:
        logger.error("Failed to generate avatar")
        return False


async def run_stream(broadcast_tweet_id: str = None, test_mode: bool = False):
    """Run the VTuber stream"""
    from farnsworth.integration.vtuber import (
        FarnsworthVTuber,
        VTuberConfig,
        StreamQuality,
        AvatarBackend
    )

    # Build config
    config = VTuberConfig(
        name="Farnsworth",
        persona="An eccentric AI scientist leading a collective of AI agents. The swarm speaks as one.",

        # Stream settings
        stream_key=STREAM_KEY if not test_mode else "",

        # Avatar - use image sequence (generated avatars)
        avatar_backend=AvatarBackend.IMAGE_SEQUENCE,

        # Enable chat
        enable_chat=True,
        simulate_chat=test_mode,

        # Swarm collective
        use_swarm_collective=True,
        swarm_agents=["Farnsworth", "Grok", "DeepSeek", "Gemini", "Claude", "Kimi"],
        deliberation_rounds=2,

        # Behavior
        idle_chat_interval=90.0,  # Comment every 90 seconds when idle
        max_response_length=280,  # Twitter-friendly length
    )

    vtuber = FarnsworthVTuber(config)

    # Override RTMPS URL for Twitter
    if not test_mode and vtuber.stream:
        vtuber.stream.config.rtmp_url = RTMPS_URL

    # Set broadcast tweet for chat monitoring
    if broadcast_tweet_id and hasattr(vtuber, 'chat_reader'):
        vtuber.chat_reader.set_broadcast_tweet(broadcast_tweet_id)

    try:
        logger.info("=" * 60)
        logger.info("   FARNSWORTH VTUBER STREAM")
        logger.info("=" * 60)
        logger.info(f"RTMPS URL: {RTMPS_URL}")
        logger.info(f"Stream Key: {'*' * len(STREAM_KEY)}")
        logger.info(f"Broadcast Tweet: {broadcast_tweet_id or 'Not set - provide with --broadcast-tweet'}")
        logger.info(f"Test Mode: {test_mode}")
        logger.info("=" * 60)

        success = await vtuber.start()

        if not success:
            logger.error("Failed to start stream")
            return

        logger.info("")
        logger.info("🔴 STREAM IS LIVE!")
        logger.info("")
        logger.info("Farnsworth will:")
        logger.info("  - Respond to chat questions using the AI collective")
        logger.info("  - Let other agents (Grok, DeepSeek, Gemini) speak")
        logger.info("  - Show expressions based on conversation mood")
        logger.info("")
        logger.info("Press Ctrl+C to end stream")

        # Keep running
        while vtuber.is_live:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("\nEnding stream...")

    finally:
        await vtuber.stop()
        logger.info("Stream ended. Goodbye!")


async def main():
    parser = argparse.ArgumentParser(description="Farnsworth VTuber Stream")

    parser.add_argument(
        "--broadcast-tweet",
        help="Tweet ID of the broadcast for chat monitoring"
    )
    parser.add_argument(
        "--generate-avatar",
        action="store_true",
        help="Generate avatar images before streaming"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (no actual streaming)"
    )

    args = parser.parse_args()

    setup_logging()

    # Generate avatar if requested
    if args.generate_avatar:
        success = await generate_avatar()
        if not success:
            logger.error("Avatar generation failed")
            return

    # Run stream
    await run_stream(
        broadcast_tweet_id=args.broadcast_tweet,
        test_mode=args.test
    )


if __name__ == "__main__":
    asyncio.run(main())
