#!/usr/bin/env python3
"""
Start RTMPS stream to X/Twitter using SadTalker full face animation.

SadTalker provides D-ID-level face animation with head motion, lip sync,
expressions, and eye blinks. Pre-renders frames for each utterance.

Prerequisites:
  SadTalker must be installed at /workspace/SadTalker with checkpoints
  export TWITTER_STREAM_KEY=...    # Set your X stream key
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add("/tmp/vtuber_sadtalker.log", rotation="50 MB", level="DEBUG")


async def main():
    from farnsworth.integration.vtuber.vtuber_core import FarnsworthVTuber, VTuberConfig
    from farnsworth.integration.vtuber.stream_manager import StreamQuality
    from farnsworth.integration.vtuber.avatar_controller import AvatarBackend

    # Load stream key
    stream_key = os.environ.get("TWITTER_STREAM_KEY", "")
    if not stream_key:
        logger.error("TWITTER_STREAM_KEY not set in environment")
        sys.exit(1)

    face_image = "/workspace/Farnsworth/assets/farnsworth_closeup.png"
    if not Path(face_image).exists():
        logger.error(f"Face image not found: {face_image}")
        sys.exit(1)

    sadtalker_dir = "/workspace/SadTalker"
    if not Path(sadtalker_dir).exists():
        logger.error(f"SadTalker not found at {sadtalker_dir}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("  FARNSWORTH RTMPS STREAM TO X")
    logger.info("  Backend: SADTALKER (full face animation)")
    logger.info(f"  Avatar: {face_image}")
    logger.info(f"  SadTalker: {sadtalker_dir}")
    logger.info(f"  Stream key: {stream_key[:4]}...")
    logger.info("=" * 50)

    config = VTuberConfig(
        name="Farnsworth",
        persona="An eccentric AI scientist leading a collective of AI agents",
        avatar_backend=AvatarBackend.SADTALKER,
        avatar_face_image=face_image,
        sadtalker_dir=sadtalker_dir,
        sadtalker_size=256,
        avatar_width=640,
        avatar_height=360,
        avatar_fps=15,
        stream_key=stream_key,
        stream_quality=StreamQuality.LOW,
        stream_platform="twitter",
        simulate_chat=False,
        use_swarm_collective=True,
        swarm_agents=["Farnsworth", "Grok", "DeepSeek", "Gemini"],
        enable_chat=True,
        idle_chat_interval=60.0,  # Longer interval since rendering takes time
    )

    vtuber = FarnsworthVTuber(config)

    try:
        logger.info("Starting VTuber stream with SadTalker...")
        success = await vtuber.start()

        if not success:
            logger.error("Failed to start stream")
            sys.exit(1)

        logger.info("=" * 50)
        logger.info("  VTUBER IS LIVE ON X (SadTalker)")
        logger.info("=" * 50)
        logger.info("Press Ctrl+C to stop")

        while vtuber.is_live:
            await asyncio.sleep(5)
            stats = vtuber.stats
            if stats.get("stream_stats"):
                fps = stats["stream_stats"].get("avg_fps", 0)
                frames = stats["stream_stats"].get("frames_sent", 0)
                logger.info(f"Stream: {fps:.1f} fps | {frames} frames sent")

    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    finally:
        await vtuber.stop()
        logger.info("Stream stopped.")


if __name__ == "__main__":
    asyncio.run(main())
