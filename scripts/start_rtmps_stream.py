#!/usr/bin/env python3
"""
Start RTMPS stream to X/Twitter using local animation backend
with the cyborg Farnsworth avatar.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>", level="INFO")
logger.add("/tmp/vtuber_stream.log", rotation="50 MB", level="DEBUG")


async def main():
    from farnsworth.integration.vtuber.vtuber_core import FarnsworthVTuber, VTuberConfig
    from farnsworth.integration.vtuber.stream_manager import StreamConfig, StreamQuality
    from farnsworth.integration.vtuber.avatar_controller import AvatarBackend, AvatarConfig

    # Load stream key from env
    stream_key = os.environ.get("TWITTER_STREAM_KEY", "")
    if not stream_key:
        logger.error("TWITTER_STREAM_KEY not set in environment")
        sys.exit(1)

    face_image = "/workspace/Farnsworth/assets/farnsworth_cyborg.png"
    if not Path(face_image).exists():
        logger.error(f"Face image not found: {face_image}")
        sys.exit(1)

    # Manual mouth ROI for the cyborg Farnsworth portrait (875x942 original).
    # The lower face is human-looking; mouth is at ~76% down, centered.
    # Normalised coords: top-left (x, y) + size (w, h)
    mouth_roi = {"x": 0.36, "y": 0.72, "w": 0.28, "h": 0.08}

    logger.info("=" * 50)
    logger.info("  FARNSWORTH RTMPS STREAM TO X")
    logger.info("  Backend: LOCAL_ANIM (JawDrop animator)")
    logger.info(f"  Avatar: {face_image}")
    logger.info(f"  Mouth ROI: {mouth_roi}")
    logger.info(f"  Stream key: {stream_key[:4]}...")
    logger.info("=" * 50)

    # Configure VTuber with local animation backend
    config = VTuberConfig(
        name="Farnsworth",
        persona="An eccentric AI scientist leading a collective of AI agents",
        avatar_backend=AvatarBackend.LOCAL_ANIM,
        avatar_face_image=face_image,
        avatar_manual_roi=mouth_roi,
        avatar_width=854,
        avatar_height=480,
        avatar_fps=24,
        stream_key=stream_key,
        stream_quality=StreamQuality.MEDIUM,
        stream_platform="twitter",
        simulate_chat=False,
        use_swarm_collective=True,
        swarm_agents=["Farnsworth", "Grok", "DeepSeek", "Gemini"],
        enable_chat=True,
        idle_chat_interval=30.0,  # speak every 30s so we can see mouth animate
    )

    vtuber = FarnsworthVTuber(config)

    try:
        logger.info("Starting VTuber stream...")
        success = await vtuber.start()

        if not success:
            logger.error("Failed to start stream")
            sys.exit(1)

        logger.info("=" * 50)
        logger.info("  VTUBER IS LIVE ON X!")
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
