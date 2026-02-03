#!/usr/bin/env python3
"""
Farnsworth VTuber Launcher
Start the AI VTuber streaming system

Usage:
    python scripts/start_vtuber.py --stream-key YOUR_KEY
    python scripts/start_vtuber.py --test  # Simulated mode
    python scripts/start_vtuber.py --help
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


def setup_logging(debug: bool = False):
    """Configure logging"""
    level = "DEBUG" if debug else "INFO"
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level
    )
    logger.add(
        "/tmp/vtuber.log",
        rotation="100 MB",
        retention="7 days",
        level="DEBUG"
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Farnsworth VTuber Streaming System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Start streaming to Twitter:
    python scripts/start_vtuber.py --stream-key YOUR_TWITTER_KEY

  Test mode (no actual streaming):
    python scripts/start_vtuber.py --test

  High quality stream:
    python scripts/start_vtuber.py --stream-key KEY --quality high

  With custom avatar:
    python scripts/start_vtuber.py --stream-key KEY --avatar /path/to/avatar.png
        """
    )

    parser.add_argument(
        "--stream-key",
        help="Twitter/RTMPS stream key (required unless --test)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with simulated chat"
    )
    parser.add_argument(
        "--platform",
        choices=["twitter", "youtube", "twitch", "custom"],
        default="twitter",
        help="Streaming platform (default: twitter)"
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high", "ultra"],
        default="medium",
        help="Stream quality preset (default: medium)"
    )
    parser.add_argument(
        "--avatar",
        help="Path to custom avatar image"
    )
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Use neural avatar (MuseTalk) if available"
    )
    parser.add_argument(
        "--no-swarm",
        action="store_true",
        help="Disable swarm collective (use single agent)"
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["Farnsworth", "Grok", "DeepSeek", "Gemini"],
        help="Agents to use in swarm collective"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Start API server without streaming"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="API server port (default: 8081)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.debug)

    # Validate arguments
    if not args.test and not args.stream_key and not args.api_only:
        logger.error("--stream-key required (or use --test for simulation mode)")
        parser.print_help()
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("   FARNSWORTH VTUBER STREAMING SYSTEM")
    logger.info("=" * 50)

    if args.api_only:
        # Start API server only
        await run_api_server(args.port)
    else:
        # Start VTuber stream
        await run_vtuber(args)


async def run_vtuber(args):
    """Run the VTuber stream"""
    from farnsworth.integration.vtuber.vtuber_core import FarnsworthVTuber, VTuberConfig
    from farnsworth.integration.vtuber.stream_manager import StreamQuality
    from farnsworth.integration.vtuber.avatar_controller import AvatarBackend

    # Map arguments to config
    quality_map = {
        "low": StreamQuality.LOW,
        "medium": StreamQuality.MEDIUM,
        "high": StreamQuality.HIGH,
        "ultra": StreamQuality.ULTRA,
    }

    avatar_backend = AvatarBackend.NEURAL if args.neural else AvatarBackend.IMAGE_SEQUENCE

    config = VTuberConfig(
        name="Farnsworth",
        persona="An eccentric AI scientist leading a collective of AI agents",
        stream_key=args.stream_key or "",
        stream_quality=quality_map.get(args.quality, StreamQuality.MEDIUM),
        avatar_backend=avatar_backend,
        avatar_model_path=args.avatar,
        simulate_chat=args.test,
        use_swarm_collective=not args.no_swarm,
        swarm_agents=args.agents,
        debug_mode=args.debug,
    )

    logger.info(f"Platform: {args.platform}")
    logger.info(f"Quality: {args.quality}")
    logger.info(f"Avatar: {avatar_backend.value}")
    logger.info(f"Swarm Agents: {', '.join(args.agents)}")
    logger.info(f"Simulate Chat: {args.test}")

    # Create VTuber
    vtuber = FarnsworthVTuber(config)

    try:
        # Start stream
        logger.info("Starting VTuber stream...")
        success = await vtuber.start()

        if not success:
            logger.error("Failed to start VTuber stream")
            sys.exit(1)

        logger.info("=" * 50)
        logger.info("   VTUBER IS LIVE!")
        logger.info("=" * 50)
        logger.info("Press Ctrl+C to stop")

        # Keep running
        while vtuber.is_live:
            await asyncio.sleep(1)

            # Periodic status
            if vtuber.stats:
                stats = vtuber.stats
                if stats.get("stream_stats"):
                    fps = stats["stream_stats"].get("avg_fps", 0)
                    bitrate = stats["stream_stats"].get("bitrate_kbps", 0)
                    logger.debug(f"Stream: {fps:.1f} fps, {bitrate:.0f} kbps")

    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")

    finally:
        await vtuber.stop()
        logger.info("VTuber stopped. Goodbye!")


async def run_api_server(port: int):
    """Run the API server only (for remote control)"""
    import uvicorn
    from fastapi import FastAPI

    from farnsworth.integration.vtuber.server_integration import router

    app = FastAPI(
        title="Farnsworth VTuber API",
        description="Control the Farnsworth AI VTuber",
        version="1.0.0"
    )
    app.include_router(router)

    logger.info(f"Starting VTuber API server on port {port}")

    config = uvicorn.Config(app, host="0.0.0.0", port=port)
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
