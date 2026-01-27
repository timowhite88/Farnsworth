#!/usr/bin/env python3
"""
Run Farnsworth Web Interface
Token-gated chat server

Usage:
    python run_web.py
    python run_web.py --port 8080 --host 0.0.0.0
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Farnsworth Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode (no token verification)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    # Set environment variables
    os.environ["FARNSWORTH_WEB_HOST"] = args.host
    os.environ["FARNSWORTH_WEB_PORT"] = str(args.port)

    if args.demo:
        os.environ["FARNSWORTH_DEMO_MODE"] = "true"

    # Import and run
    import uvicorn

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║             🧠 Farnsworth Neural Interface v2.8              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Token-Gated AI Companion Chat                               ║
    ║                                                              ║
    ║  Server: http://{args.host}:{args.port:<5}                            ║
    ║  Demo Mode: {'Yes' if args.demo or os.getenv('FARNSWORTH_DEMO_MODE', 'true').lower() == 'true' else 'No ':<4}                                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Required Token: 9crfy4udr...wBAGS                           ║
    ║  Full Features: Install locally for Solana, P2P, Swarm       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "farnsworth.web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
