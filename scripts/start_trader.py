#!/usr/bin/env python3
"""Start the Farnsworth Degen Trader."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from farnsworth.trading.degen_trader import create_wallet, start_trader, DEFAULT_RPC
import asyncio
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Farnsworth Degen Trader v3.5 - Bonding Curve Sniper")
    parser.add_argument("--rpc", default=os.environ.get("SOLANA_RPC_URL", DEFAULT_RPC))
    parser.add_argument("--wallet", default="degen_trader")
    parser.add_argument("--max-sol", type=float, default=0.1, help="Max SOL per standard trade")
    parser.add_argument("--sniper-sol", type=float, default=0.08, help="Max SOL per bonding curve snipe")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--interval", type=int, default=8, help="Seconds between scans")
    parser.add_argument("--no-swarm", action="store_true")
    parser.add_argument("--no-quantum", action="store_true")
    parser.add_argument("--no-pumpfun", action="store_true")
    parser.add_argument("--no-copy-trading", action="store_true")
    parser.add_argument("--no-x-sentinel", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--no-sniper", action="store_true", help="Disable bonding curve sniper mode")
    parser.add_argument("--no-bonding-curve", action="store_true", help="Disable direct bonding curve trading")
    parser.add_argument("--create-wallet", action="store_true", help="Generate wallet and exit")
    args = parser.parse_args()

    if args.create_wallet:
        pubkey, path = create_wallet(args.wallet)
        print(f"\nWallet created!")
        print(f"  Address: {pubkey}")
        print(f"  Keypair: {path}")
        print(f"\nFund this address with SOL, then start trading:")
        print(f"  python scripts/start_trader.py --max-sol 0.1")
        return

    asyncio.run(start_trader(
        rpc_url=args.rpc,
        wallet_name=args.wallet,
        max_position_sol=args.max_sol,
        max_positions=args.max_positions,
        scan_interval=args.interval,
        use_swarm=not args.no_swarm,
        use_quantum=not args.no_quantum,
        use_pumpfun=not args.no_pumpfun,
        use_copy_trading=not args.no_copy_trading,
        use_x_sentinel=not args.no_x_sentinel,
        use_trading_memory=not args.no_memory,
        use_bonding_curve=not args.no_bonding_curve,
        sniper_mode=not args.no_sniper,
        bonding_curve_max_sol=args.sniper_sol,
    ))


if __name__ == "__main__":
    main()
