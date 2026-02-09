#!/usr/bin/env python3
"""Test all trading data sources — Jupiter, DexScreener, PumpPortal WS."""
import asyncio
import aiohttp
import time
import json

BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
JUP = "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"


async def test_jupiter():
    print("=" * 50)
    print("JUPITER PRICE API v2")
    print("=" * 50)
    async with aiohttp.ClientSession() as session:
        start = time.time()
        async with session.get(
            "https://api.jup.ag/price/v2",
            params={"ids": f"{BONK},{JUP}"},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            elapsed = (time.time() - start) * 1000
            data = await resp.json()
            print(f"Status: {resp.status} | Latency: {elapsed:.0f}ms")
            for mint, info in data.get("data", {}).items():
                mid = info.get("id", "")[:16]
                price = info.get("price", "N/A")
                print(f"  {mid}... = ${price}")
    print()


async def test_dexscreener():
    print("=" * 50)
    print("DEXSCREENER API")
    print("=" * 50)
    async with aiohttp.ClientSession() as session:
        start = time.time()
        async with session.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{BONK}",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            elapsed = (time.time() - start) * 1000
            data = await resp.json()
            pairs = data.get("pairs", [])
            print(f"Status: {resp.status} | Latency: {elapsed:.0f}ms | Pairs: {len(pairs)}")
            if pairs:
                p = pairs[0]
                sym = p.get("baseToken", {}).get("symbol", "?")
                price = p.get("priceUsd", "N/A")
                liq = p.get("liquidity", {}).get("usd", 0)
                print(f"  {sym} = ${price}  liq=${liq:,.0f}")
    print()


async def test_pumpportal_ws():
    print("=" * 50)
    print("PUMPPORTAL WEBSOCKET")
    print("=" * 50)
    try:
        import websockets
    except ImportError:
        print("websockets not installed, skipping")
        return

    try:
        async with websockets.connect("wss://pumpportal.fun/api/data", close_timeout=5) as ws:
            await ws.send(json.dumps({"method": "subscribeNewToken"}))
            print("Connected & subscribed to newToken events")
            count = 0
            start = time.time()
            async for msg in ws:
                data = json.loads(msg)
                mint = data.get("mint", "")
                mc = data.get("marketCapSol", 0)
                v_sol = data.get("vSolInBondingCurve", 0)
                v_tok = data.get("vTokensInBondingCurve", 0)
                sol_amt = data.get("solAmount", 0) / 1e9 if data.get("solAmount") else 0
                has_price = "YES" if mc or v_sol else "NO"
                print(f"  mint={mint[:16]}... mcSol={mc:.2f} vSol={v_sol} vTok={v_tok} trade={sol_amt:.4f}SOL price_data={has_price}")
                count += 1
                if count >= 5 or time.time() - start > 15:
                    break
            print(f"Received {count} events")
    except Exception as e:
        print(f"PumpPortal WS error: {e}")
    print()


async def test_solana_rpc():
    print("=" * 50)
    print("SOLANA RPC (Helius)")
    print("=" * 50)
    import os
    helius_key = os.environ.get("HELIUS_API_KEY", "")
    if not helius_key:
        print("No HELIUS_API_KEY in env, skipping")
        return
    rpc_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
    async with aiohttp.ClientSession() as session:
        start = time.time()
        async with session.post(
            rpc_url,
            json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            elapsed = (time.time() - start) * 1000
            data = await resp.json()
            print(f"Status: {resp.status} | Latency: {elapsed:.0f}ms")
            print(f"  Result: {data.get('result', data.get('error', 'unknown'))}")
    print()


async def test_api_auth():
    import os as _os
    print("=" * 50)
    print("API AUTH TEST — trading endpoints")
    print("=" * 50)
    async with aiohttp.ClientSession() as session:
        # Test WITHOUT key — should get 403
        try:
            async with session.get(
                "https://ai.farnsworth.cloud/api/trading/status",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                print(f"  No key:   status={resp.status} (expect 403)")
        except Exception as e:
            print(f"  No key:   error={e}")

        # Test WITH correct key
        key = _os.environ.get("FARNSWORTH_TRADING_KEY", "")
        if key:
            try:
                async with session.get(
                    "https://ai.farnsworth.cloud/api/trading/status",
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()
                    print(f"  With key: status={resp.status} data={json.dumps(data)[:100]}")
            except Exception as e:
                print(f"  With key: error={e}")
        else:
            print("  FARNSWORTH_TRADING_KEY not in env, skipping auth test")

        # Test WITH wrong key — should get 403
        try:
            async with session.get(
                "https://ai.farnsworth.cloud/api/trading/status",
                headers={"Authorization": "Bearer wrong_key_12345"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                print(f"  Bad key:  status={resp.status} (expect 403)")
        except Exception as e:
            print(f"  Bad key:  error={e}")
    print()


async def main():
    import os
    # Load .env
    env_path = "/workspace/Farnsworth/.env"
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    await test_jupiter()
    await test_dexscreener()
    await test_pumpportal_ws()
    await test_solana_rpc()
    await test_api_auth()
    print("=" * 50)
    print("ALL DATA SOURCE TESTS COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
