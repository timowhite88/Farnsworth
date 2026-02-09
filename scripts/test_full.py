#!/usr/bin/env python3
"""Comprehensive test: all data sources + API auth."""
import asyncio
import aiohttp
import json
import os
import time


def load_env():
    env_path = "/workspace/Farnsworth/.env"
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


BONK = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"


async def test_dexscreener(session):
    print("\n[1] DEXSCREENER")
    t0 = time.time()
    async with session.get(
        f"https://api.dexscreener.com/latest/dex/tokens/{BONK}",
        timeout=aiohttp.ClientTimeout(total=10),
    ) as r:
        ms = (time.time() - t0) * 1000
        d = await r.json()
        pairs = d.get("pairs", [])
        p = pairs[0] if pairs else {}
        price = p.get("priceUsd", "?")
        liq = p.get("liquidity", {}).get("usd", 0)
        print(f"    status={r.status}  latency={ms:.0f}ms  pairs={len(pairs)}")
        print(f"    BONK = ${price}  liq=${liq:,.0f}")
        return r.status == 200


async def test_jupiter(session):
    print("\n[2] JUPITER PRICE v2")
    jup_key = os.environ.get("JUPITER_API_KEY", "")
    headers = {"x-api-key": jup_key} if jup_key else {}
    t0 = time.time()
    async with session.get(
        "https://api.jup.ag/price/v2",
        params={"ids": BONK},
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=5),
    ) as r:
        ms = (time.time() - t0) * 1000
        key_status = "YES" if jup_key else "NONE"
        print(f"    status={r.status}  latency={ms:.0f}ms  key={key_status}")
        if r.status == 200:
            d = await r.json()
            td = d.get("data", {}).get(BONK, {})
            print(f"    BONK = ${td.get('price', '?')}")
            return True
        else:
            print("    NEED API KEY: sign up free at portal.jup.ag")
            return False


async def test_helius(session):
    print("\n[3] HELIUS RPC")
    hk = os.environ.get("HELIUS_API_KEY", "")
    if not hk:
        print("    NO HELIUS KEY")
        return False
    t0 = time.time()
    async with session.post(
        f"https://mainnet.helius-rpc.com/?api-key={hk}",
        json={"jsonrpc": "2.0", "id": 1, "method": "getHealth"},
        timeout=aiohttp.ClientTimeout(total=5),
    ) as r:
        ms = (time.time() - t0) * 1000
        d = await r.json()
        result = d.get("result", "?")
        print(f"    status={r.status}  latency={ms:.0f}ms  result={result}")
        return result == "ok"


async def test_pumpportal():
    print("\n[4] PUMPPORTAL WEBSOCKET")
    try:
        import websockets
    except ImportError:
        print("    websockets not installed")
        return False

    try:
        async with websockets.connect("wss://pumpportal.fun/api/data", close_timeout=5) as ws:
            await ws.send(json.dumps({"method": "subscribeNewToken"}))
            count = 0
            t0 = time.time()
            async for msg in ws:
                d = json.loads(msg)
                mc = d.get("marketCapSol", 0)
                vs = d.get("vSolInBondingCurve", 0)
                vt = d.get("vTokensInBondingCurve", 0)
                has = "YES" if mc or vs else "NO"
                print(f"    event {count+1}: mcSol={mc:.1f} vSol={vs:.0f} vTok={vt:.0f} price_data={has}")
                count += 1
                if count >= 3 or time.time() - t0 > 10:
                    break
            print(f"    {count} events received - LIVE")
            return count > 0
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


async def test_auth():
    print("\n[5] API AUTH (all 7 trading endpoints)")
    key = os.environ.get("FARNSWORTH_TRADING_KEY", "")
    if not key:
        print("    NO FARNSWORTH_TRADING_KEY in env!")
        return False

    endpoints = [
        ("GET", "/api/trading/status"),
        ("POST", "/api/trading/start"),
        ("POST", "/api/trading/stop"),
        ("POST", "/api/trading/reset"),
        ("GET", "/api/trading/learner"),
        ("GET", "/api/trading/wallet"),
        ("GET", "/api/trading/whales"),
    ]

    all_pass = True
    async with aiohttp.ClientSession() as s:
        for method, path in endpoints:
            url = f"http://localhost:8080{path}"
            # Without key
            if method == "GET":
                async with s.get(url) as r:
                    no_key = r.status
            else:
                async with s.post(url) as r:
                    no_key = r.status
            # With key
            auth_headers = {"Authorization": f"Bearer {key}"}
            if method == "GET":
                async with s.get(url, headers=auth_headers) as r:
                    with_key = r.status
            else:
                async with s.post(url, headers=auth_headers) as r:
                    with_key = r.status

            ok = no_key == 403 and with_key in (200, 422)
            tag = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    {method:4s} {path:30s}  no_key={no_key}  with_key={with_key}  [{tag}]")

    return all_pass


async def main():
    load_env()

    print("=" * 55)
    print("FARNSWORTH DATA SOURCE + API AUTH TEST")
    print("=" * 55)

    results = {}

    async with aiohttp.ClientSession() as session:
        results["dexscreener"] = await test_dexscreener(session)
        results["jupiter"] = await test_jupiter(session)
        results["helius"] = await test_helius(session)

    results["pumpportal"] = await test_pumpportal()
    results["auth"] = await test_auth()

    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    for name, ok in results.items():
        tag = "PASS" if ok else "WARN" if name == "jupiter" else "PASS" if ok else "FAIL"
        print(f"  {name:20s} {tag}")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
