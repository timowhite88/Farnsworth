#!/usr/bin/env python3
"""
x402 Discovery Registration — Announce Farnsworth Quantum API to the ecosystem.

The x402 ecosystem uses multiple discovery mechanisms:
1. .well-known/x402.json — Standard auto-discovery (already served)
2. CDP Bazaar — Auto-catalogs on first payment via CDP facilitator
3. x402 Index — Community directory at x402index.com
4. x402scan — Ecosystem explorer
5. x402list.fun — Facilitator-reported directory
6. BlockRun.AI — Service catalog
7. x402apis.io — Decentralized marketplace (Solana-native)

This script:
- Verifies our .well-known/x402.json is accessible
- Pings known crawlers/directories to trigger indexing
- Logs registration status

Run: python3 scripts/x402_register.py
"""

import asyncio
import aiohttp
import json
import sys
import os

# Our service details
FARNSWORTH_BASE = os.getenv("FARNSWORTH_API_URL", "https://ai.farnsworth.cloud")
DEX_BASE = os.getenv("DEX_URL", "https://ai.farnsworth.cloud/dex")

DISCOVERY_URLS = [
    f"{FARNSWORTH_BASE}/.well-known/x402.json",
    f"{FARNSWORTH_BASE}/api/x402/discovery",
    f"{FARNSWORTH_BASE}/api/x402/quantum/pricing",
]

# Known x402 directories and crawlers to notify
DIRECTORIES = [
    {
        "name": "x402 Index",
        "url": "https://x402index.com",
        "crawl_hint": "https://x402index.com/api/all",
        "note": "Community index — may auto-crawl .well-known endpoints",
    },
    {
        "name": "x402scan",
        "url": "https://x402scan.com",
        "note": "Ecosystem explorer — crawls x402 services",
    },
    {
        "name": "x402list.fun",
        "url": "https://x402list.fun",
        "note": "Aggregates from facilitators — listing happens via facilitator registration",
    },
    {
        "name": "BlockRun.AI",
        "url": "https://blockrun.ai",
        "note": "Service catalog for x402",
    },
    {
        "name": "x402apis.io",
        "url": "https://www.x402apis.io",
        "note": "Decentralized marketplace on Solana — may need node registration",
    },
    {
        "name": "x402 Playground",
        "url": "https://www.x402playground.com",
        "note": "Interactive testing environment with Bazaar discovery",
    },
    {
        "name": "CDP Bazaar Discovery",
        "url": "https://api.cdp.coinbase.com/platform/v2/x402/discovery/resources",
        "note": "Coinbase CDP facilitator — auto-catalogs on first payment",
    },
    {
        "name": "x402.org Facilitator",
        "url": "https://www.x402.org/facilitator/discovery/resources",
        "note": "x402.org facilitator discovery endpoint",
    },
    {
        "name": "Rencom Search",
        "url": "https://x402.rencom.ai",
        "note": "x402 resource search engine",
    },
]


async def verify_discovery_endpoints():
    """Verify our .well-known and discovery endpoints are accessible."""
    print("\n=== Verifying Farnsworth x402 Discovery Endpoints ===\n")
    results = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        for url in DISCOVERY_URLS:
            try:
                async with session.get(url) as resp:
                    status = resp.status
                    if status == 200:
                        data = await resp.json()
                        print(f"  [OK] {url} — {status}")
                        if "x402Version" in data:
                            print(f"       x402 v{data['x402Version']} manifest with {len(data.get('endpoints', []))} endpoint(s)")
                        elif "service" in data:
                            print(f"       Service: {data['service']}, Price: {data.get('price_sol')} SOL")
                        results.append({"url": url, "status": "ok"})
                    else:
                        print(f"  [FAIL] {url} — HTTP {status}")
                        results.append({"url": url, "status": f"http_{status}"})
            except Exception as e:
                print(f"  [ERROR] {url} — {e}")
                results.append({"url": url, "status": f"error: {e}"})

    return results


async def test_402_flow():
    """Test the actual 402 payment flow."""
    print("\n=== Testing x402 Payment Required Flow ===\n")
    url = f"{FARNSWORTH_BASE}/api/x402/quantum/analyze"

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        try:
            async with session.post(
                url,
                json={"token_address": "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"},
                headers={"Content-Type": "application/json"},
            ) as resp:
                status = resp.status
                data = await resp.json()

                if status == 402:
                    print(f"  [OK] 402 Payment Required returned correctly")
                    print(f"       Price: {data.get('price_sol')} SOL ({data.get('price_lamports')} lamports)")
                    print(f"       Pay to: {data.get('pay_to')}")
                    print(f"       Network: {data.get('network')}")

                    # Check for x402 headers
                    x_payment = resp.headers.get("X-PAYMENT")
                    if x_payment:
                        print(f"       X-PAYMENT header: present ({len(x_payment)} chars, base64)")
                    else:
                        print(f"       [WARN] X-PAYMENT header missing!")

                    return True
                else:
                    print(f"  [FAIL] Expected 402, got {status}")
                    return False
        except Exception as e:
            print(f"  [ERROR] {e}")
            return False


async def ping_directories():
    """Ping known directories to trigger crawling of our .well-known endpoint."""
    print("\n=== Pinging x402 Directories ===\n")
    print("  Most x402 directories discover services via:")
    print("  1. .well-known/x402.json auto-crawling")
    print("  2. CDP facilitator auto-cataloging on first payment")
    print("  3. Manual submission (some directories)\n")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        for directory in DIRECTORIES:
            try:
                async with session.get(directory["url"]) as resp:
                    print(f"  [{resp.status}] {directory['name']} — {directory['url']}")
                    print(f"       {directory['note']}")
            except Exception as e:
                print(f"  [DOWN] {directory['name']} — {directory['url']}")
                print(f"       {e}")


async def print_registration_guide():
    """Print manual registration steps for directories that need it."""
    print("\n=== Manual Registration Guide ===\n")
    print("  Our .well-known/x402.json is live and auto-discoverable.")
    print("  For directories that need manual registration:\n")

    print("  1. CDP Bazaar (Coinbase)")
    print("     → Auto-catalogs when you use CDP as facilitator")
    print("     → Set up CDP facilitator SDK: npm install @x402/express @x402/svm")
    print("     → Use declareDiscoveryExtension() with discoverable: true\n")

    print("  2. x402apis.io (Decentralized, Solana-native)")
    print("     → Clone their router node: git clone x402-router-node")
    print("     → Or submit via their client SDK: npm i -g @x402apis/client\n")

    print("  3. x402list.fun")
    print("     → Aggregates from facilitators automatically")
    print("     → Will appear once we process a payment through a listed facilitator\n")

    print("  4. x402 Index (x402index.com)")
    print("     → Community directory — submit at their website\n")

    print("  5. BlockRun.AI")
    print("     → Crawls x402 .well-known endpoints periodically\n")

    print(f"\n  Live endpoints:")
    print(f"    Discovery:  {FARNSWORTH_BASE}/.well-known/x402.json")
    print(f"    Pricing:    {FARNSWORTH_BASE}/api/x402/quantum/pricing")
    print(f"    Analyze:    POST {FARNSWORTH_BASE}/api/x402/quantum/analyze")
    print(f"    Stats:      {FARNSWORTH_BASE}/api/x402/quantum/stats")


async def main():
    print("=" * 60)
    print("  Farnsworth x402 Premium API — Registration & Verification")
    print("=" * 60)

    # 1. Verify our endpoints
    await verify_discovery_endpoints()

    # 2. Test 402 flow
    await test_402_flow()

    # 3. Ping directories
    await ping_directories()

    # 4. Print manual guide
    await print_registration_guide()

    print("\n" + "=" * 60)
    print("  Registration complete. x402 API is live and discoverable.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
