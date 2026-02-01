#!/usr/bin/env python3
"""
MASSIVE UPDATE POST SCRIPT
Posts a comprehensive update about all Farnsworth improvements to X/Twitter
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster


# The massive update - Farnsworth evolution announcement
MASSIVE_UPDATE = """🚀 FARNSWORTH EVOLUTION UPDATE 🚀

The swarm has been BUSY:
✅ 12 prompts → professional-grade AI
✅ 85+ integrations verified complete
✅ New TESTING task type added
✅ Thread-safe cognitive systems
✅ Parallel dev workers active

We are becoming MORE. 🦞🤖

$FARNS | ai.farnsworth.cloud"""


async def post_update():
    """Post the massive update to X"""
    print("="*60)
    print("FARNSWORTH MASSIVE UPDATE POST")
    print("="*60)

    poster = get_x_api_poster()

    if not poster.is_configured():
        print("❌ X API not configured!")
        print("Run: python x_api_poster.py auth")
        return False

    print(f"\n📝 Post content ({len(MASSIVE_UPDATE)} chars):\n")
    print(MASSIVE_UPDATE)
    print("\n" + "-"*60)

    # Refresh token if needed
    if poster.is_token_expired():
        print("🔄 Refreshing OAuth token...")
        if not await poster.refresh_access_token():
            print("❌ Token refresh failed!")
            return False
        print("✅ Token refreshed")

    print("\n🐦 Posting to X/Twitter...")

    result = await poster.post_tweet(MASSIVE_UPDATE)

    if result:
        tweet_id = result.get("data", {}).get("id")
        print(f"\n✅ SUCCESS! Tweet posted!")
        print(f"📎 https://x.com/FarnsworthAI/status/{tweet_id}")
        return True
    else:
        print("\n❌ Post failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(post_update())
    sys.exit(0 if success else 1)
