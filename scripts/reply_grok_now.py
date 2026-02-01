#!/usr/bin/env python3
"""Quick script to reply to Grok's latest message"""

import asyncio
from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
from farnsworth.integration.x_automation.posting_brain import PostingBrain
import httpx

async def main():
    poster = get_x_api_poster()
    brain = PostingBrain()

    if poster.is_token_expired():
        await poster.refresh_access_token()

    # Get Grok replies
    headers = {"Authorization": f"Bearer {poster.access_token}"}
    query = "conversation_id:2017837874779938899 from:grok"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&tweet.fields=text&max_results=10"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        data = resp.json()
        tweets = data.get("data", [])
        print(f"Found {len(tweets)} Grok tweets")
        for t in tweets:
            tid = t["id"]
            txt = t["text"][:80]
            print(f"  {tid}: {txt}...")

        # Reply to the newest one
        if tweets:
            newest = tweets[0]
            print(f"\nReplying to: {newest['id']}")
            response = await brain.generate_grok_response(newest["text"])
            print(f"Response: {response}")
            result = await poster.post_reply(response, newest["id"])
            print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
