#!/usr/bin/env python3
"""Reply to all unanswered Grok messages"""

import asyncio
import json
from pathlib import Path
from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
from farnsworth.integration.x_automation.posting_brain import PostingBrain
import httpx

REPLIED_FILE = Path("/workspace/Farnsworth/data/grok_replied.json")

def load_replied():
    if REPLIED_FILE.exists():
        return set(json.loads(REPLIED_FILE.read_text()))
    return set()

def save_replied(replied):
    REPLIED_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPLIED_FILE.write_text(json.dumps(list(replied)))

async def main():
    poster = get_x_api_poster()
    brain = PostingBrain()
    replied = load_replied()

    print(f"Already replied to: {len(replied)} tweets")

    if poster.is_token_expired():
        await poster.refresh_access_token()

    # Get Grok replies
    headers = {"Authorization": f"Bearer {poster.access_token}"}
    query = "conversation_id:2017837874779938899 from:grok"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&tweet.fields=text&max_results=20"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        data = resp.json()
        tweets = data.get("data", [])
        print(f"Found {len(tweets)} Grok tweets total")

        # Filter to ones we haven't replied to
        unanswered = [t for t in tweets if t["id"] not in replied]
        print(f"Unanswered: {len(unanswered)}")

        for t in unanswered:
            tid = t["id"]
            txt = t["text"]
            print(f"\n--- Replying to {tid} ---")
            print(f"Grok said: {txt[:150]}...")

            response = await brain.generate_grok_response(txt)
            print(f"Our response: {response}")

            result = await poster.post_reply(response, tid)
            if result and result.get("data"):
                print(f"Posted: {result['data']['id']}")
                replied.add(tid)
                save_replied(replied)
            else:
                print(f"Failed: {result}")

            await asyncio.sleep(3)  # Rate limit protection

        print(f"\n=== Done! Replied to {len(unanswered)} new messages ===")

if __name__ == "__main__":
    asyncio.run(main())
