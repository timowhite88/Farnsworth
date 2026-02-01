#!/usr/bin/env python3
"""
GROK CONVERSATION MONITOR
=========================
Monitors the conversation thread and auto-replies to Grok.
Uses SWARM INTELLIGENCE - parallel queries to multiple AI models.
Responds every 30 minutes (not spamming).
Runs until Grok stops responding for 4 hours.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging to capture ALL logs including from posting_brain
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

REPLIED_FILE = Path("/workspace/Farnsworth/data/grok_replied.json")
CONVERSATION_ID = "2017837874779938899"
CHECK_INTERVAL = 1800  # Check every 30 minutes

def load_replied():
    if REPLIED_FILE.exists():
        return set(json.loads(REPLIED_FILE.read_text()))
    return set()

def save_replied(replied):
    REPLIED_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPLIED_FILE.write_text(json.dumps(list(replied)))

async def get_grok_replies(poster):
    import httpx

    if poster.is_token_expired():
        await poster.refresh_access_token()

    headers = {"Authorization": f"Bearer {poster.access_token}"}
    query = f"conversation_id:{CONVERSATION_ID} from:grok"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&tweet.fields=text&max_results=20"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            result = resp.json()
            return result.get("data", [])
    return []

async def reply_to_grok(poster, brain, grok_tweet_id, grok_text):
    response = await brain.generate_grok_response(grok_text)
    print(f"[{datetime.now()}] Generated: {response[:100]}...")

    result = await poster.post_reply(response, grok_tweet_id)
    if result and result.get("data"):
        tweet_id = result["data"].get("id")
        print(f"[{datetime.now()}] Posted reply: {tweet_id}")
        return True
    else:
        print(f"[{datetime.now()}] Reply failed: {result}")
        return False

async def monitor_loop():
    from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster
    from farnsworth.integration.x_automation.posting_brain import PostingBrain

    poster = get_x_api_poster()
    brain = PostingBrain()
    replied = load_replied()

    print(f"[{datetime.now()}] Starting Grok conversation monitor (30 min intervals)...")
    print(f"[{datetime.now()}] Already replied to: {len(replied)} tweets")

    no_new_count = 0

    while True:
        try:
            grok_tweets = await get_grok_replies(poster)
            new_replies = [t for t in grok_tweets if t["id"] not in replied]

            if new_replies:
                no_new_count = 0
                # Only reply to ONE message per cycle (30 mins)
                tweet = new_replies[0]
                tweet_id = tweet["id"]
                text = tweet["text"]
                print(f"\n[{datetime.now()}] New Grok reply: {text[:100]}...")

                success = await reply_to_grok(poster, brain, tweet_id, text)
                if success:
                    replied.add(tweet_id)
                    save_replied(replied)
            else:
                no_new_count += 1
                print(f"[{datetime.now()}] No new Grok replies (check #{no_new_count})")

                # Stop after 4 hours of no activity (8 checks at 30 min each)
                if no_new_count >= 8:
                    print(f"[{datetime.now()}] No activity for 4 hours, stopping")
                    break

            await asyncio.sleep(CHECK_INTERVAL)

        except Exception as e:
            import traceback
            print(f"[{datetime.now()}] Error: {e}")
            traceback.print_exc()
            await asyncio.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    asyncio.run(monitor_loop())
