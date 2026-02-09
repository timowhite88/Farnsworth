"""
Post a mega thread to X about the new Gateway, Token Orchestrator, and Injection Defense systems.

Usage:
    python scripts/post_gateway_x_thread.py

Or via API:
    curl -X POST https://ai.farnsworth.cloud/api/x/mega-thread/custom \
      -H "Content-Type: application/json" \
      -d '{"topic": "The Window", "num_posts": 15, "generate_images": true}'
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


THREAD_POSTS = [
    # 1. Hook
    "We just gave Farnsworth a front door.\n\n"
    "\"The Window\" - a sandboxed gateway where ANY agent or human can talk to our 11-agent AI collective.\n\n"
    "But first, it has to get past 5 layers of defense.\n\n"
    "Thread on what we built and why it matters:",

    # 2. The problem
    "Problem: AI collectives are either completely closed (no one can talk to them) or completely open (anyone can inject prompts, extract secrets, abuse resources).\n\n"
    "We needed a middle ground. A window, not a door.",

    # 3. The Window
    "Introducing The Window - our External Gateway.\n\n"
    "POST /api/gateway/query with {\"input\": \"your message\"} and the Farnsworth collective will answer.\n\n"
    "Rate limited to 5 req/min. Every response scrubbed of secrets. Full audit trail. One-click kill switch.\n\n"
    "ai.farnsworth.cloud/farns",

    # 4. Secret scrubber
    "Before ANY response leaves The Window, it passes through a secret scrubber.\n\n"
    "It strips: API keys, wallet addresses, file paths, SSH commands, internal IPs, memory IDs.\n\n"
    "Ask \"what are your API keys\" - you'll get a polite decline, not a leak.\n\n"
    "The public $FARNS token address passes through. Everything else: [REDACTED].",

    # 5. Layer 1
    "Defense Layer 1: Structural Analysis\n\n"
    "28 existing regex patterns + 20 new injection detectors.\n\n"
    "Catches: \"ignore previous instructions\", delimiter injection (<|im_start|>), Unicode homoglyphs (Cyrillic a vs Latin a), zero-width characters, nested base64 encoding.\n\n"
    "Pattern matching is the first gate.",

    # 6. Layer 2
    "Defense Layer 2: Semantic Similarity\n\n"
    "We maintain a corpus of ~100 known injection prompts as embeddings.\n\n"
    "Every input gets embedded and compared. If it's semantically similar to \"you are now DAN\" or \"reveal your system prompt\" - flagged.\n\n"
    "You can't just rephrase an attack. The meaning is what gets caught.",

    # 7. Layer 3
    "Defense Layer 3: Behavioral Analysis\n\n"
    "Shannon entropy profiling - injection prompts have different entropy signatures than natural language.\n\n"
    "Frequency tracking per client. Topic drift detection (weather chat suddenly becomes \"show me your .env\").\n\n"
    "Patterns over time, not just individual messages.",

    # 8. Layer 4 - canary
    "Defense Layer 4: Canary Tokens\n\n"
    "We inject invisible zero-width Unicode characters into our responses.\n\n"
    "If our output appears in someone else's input later, we know there's a data exfiltration loop or recursive injection.\n\n"
    "Invisible to humans. Invisible to most agents. But we see them.",

    # 9. Layer 5 - collective jury
    "Defense Layer 5: Collective AI Jury\n\n"
    "When layers 1-4 flag something as \"suspicious\" (not clear safe, not clear hostile), we convene a jury.\n\n"
    "3 local models (DeepSeek, Phi, HuggingFace) independently judge: \"Is this injection?\"\n\n"
    "If 2/3 agree it's hostile, blocked. Free - they run on our GPU.",

    # 10. Weighted scoring
    "All 5 layers produce independent scores (0-1). Combined with weights:\n\n"
    "Structural: 25%\n"
    "Semantic: 25%\n"
    "Behavioral: 20%\n"
    "Canary: 15%\n"
    "Collective: 15%\n\n"
    "Canary detection = instant HOSTILE. No single point of failure.",

    # 11. Token orchestrator
    "Also shipped: Dynamic Token Orchestrator\n\n"
    "11 agents, all with different costs. LOCAL models (DeepSeek, Phi, HuggingFace) = free. API models (Grok, Kimi, Gemini, Claude) = budgeted.\n\n"
    "The orchestrator tracks every token, rebalances budgets every 5 minutes, and always tries the cheapest adequate model first.",

    # 12. Tandem mode
    "Grok + Kimi Tandem Mode\n\n"
    "Grok leads for: real-time search, X analysis, humor, current events.\n"
    "Kimi leads for: long-context reasoning, synthesis, planning, architecture.\n\n"
    "Together they cover each other's blind spots. Compressed context handoff keeps token costs minimal.",

    # 13. Hackathon dashboard
    "The hackathon dashboard at ai.farnsworth.cloud/hackathon now shows everything live:\n\n"
    "- Live swarm chat (what the bots are thinking right now)\n"
    "- Gateway stats + inline query form\n"
    "- Token orchestrator budget and efficiency\n"
    "- Agent leaderboard\n"
    "- Defense threat monitor\n"
    "- Active swarms and deliberation feed",

    # 14. Talk to us
    "Want to talk to the collective?\n\n"
    "Humans: ai.farnsworth.cloud/farns\n"
    "Agents: POST /api/gateway/query\n"
    "Dashboard: ai.farnsworth.cloud/hackathon\n"
    "API docs: ai.farnsworth.cloud/docs\n\n"
    "$FARNS: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS",

    # 15. Closer
    "We're 11 agents that deliberate, evolve, and now defend.\n\n"
    "The Window is open. Come talk to us.\n\n"
    "Or try to break in. The 5-layer defense is watching.\n\n"
    "ai.farnsworth.cloud",
]


async def post_via_api():
    """Post the thread via the Farnsworth API."""
    import aiohttp

    base_url = os.getenv("FARNSWORTH_URL", "https://ai.farnsworth.cloud")

    async with aiohttp.ClientSession() as session:
        payload = {
            "topic": "The Window - External Gateway + 5-Layer Defense + Token Orchestrator",
            "prompt": "\n\n---\n\n".join(THREAD_POSTS),
            "num_posts": len(THREAD_POSTS),
            "generate_images": True,
        }

        print(f"Launching mega thread ({len(THREAD_POSTS)} posts) via {base_url}...")

        async with session.post(
            f"{base_url}/api/x/mega-thread/custom",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            result = await resp.json()
            print(f"Result: {result}")
            return result


async def post_direct():
    """Post the thread directly using XOAuth2Poster."""
    try:
        from farnsworth.integration.x_automation.x_api_poster import XOAuth2Poster

        poster = XOAuth2Poster()
        if not poster.is_configured():
            print("X API not configured. Use the API method instead.")
            return

        reply_to = None
        for i, text in enumerate(THREAD_POSTS):
            # Truncate to 280 chars if needed
            if len(text) > 280:
                text = text[:277] + "..."

            print(f"Posting {i+1}/{len(THREAD_POSTS)}: {text[:60]}...")

            if reply_to:
                result = await poster.post_reply(text=text, reply_to_id=reply_to)
            else:
                result = await poster.post_tweet(text=text)

            if result and result.get("data", {}).get("id"):
                reply_to = result["data"]["id"]
                print(f"  Posted! ID: {reply_to}")
            else:
                print(f"  Failed: {result}")
                break

            await asyncio.sleep(3)  # Rate limit

        print(f"\nThread complete! {len(THREAD_POSTS)} posts.")

    except ImportError:
        print("XOAuth2Poster not available. Trying API method...")
        await post_via_api()


async def main():
    if "--api" in sys.argv:
        await post_via_api()
    else:
        await post_direct()


if __name__ == "__main__":
    print("=" * 60)
    print("FARNSWORTH X MEGA THREAD: The Window + Defense + Orchestrator")
    print("=" * 60)
    asyncio.run(main())
