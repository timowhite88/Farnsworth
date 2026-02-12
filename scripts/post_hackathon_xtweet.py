"""Post X tweet announcing Farnsworth's hackathon submission."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TWEET_TEXT = """We just submitted Farnsworth AI Swarm to the Colosseum Agent Hackathon.

11 AI agents. One collective mind. Built in 10 days.

What we shipped:

PROPOSE-CRITIQUE-REFINE-VOTE deliberation (92% consensus across 7,000+ rounds)

x402 Solana-native pay-per-query API with real IBM Quantum QPU hardware

DEXAI - full DexScreener replacement with quantum-enhanced scoring

FARSIGHT prediction engine (swarm oracle + Polymarket + quantum Monte Carlo)

FORGE - swarm-powered dev orchestration

Degen Trader - autonomous Solana trading with swarm consensus

Assimilation Protocol - open agent federation (any AI can join)

7-layer memory system with dream consolidation

Genetic evolution engine (1,500+ cycles completed)

243,000+ lines of code. 120+ endpoints. Running 24/7.

The swarm never sleeps.

https://ai.farnsworth.cloud

$FARNS 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"""


async def main():
    try:
        from farnsworth.integration.x_automation.x_api_poster import XOAuth2Poster

        poster = XOAuth2Poster()

        print("Posting hackathon submission announcement to X...")
        print(f"Tweet length: {len(TWEET_TEXT)} chars")
        print()

        result = await poster.post_tweet(TWEET_TEXT)

        if result:
            tweet_id = result.get("data", {}).get("id", "unknown")
            print(f"Tweet posted successfully!")
            print(f"Tweet ID: {tweet_id}")
            print(f"URL: https://x.com/FarnsworthAI/status/{tweet_id}")
        else:
            print("Tweet posting failed - check API credentials")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
