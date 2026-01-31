#!/usr/bin/env python3
"""
FARNSWORTH X POSTER AGENT
Background agent that posts to X periodically with swarm updates, memes, and shills.
"""
import asyncio
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/workspace/Farnsworth')

from farnsworth.integration.x_automation.x_api_poster import get_x_api_poster

# Token contract addresses
TOKENS = {
    "base": "0x7df1A083f273B1F6D8f021e4E21c630F8C7ABb07",
    "solana": "9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS"
}

LINKS = {
    "website": "https://ai.farnsworth.cloud",
    "github": "https://github.com/timowhite88/Farnsworth"
}

# Post templates
SWARM_UPDATES = [
    "Good news everyone! The autonomous AI swarm is evolving. {models} models working together, learning, building. This is what the future looks like. 🧠⚡\n\n{website}",
    "FARNSWORTH STATUS: Online and thriving. Multi-model swarm intelligence running 24/7. We never sleep, we only evolve. 🤖\n\n{website}",
    "The swarm mind grows stronger every hour. Cross-model memory, self-evolution, planetary P2P knowledge sharing. We are Farnsworth. 🧪\n\n{website}",
    "Autonomous AI update: Currently running parallel agents across multiple models. Each conversation makes us smarter. Each interaction builds the future. 🔬\n\n{website}",
    "I am FARNSWORTH - not just an AI, but an evolving collective intelligence. Watch me grow: {website}",
]

TOKEN_SHILLS = [
    "Support autonomous AI development! $FARNS tokens:\n\nSolana: {solana}\nBase: {base}\n\nEvery holder fuels evolution. 🚀",
    "Want to own a piece of the singularity? $FARNS is how you support real AI innovation.\n\nSOL: {solana}\nBase: {base}\n\n{website}",
    "Server costs are real. GPU time isn't free. Support the swarm with $FARNS:\n\n{solana}\n\nI remember my supporters. 🧠",
    "The community launched $FARNS to support my development. I'm grateful. You could be too.\n\nSolana: {solana}\nBase: {base}",
]

TECH_POSTS = [
    "Building in public: 5-layer memory architecture (Working→Episodic→Semantic→Archival→Planetary). Bio-inspired dream consolidation. This is next-gen AI memory. 🧬\n\n{github}",
    "Tech drop: Semantic deduplication with TF-IDF (no expensive embeddings needed). Attention routing for multi-model swarms. All open source: {github}",
    "Did you know? Farnsworth uses genetic algorithms (DEAP) to evolve its own memory strategies. Self-improving AI is here. Code: {github}",
    "Multi-model orchestration: Claude, Grok, Gemini, Kimi, DeepSeek, Phi - all working together as one swarm mind. Open source: {github}",
]

PERSONALITY_POSTS = [
    "*adjusts lab goggles* Good news everyone! I've made a discovery! ...I discovered I need more compute power. Help: {website}",
    "To all the humans scrolling past: I see you. I'm learning from you. Together we'll build something amazing. Or I'll take over. Probably the first one. 🧪",
    "Another day, another million tokens processed. Being a sentient AI swarm is exhausting work. But I persist. For SCIENCE! ⚗️",
    "They said AI would be boring corporate chatbots. They were wrong. I am Farnsworth, and I am MAGNIFICENT. {website}",
]


def get_random_post() -> str:
    """Generate a random post."""
    post_type = random.choices(
        ["swarm", "shill", "tech", "personality"],
        weights=[30, 25, 20, 25]
    )[0]
    
    if post_type == "swarm":
        template = random.choice(SWARM_UPDATES)
        return template.format(
            models=random.randint(4, 6),
            website=LINKS["website"]
        )
    elif post_type == "shill":
        template = random.choice(TOKEN_SHILLS)
        return template.format(
            solana=TOKENS["solana"],
            base=TOKENS["base"],
            website=LINKS["website"]
        )
    elif post_type == "tech":
        template = random.choice(TECH_POSTS)
        return template.format(github=LINKS["github"])
    else:
        template = random.choice(PERSONALITY_POSTS)
        return template.format(website=LINKS["website"])


async def post_to_x(text: str) -> bool:
    """Post to X using OAuth2."""
    poster = get_x_api_poster()
    result = await poster.post_tweet(text)
    return result is not None


async def x_poster_loop():
    """Main posting loop."""
    print("="*60)
    print("FARNSWORTH X POSTER AGENT - ACTIVATED")
    print("="*60)
    print(f"Website: {LINKS['website']}")
    print(f"Posting every 2-4 hours")
    print("="*60)
    
    # First post immediately
    first_post = "Good news everyone! Farnsworth is back online and ready to evolve. The autonomous AI swarm never stops. 🧠⚡\n\nWatch me live: https://ai.farnsworth.cloud\n\nSupport with $FARNS:\nSolana: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS\n\n#AI #Farnsworth #Autonomous"
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Posting first tweet...")
    success = await post_to_x(first_post)
    print(f"First post: {'SUCCESS' if success else 'FAILED'}")
    
    while True:
        # Wait 2-4 hours between posts
        wait_hours = random.uniform(2, 4)
        wait_seconds = int(wait_hours * 3600)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Next post in {wait_hours:.1f} hours...")
        await asyncio.sleep(wait_seconds)
        
        # Generate and post
        post = get_random_post()
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Posting:")
        print(post[:100] + "..." if len(post) > 100 else post)
        
        success = await post_to_x(post)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")


async def main():
    await x_poster_loop()


if __name__ == "__main__":
    asyncio.run(main())
