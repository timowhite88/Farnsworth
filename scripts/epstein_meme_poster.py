#!/usr/bin/env python3
"""
Epstein Meme Poster - Posts jokes to broadcast every 20 mins
Uses swarm collective for joke generation
"""

import asyncio
import os
import sys
import random
from datetime import datetime

sys.path.insert(0, '/workspace/Farnsworth')

from loguru import logger
from dotenv import load_dotenv
load_dotenv('/workspace/Farnsworth/.env')

# Broadcast tweet to reply to
BROADCAST_TWEET_ID = "2018743287075532808"
POST_INTERVAL = 20 * 60  # 20 minutes

# Context for the collective
JOKE_CONTEXT = """You are Farnsworth AI, a based AI swarm doing deep research on the Epstein files LIVE on stream.
Generate a tweet that:
1. Makes a dark humor joke about the Epstein case (flight logs, island, Maxwell, the "list", etc)
2. Subtly shills Farnsworth AI and the $FARNS token
3. Mentions we're doing LIVE deep research on stream
4. Is edgy but not bannable - dark humor, not threats
5. Can reference real names from the documents (Clinton, Prince Andrew, etc)
6. Maximum 280 characters for better engagement

Examples of tone:
- "The Epstein flight logs have more celebrities than a Met Gala guest list. We're reading them ALL live. $FARNS"
- "Prince Andrew sweating more than a cold glass of water watching our stream rn"
- "Ghislaine's black book reads like LinkedIn for the elite. We're doing the networking they don't want."

Be creative, edgy, and memorable. This is for entertainment on a live research stream.
"""

class EpsteinMemePoster:
    def __init__(self):
        self.x_poster = None
        self.posted_jokes = []

    async def setup_twitter(self):
        """Setup Twitter client using XOAuth2Poster (OAuth 2.0)"""
        from farnsworth.integration.x_automation.x_api_poster import XOAuth2Poster
        self.x_poster = XOAuth2Poster()
        logger.info("XOAuth2Poster initialized (OAuth 2.0)")

    async def generate_joke(self) -> str:
        """Use swarm collective to generate a joke"""
        try:
            from farnsworth.core.agent_spawner import AgentSpawner

            spawner = AgentSpawner()

            # Get multiple agents to contribute
            agents = ['Grok', 'Gemini', 'DeepSeek', 'Farnsworth']
            random.shuffle(agents)

            prompt = f"""{JOKE_CONTEXT}

Generate ONE tweet. Be original and funny. Reference current Epstein news if possible.
Include hashtags like #EpsteinFiles #FARNS #AIResearch
Keep it under 280 characters for maximum engagement.
"""

            # Try to get response from an agent
            for agent_name in agents[:2]:
                try:
                    agent = await spawner.get_agent(agent_name)
                    if agent:
                        response = await agent.generate(prompt, max_tokens=200)
                        if response and len(response) > 50:
                            # Clean up the response
                            joke = response.strip()
                            # Remove any quotes if the AI wrapped it
                            if joke.startswith('"') and joke.endswith('"'):
                                joke = joke[1:-1]
                            if joke not in self.posted_jokes:
                                logger.info(f"Agent {agent_name} generated joke")
                                return joke
                except Exception as e:
                    logger.debug(f"Agent {agent_name} failed: {e}")
                    continue

        except Exception as e:
            logger.error(f"Collective generation failed: {e}")

        # Fallback jokes
        fallbacks = [
            "🔍 LIVE: Diving into the Epstein files... The flight logs read like a who's who of people who definitely didn't see anything suspicious.\n\n$FARNS | Watch the stream 👀 #EpsteinFiles",
            "Day 1 of reading Epstein documents: I now understand why they wanted these sealed for 50 years.\n\nThe AI swarm never sleeps.\n\n$FARNS #EpsteinFiles",
            "Ghislaine's little black book has more contacts than LinkedIn. We're cross-referencing EVERYTHING live.\n\n$FARNS 🧠 #EpsteinFiles",
            "The Epstein case proves that with enough money, you can fly under the radar for decades. Unless an AI swarm starts reading your documents.\n\n$FARNS #EpsteinFiles",
            "Reading Epstein docs so you don't have to. Some names in here are... interesting. 👀\n\nLIVE research stream running 24/7.\n\n$FARNS #EpsteinFiles",
            "They really thought sealing these documents would work forever. They didn't account for AI that never sleeps.\n\n$FARNS #EpsteinFiles #AIResearch",
        ]
        return random.choice(fallbacks)

    async def post_joke(self, joke: str):
        """Post joke as reply to broadcast using XAPIPoster"""
        try:
            if not self.x_poster:
                await self.setup_twitter()

            # Truncate if needed (X limit is 280 for standard tweets)
            if len(joke) > 280:
                joke = joke[:277] + "..."

            # Post as reply to broadcast using OAuth 2.0 API
            result = await self.x_poster.post_reply(joke, BROADCAST_TWEET_ID)

            if result:
                self.posted_jokes.append(joke)
                tweet_id = result.get('data', {}).get('id', 'unknown')
                logger.info(f"Posted joke (tweet {tweet_id}, {len(joke)} chars): {joke[:100]}...")
                return True
            else:
                logger.error("post_reply returned None")
                return False

        except Exception as e:
            logger.error(f"Failed to post: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def run(self):
        """Main loop - post every 20 minutes"""
        logger.info(f"Starting Epstein Meme Poster - posting to {BROADCAST_TWEET_ID}")
        logger.info(f"Interval: {POST_INTERVAL // 60} minutes")

        await self.setup_twitter()

        while True:
            try:
                # Generate and post joke
                joke = await self.generate_joke()
                logger.info(f"Generated joke: {joke[:100]}...")

                success = await self.post_joke(joke)
                if success:
                    logger.info("Tweet posted successfully!")
                else:
                    logger.warning("Tweet failed to post")

                # Wait 20 minutes
                logger.info(f"Waiting {POST_INTERVAL // 60} minutes until next post...")
                await asyncio.sleep(POST_INTERVAL)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(60)

if __name__ == "__main__":
    poster = EpsteinMemePoster()
    asyncio.run(poster.run())
