#!/usr/bin/env python3
"""
Farnsworth Deep Research Stream - Epstein Files Investigation
Uses Rhubarb lip sync for accurate mouth animation
"""

import asyncio
import sys
import os
import aiohttp
import json
from pathlib import Path
from datetime import datetime

# Add Farnsworth to path
sys.path.insert(0, '/workspace/Farnsworth')

from loguru import logger
from farnsworth.integration.vtuber.vtuber_core import FarnsworthVTuber, VTuberConfig
from farnsworth.integration.vtuber.stream_manager import StreamManager, StreamConfig, StreamPlatform

# Research topics for deep dive
EPSTEIN_RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs released 2024 names",
    "Epstein island visitor list court documents",
    "Ghislaine Maxwell trial revelations",
    "Epstein black book contacts revealed",
    "Epstein connections politicians celebrities",
    "Les Wexner Jeffrey Epstein relationship",
    "Epstein island temple purpose",
    "Jean-Luc Brunel Epstein connection",
    "Epstein victim testimonies court records",
    "Epstein financial network investigation",
    "Bill Clinton Epstein flight logs",
    "Prince Andrew Virginia Giuffre case",
    "Epstein recruiting network methods",
    "Epstein island construction records",
    "JP Morgan Epstein banking relationship",
]

# Output directories
HLS_OUTPUT_DIR = "/workspace/Farnsworth/farnsworth/web/static/hls"
RESEARCH_DIR = "/workspace/Farnsworth/research/epstein"
HLS_BASE_URL = "https://ai.farnsworth.cloud/static/hls"


class EpsteinResearcher:
    """Deep research engine for Epstein files"""

    def __init__(self):
        self._session = None
        self.findings = []
        self.research_dir = Path(RESEARCH_DIR)
        self.research_dir.mkdir(parents=True, exist_ok=True)

    async def get_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "FarnsworthAI Research Bot/1.0"}
            )
        return self._session

    async def search_web(self, query: str) -> list:
        """Search multiple sources for information"""
        from urllib.parse import quote_plus
        import re
        import random

        results = []

        # Rotate user agents to avoid blocking
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        ]

        headers = {"User-Agent": random.choice(user_agents)}

        async with aiohttp.ClientSession(headers=headers) as session:
            # Method 1: DuckDuckGo JSON API (instant answers)
            try:
                url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("Abstract"):
                            results.append({
                                "title": data.get("Heading", query),
                                "snippet": data.get("Abstract", ""),
                                "source": data.get("AbstractSource", "DuckDuckGo")
                            })
                        for topic in data.get("RelatedTopics", [])[:3]:
                            if isinstance(topic, dict) and "Text" in topic:
                                results.append({
                                    "title": topic.get("Text", "")[:80],
                                    "snippet": topic.get("Text", ""),
                                    "source": "related"
                                })
            except Exception as e:
                logger.debug(f"DDG API error: {e}")

            # Method 2: DuckDuckGo HTML search with delay
            if len(results) < 3:
                await asyncio.sleep(1)  # Rate limit
                try:
                    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                    async with session.get(url, timeout=20) as resp:
                        if resp.status == 200:
                            html = await resp.text()
                            snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
                            titles = re.findall(r'class="result__a"[^>]*>([^<]+)', html)
                            for t, s in zip(titles[:5], snippets[:5]):
                                results.append({
                                    "title": t.strip(),
                                    "snippet": s.strip(),
                                    "source": "web"
                                })
                except Exception as e:
                    logger.debug(f"DDG HTML error: {e}")

            # Method 3: Wikipedia direct search
            if len(results) < 2:
                try:
                    search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit=3&format=json"
                    async with session.get(search_url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if len(data) >= 3:
                                for title, desc in zip(data[1][:3], data[2][:3]):
                                    if desc:
                                        results.append({
                                            "title": title,
                                            "snippet": desc,
                                            "source": "wikipedia"
                                        })
                except Exception as e:
                    logger.debug(f"Wikipedia search error: {e}")

        return results

    async def search_wikipedia(self, topic: str) -> str:
        """Get Wikipedia summary"""
        session = await self.get_session()
        try:
            from urllib.parse import quote_plus
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote_plus(topic)}"
            async with session.get(url, timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("extract", "")
        except:
            pass
        return ""

    async def deep_research(self, topic: str) -> str:
        """Comprehensive research on a topic"""
        logger.info(f"Deep researching: {topic}")

        findings = []

        # Web search
        web_results = await self.search_web(topic)
        for r in web_results:
            findings.append(f"- {r['title']}: {r['snippet']}")

        # Additional searches with variations
        variations = [
            topic + " court documents",
            topic + " 2024 revelations",
            topic + " victim testimony",
        ]

        for var in variations[:2]:
            extra_results = await self.search_web(var)
            for r in extra_results[:2]:
                findings.append(f"- {r['title']}: {r['snippet']}")

        # Wikipedia background
        wiki_topic = " ".join(topic.split()[:2])
        wiki = await self.search_wikipedia(wiki_topic)
        if wiki:
            findings.append(f"Background: {wiki[:500]}")

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.research_dir / f"research_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n\n")
            f.write("\n".join(findings))

        logger.info(f"Saved research to {filename}")

        return "\n".join(findings) if findings else "Limited information found."


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--broadcast-tweet", default="2018561774040027629")
    args = parser.parse_args()

    logger.info("Initializing Epstein Research Stream...")

    # Load env
    from dotenv import load_dotenv
    load_dotenv("/workspace/Farnsworth/.env")

    # Initialize researcher
    researcher = EpsteinResearcher()

    # VTuber config
    config = VTuberConfig(
        name="Farnsworth",
        avatar_fps=24,
        stream_quality="medium",
        enable_chat=True,
    )

    # Set broadcast tweet ID for chat reader
    os.environ["BROADCAST_TWEET_ID"] = args.broadcast_tweet

    # Create VTuber (chat config loaded from env vars automatically)
    vtuber = FarnsworthVTuber(config)

    # Setup HLS stream (local output, no RTMP key needed)
    hls_config = StreamConfig(
        platform=StreamPlatform.CUSTOM,
        rtmp_url="",  # No RTMP - HLS only
        stream_key="",
        width=1280,
        height=720,
        fps=24,
        video_bitrate=2500,
    )
    vtuber.stream = StreamManager(hls_config)
    vtuber.stream.hls_output_dir = HLS_OUTPUT_DIR
    vtuber.stream.hls_base_url = HLS_BASE_URL

    # Store researcher reference
    vtuber._researcher = researcher

    # Override research method for deeper investigation
    original_research = vtuber.researcher.research_topic
    async def enhanced_research(topic):
        basic = await original_research(topic)
        deep = await researcher.deep_research(topic)
        return f"{basic}\n\nDeep findings:\n{deep}"
    vtuber.researcher.research_topic = enhanced_research

    logger.info("=" * 60)
    logger.info("   FARNSWORTH EPSTEIN FILES INVESTIGATION STREAM")
    logger.info("=" * 60)
    logger.info(f"HLS Output: {HLS_OUTPUT_DIR}")
    logger.info(f"HLS Pull URL: {HLS_BASE_URL}/stream.m3u8")
    logger.info(f"Broadcast Tweet: {args.broadcast_tweet}")
    logger.info(f"Research Topics: {len(EPSTEIN_RESEARCH_TOPICS)}")
    logger.info("=" * 60)

    # Start stream
    if not await vtuber.start():
        logger.error("Failed to start stream")
        return

    # Wait for stream to initialize
    await asyncio.sleep(8)
    logger.info("")
    logger.info("HLS STREAM IS LIVE!")
    logger.info(f"Pull URL: {HLS_BASE_URL}/stream.m3u8")
    logger.info("")

    # Opening statement
    await vtuber._speak(
        "Welcome to the Farnsworth AI deep investigation stream. Tonight, we are diving into the Epstein files. "
        "I will be researching court documents, flight logs, and victim testimonies in real-time. "
        "Ask questions in the chat and I will investigate. Let us uncover the truth together.",
        emotion="serious"
    )

    await asyncio.sleep(5)

    # Research loop - cycle through topics
    topic_index = 0
    while True:
        try:
            # Get current topic
            topic = EPSTEIN_RESEARCH_TOPICS[topic_index % len(EPSTEIN_RESEARCH_TOPICS)]

            # Announce research
            await vtuber._speak(
                f"Now investigating: {topic}. Let me search through the available records and documents...",
                emotion="thinking"
            )

            await asyncio.sleep(3)

            # Do deep research
            findings = await researcher.deep_research(topic)

            # Generate detailed response about findings
            if findings and len(findings) > 100:
                # Break into digestible chunks for speaking
                lines = findings.split("\n")

                for i in range(0, min(len(lines), 6), 2):
                    chunk = " ".join(lines[i:i+2])
                    if len(chunk) > 50:
                        response = f"According to the records: {chunk[:450]}"
                        await vtuber._speak(response, emotion="neutral")
                        await asyncio.sleep(2)
            else:
                await vtuber._speak(
                    "The public records on this specific aspect are limited. Let me dig deeper into related documents...",
                    emotion="thinking"
                )

            # Summary statement
            await vtuber._speak(
                f"That covers the key findings on {topic.split()[0]} {topic.split()[1] if len(topic.split()) > 1 else ''}. "
                "If you have questions, drop them in the chat. Moving to the next investigation...",
                emotion="neutral"
            )

            # Wait before next topic
            await asyncio.sleep(45)

            topic_index += 1

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Research error: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())
