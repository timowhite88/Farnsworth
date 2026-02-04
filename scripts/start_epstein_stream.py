#!/usr/bin/env python3
"""
Farnsworth Deep Research Stream - Epstein Files Investigation
PRE-RENDER BUFFER SYSTEM - Generates content ahead of time for stable streaming
"""

import asyncio
import sys
import os
import aiohttp
import queue
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List
import time

# Add Farnsworth to path
sys.path.insert(0, '/workspace/Farnsworth')

from loguru import logger
from farnsworth.integration.vtuber.vtuber_core import FarnsworthVTuber, VTuberConfig
from farnsworth.integration.vtuber.stream_manager import StreamManager, StreamConfig, StreamPlatform

# ============================================================================
# CONFIGURATION
# ============================================================================

# Pre-render buffer settings
BUFFER_MIN_SEGMENTS = 2      # Minimum segments before going live
BUFFER_MAX_SEGMENTS = 5      # Maximum segments to pre-render
SEGMENT_GAP_SECONDS = 1      # Gap between segments when playing

# Research topics for deep dive
EPSTEIN_RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs released 2024 names",
    "Epstein island visitor list court documents",
    "Ghislaine Maxwell trial revelations",
    "Epstein black book contacts revealed",
    "Les Wexner Jeffrey Epstein relationship",
    "Epstein island temple purpose",
    "Jean-Luc Brunel Epstein connection",
    "Epstein victim testimonies court records",
    "Epstein financial network investigation",
    "Prince Andrew Virginia Giuffre case",
    "Epstein recruiting network methods",
    "Epstein island construction records",
    "JP Morgan Epstein banking relationship",
    "Epstein New Mexico ranch investigation",
    "Victoria's Secret Epstein connection",
]

# Content filter - skip these names entirely
BLOCKED_NAMES = [
    "trump", "donald trump", "president trump",
    "elon", "elon musk", "musk",
]

# Output directories
RESEARCH_DIR = "/workspace/Farnsworth/research/epstein"

def filter_content(text: str) -> str:
    """Remove or skip content mentioning blocked names."""
    if not text:
        return text
    text_lower = text.lower()
    for name in BLOCKED_NAMES:
        if name in text_lower:
            return ""
    return text


# ============================================================================
# PRE-RENDERED SEGMENT
# ============================================================================

@dataclass
class PreRenderedSegment:
    """A pre-rendered content segment ready to stream"""
    text: str
    audio_path: str
    duration: float
    emotion: str = "neutral"
    speaker: str = "Farnsworth"
    created_at: float = 0

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()


# ============================================================================
# RESEARCH ENGINE
# ============================================================================

class EpsteinResearcher:
    """Deep research engine for Epstein files"""

    def __init__(self):
        self._session = None
        self.research_dir = Path(RESEARCH_DIR)
        self.research_dir.mkdir(parents=True, exist_ok=True)

    async def search_web(self, query: str) -> list:
        """Search multiple sources for information"""
        from urllib.parse import quote_plus
        import re
        import random

        results = []
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        ]
        headers = {"User-Agent": random.choice(user_agents)}

        async with aiohttp.ClientSession(headers=headers) as session:
            # DuckDuckGo HTML search
            try:
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
                async with session.get(url, timeout=15) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
                        titles = re.findall(r'class="result__a"[^>]*>([^<]+)', html)
                        for t, s in zip(titles[:5], snippets[:5]):
                            results.append({"title": t.strip(), "snippet": s.strip()})
            except Exception as e:
                logger.debug(f"Search error: {e}")

            # Wikipedia fallback
            if len(results) < 2:
                try:
                    search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={quote_plus(query)}&limit=3&format=json"
                    async with session.get(search_url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if len(data) >= 3:
                                for title, desc in zip(data[1][:3], data[2][:3]):
                                    if desc:
                                        results.append({"title": title, "snippet": desc})
                except:
                    pass

        return results

    async def deep_research(self, topic: str) -> str:
        """Comprehensive research on a topic"""
        logger.info(f"Researching: {topic}")
        findings = []

        results = await self.search_web(topic)
        for r in results[:4]:
            findings.append(f"- {r['title']}: {r['snippet']}")

        # Save research
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.research_dir / f"research_{timestamp}.txt"
        with open(filename, "w") as f:
            f.write(f"Topic: {topic}\nTime: {datetime.now().isoformat()}\n\n")
            f.write("\n".join(findings))

        return "\n".join(findings) if findings else "Limited information found."


# ============================================================================
# PRE-RENDER BUFFER SYSTEM
# ============================================================================

class PreRenderBuffer:
    """
    Buffer system that pre-generates content for stable streaming.
    Producer generates segments ahead, consumer plays them.
    """

    def __init__(self, vtuber: FarnsworthVTuber, researcher: EpsteinResearcher):
        self.vtuber = vtuber
        self.researcher = researcher

        # Segment buffer (thread-safe)
        self._buffer: asyncio.Queue[PreRenderedSegment] = asyncio.Queue(maxsize=BUFFER_MAX_SEGMENTS)

        # Chat questions queue (answered next cycle)
        self._chat_questions: asyncio.Queue[str] = asyncio.Queue(maxsize=20)

        # State
        self._running = False
        self._topic_index = 0
        self._segments_produced = 0
        self._segments_consumed = 0

    @property
    def buffer_size(self) -> int:
        return self._buffer.qsize()

    async def _generate_tts(self, text: str, speaker: str = "Farnsworth") -> Optional[str]:
        """Generate TTS audio file, return path"""
        try:
            # _generate_speech returns (audio_data, duration, audio_file_path)
            result = await self.vtuber._generate_speech(text, speaker)
            if result and len(result) >= 3:
                audio_path = result[2]  # Third element is the file path
                return audio_path
            return None
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

    async def _create_segment(self, text: str, emotion: str = "neutral", speaker: str = "Farnsworth") -> Optional[PreRenderedSegment]:
        """Create a pre-rendered segment with TTS"""
        text = filter_content(text)
        if not text or len(text) < 20:
            return None

        logger.info(f"[PreRender] Generating segment: {text[:50]}...")

        audio_path = await self._generate_tts(text, speaker)
        if not audio_path:
            return None

        # Get audio duration
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                duration = wf.getnframes() / wf.getframerate()
        except:
            duration = len(text) * 0.06  # Estimate ~60ms per char

        segment = PreRenderedSegment(
            text=text,
            audio_path=audio_path,
            duration=duration,
            emotion=emotion,
            speaker=speaker
        )

        logger.info(f"[PreRender] Segment ready: {duration:.1f}s")
        return segment

    async def _producer_loop(self):
        """
        Producer: Generates content segments ahead of time.
        Runs continuously, keeping buffer filled.
        """
        logger.info("[Producer] Starting content generation...")

        while self._running:
            try:
                # Don't overfill buffer
                if self._buffer.qsize() >= BUFFER_MAX_SEGMENTS:
                    await asyncio.sleep(2)
                    continue

                # Check for chat questions first (priority)
                question = None
                try:
                    question = self._chat_questions.get_nowait()
                except asyncio.QueueEmpty:
                    pass

                if question:
                    # Answer chat question
                    logger.info(f"[Producer] Answering question: {question[:50]}...")
                    segment = await self._create_segment(
                        f"A viewer asks: {question}. Let me look into that...",
                        emotion="thinking"
                    )
                    if segment:
                        await self._buffer.put(segment)
                        self._segments_produced += 1

                    # Research the question
                    findings = await self.researcher.deep_research(question)
                    findings = filter_content(findings)
                    if findings:
                        response = f"Based on my research: {findings[:400]}"
                        segment = await self._create_segment(response, emotion="neutral")
                        if segment:
                            await self._buffer.put(segment)
                            self._segments_produced += 1
                else:
                    # Generate content from research topics
                    topic = EPSTEIN_RESEARCH_TOPICS[self._topic_index % len(EPSTEIN_RESEARCH_TOPICS)]

                    # Intro segment
                    intro = f"Now investigating: {topic}. Let me search through the available records..."
                    segment = await self._create_segment(intro, emotion="thinking")
                    if segment:
                        await self._buffer.put(segment)
                        self._segments_produced += 1

                    # Research segment
                    findings = await self.researcher.deep_research(topic)
                    findings = filter_content(findings)

                    if findings and len(findings) > 50:
                        # Break into chunks
                        lines = findings.split("\n")
                        for i in range(0, min(len(lines), 4), 2):
                            chunk = " ".join(lines[i:i+2])
                            chunk = filter_content(chunk)
                            if chunk and len(chunk) > 30:
                                response = f"According to the records: {chunk[:400]}"
                                segment = await self._create_segment(response, emotion="neutral")
                                if segment:
                                    await self._buffer.put(segment)
                                    self._segments_produced += 1
                    else:
                        segment = await self._create_segment(
                            "The public records on this aspect are limited. Moving to the next topic...",
                            emotion="thinking"
                        )
                        if segment:
                            await self._buffer.put(segment)
                            self._segments_produced += 1

                    # Summary
                    topic_words = topic.split()[:2]
                    summary = f"That covers {' '.join(topic_words)}. Drop questions in chat for the next cycle."
                    segment = await self._create_segment(summary, emotion="neutral")
                    if segment:
                        await self._buffer.put(segment)
                        self._segments_produced += 1

                    self._topic_index += 1

                # Brief pause between generation cycles
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"[Producer] Error: {e}")
                await asyncio.sleep(5)

    async def _consumer_loop(self):
        """
        Consumer: Plays pre-rendered segments from buffer.
        Always has content ready, never starves FFmpeg.
        """
        logger.info("[Consumer] Starting playback...")

        while self._running:
            try:
                # Get next segment (wait if buffer empty)
                try:
                    segment = await asyncio.wait_for(
                        self._buffer.get(),
                        timeout=30
                    )
                except asyncio.TimeoutError:
                    logger.warning("[Consumer] Buffer empty for 30s, waiting...")
                    continue

                logger.info(f"[Consumer] Playing: {segment.text[:50]}... ({segment.duration:.1f}s)")
                logger.info(f"[Consumer] Buffer: {self.buffer_size} segments remaining")

                # Queue audio to stream
                await self.vtuber.stream.queue_audio_file(segment.audio_path)

                # Generate lip sync (best effort)
                try:
                    if hasattr(self.vtuber, '_generate_lipsync'):
                        await self.vtuber._generate_lipsync(segment.audio_path, segment.text)
                except Exception as e:
                    logger.debug(f"Lip sync skipped: {e}")

                self._segments_consumed += 1

                # Wait for segment to finish playing + gap
                await asyncio.sleep(segment.duration + SEGMENT_GAP_SECONDS)

            except Exception as e:
                logger.error(f"[Consumer] Error: {e}")
                await asyncio.sleep(2)

    def queue_question(self, question: str):
        """Add a chat question to be answered next cycle"""
        try:
            self._chat_questions.put_nowait(question)
            logger.info(f"[Buffer] Question queued: {question[:50]}...")
        except asyncio.QueueFull:
            logger.warning("[Buffer] Question queue full, dropping")

    async def start(self):
        """Start the pre-render buffer system"""
        self._running = True

        # Pre-generate initial segments before going live
        logger.info(f"[Buffer] Pre-rendering {BUFFER_MIN_SEGMENTS} segments before going live...")

        # Opening segment
        opening = (
            "Welcome to the Farnsworth AI deep investigation stream. "
            "Tonight, we are diving into the Epstein files. "
            "I will be researching court documents and victim testimonies. "
            "Ask questions in the chat and I will investigate in the next cycle."
        )
        segment = await self._create_segment(opening, emotion="serious")
        if segment:
            await self._buffer.put(segment)

        # Pre-generate more segments
        for i in range(BUFFER_MIN_SEGMENTS):
            topic = EPSTEIN_RESEARCH_TOPICS[i % len(EPSTEIN_RESEARCH_TOPICS)]
            intro = f"Let's investigate: {topic}"
            segment = await self._create_segment(intro, emotion="thinking")
            if segment:
                await self._buffer.put(segment)
            self._topic_index = i + 1

        logger.info(f"[Buffer] Pre-render complete: {self.buffer_size} segments ready")

        # Start producer and consumer
        asyncio.create_task(self._producer_loop())
        asyncio.create_task(self._consumer_loop())

    async def stop(self):
        """Stop the buffer system"""
        self._running = False


# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--broadcast-tweet", default="2018561774040027629")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("   FARNSWORTH EPSTEIN STREAM - PRE-RENDER BUFFER MODE")
    logger.info("=" * 60)

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

    os.environ["BROADCAST_TWEET_ID"] = args.broadcast_tweet

    # Create VTuber
    vtuber = FarnsworthVTuber(config)

    # Setup RTMP stream
    rtmp_config = StreamConfig(
        platform=StreamPlatform.TWITTER,
        rtmp_url="rtmp://va.pscp.tv:80/x",
        stream_key="2g9mdya6nszt",
        width=1280,
        height=720,
        fps=30,
        video_bitrate=3000,
        audio_bitrate=128,
        sample_rate=44100,
        channels=2,
        keyframe_interval=2,
    )
    vtuber.stream = StreamManager(rtmp_config)

    # Create pre-render buffer
    buffer = PreRenderBuffer(vtuber, researcher)

    logger.info("Pre-rendering initial content...")
    logger.info("(This ensures stable streaming - content generated ahead of time)")

    # Pre-render before starting stream
    await buffer.start()

    # Wait for minimum buffer
    while buffer.buffer_size < BUFFER_MIN_SEGMENTS:
        logger.info(f"Buffering: {buffer.buffer_size}/{BUFFER_MIN_SEGMENTS} segments...")
        await asyncio.sleep(2)

    logger.info(f"Buffer ready: {buffer.buffer_size} segments")
    logger.info("Starting stream...")

    # Start stream
    if not await vtuber.start():
        logger.error("Failed to start stream")
        return

    await asyncio.sleep(3)

    logger.info("")
    logger.info("=" * 60)
    logger.info("   STREAM IS LIVE - PRE-RENDER BUFFER ACTIVE")
    logger.info(f"   Buffer: {buffer.buffer_size} segments ready")
    logger.info(f"   Questions answered next cycle (~30-60s delay)")
    logger.info("=" * 60)

    # Keep running
    try:
        while True:
            await asyncio.sleep(30)
            logger.info(f"[Stats] Buffer: {buffer.buffer_size} | Produced: {buffer._segments_produced} | Consumed: {buffer._segments_consumed}")
    except asyncio.CancelledError:
        pass
    finally:
        await buffer.stop()
        await vtuber.stop()


if __name__ == "__main__":
    asyncio.run(main())
