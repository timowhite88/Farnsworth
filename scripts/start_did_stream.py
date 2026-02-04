#!/usr/bin/env python3
"""
Farnsworth D-ID Stream - Premium Avatar with Lip Sync
Uses ElevenLabs (voice) + D-ID (avatar) + FFmpeg (stream to Twitter)

Pipeline:
1. Generate text content (research, chat responses)
2. ElevenLabs: Text → Audio (Farnsworth voice)
3. D-ID: Audio → Video with lip sync (Farnsworth avatar)
4. FFmpeg: Video → RTMP stream to Twitter
"""

import asyncio
import aiohttp
import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

sys.path.insert(0, '/workspace/Farnsworth')

from loguru import logger
from dotenv import load_dotenv
load_dotenv('/workspace/Farnsworth/.env')

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Keys (from environment)
DID_API_KEY = os.getenv("DID_API_KEY")
DID_AVATAR_URL = os.getenv("DID_AVATAR_URL")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_FARNSWORTH", "dxvY1G6UilzEKgCy370m")

# Stream settings
RTMP_URL = "rtmp://va.pscp.tv:80/x"
STREAM_KEY = os.getenv("TWITTER_STREAM_KEY", "")

# Buffer settings
BUFFER_MIN_BEFORE_LIVE = 3  # Minimum videos before going live
BUFFER_TARGET = 5           # Target buffer size
PARALLEL_GENERATIONS = 3    # How many to generate in parallel

# Content settings
SEGMENT_MAX_CHARS = 300     # Max characters per segment (keeps videos short)

# Directories
CACHE_DIR = Path("/workspace/Farnsworth/cache/did_stream")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Research topics
RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs 2024 names revealed",
    "Epstein island visitor list court documents",
    "Ghislaine Maxwell trial key revelations",
    "Epstein black book contacts",
    "Les Wexner Jeffrey Epstein connection",
    "Prince Andrew Virginia Giuffre settlement",
    "JP Morgan Epstein banking relationship",
    "Epstein victim testimonies",
]

# Content filter
BLOCKED_NAMES = ["trump", "donald trump", "elon", "elon musk", "musk"]


# ============================================================================
# VIDEO SEGMENT
# ============================================================================

@dataclass
class VideoSegment:
    """A pre-rendered video segment ready to stream"""
    video_path: str
    video_url: str
    text: str
    duration: float
    created_at: float

    @property
    def age(self) -> float:
        return time.time() - self.created_at


# ============================================================================
# D-ID VIDEO GENERATOR
# ============================================================================

class DIDVideoGenerator:
    """Generates video segments using ElevenLabs + D-ID"""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._segment_counter = 0

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate_segment(self, text: str) -> Optional[VideoSegment]:
        """Generate a video segment from text"""
        await self._ensure_session()
        self._segment_counter += 1
        seg_id = self._segment_counter
        start_time = time.time()

        # Filter content
        text_lower = text.lower()
        for blocked in BLOCKED_NAMES:
            if blocked in text_lower:
                logger.warning(f"Segment {seg_id}: Blocked content, skipping")
                return None

        if len(text) < 20:
            logger.warning(f"Segment {seg_id}: Text too short, skipping")
            return None

        try:
            # Step 1: ElevenLabs TTS
            logger.info(f"Segment {seg_id}: Generating audio...")
            audio_data = await self._generate_audio(text)
            if not audio_data:
                return None

            # Step 2: Upload audio to D-ID
            logger.info(f"Segment {seg_id}: Uploading audio to D-ID...")
            audio_url = await self._upload_audio(audio_data, seg_id)
            if not audio_url:
                return None

            # Step 3: Create D-ID talk
            logger.info(f"Segment {seg_id}: Creating D-ID video...")
            talk_id = await self._create_talk(audio_url)
            if not talk_id:
                return None

            # Step 4: Wait for render
            logger.info(f"Segment {seg_id}: Waiting for D-ID render...")
            video_url = await self._wait_for_talk(talk_id)
            if not video_url:
                return None

            # Step 5: Download video
            video_path = CACHE_DIR / f"segment_{seg_id}_{int(time.time())}.mp4"
            await self._download_video(video_url, video_path)

            # Get video duration
            duration = await self._get_video_duration(video_path)

            total_time = time.time() - start_time
            logger.info(f"Segment {seg_id}: Complete in {total_time:.1f}s ({duration:.1f}s video)")

            return VideoSegment(
                video_path=str(video_path),
                video_url=video_url,
                text=text,
                duration=duration,
                created_at=time.time()
            )

        except Exception as e:
            logger.error(f"Segment {seg_id}: Failed - {e}")
            return None

    async def _generate_audio(self, text: str) -> Optional[bytes]:
        """Generate audio with ElevenLabs"""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }

        async with self._session.post(url, json=payload, headers=headers) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                logger.error(f"ElevenLabs failed: {resp.status}")
                return None

    async def _upload_audio(self, audio_data: bytes, seg_id: int) -> Optional[str]:
        """Upload audio to D-ID"""
        data = aiohttp.FormData()
        data.add_field("audio", audio_data, filename=f"seg{seg_id}.mp3", content_type="audio/mpeg")

        async with self._session.post(
            "https://api.d-id.com/audios",
            data=data,
            headers={"Authorization": f"Basic {DID_API_KEY}"}
        ) as resp:
            if resp.status == 201:
                result = await resp.json()
                return result.get("url")
            else:
                logger.error(f"D-ID audio upload failed: {resp.status}")
                return None

    async def _create_talk(self, audio_url: str) -> Optional[str]:
        """Create D-ID talk"""
        payload = {
            "source_url": DID_AVATAR_URL,
            "script": {"type": "audio", "audio_url": audio_url},
            "config": {"fluent": True, "pad_audio": 0.3},
            "driver_url": "bank://lively",
        }

        async with self._session.post(
            "https://api.d-id.com/talks",
            json=payload,
            headers={"Authorization": f"Basic {DID_API_KEY}", "Content-Type": "application/json"}
        ) as resp:
            if resp.status == 201:
                result = await resp.json()
                return result.get("id")
            else:
                error = await resp.text()
                logger.error(f"D-ID create talk failed: {resp.status} - {error}")
                return None

    async def _wait_for_talk(self, talk_id: str, timeout: float = 60) -> Optional[str]:
        """Wait for D-ID talk to complete"""
        start = time.time()
        while time.time() - start < timeout:
            async with self._session.get(
                f"https://api.d-id.com/talks/{talk_id}",
                headers={"Authorization": f"Basic {DID_API_KEY}"}
            ) as resp:
                data = await resp.json()
                status = data.get("status")
                if status == "done":
                    return data.get("result_url")
                elif status == "error":
                    logger.error(f"D-ID talk error: {data.get('error')}")
                    return None
            await asyncio.sleep(2)
        logger.error("D-ID talk timed out")
        return None

    async def _download_video(self, url: str, path: Path):
        """Download video file"""
        async with self._session.get(url) as resp:
            with open(path, "wb") as f:
                f.write(await resp.read())

    async def _get_video_duration(self, path: Path) -> float:
        """Get video duration using ffprobe"""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 8.0  # Default estimate


# ============================================================================
# CONTENT GENERATOR
# ============================================================================

class ContentGenerator:
    """Generates research content for the stream"""

    def __init__(self):
        self._topic_index = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def get_opening(self) -> str:
        return (
            "Good news everyone! Welcome to the Farnsworth AI deep research stream. "
            "Tonight we are diving into the Epstein files, analyzing court documents "
            "and victim testimonies. Drop your questions in the chat."
        )

    def get_topic_intro(self) -> str:
        topic = RESEARCH_TOPICS[self._topic_index % len(RESEARCH_TOPICS)]
        self._topic_index += 1
        return f"Now investigating: {topic}. Let me search through the available records."

    async def research_topic(self, topic: str) -> str:
        """Do web research on a topic"""
        await self._ensure_session()
        from urllib.parse import quote_plus

        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}"
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

            async with self._session.get(url, timeout=15, headers=headers) as resp:
                if resp.status == 200:
                    import re
                    html = await resp.text()
                    snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
                    if snippets:
                        # Combine first 2 snippets
                        content = " ".join(snippets[:2])
                        # Filter and truncate
                        for blocked in BLOCKED_NAMES:
                            if blocked in content.lower():
                                return "The records on this topic contain restricted names. Moving on."
                        if len(content) > SEGMENT_MAX_CHARS:
                            content = content[:SEGMENT_MAX_CHARS - 3] + "..."
                        return f"According to the records: {content}"
        except Exception as e:
            logger.debug(f"Research failed: {e}")

        return "The public records on this topic are limited. Let me move to the next area of investigation."

    def get_transition(self) -> str:
        transitions = [
            "Interesting findings. Let me dig deeper into the next topic.",
            "That covers this area. Moving on to the next investigation.",
            "The records reveal much. Continuing our deep dive.",
        ]
        import random
        return random.choice(transitions)


# ============================================================================
# STREAM MANAGER
# ============================================================================

class DIDStreamManager:
    """Manages the video buffer and FFmpeg streaming"""

    def __init__(self):
        self.generator = DIDVideoGenerator()
        self.content = ContentGenerator()
        self.buffer: deque[VideoSegment] = deque()
        self._running = False
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._current_video: Optional[str] = None

    async def start(self):
        """Start the stream"""
        self._running = True

        # Pre-buffer videos
        logger.info(f"Pre-buffering {BUFFER_MIN_BEFORE_LIVE} videos...")
        await self._prebuffer()

        if len(self.buffer) < BUFFER_MIN_BEFORE_LIVE:
            logger.error("Failed to pre-buffer enough videos")
            return False

        logger.info(f"Buffer ready: {len(self.buffer)} videos")

        # Start producer and consumer
        producer_task = asyncio.create_task(self._producer_loop())
        consumer_task = asyncio.create_task(self._consumer_loop())

        try:
            await asyncio.gather(producer_task, consumer_task)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

        return True

    async def stop(self):
        """Stop the stream"""
        self._running = False
        if self._ffmpeg_process:
            self._ffmpeg_process.terminate()
        await self.generator.close()
        await self.content.close()

    async def _prebuffer(self):
        """Pre-generate videos before going live"""
        # Generate in batches of 3 for optimal parallel performance
        # Batch 1: Opening + first topic
        batch1_texts = [
            self.content.get_opening(),
            self.content.get_topic_intro(),
            await self.content.research_topic(RESEARCH_TOPICS[0]),
        ]

        logger.info("Generating batch 1 (3 segments)...")
        tasks = [self.generator.generate_segment(text) for text in batch1_texts]
        results = await asyncio.gather(*tasks)

        for segment in results:
            if segment:
                self.buffer.append(segment)
                logger.info(f"Buffered: {segment.text[:50]}... ({segment.duration:.1f}s)")

        # Batch 2: More topics
        batch2_texts = [
            self.content.get_topic_intro(),
            await self.content.research_topic(RESEARCH_TOPICS[1]),
            self.content.get_transition(),
        ]

        logger.info("Generating batch 2 (3 segments)...")
        tasks = [self.generator.generate_segment(text) for text in batch2_texts]
        results = await asyncio.gather(*tasks)

        for segment in results:
            if segment:
                self.buffer.append(segment)
                logger.info(f"Buffered: {segment.text[:50]}... ({segment.duration:.1f}s)")

    async def _producer_loop(self):
        """Continuously generate videos to keep buffer filled"""
        while self._running:
            try:
                # Check if we need more videos
                if len(self.buffer) >= BUFFER_TARGET:
                    await asyncio.sleep(2)
                    continue

                # Generate content
                needed = BUFFER_TARGET - len(self.buffer)
                texts = []
                for _ in range(min(needed, PARALLEL_GENERATIONS)):
                    # Alternate between intro and research
                    if len(texts) % 2 == 0:
                        texts.append(self.content.get_topic_intro())
                    else:
                        topic = RESEARCH_TOPICS[self.content._topic_index % len(RESEARCH_TOPICS)]
                        texts.append(await self.content.research_topic(topic))

                # Generate in parallel
                if texts:
                    logger.info(f"Generating {len(texts)} segments in parallel...")
                    tasks = [self.generator.generate_segment(text) for text in texts]
                    results = await asyncio.gather(*tasks)

                    for segment in results:
                        if segment:
                            self.buffer.append(segment)

                    logger.info(f"Buffer: {len(self.buffer)} videos")

            except Exception as e:
                logger.error(f"Producer error: {e}")
                await asyncio.sleep(5)

    async def _consumer_loop(self):
        """Play videos from buffer via FFmpeg"""
        logger.info("Starting FFmpeg stream...")

        while self._running:
            try:
                # Wait for video in buffer
                if not self.buffer:
                    logger.warning("Buffer empty! Waiting...")
                    await asyncio.sleep(1)
                    continue

                # Get next video
                segment = self.buffer.popleft()
                logger.info(f"Playing: {segment.text[:50]}... ({segment.duration:.1f}s)")
                logger.info(f"Buffer remaining: {len(self.buffer)}")

                # Stream video via FFmpeg
                await self._stream_video(segment.video_path)

                # Clean up old video file
                try:
                    os.remove(segment.video_path)
                except:
                    pass

            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(2)

    async def _stream_video(self, video_path: str):
        """Stream a single video file to RTMP"""
        cmd = [
            "ffmpeg",
            "-re",  # Read at native framerate
            "-i", video_path,
            # Video encoding
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-profile:v", "main",
            "-b:v", "3000k",
            "-maxrate", "3500k",
            "-bufsize", "6000k",
            "-g", "60",
            "-keyint_min", "60",
            "-pix_fmt", "yuv420p",
            # Audio encoding
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            # Output
            "-f", "flv",
            f"{RTMP_URL}/{STREAM_KEY}",
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )

        self._ffmpeg_process = process

        # Wait for FFmpeg to finish
        _, stderr = await process.communicate()

        if process.returncode != 0 and self._running:
            logger.warning(f"FFmpeg exited: {stderr.decode()[-200:]}")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-key", default=os.getenv("TWITTER_STREAM_KEY", ""))
    args = parser.parse_args()

    global STREAM_KEY
    if args.stream_key:
        STREAM_KEY = args.stream_key

    if not STREAM_KEY:
        logger.error("No stream key! Set TWITTER_STREAM_KEY or use --stream-key")
        return

    if not DID_API_KEY or not DID_AVATAR_URL:
        logger.error("D-ID not configured! Set DID_API_KEY and DID_AVATAR_URL")
        return

    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs not configured! Set ELEVENLABS_API_KEY")
        return

    logger.info("=" * 60)
    logger.info("   FARNSWORTH D-ID STREAM")
    logger.info("   ElevenLabs Voice + D-ID Avatar + Twitter RTMP")
    logger.info("=" * 60)
    logger.info(f"Avatar: {DID_AVATAR_URL[:50]}...")
    logger.info(f"Voice: {ELEVENLABS_VOICE_ID}")
    logger.info(f"Stream: {RTMP_URL}")

    manager = DIDStreamManager()

    # Handle shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown requested...")
        asyncio.create_task(manager.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())
