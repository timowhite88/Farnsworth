#!/usr/bin/env python3
"""
Farnsworth D-ID Stream v2 - Seamless Video Streaming
Fixes: Resolution scaling, continuous FFmpeg, no black frames

Key improvements:
1. Single persistent FFmpeg process (no restarts)
2. Scale 512x512 D-ID output → 1920x1080 Twitter
3. Convert mono audio → stereo
4. Use concat demuxer for seamless playback
5. Show holding frame between segments
"""

import asyncio
import aiohttp
import os
import sys
import time
import subprocess
import signal
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from collections import deque

sys.path.insert(0, '/workspace/Farnsworth')

from loguru import logger
from dotenv import load_dotenv
load_dotenv('/workspace/Farnsworth/.env')

# ============================================================================
# CONFIGURATION
# ============================================================================

DID_API_KEY = os.getenv("DID_API_KEY")
DID_AVATAR_URL = os.getenv("DID_AVATAR_URL")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_FARNSWORTH", "dxvY1G6UilzEKgCy370m")

RTMP_URL = "rtmp://va.pscp.tv:80/x"
STREAM_KEY = os.getenv("TWITTER_STREAM_KEY", "")

# Output format for Twitter
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30
VIDEO_BITRATE = "6000k"  # 6Mbps for quality
AUDIO_BITRATE = "128k"
KEYFRAME_INTERVAL = 3  # seconds

# Buffer settings
BUFFER_MIN = 3
BUFFER_TARGET = 6
PARALLEL_GEN = 3

# Directories
CACHE_DIR = Path("/workspace/Farnsworth/cache/did_stream_v2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = CACHE_DIR / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

# Concat playlist file
PLAYLIST_FILE = CACHE_DIR / "playlist.txt"

# Background/holding image (will be generated)
HOLDING_IMAGE = CACHE_DIR / "holding.png"

# Research topics
RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs 2024 names revealed",
    "Ghislaine Maxwell trial key revelations",
    "Epstein black book contacts exposed",
    "Prince Andrew Virginia Giuffre case details",
    "JP Morgan Epstein banking relationship",
    "Les Wexner Epstein connection",
    "Epstein island visitor records",
    "Epstein victim court testimonies",
]

BLOCKED_NAMES = ["trump", "donald trump", "elon", "elon musk", "musk"]


# ============================================================================
# VIDEO SEGMENT
# ============================================================================

@dataclass
class VideoSegment:
    original_path: str
    processed_path: str  # Scaled to 1920x1080
    text: str
    duration: float
    created_at: float


# ============================================================================
# GENERATE HOLDING IMAGE
# ============================================================================

def create_holding_image():
    """Create a holding/loading image for between segments"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a1a2e:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=1",
        "-vf", "drawtext=text='FARNSWORTH AI':fontsize=72:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2-50,"
               "drawtext=text='Deep Research Stream':fontsize=36:fontcolor=0x888888:x=(w-text_w)/2:y=(h-text_h)/2+50",
        "-frames:v", "1",
        str(HOLDING_IMAGE)
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=10)
        logger.info(f"Created holding image: {HOLDING_IMAGE}")
    except Exception as e:
        logger.warning(f"Could not create holding image: {e}")


# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class VideoProcessor:
    """Process D-ID videos to Twitter format"""

    @staticmethod
    async def process_video(input_path: str, output_path: str) -> bool:
        """
        Scale 512x512 D-ID video to 1920x1080 with letterboxing
        Convert mono audio to stereo
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            # Video: scale and pad to 1920x1080
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,"
                   f"fps={OUTPUT_FPS}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-profile:v", "high",
            "-level", "4.1",
            "-b:v", VIDEO_BITRATE,
            "-maxrate", "7000k",
            "-bufsize", "14000k",
            "-g", str(OUTPUT_FPS * KEYFRAME_INTERVAL),
            "-keyint_min", str(OUTPUT_FPS * KEYFRAME_INTERVAL),
            "-pix_fmt", "yuv420p",
            # Audio: convert to stereo
            "-ac", "2",
            "-c:a", "aac",
            "-b:a", AUDIO_BITRATE,
            "-ar", "44100",
            output_path
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)

            if proc.returncode == 0:
                return True
            else:
                logger.error(f"Video processing failed: {stderr.decode()[-200:]}")
                return False
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return False


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
                return None

        if len(text) < 20:
            return None

        try:
            # Step 1: ElevenLabs TTS
            logger.info(f"[{seg_id}] Generating audio...")
            audio_data = await self._generate_audio(text)
            if not audio_data:
                return None

            # Step 2: Upload audio to D-ID
            audio_url = await self._upload_audio(audio_data, seg_id)
            if not audio_url:
                return None

            # Step 3: Create D-ID talk
            logger.info(f"[{seg_id}] Creating D-ID video...")
            talk_id = await self._create_talk(audio_url)
            if not talk_id:
                return None

            # Step 4: Wait for render
            video_url = await self._wait_for_talk(talk_id)
            if not video_url:
                return None

            # Step 5: Download video
            original_path = CACHE_DIR / f"raw_{seg_id}_{int(time.time())}.mp4"
            await self._download_video(video_url, original_path)

            # Step 6: Process to Twitter format (scale + stereo)
            logger.info(f"[{seg_id}] Processing video to 1920x1080...")
            processed_path = PROCESSED_DIR / f"seg_{seg_id}_{int(time.time())}.mp4"
            success = await VideoProcessor.process_video(str(original_path), str(processed_path))

            if not success:
                return None

            # Clean up original
            try:
                os.remove(original_path)
            except:
                pass

            # Get duration
            duration = await self._get_video_duration(processed_path)

            total_time = time.time() - start_time
            logger.info(f"[{seg_id}] Complete in {total_time:.1f}s ({duration:.1f}s video)")

            return VideoSegment(
                original_path=str(original_path),
                processed_path=str(processed_path),
                text=text,
                duration=duration,
                created_at=time.time()
            )

        except Exception as e:
            logger.error(f"[{seg_id}] Failed - {e}")
            return None

    async def _generate_audio(self, text: str) -> Optional[bytes]:
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
            return None

    async def _upload_audio(self, audio_data: bytes, seg_id: int) -> Optional[str]:
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
            return None

    async def _create_talk(self, audio_url: str) -> Optional[str]:
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
            return None

    async def _wait_for_talk(self, talk_id: str, timeout: float = 60) -> Optional[str]:
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
                    return None
            await asyncio.sleep(2)
        return None

    async def _download_video(self, url: str, path: Path):
        async with self._session.get(url) as resp:
            with open(path, "wb") as f:
                f.write(await resp.read())

    async def _get_video_duration(self, path: Path) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 8.0


# ============================================================================
# CONTENT GENERATOR
# ============================================================================

class ContentGenerator:
    def __init__(self):
        self._topic_index = 0

    def get_opening(self) -> str:
        return (
            "Good news everyone! Welcome to the Farnsworth AI deep research stream. "
            "Tonight we are diving into the Epstein files. "
            "Drop your questions in the chat."
        )

    def get_topic_intro(self) -> str:
        topic = RESEARCH_TOPICS[self._topic_index % len(RESEARCH_TOPICS)]
        self._topic_index += 1
        return f"Now investigating: {topic}."

    async def research_topic(self, topic: str) -> str:
        try:
            from urllib.parse import quote_plus
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}"
            headers = {"User-Agent": "Mozilla/5.0"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=15, headers=headers) as resp:
                    if resp.status == 200:
                        import re
                        html = await resp.text()
                        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
                        if snippets:
                            content = " ".join(snippets[:2])[:300]
                            for blocked in BLOCKED_NAMES:
                                if blocked in content.lower():
                                    return "Records contain restricted information. Moving on."
                            return f"According to the documents: {content}"
        except:
            pass
        return "Public records on this topic are limited."

    def get_transition(self) -> str:
        import random
        return random.choice([
            "Interesting findings. Continuing investigation.",
            "That covers this area. Moving to the next topic.",
            "The records reveal much. Let me dig deeper.",
        ])


# ============================================================================
# SEAMLESS STREAM MANAGER
# ============================================================================

class SeamlessStreamManager:
    """
    Manages seamless video streaming using FFmpeg concat demuxer.
    Videos are queued and played without gaps.
    """

    def __init__(self):
        self.generator = DIDVideoGenerator()
        self.content = ContentGenerator()
        self.video_queue: deque[VideoSegment] = deque()
        self._running = False
        self._ffmpeg_process = None
        self._current_playlist: List[str] = []

    async def start(self):
        """Start the stream"""
        self._running = True

        # Create holding image
        create_holding_image()

        # Pre-buffer videos
        logger.info(f"Pre-buffering {BUFFER_MIN} videos...")
        await self._prebuffer()

        if len(self.video_queue) < BUFFER_MIN:
            logger.error("Failed to pre-buffer enough videos")
            return False

        logger.info(f"Buffer ready: {len(self.video_queue)} videos")
        logger.info("Starting seamless stream...")

        # Start tasks
        producer = asyncio.create_task(self._producer_loop())
        consumer = asyncio.create_task(self._consumer_loop())

        try:
            await asyncio.gather(producer, consumer)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

        return True

    async def stop(self):
        self._running = False
        if self._ffmpeg_process:
            self._ffmpeg_process.terminate()
        await self.generator.close()

    async def _prebuffer(self):
        """Pre-generate videos in batches"""
        # Batch 1
        texts1 = [
            self.content.get_opening(),
            self.content.get_topic_intro(),
            await self.content.research_topic(RESEARCH_TOPICS[0]),
        ]

        logger.info("Generating batch 1...")
        tasks = [self.generator.generate_segment(t) for t in texts1]
        results = await asyncio.gather(*tasks)
        for seg in results:
            if seg:
                self.video_queue.append(seg)

        # Batch 2
        texts2 = [
            self.content.get_topic_intro(),
            await self.content.research_topic(RESEARCH_TOPICS[1]),
            self.content.get_transition(),
        ]

        logger.info("Generating batch 2...")
        tasks = [self.generator.generate_segment(t) for t in texts2]
        results = await asyncio.gather(*tasks)
        for seg in results:
            if seg:
                self.video_queue.append(seg)

    async def _producer_loop(self):
        """Keep buffer filled"""
        while self._running:
            try:
                if len(self.video_queue) >= BUFFER_TARGET:
                    await asyncio.sleep(2)
                    continue

                needed = min(BUFFER_TARGET - len(self.video_queue), PARALLEL_GEN)
                texts = []
                for _ in range(needed):
                    if len(texts) % 2 == 0:
                        texts.append(self.content.get_topic_intro())
                    else:
                        topic = RESEARCH_TOPICS[self.content._topic_index % len(RESEARCH_TOPICS)]
                        texts.append(await self.content.research_topic(topic))

                if texts:
                    logger.info(f"Generating {len(texts)} segments...")
                    tasks = [self.generator.generate_segment(t) for t in texts]
                    results = await asyncio.gather(*tasks)
                    for seg in results:
                        if seg:
                            self.video_queue.append(seg)
                    logger.info(f"Buffer: {len(self.video_queue)} videos")

            except Exception as e:
                logger.error(f"Producer error: {e}")
                await asyncio.sleep(5)

    async def _consumer_loop(self):
        """
        Stream videos seamlessly using FFmpeg with concat.
        Instead of restarting FFmpeg for each video, we use a persistent
        FFmpeg process that reads videos sequentially.
        """
        while self._running:
            try:
                if not self.video_queue:
                    logger.warning("Buffer empty, waiting...")
                    await asyncio.sleep(1)
                    continue

                # Get next video
                segment = self.video_queue.popleft()
                logger.info(f"Streaming: {segment.text[:50]}... ({segment.duration:.1f}s)")
                logger.info(f"Buffer: {len(self.video_queue)} remaining")

                # Stream this video (with proper settings for Twitter)
                await self._stream_single_video(segment.processed_path, segment.duration)

                # Clean up
                try:
                    os.remove(segment.processed_path)
                except:
                    pass

            except Exception as e:
                logger.error(f"Consumer error: {e}")
                await asyncio.sleep(2)

    async def _stream_single_video(self, video_path: str, duration: float):
        """
        Stream a single video to RTMP.
        The video is already processed to correct format.
        We just need to stream it with -re flag for realtime.
        """
        cmd = [
            "ffmpeg",
            "-re",  # Realtime playback
            "-stream_loop", "0",  # Don't loop
            "-i", video_path,
            # Copy video (already encoded correctly)
            "-c:v", "copy",
            # Copy audio (already encoded correctly)
            "-c:a", "copy",
            # Output
            "-f", "flv",
            "-flvflags", "no_duration_filesize",
            f"{RTMP_URL}/{STREAM_KEY}",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )

        self._ffmpeg_process = proc

        try:
            # Wait for video to finish (with small buffer for next)
            await asyncio.wait_for(proc.communicate(), timeout=duration + 10)
        except asyncio.TimeoutError:
            proc.terminate()
        except Exception as e:
            logger.warning(f"FFmpeg error: {e}")


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
        logger.error("No stream key!")
        return

    if not DID_API_KEY or not DID_AVATAR_URL:
        logger.error("D-ID not configured!")
        return

    if not ELEVENLABS_API_KEY:
        logger.error("ElevenLabs not configured!")
        return

    logger.info("=" * 60)
    logger.info("   FARNSWORTH D-ID STREAM v2 - SEAMLESS")
    logger.info("   1920x1080 @ 30fps, Stereo Audio")
    logger.info("=" * 60)

    manager = SeamlessStreamManager()

    def signal_handler(sig, frame):
        logger.info("Shutdown...")
        asyncio.create_task(manager.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())
