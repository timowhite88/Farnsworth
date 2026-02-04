#!/usr/bin/env python3
"""
Farnsworth D-ID Stream v3 - Truly Continuous Stream
Uses a SINGLE persistent FFmpeg process with video concatenation.

Key: Concatenate multiple videos into batches, stream batches seamlessly.
"""

import asyncio
import aiohttp
import os
import sys
import time
import subprocess
import signal
import tempfile
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

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30
VIDEO_BITRATE = "6000k"
KEYFRAME_INTERVAL = 3

CACHE_DIR = Path("/workspace/Farnsworth/cache/did_v3")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# How many videos to batch together before streaming
BATCH_SIZE = 3

RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs 2024 revealed",
    "Ghislaine Maxwell trial revelations",
    "Epstein black book contacts",
    "Prince Andrew settlement details",
    "JP Morgan Epstein banking",
    "Les Wexner connection",
    "Epstein island visitors",
    "Victim testimonies",
]

BLOCKED_NAMES = ["trump", "donald trump", "elon", "musk"]


@dataclass
class VideoSegment:
    path: str
    text: str
    duration: float


class DIDGenerator:
    def __init__(self):
        self._session = None
        self._counter = 0

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()

    async def generate(self, text: str) -> Optional[VideoSegment]:
        await self._ensure_session()
        self._counter += 1
        seg_id = self._counter

        if any(b in text.lower() for b in BLOCKED_NAMES) or len(text) < 20:
            return None

        try:
            # ElevenLabs
            logger.info(f"[{seg_id}] Audio...")
            audio = await self._elevenlabs(text)
            if not audio:
                return None

            # D-ID upload
            audio_url = await self._upload_audio(audio, seg_id)
            if not audio_url:
                return None

            # D-ID talk
            logger.info(f"[{seg_id}] D-ID video...")
            talk_id = await self._create_talk(audio_url)
            if not talk_id:
                return None

            video_url = await self._wait_talk(talk_id)
            if not video_url:
                return None

            # Download
            raw_path = CACHE_DIR / f"raw_{seg_id}.mp4"
            await self._download(video_url, raw_path)

            # Process to correct format
            logger.info(f"[{seg_id}] Processing...")
            final_path = CACHE_DIR / f"final_{seg_id}.mp4"
            if not await self._process(raw_path, final_path):
                return None

            os.remove(raw_path)

            duration = self._get_duration(final_path)
            logger.info(f"[{seg_id}] Done ({duration:.1f}s)")

            return VideoSegment(str(final_path), text, duration)

        except Exception as e:
            logger.error(f"[{seg_id}] Error: {e}")
            return None

    async def _elevenlabs(self, text):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
        payload = {"text": text, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}}
        async with self._session.post(url, json=payload, headers=headers) as r:
            return await r.read() if r.status == 200 else None

    async def _upload_audio(self, data, seg_id):
        form = aiohttp.FormData()
        form.add_field("audio", data, filename=f"s{seg_id}.mp3", content_type="audio/mpeg")
        async with self._session.post("https://api.d-id.com/audios", data=form, headers={"Authorization": f"Basic {DID_API_KEY}"}) as r:
            return (await r.json()).get("url") if r.status == 201 else None

    async def _create_talk(self, audio_url):
        payload = {"source_url": DID_AVATAR_URL, "script": {"type": "audio", "audio_url": audio_url}, "config": {"fluent": True}, "driver_url": "bank://lively"}
        async with self._session.post("https://api.d-id.com/talks", json=payload, headers={"Authorization": f"Basic {DID_API_KEY}", "Content-Type": "application/json"}) as r:
            return (await r.json()).get("id") if r.status == 201 else None

    async def _wait_talk(self, talk_id, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            async with self._session.get(f"https://api.d-id.com/talks/{talk_id}", headers={"Authorization": f"Basic {DID_API_KEY}"}) as r:
                data = await r.json()
                if data.get("status") == "done":
                    return data.get("result_url")
                if data.get("status") == "error":
                    return None
            await asyncio.sleep(2)
        return None

    async def _download(self, url, path):
        async with self._session.get(url) as r:
            with open(path, "wb") as f:
                f.write(await r.read())

    async def _process(self, inp, out):
        """Scale to 1920x1080, stereo audio, Twitter-compatible encoding"""
        cmd = [
            "ffmpeg", "-y", "-i", str(inp),
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,fps={OUTPUT_FPS}",
            "-c:v", "libx264", "-preset", "fast", "-profile:v", "high",
            "-b:v", VIDEO_BITRATE, "-maxrate", "7000k", "-bufsize", "14000k",
            "-g", str(OUTPUT_FPS * KEYFRAME_INTERVAL),
            "-keyint_min", str(OUTPUT_FPS * KEYFRAME_INTERVAL),
            "-pix_fmt", "yuv420p",
            "-ac", "2", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            str(out)
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await proc.wait()
        return proc.returncode == 0

    def _get_duration(self, path):
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True)
            return float(r.stdout.strip())
        except:
            return 8.0


class ContentGen:
    def __init__(self):
        self._idx = 0

    def opening(self):
        return "Good news everyone! Welcome to the Farnsworth AI research stream. Tonight we investigate the Epstein files."

    def topic(self):
        t = RESEARCH_TOPICS[self._idx % len(RESEARCH_TOPICS)]
        self._idx += 1
        return f"Now investigating: {t}."

    async def research(self, topic):
        try:
            from urllib.parse import quote_plus
            async with aiohttp.ClientSession() as s:
                async with s.get(f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}", headers={"User-Agent": "Mozilla/5.0"}, timeout=10) as r:
                    if r.status == 200:
                        import re
                        html = await r.text()
                        snip = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
                        if snip:
                            txt = " ".join(snip[:2])[:250]
                            if not any(b in txt.lower() for b in BLOCKED_NAMES):
                                return f"The documents show: {txt}"
        except:
            pass
        return "Records on this topic are limited."


async def concat_videos(videos: List[str], output: str) -> bool:
    """Concatenate multiple videos into one using FFmpeg concat demuxer"""
    # Create concat file
    concat_file = CACHE_DIR / "concat.txt"
    with open(concat_file, "w") as f:
        for v in videos:
            f.write(f"file '{v}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",  # No re-encoding since all videos are same format
        output
    ]

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
    await proc.wait()
    return proc.returncode == 0


async def stream_video(path: str, duration: float):
    """Stream a video file to RTMP"""
    cmd = [
        "ffmpeg",
        "-re",
        "-i", path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-b:v", "6000k",
        "-maxrate", "7000k",
        "-bufsize", "14000k",
        "-g", str(OUTPUT_FPS * KEYFRAME_INTERVAL),
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "44100",
        "-ac", "2",
        "-f", "flv",
        f"{RTMP_URL}/{STREAM_KEY}",
    ]

    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
    try:
        await asyncio.wait_for(proc.communicate(), timeout=duration + 30)
    except asyncio.TimeoutError:
        proc.terminate()


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-key", default="")
    args = parser.parse_args()

    global STREAM_KEY
    STREAM_KEY = args.stream_key or os.getenv("TWITTER_STREAM_KEY", "")

    if not all([STREAM_KEY, DID_API_KEY, DID_AVATAR_URL, ELEVENLABS_API_KEY]):
        logger.error("Missing configuration")
        return

    logger.info("=" * 50)
    logger.info("  FARNSWORTH D-ID STREAM v3")
    logger.info("  Batch concatenation for seamless playback")
    logger.info("=" * 50)

    gen = DIDGenerator()
    content = ContentGen()

    # Pre-generate initial batch
    logger.info("Generating initial batch...")
    texts = [
        content.opening(),
        content.topic(),
        await content.research(RESEARCH_TOPICS[0]),
        content.topic(),
        await content.research(RESEARCH_TOPICS[1]),
        content.topic(),
    ]

    # Generate in parallel
    tasks = [gen.generate(t) for t in texts]
    segments = [s for s in await asyncio.gather(*tasks) if s]

    if len(segments) < 3:
        logger.error("Not enough segments generated")
        return

    logger.info(f"Generated {len(segments)} segments")

    # Main loop: concatenate batch → stream → generate next batch
    batch_num = 0
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running and segments:
        batch_num += 1
        logger.info(f"\n=== BATCH {batch_num} ({len(segments)} videos) ===")

        # Concatenate current segments into one video
        video_paths = [s.path for s in segments]
        total_duration = sum(s.duration for s in segments)
        batch_file = CACHE_DIR / f"batch_{batch_num}.mp4"

        logger.info(f"Concatenating {len(video_paths)} videos ({total_duration:.0f}s total)...")
        if not await concat_videos(video_paths, str(batch_file)):
            logger.error("Concat failed")
            break

        # Clean up individual files
        for p in video_paths:
            try:
                os.remove(p)
            except:
                pass

        # Start streaming current batch while generating next batch
        logger.info(f"Streaming batch {batch_num}...")

        # Generate next batch in background
        next_texts = []
        for _ in range(BATCH_SIZE * 2):  # Generate more to have buffer
            if len(next_texts) % 2 == 0:
                next_texts.append(content.topic())
            else:
                topic = RESEARCH_TOPICS[content._idx % len(RESEARCH_TOPICS)]
                next_texts.append(await content.research(topic))

        gen_task = asyncio.create_task(
            asyncio.gather(*[gen.generate(t) for t in next_texts])
        )

        # Stream the batch
        await stream_video(str(batch_file), total_duration)

        # Get next segments
        next_results = await gen_task
        segments = [s for s in next_results if s]

        # Clean up batch file
        try:
            os.remove(batch_file)
        except:
            pass

        if not segments:
            logger.warning("No more segments generated")
            break

    await gen.close()
    logger.info("Stream ended")


if __name__ == "__main__":
    asyncio.run(main())
