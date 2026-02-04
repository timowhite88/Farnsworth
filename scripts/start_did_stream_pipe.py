#!/usr/bin/env python3
"""
Farnsworth D-ID Stream - TRUE PERSISTENT via Named Pipe
ONE FFmpeg process that NEVER restarts.

How it works:
1. Create a named pipe (FIFO) for raw video/audio
2. Start FFmpeg reading from the pipe (runs forever)
3. Decode each video segment and write raw frames to pipe
4. FFmpeg continuously encodes and streams - no restarts ever
"""

import asyncio
import aiohttp
import os
import sys
import time
import subprocess
import signal
import threading
import queue
import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, '/workspace/Farnsworth')

from loguru import logger
from dotenv import load_dotenv
load_dotenv('/workspace/Farnsworth/.env')

# ============================================================================
# CONFIG
# ============================================================================

DID_API_KEY = os.getenv("DID_API_KEY")
DID_AVATAR_URL = os.getenv("DID_AVATAR_URL")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_FARNSWORTH", "dxvY1G6UilzEKgCy370m")

RTMP_URL = "rtmp://va.pscp.tv:80/x"
STREAM_KEY = ""

OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
OUTPUT_FPS = 30
KEYFRAME_SEC = 3

CACHE_DIR = Path("/workspace/Farnsworth/cache/did_pipe")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs 2024",
    "Ghislaine Maxwell trial",
    "Epstein black book contacts",
    "Prince Andrew case details",
    "JP Morgan Epstein",
    "Les Wexner connection",
]

BLOCKED = ["trump", "elon", "musk"]


@dataclass
class Segment:
    path: str
    duration: float


class VideoGen:
    """Generate D-ID video segments"""

    def __init__(self):
        self._session = None
        self._n = 0

    async def _session_ok(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()

    async def generate(self, text: str) -> Optional[Segment]:
        await self._session_ok()
        self._n += 1
        n = self._n

        if any(b in text.lower() for b in BLOCKED) or len(text) < 20:
            return None

        try:
            logger.info(f"[{n}] Generating...")

            # ElevenLabs TTS
            audio = await self._tts(text)
            if not audio:
                return None

            # D-ID
            url = await self._did_upload(audio, n)
            if not url:
                return None

            tid = await self._did_talk(url)
            if not tid:
                return None

            vurl = await self._did_wait(tid)
            if not vurl:
                return None

            # Download
            raw = CACHE_DIR / f"r{n}.mp4"
            await self._dl(vurl, raw)

            # Process to standard format
            final = CACHE_DIR / f"v{n}.mp4"
            ok = await self._proc(raw, final)
            os.remove(raw) if os.path.exists(raw) else None

            if not ok:
                return None

            dur = self._dur(final)
            logger.info(f"[{n}] Ready ({dur:.1f}s)")
            return Segment(str(final), dur)

        except Exception as e:
            logger.error(f"[{n}] {e}")
            return None

    async def _tts(self, text):
        async with self._session.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            json={"text": text, "model_id": "eleven_monolingual_v1",
                  "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            headers={"Accept": "audio/mpeg", "Content-Type": "application/json",
                     "xi-api-key": ELEVENLABS_API_KEY}
        ) as r:
            return await r.read() if r.status == 200 else None

    async def _did_upload(self, data, n):
        form = aiohttp.FormData()
        form.add_field("audio", data, filename=f"a{n}.mp3", content_type="audio/mpeg")
        async with self._session.post(
            "https://api.d-id.com/audios", data=form,
            headers={"Authorization": f"Basic {DID_API_KEY}"}
        ) as r:
            return (await r.json()).get("url") if r.status == 201 else None

    async def _did_talk(self, audio_url):
        async with self._session.post(
            "https://api.d-id.com/talks",
            json={"source_url": DID_AVATAR_URL,
                  "script": {"type": "audio", "audio_url": audio_url},
                  "config": {"fluent": True}, "driver_url": "bank://lively"},
            headers={"Authorization": f"Basic {DID_API_KEY}",
                     "Content-Type": "application/json"}
        ) as r:
            return (await r.json()).get("id") if r.status == 201 else None

    async def _did_wait(self, tid):
        for _ in range(30):
            async with self._session.get(
                f"https://api.d-id.com/talks/{tid}",
                headers={"Authorization": f"Basic {DID_API_KEY}"}
            ) as r:
                d = await r.json()
                if d.get("status") == "done":
                    return d.get("result_url")
                if d.get("status") == "error":
                    return None
            await asyncio.sleep(2)
        return None

    async def _dl(self, url, path):
        async with self._session.get(url) as r:
            with open(path, "wb") as f:
                f.write(await r.read())

    async def _proc(self, inp, out):
        cmd = [
            "ffmpeg", "-y", "-i", str(inp),
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,"
                   f"fps={OUTPUT_FPS}",
            "-c:v", "libx264", "-preset", "fast", "-profile:v", "high",
            "-b:v", "6000k", "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-pix_fmt", "yuv420p",
            "-ac", "2", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            str(out)
        ]
        p = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await p.wait()
        return p.returncode == 0

    def _dur(self, path):
        try:
            r = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path)
            ], capture_output=True, text=True)
            return float(r.stdout.strip())
        except:
            return 10.0


class Content:
    def __init__(self):
        self._i = 0

    def opening(self):
        return (
            "Good news everyone! Welcome to the Farnsworth AI research stream. "
            "Tonight we investigate the Epstein documents. "
            "Drop your questions in chat and I will research them live."
        )

    async def full_segment(self):
        topic = RESEARCH_TOPICS[self._i % len(RESEARCH_TOPICS)]
        self._i += 1
        research = await self._research(topic)
        return f"Investigating {topic}. {research} The patterns reveal deeper connections."

    async def _research(self, topic):
        try:
            from urllib.parse import quote_plus
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=10
                ) as r:
                    if r.status == 200:
                        import re
                        snip = re.findall(r'class="result__snippet"[^>]*>([^<]+)', await r.text())
                        if snip:
                            txt = " ".join(snip[:3])[:400]
                            if not any(b in txt.lower() for b in BLOCKED):
                                return f"Documents show: {txt}"
        except:
            pass
        return "Records are limited but suggest deeper patterns."


class PipeStreamer:
    """
    TRUE persistent streaming via stdin pipe.
    ONE FFmpeg runs forever, we pipe video data to it.
    """

    def __init__(self):
        self.video_queue = queue.Queue()
        self._running = False
        self._ffmpeg = None
        self._feeder_thread = None

    def start(self):
        self._running = True

        # Start FFmpeg reading from stdin (pipe)
        # This process runs FOREVER
        cmd = [
            "ffmpeg",
            "-re",  # Read at realtime speed
            "-f", "mpegts",  # Input format: MPEG-TS (good for piping)
            "-i", "pipe:0",  # Read from stdin
            # Output encoding
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-profile:v", "high",
            "-b:v", "6000k",
            "-maxrate", "7000k",
            "-bufsize", "14000k",
            "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-keyint_min", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-ac", "2",
            "-f", "flv",
            "-flvflags", "no_duration_filesize",
            f"{RTMP_URL}/{STREAM_KEY}",
        ]

        logger.info("[FFmpeg] Starting PERSISTENT stream (never restarts)...")
        self._ffmpeg = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # Start feeder thread
        self._feeder_thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._feeder_thread.start()

        logger.info("[FFmpeg] Persistent stream started!")

    def stop(self):
        self._running = False
        if self._ffmpeg and self._ffmpeg.stdin:
            self._ffmpeg.stdin.close()
        if self._ffmpeg:
            self._ffmpeg.terminate()

    def add_video(self, path: str, duration: float):
        self.video_queue.put((path, duration))

    def _feed_loop(self):
        """
        Feed video files to FFmpeg's stdin as MPEG-TS stream.
        Each video is converted to MPEG-TS and piped.
        """
        while self._running:
            try:
                # Get next video
                try:
                    path, duration = self.video_queue.get(timeout=5)
                except queue.Empty:
                    logger.warning("[Feeder] Queue empty, waiting...")
                    continue

                logger.info(f"[Feeder] Feeding video ({duration:.1f}s), queue: {self.video_queue.qsize()}")

                # Convert video to MPEG-TS and pipe to FFmpeg
                self._pipe_video(path)

                # Clean up
                try:
                    os.remove(path)
                except:
                    pass

            except Exception as e:
                logger.error(f"[Feeder] Error: {e}")
                if not self._running:
                    break
                time.sleep(1)

    def _pipe_video(self, video_path: str):
        """
        Convert a video file to MPEG-TS and write to FFmpeg's stdin.
        This is done in realtime (-re flag on the reader).
        """
        if not self._ffmpeg or not self._ffmpeg.stdin:
            logger.error("[Feeder] FFmpeg not running!")
            return

        # Use FFmpeg to convert the video to MPEG-TS and output to stdout
        # We then read that and write to main FFmpeg's stdin
        read_cmd = [
            "ffmpeg",
            "-re",  # Realtime playback
            "-i", video_path,
            "-c:v", "copy",  # No re-encode (already encoded)
            "-c:a", "copy",
            "-f", "mpegts",
            "-muxdelay", "0",
            "-muxpreload", "0",
            "pipe:1"  # Output to stdout
        ]

        try:
            reader = subprocess.Popen(
                read_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            # Read from reader and write to main FFmpeg
            while True:
                chunk = reader.stdout.read(65536)  # 64KB chunks
                if not chunk:
                    break
                try:
                    self._ffmpeg.stdin.write(chunk)
                    self._ffmpeg.stdin.flush()
                except BrokenPipeError:
                    logger.error("[Feeder] FFmpeg pipe broken!")
                    self._running = False
                    break

            reader.wait()

        except Exception as e:
            logger.error(f"[Feeder] Pipe error: {e}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-key", default="")
    args = parser.parse_args()

    global STREAM_KEY
    STREAM_KEY = args.stream_key or os.getenv("TWITTER_STREAM_KEY", "")

    if not all([STREAM_KEY, DID_API_KEY, DID_AVATAR_URL, ELEVENLABS_API_KEY]):
        logger.error("Missing config")
        return

    logger.info("=" * 60)
    logger.info("  FARNSWORTH TRUE PERSISTENT STREAM")
    logger.info("  ONE FFmpeg process - ZERO restarts")
    logger.info("=" * 60)

    gen = VideoGen()
    content = Content()
    streamer = PipeStreamer()

    # Generate initial videos
    logger.info("Generating initial videos...")
    texts = [
        content.opening(),
        await content.full_segment(),
        await content.full_segment(),
        await content.full_segment(),
    ]

    tasks = [gen.generate(t) for t in texts]
    segments = [s for s in await asyncio.gather(*tasks) if s]

    if not segments:
        logger.error("No initial segments")
        return

    for seg in segments:
        streamer.add_video(seg.path, seg.duration)

    logger.info(f"Queued {len(segments)} initial videos")

    # Start persistent stream
    streamer.start()

    # Production loop
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False
        streamer.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        try:
            qsize = streamer.video_queue.qsize()
            logger.info(f"Queue: {qsize}")

            if qsize < 4:
                new_texts = [
                    await content.full_segment(),
                    await content.full_segment(),
                ]
                tasks = [gen.generate(t) for t in new_texts]
                new_segs = [s for s in await asyncio.gather(*tasks) if s]

                for seg in new_segs:
                    streamer.add_video(seg.path, seg.duration)

            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(5)

    await gen.close()


if __name__ == "__main__":
    asyncio.run(main())
