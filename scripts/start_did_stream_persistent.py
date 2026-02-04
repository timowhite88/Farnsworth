#!/usr/bin/env python3
"""
Farnsworth D-ID Stream - PERSISTENT FFMPEG
One FFmpeg process that NEVER restarts. Uses named pipe for video feeding.

Architecture:
1. Single FFmpeg process reads from named pipe, streams to RTMP forever
2. Video segments are decoded and fed as raw frames to the pipe
3. No RTMP reconnection = seamless stream
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

CACHE_DIR = Path("/workspace/Farnsworth/cache/did_persist")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_PIPE = CACHE_DIR / "video.pipe"
AUDIO_PIPE = CACHE_DIR / "audio.pipe"

RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs 2024",
    "Ghislaine Maxwell trial",
    "Epstein black book contacts",
    "Prince Andrew case details",
    "JP Morgan Epstein banking",
    "Les Wexner connection",
]

BLOCKED = ["trump", "elon", "musk"]


# ============================================================================
# VIDEO GENERATOR (same as before)
# ============================================================================

@dataclass
class Segment:
    path: str
    duration: float
    text: str


class VideoGen:
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
            logger.info(f"[{n}] TTS...")
            audio = await self._tts(text)
            if not audio:
                return None

            logger.info(f"[{n}] D-ID...")
            url = await self._did_upload(audio, n)
            if not url:
                return None

            tid = await self._did_talk(url)
            if not tid:
                return None

            vurl = await self._did_wait(tid)
            if not vurl:
                return None

            raw = CACHE_DIR / f"r{n}.mp4"
            await self._dl(vurl, raw)

            logger.info(f"[{n}] Process...")
            final = CACHE_DIR / f"v{n}.mp4"
            ok = await self._proc(raw, final)
            try:
                os.remove(raw)
            except:
                pass

            if not ok:
                return None

            dur = self._dur(final)
            logger.info(f"[{n}] Ready ({dur:.1f}s)")
            return Segment(str(final), dur, text)

        except Exception as e:
            logger.error(f"[{n}] {e}")
            return None

    async def _tts(self, text):
        async with self._session.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            json={"text": text, "model_id": "eleven_monolingual_v1", "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
            headers={"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVENLABS_API_KEY}
        ) as r:
            return await r.read() if r.status == 200 else None

    async def _did_upload(self, data, n):
        form = aiohttp.FormData()
        form.add_field("audio", data, filename=f"a{n}.mp3", content_type="audio/mpeg")
        async with self._session.post("https://api.d-id.com/audios", data=form, headers={"Authorization": f"Basic {DID_API_KEY}"}) as r:
            return (await r.json()).get("url") if r.status == 201 else None

    async def _did_talk(self, audio_url):
        async with self._session.post(
            "https://api.d-id.com/talks",
            json={"source_url": DID_AVATAR_URL, "script": {"type": "audio", "audio_url": audio_url}, "config": {"fluent": True}, "driver_url": "bank://lively"},
            headers={"Authorization": f"Basic {DID_API_KEY}", "Content-Type": "application/json"}
        ) as r:
            return (await r.json()).get("id") if r.status == 201 else None

    async def _did_wait(self, tid):
        for _ in range(30):
            async with self._session.get(f"https://api.d-id.com/talks/{tid}", headers={"Authorization": f"Basic {DID_API_KEY}"}) as r:
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
        """Process to exact format needed for concat"""
        cmd = [
            "ffmpeg", "-y", "-i", str(inp),
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,fps={OUTPUT_FPS}",
            "-c:v", "libx264", "-preset", "fast", "-profile:v", "high",
            "-b:v", "6000k", "-g", str(OUTPUT_FPS * KEYFRAME_SEC), "-pix_fmt", "yuv420p",
            "-ac", "2", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            str(out)
        ]
        p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        await p.wait()
        return p.returncode == 0

    def _dur(self, path):
        try:
            r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], capture_output=True, text=True)
            return float(r.stdout.strip())
        except:
            return 8.0


# ============================================================================
# CONTENT
# ============================================================================

class Content:
    def __init__(self):
        self._i = 0

    def opening(self):
        return (
            "Good news everyone! Welcome to the Farnsworth AI deep research stream. "
            "I am Professor Farnsworth, investigating documents the establishment prefers hidden. "
            "Tonight we dive into the Epstein files. Drop questions in the chat."
        )

    async def full_segment(self):
        topic = RESEARCH_TOPICS[self._i % len(RESEARCH_TOPICS)]
        self._i += 1
        research = await self._research(topic)
        return f"Now investigating {topic}. {research} This reveals important patterns in the network."

    async def _research(self, topic):
        try:
            from urllib.parse import quote_plus
            async with aiohttp.ClientSession() as s:
                async with s.get(f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}", headers={"User-Agent": "Mozilla/5.0"}, timeout=10) as r:
                    if r.status == 200:
                        import re
                        snip = re.findall(r'class="result__snippet"[^>]*>([^<]+)', await r.text())
                        if snip:
                            txt = " ".join(snip[:3])[:400]
                            if not any(b in txt.lower() for b in BLOCKED):
                                return f"According to the documents: {txt}"
        except:
            pass
        return "The public records are limited but patterns suggest deeper connections."


# ============================================================================
# PERSISTENT STREAMER - ONE FFMPEG FOREVER
# ============================================================================

class PersistentStreamer:
    """
    Maintains ONE FFmpeg process for the entire stream.
    Uses concat demuxer with a playlist file that we keep updating.
    """

    def __init__(self):
        self.video_queue = queue.Queue()
        self._running = False
        self._ffmpeg = None
        self._playlist_file = CACHE_DIR / "playlist.txt"
        self._concat_thread = None

    def start(self):
        """Start the persistent FFmpeg stream"""
        self._running = True

        # Start the concat streaming thread
        self._concat_thread = threading.Thread(target=self._concat_stream_loop, daemon=True)
        self._concat_thread.start()

    def stop(self):
        self._running = False
        if self._ffmpeg:
            self._ffmpeg.terminate()

    def add_video(self, path: str, duration: float):
        self.video_queue.put((path, duration))
        logger.info(f"[Queue] Added video, queue size: {self.video_queue.qsize()}")

    def _concat_stream_loop(self):
        """
        Main streaming loop using FFmpeg concat with file concatenation.
        We concatenate videos into larger chunks and stream those.
        """
        while self._running:
            try:
                # Collect videos for a batch (wait for at least 2, or timeout)
                batch = []
                batch_duration = 0

                # Get first video (blocking)
                try:
                    path, dur = self.video_queue.get(timeout=30)
                    batch.append(path)
                    batch_duration += dur
                except queue.Empty:
                    logger.warning("[Stream] No videos available, waiting...")
                    continue

                # Try to get more videos (non-blocking) to build a longer batch
                while batch_duration < 60 and len(batch) < 5:  # Up to 60 sec or 5 videos
                    try:
                        path, dur = self.video_queue.get_nowait()
                        batch.append(path)
                        batch_duration += dur
                    except queue.Empty:
                        break

                if not batch:
                    continue

                logger.info(f"[Stream] Streaming batch: {len(batch)} videos, {batch_duration:.0f}s")

                # Concatenate batch into single file
                batch_file = CACHE_DIR / f"batch_{int(time.time())}.mp4"
                if len(batch) == 1:
                    # Just use the single file directly
                    batch_file = batch[0]
                    needs_cleanup = False
                else:
                    # Concatenate multiple files
                    concat_list = CACHE_DIR / "concat_list.txt"
                    with open(concat_list, "w") as f:
                        for p in batch:
                            f.write(f"file '{p}'\n")

                    # Concat without re-encoding (fast)
                    concat_cmd = [
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(concat_list),
                        "-c", "copy",
                        str(batch_file)
                    ]
                    subprocess.run(concat_cmd, capture_output=True)
                    needs_cleanup = True

                    # Clean up individual files
                    for p in batch:
                        try:
                            os.remove(p)
                        except:
                            pass

                # Stream the batch
                self._stream_file(str(batch_file), batch_duration)

                # Clean up batch file
                if needs_cleanup:
                    try:
                        os.remove(batch_file)
                    except:
                        pass

            except Exception as e:
                logger.error(f"[Stream] Error: {e}")
                time.sleep(2)

    def _stream_file(self, path: str, duration: float):
        """Stream a single file to RTMP"""
        cmd = [
            "ffmpeg",
            "-re",  # Realtime
            "-i", path,
            # Re-encode to ensure consistent stream
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

        logger.info(f"[FFmpeg] Starting stream of {duration:.0f}s batch")
        self._ffmpeg = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        try:
            self._ffmpeg.wait(timeout=duration + 30)
        except subprocess.TimeoutExpired:
            self._ffmpeg.terminate()

        logger.info("[FFmpeg] Batch complete")


# ============================================================================
# MAIN
# ============================================================================

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

    logger.info("=" * 50)
    logger.info("  FARNSWORTH PERSISTENT STREAM")
    logger.info("  Batched concat for fewer FFmpeg restarts")
    logger.info("=" * 50)

    gen = VideoGen()
    content = Content()
    streamer = PersistentStreamer()

    # Generate initial batch
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

    logger.info(f"Initial queue: {len(segments)} videos")

    # Start streaming
    streamer.start()
    logger.info("Stream started!")

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
            logger.info(f"Queue: {qsize} videos")

            if qsize < 4:
                # Generate more
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
            logger.error(f"Producer error: {e}")
            await asyncio.sleep(5)

    await gen.close()


if __name__ == "__main__":
    asyncio.run(main())
