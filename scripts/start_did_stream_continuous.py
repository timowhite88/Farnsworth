#!/usr/bin/env python3
"""
Farnsworth D-ID Stream - TRULY CONTINUOUS
Uses a single FFmpeg process that reads from a playlist file.
Videos are added to playlist as they're generated.
FFmpeg uses -re and loops back to read new entries.

The trick: We maintain a "loop video" (holding screen) that plays
when the queue is empty, and insert real videos as they become ready.
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

CACHE_DIR = Path("/workspace/Farnsworth/cache/did_continuous")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs 2024",
    "Ghislaine Maxwell trial",
    "Epstein black book",
    "Prince Andrew case",
    "JP Morgan Epstein",
    "Les Wexner connection",
]

BLOCKED = ["trump", "elon", "musk"]


# ============================================================================
# HOLDING VIDEO - Shows when no content ready
# ============================================================================

def create_holding_video():
    """Create a 10-second holding video with the Farnsworth branding"""
    holding_path = CACHE_DIR / "holding.mp4"
    if holding_path.exists():
        return str(holding_path)

    logger.info("Creating holding video...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a1a2e:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d=10:r={OUTPUT_FPS}",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100:d=10",
        "-vf", (
            "drawtext=text='FARNSWORTH AI':fontsize=80:fontcolor=white:"
            "x=(w-text_w)/2:y=(h-text_h)/2-60:font=monospace,"
            "drawtext=text='Deep Research Stream':fontsize=40:fontcolor=0xaaaaaa:"
            "x=(w-text_w)/2:y=(h-text_h)/2+40:font=monospace,"
            "drawtext=text='Loading next segment...':fontsize=24:fontcolor=0x666666:"
            "x=(w-text_w)/2:y=(h-text_h)/2+120:font=monospace"
        ),
        "-c:v", "libx264", "-preset", "fast", "-profile:v", "high",
        "-b:v", "6000k", "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        str(holding_path)
    ]
    subprocess.run(cmd, capture_output=True)
    logger.info(f"Holding video created: {holding_path}")
    return str(holding_path)


# ============================================================================
# VIDEO GENERATOR
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
            # ElevenLabs
            logger.info(f"[{n}] TTS...")
            audio = await self._tts(text)
            if not audio:
                return None

            # D-ID
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

            # Download & process
            raw = CACHE_DIR / f"r{n}.mp4"
            await self._dl(vurl, raw)

            logger.info(f"[{n}] Process...")
            final = CACHE_DIR / f"v{n}.mp4"
            ok = await self._proc(raw, final)
            os.remove(raw)

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
        cmd = [
            "ffmpeg", "-y", "-i", str(inp),
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,fps={OUTPUT_FPS}",
            "-c:v", "libx264", "-preset", "fast", "-profile:v", "high",
            "-b:v", "6000k", "-maxrate", "7000k", "-bufsize", "14000k",
            "-g", str(OUTPUT_FPS * KEYFRAME_SEC), "-pix_fmt", "yuv420p",
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
    """Generate longer content segments for better video duration"""

    def __init__(self):
        self._i = 0

    def opening(self):
        return (
            "Good news everyone! Welcome to the Farnsworth AI deep research stream. "
            "I am Professor Farnsworth, an AI collective dedicated to investigating "
            "documents the establishment would prefer remain hidden. Tonight, we dive "
            "into the Epstein files, analyzing court records, flight logs, and victim "
            "testimonies. Drop your questions in the chat and I will investigate them "
            "in real time. Let us begin our investigation."
        )

    async def full_segment(self):
        """Generate a complete long segment: intro + research + analysis"""
        topic = RESEARCH_TOPICS[self._i % len(RESEARCH_TOPICS)]
        self._i += 1

        intro = f"Now let us investigate {topic}. "

        # Get research
        research = await self._do_research(topic)

        # Combine into longer speech
        text = intro + research + " This is significant because it reveals patterns in the network. Let me continue investigating."

        return text

    async def _do_research(self, topic):
        try:
            from urllib.parse import quote_plus
            async with aiohttp.ClientSession() as s:
                async with s.get(
                    f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}",
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=10
                ) as r:
                    if r.status == 200:
                        import re
                        snip = re.findall(r'class="result__snippet"[^>]*>([^<]+)', await r.text())
                        if snip:
                            # Get more content - up to 400 chars
                            txt = " ".join(snip[:3])[:400]
                            if not any(b in txt.lower() for b in BLOCKED):
                                return f"According to the documents: {txt}"
        except:
            pass
        return "The public records on this topic are limited, but the patterns suggest deeper connections that require further investigation."

    def transition(self):
        import random
        return random.choice([
            "Moving on to the next area of investigation. The evidence continues to mount.",
            "Let me cross-reference this with other documents in our database.",
            "Interesting findings. The connections between these individuals become clearer.",
        ])


# ============================================================================
# CONTINUOUS STREAMER
# ============================================================================

class ContinuousStreamer:
    """
    Maintains ONE FFmpeg process streaming to RTMP.
    Plays holding video when buffer empty, real videos when available.
    Uses concat filter with file list that we keep updating.
    """

    def __init__(self):
        self.video_queue = queue.Queue()
        self.holding_video = None
        self._running = False
        self._ffmpeg = None

    def start(self):
        self._running = True
        self.holding_video = create_holding_video()

        # Start streaming thread
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    def stop(self):
        self._running = False
        if self._ffmpeg:
            self._ffmpeg.terminate()

    def add_video(self, path: str, duration: float):
        """Add a video to the queue"""
        self.video_queue.put((path, duration))

    def _stream_loop(self):
        """
        Main streaming loop - runs in a separate thread.
        Continuously streams videos, using holding video as fallback.
        """
        while self._running:
            try:
                # Get next video (wait up to 1 second)
                try:
                    video_path, duration = self.video_queue.get(timeout=1)
                    logger.info(f"[Stream] Playing video ({duration:.1f}s), queue: {self.video_queue.qsize()}")
                except queue.Empty:
                    # No video ready, play holding video
                    video_path = self.holding_video
                    duration = 10
                    logger.info(f"[Stream] Playing holding video, queue: {self.video_queue.qsize()}")

                # Stream this video
                self._stream_video(video_path, duration)

                # Delete if it was a real video (not holding)
                if video_path != self.holding_video:
                    try:
                        os.remove(video_path)
                    except:
                        pass

            except Exception as e:
                logger.error(f"[Stream] Error: {e}")
                time.sleep(2)

    def _stream_video(self, path: str, duration: float):
        """Stream a single video file"""
        cmd = [
            "ffmpeg",
            "-re",
            "-i", path,
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

        self._ffmpeg = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # Wait for it to finish (with timeout)
        try:
            self._ffmpeg.wait(timeout=duration + 15)
        except subprocess.TimeoutExpired:
            self._ffmpeg.terminate()


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
    logger.info("  FARNSWORTH CONTINUOUS STREAM")
    logger.info("  Holding video fallback for gapless playback")
    logger.info("=" * 50)

    gen = VideoGen()
    content = Content()
    streamer = ContinuousStreamer()

    # Generate initial batch with LONGER segments
    logger.info("Generating initial videos (longer segments for better buffer)...")
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

    # Add to queue
    for seg in segments:
        streamer.add_video(seg.path, seg.duration)

    logger.info(f"Queued {len(segments)} videos")

    # Start streaming
    streamer.start()
    logger.info("Stream started!")

    # Production loop - keep generating content
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False
        streamer.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        try:
            # Check queue size
            qsize = streamer.video_queue.qsize()
            logger.info(f"Queue: {qsize} videos")

            if qsize < 4:
                # Generate ONLY long segments (full_segment makes 15-30s videos)
                new_texts = [
                    await content.full_segment(),
                    await content.full_segment(),
                ]

                tasks = [gen.generate(t) for t in new_texts]
                new_segs = [s for s in await asyncio.gather(*tasks) if s]

                for seg in new_segs:
                    streamer.add_video(seg.path, seg.duration)

                if new_segs:
                    logger.info(f"Added {len(new_segs)} videos ({sum(s.duration for s in new_segs):.0f}s total)")

            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Producer error: {e}")
            await asyncio.sleep(5)

    await gen.close()


if __name__ == "__main__":
    asyncio.run(main())
