#!/usr/bin/env python3
"""
Farnsworth D-ID Stream - MPEG-TS PIPE ARCHITECTURE
True seamless streaming using named pipe + TS concatenation.

Key insight: MPEG-TS is designed for byte-level appending.
- One FFmpeg process reads from named pipe forever
- Convert D-ID MP4s to TS segments
- Cat TS bytes directly to pipe - instant, seamless
- No restarts, no reloads, no gaps

Based on broadcast-grade 24/7 streaming architecture.
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
from datetime import datetime

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

# Twitter/X RTMP
RTMP_URL = "rtmp://va.pscp.tv:80/x"
STREAM_KEY = ""

# Video specs (X-optimized)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30
VIDEO_BITRATE = "4000k"
KEYFRAME_SEC = 2  # X wants keyframes every 2-4s

# Pipe and cache
PIPE_PATH = Path("/tmp/avatar_stream.ts")
CACHE_DIR = Path("/workspace/Farnsworth/cache/did_ts_pipe")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Buffer settings
MIN_BUFFER_CLIPS = 4   # Minimum clips before going live
MAX_BUFFER_CLIPS = 10  # Maximum clips to pre-generate
IDLE_CLIP_DURATION = 15  # Seconds for idle/thinking clip

RESEARCH_TOPICS = [
    "Jeffrey Epstein flight logs 2024",
    "Ghislaine Maxwell trial revelations",
    "Epstein black book contacts",
    "Prince Andrew case details",
    "JP Morgan Epstein banking",
    "Les Wexner connection",
    "Epstein island visitor records",
    "Victim testimonies court documents",
]

BLOCKED = ["trump", "elon", "musk"]


# ============================================================================
# IDLE CLIP GENERATOR
# ============================================================================

def create_idle_clip() -> Path:
    """Create a pre-rendered idle/thinking clip (avatar subtle movement)"""
    idle_path = CACHE_DIR / "idle_clip.mp4"
    idle_ts = CACHE_DIR / "idle_clip.ts"

    if idle_ts.exists():
        return idle_ts

    logger.info("Creating idle clip (one-time)...")

    # Generate a simple idle video with text overlay
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x1a1a2e:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:d={IDLE_CLIP_DURATION}:r={OUTPUT_FPS}",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:d={IDLE_CLIP_DURATION}",
        "-vf", (
            f"drawtext=text='FARNSWORTH AI':fontsize=72:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2-80:font=monospace,"
            f"drawtext=text='Deep Research Stream':fontsize=36:fontcolor=0xaaaaaa:"
            f"x=(w-text_w)/2:y=(h-text_h)/2+20:font=monospace,"
            f"drawtext=text='Researching...':fontsize=24:fontcolor=0x666666:"
            f"x=(w-text_w)/2:y=(h-text_h)/2+100:font=monospace"
        ),
        "-c:v", "libx264", "-preset", "fast", "-profile:v", "main",
        "-b:v", VIDEO_BITRATE, "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        str(idle_path)
    ]
    subprocess.run(cmd, capture_output=True)

    # Convert to TS
    ts_cmd = [
        "ffmpeg", "-y", "-i", str(idle_path),
        "-c", "copy", "-f", "mpegts",
        "-bsf:v", "h264_mp4toannexb",
        str(idle_ts)
    ]
    subprocess.run(ts_cmd, capture_output=True)

    logger.info(f"Idle clip ready: {idle_ts}")
    return idle_ts


# ============================================================================
# D-ID VIDEO GENERATOR
# ============================================================================

@dataclass
class Clip:
    mp4_path: Path
    ts_path: Path
    duration: float
    text: str


class ClipGenerator:
    """Generate D-ID avatar clips with ElevenLabs voice"""

    def __init__(self):
        self._session = None
        self._n = 0

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session:
            await self._session.close()

    async def generate(self, text: str) -> Optional[Clip]:
        """Generate a D-ID clip from text"""
        await self._ensure_session()
        self._n += 1
        n = self._n

        # Content filter
        if any(b in text.lower() for b in BLOCKED) or len(text) < 20:
            return None

        try:
            # 1. ElevenLabs TTS
            logger.info(f"[Clip {n}] TTS ({len(text)} chars)...")
            audio = await self._tts(text)
            if not audio:
                return None

            # 2. Upload audio to D-ID
            logger.info(f"[Clip {n}] Upload to D-ID...")
            audio_url = await self._did_upload_audio(audio, n)
            if not audio_url:
                return None

            # 3. Create D-ID talk
            logger.info(f"[Clip {n}] D-ID talk...")
            talk_id = await self._did_create_talk(audio_url)
            if not talk_id:
                return None

            # 4. Wait for completion
            logger.info(f"[Clip {n}] Waiting for video...")
            video_url = await self._did_wait(talk_id)
            if not video_url:
                return None

            # 5. Download MP4
            mp4_path = CACHE_DIR / f"clip_{n}.mp4"
            await self._download(video_url, mp4_path)

            # 6. Process to standard format
            logger.info(f"[Clip {n}] Processing...")
            processed_path = CACHE_DIR / f"processed_{n}.mp4"
            if not await self._process_video(mp4_path, processed_path):
                return None

            # 7. Convert to TS (the key step!)
            ts_path = CACHE_DIR / f"clip_{n}.ts"
            if not self._convert_to_ts(processed_path, ts_path):
                return None

            # Get duration
            duration = self._get_duration(ts_path)

            # Cleanup intermediate files
            mp4_path.unlink(missing_ok=True)
            processed_path.unlink(missing_ok=True)

            logger.info(f"[Clip {n}] Ready: {duration:.1f}s TS segment")
            return Clip(processed_path, ts_path, duration, text)

        except Exception as e:
            logger.error(f"[Clip {n}] Error: {e}")
            return None

    async def _tts(self, text: str) -> Optional[bytes]:
        """ElevenLabs TTS"""
        try:
            async with self._session.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
                },
                headers={
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as r:
                return await r.read() if r.status == 200 else None
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    async def _did_upload_audio(self, audio: bytes, n: int) -> Optional[str]:
        """Upload audio to D-ID (required - D-ID needs HTTPS URL)"""
        form = aiohttp.FormData()
        form.add_field("audio", audio, filename=f"audio_{n}.mp3", content_type="audio/mpeg")

        async with self._session.post(
            "https://api.d-id.com/audios",
            data=form,
            headers={"Authorization": f"Basic {DID_API_KEY}"}
        ) as r:
            if r.status == 201:
                return (await r.json()).get("url")
            logger.error(f"D-ID upload failed: {r.status}")
            return None

    async def _did_create_talk(self, audio_url: str) -> Optional[str]:
        """Create D-ID talk"""
        async with self._session.post(
            "https://api.d-id.com/talks",
            json={
                "source_url": DID_AVATAR_URL,
                "script": {"type": "audio", "audio_url": audio_url},
                "config": {"fluent": True},
                "driver_url": "bank://lively"
            },
            headers={
                "Authorization": f"Basic {DID_API_KEY}",
                "Content-Type": "application/json"
            }
        ) as r:
            if r.status == 201:
                return (await r.json()).get("id")
            logger.error(f"D-ID talk failed: {r.status}")
            return None

    async def _did_wait(self, talk_id: str, timeout: int = 120) -> Optional[str]:
        """Wait for D-ID video completion"""
        start = time.time()
        while time.time() - start < timeout:
            async with self._session.get(
                f"https://api.d-id.com/talks/{talk_id}",
                headers={"Authorization": f"Basic {DID_API_KEY}"}
            ) as r:
                data = await r.json()
                if data.get("status") == "done":
                    return data.get("result_url")
                if data.get("status") == "error":
                    logger.error(f"D-ID error: {data}")
                    return None
            await asyncio.sleep(2)
        return None

    async def _download(self, url: str, path: Path):
        """Download file"""
        async with self._session.get(url) as r:
            with open(path, "wb") as f:
                f.write(await r.read())

    async def _process_video(self, inp: Path, out: Path) -> bool:
        """Process video to X-compatible format"""
        cmd = [
            "ffmpeg", "-y", "-i", str(inp),
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,"
                   f"fps={OUTPUT_FPS}",
            "-c:v", "libx264", "-preset", "fast", "-profile:v", "main",
            "-b:v", VIDEO_BITRATE, "-maxrate", VIDEO_BITRATE, "-bufsize", "8000k",
            "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-keyint_min", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            str(out)
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()
        return proc.returncode == 0

    def _convert_to_ts(self, mp4_path: Path, ts_path: Path) -> bool:
        """Convert MP4 to MPEG-TS (the seamless format)"""
        cmd = [
            "ffmpeg", "-y", "-i", str(mp4_path),
            "-c", "copy",  # No re-encode
            "-f", "mpegts",
            "-bsf:v", "h264_mp4toannexb",  # Required for TS compatibility
            str(ts_path)
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0

    def _get_duration(self, path: Path) -> float:
        """Get video duration"""
        try:
            cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 10.0


# ============================================================================
# CONTENT GENERATOR
# ============================================================================

class ContentGenerator:
    """Generate research content"""

    def __init__(self):
        self._idx = 0

    def opening(self) -> str:
        return (
            "Good news everyone! Welcome to the Farnsworth AI deep research stream. "
            "I am Professor Farnsworth, investigating documents the establishment "
            "prefers remain hidden. Tonight we dive into the Epstein files. "
            "Drop questions in the chat."
        )

    async def segment(self) -> str:
        """Generate a content segment"""
        topic = RESEARCH_TOPICS[self._idx % len(RESEARCH_TOPICS)]
        self._idx += 1
        research = await self._research(topic)
        return f"Now investigating {topic}. {research} The patterns reveal deeper connections."

    async def _research(self, topic: str) -> str:
        """Search for research content"""
        try:
            from urllib.parse import quote_plus
            async with aiohttp.ClientSession() as s:
                url = f"https://html.duckduckgo.com/html/?q={quote_plus(topic)}"
                async with s.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10) as r:
                    if r.status == 200:
                        import re
                        html = await r.text()
                        snippets = re.findall(r'class="result__snippet"[^>]*>([^<]+)', html)
                        if snippets:
                            text = " ".join(snippets[:3])[:400]
                            if not any(b in text.lower() for b in BLOCKED):
                                return f"According to the documents: {text}"
        except:
            pass
        return "Records are limited but patterns suggest deeper connections."


# ============================================================================
# TS PIPE STREAMER - THE CORE INNOVATION
# ============================================================================

class TSPipeStreamer:
    """
    Seamless streaming via MPEG-TS named pipe.

    Architecture:
    1. Create named pipe (FIFO)
    2. FFmpeg reads from pipe forever, pushes to RTMP
    3. Cat TS segments to pipe - instant append, no gaps
    4. Idle clip fills silence
    """

    def __init__(self, stream_key: str):
        self.stream_key = stream_key
        self.pipe_path = PIPE_PATH
        self.clip_queue = queue.Queue()
        self.idle_clip = None

        self._running = False
        self._ffmpeg = None
        self._feeder_thread = None
        self._idle_thread = None
        self._feeding_clip = threading.Event()

    def start(self):
        """Start the streaming pipeline"""
        self._running = True

        # Create idle clip
        self.idle_clip = create_idle_clip()

        # Create named pipe
        if self.pipe_path.exists():
            self.pipe_path.unlink()
        os.mkfifo(str(self.pipe_path))
        logger.info(f"Created named pipe: {self.pipe_path}")

        # Start FFmpeg reader (runs forever)
        self._start_ffmpeg()

        # Start feeder thread
        self._feeder_thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._feeder_thread.start()

        # Start idle filler thread
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()

        logger.info("TS Pipe Streamer started!")

    def stop(self):
        """Stop streaming"""
        self._running = False
        if self._ffmpeg:
            self._ffmpeg.terminate()
        logger.info("Streamer stopped")

    def add_clip(self, clip: Clip):
        """Add a clip to the queue"""
        self.clip_queue.put(clip)
        logger.info(f"[Queue] Added clip ({clip.duration:.1f}s), queue: {self.clip_queue.qsize()}")

    def _start_ffmpeg(self):
        """Start the eternal FFmpeg pusher"""
        cmd = [
            "ffmpeg",
            "-re",  # Read at realtime speed
            "-fflags", "+genpts",  # Generate PTS for smooth playback
            "-f", "mpegts",
            "-i", str(self.pipe_path),
            # Output encoding
            "-c", "copy",  # No re-encode (already encoded)
            "-bsf:v", "h264_mp4toannexb",
            # Force keyframes for X compatibility
            "-r", str(OUTPUT_FPS),
            "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-keyint_min", str(OUTPUT_FPS * KEYFRAME_SEC),
            # Audio
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-ac", "2",
            # Output to RTMP
            "-f", "flv",
            "-flvflags", "no_duration_filesize",
            f"{RTMP_URL}/{self.stream_key}"
        ]

        logger.info("[FFmpeg] Starting eternal RTMP pusher...")
        logger.info(f"[FFmpeg] Output: {RTMP_URL}/***")

        self._ffmpeg = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        # Monitor thread
        def monitor():
            while self._running:
                if self._ffmpeg and self._ffmpeg.poll() is not None:
                    stderr = self._ffmpeg.stderr.read().decode() if self._ffmpeg.stderr else ""
                    logger.error(f"[FFmpeg] Died: {stderr[:200]}")
                    if self._running:
                        logger.info("[FFmpeg] Restarting...")
                        time.sleep(2)
                        self._start_ffmpeg()
                time.sleep(5)

        threading.Thread(target=monitor, daemon=True).start()

    def _feed_loop(self):
        """Feed clips to the pipe"""
        logger.info("[Feeder] Starting feed loop...")

        # Open pipe for writing (blocks until FFmpeg opens read end)
        with open(str(self.pipe_path), 'wb') as pipe:
            logger.info("[Feeder] Pipe opened, ready to stream!")

            while self._running:
                try:
                    # Get next clip
                    try:
                        clip = self.clip_queue.get(timeout=1)
                    except queue.Empty:
                        continue

                    logger.info(f"[Feeder] Feeding clip ({clip.duration:.1f}s)...")

                    # Signal that we're feeding a clip (pause idle)
                    self._feeding_clip.set()

                    # Cat the TS file to pipe - THIS IS THE MAGIC
                    # No re-encoding, no restarts, just byte append
                    try:
                        with open(clip.ts_path, 'rb') as ts_file:
                            while True:
                                chunk = ts_file.read(65536)  # 64KB chunks
                                if not chunk:
                                    break
                                pipe.write(chunk)
                                pipe.flush()
                    except Exception as e:
                        logger.error(f"[Feeder] Write error: {e}")

                    # Cleanup TS file
                    clip.ts_path.unlink(missing_ok=True)

                    # Clear feeding flag
                    self._feeding_clip.clear()

                    logger.info(f"[Feeder] Clip fed, queue: {self.clip_queue.qsize()}")

                except Exception as e:
                    logger.error(f"[Feeder] Error: {e}")
                    time.sleep(1)

    def _idle_loop(self):
        """Feed idle clip when queue is empty"""
        logger.info("[Idle] Starting idle filler...")

        # Wait for pipe to be ready
        time.sleep(5)

        # Open pipe for writing
        with open(str(self.pipe_path), 'wb') as pipe:
            while self._running:
                try:
                    # Only feed idle if queue is empty and not currently feeding
                    if self.clip_queue.empty() and not self._feeding_clip.is_set():
                        logger.info("[Idle] Queue empty, feeding idle clip...")

                        with open(self.idle_clip, 'rb') as ts_file:
                            while True:
                                # Check if real clip arrived
                                if not self.clip_queue.empty() or self._feeding_clip.is_set():
                                    break
                                chunk = ts_file.read(65536)
                                if not chunk:
                                    break
                                try:
                                    pipe.write(chunk)
                                    pipe.flush()
                                except:
                                    break

                    time.sleep(0.5)

                except Exception as e:
                    logger.debug(f"[Idle] {e}")
                    time.sleep(1)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Farnsworth D-ID TS Pipe Stream")
    parser.add_argument("--stream-key", required=True, help="Twitter stream key")
    parser.add_argument("--initial-clips", type=int, default=4, help="Clips to pre-generate")
    args = parser.parse_args()

    # Validate config
    if not all([DID_API_KEY, DID_AVATAR_URL, ELEVENLABS_API_KEY]):
        logger.error("Missing API keys in .env")
        return

    logger.info("=" * 60)
    logger.info("  FARNSWORTH D-ID STREAM - TS PIPE ARCHITECTURE")
    logger.info("  True seamless streaming via MPEG-TS concatenation")
    logger.info("=" * 60)

    # Initialize
    gen = ClipGenerator()
    content = ContentGenerator()
    streamer = TSPipeStreamer(args.stream_key)

    # Pre-generate initial clips
    logger.info(f"Pre-generating {args.initial_clips} clips...")

    texts = [content.opening()]
    for _ in range(args.initial_clips - 1):
        texts.append(await content.segment())

    # Generate in parallel
    tasks = [gen.generate(t) for t in texts]
    clips = [c for c in await asyncio.gather(*tasks) if c]

    if len(clips) < 2:
        logger.error("Failed to generate enough initial clips")
        return

    # Queue clips
    for clip in clips:
        streamer.add_clip(clip)

    logger.info(f"Initial buffer: {len(clips)} clips")

    # Start streaming
    streamer.start()

    # Production loop
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False
        streamer.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  STREAM IS LIVE - TS PIPE ACTIVE")
    logger.info("  Clips append seamlessly to pipe")
    logger.info("=" * 60)

    while running:
        try:
            qsize = streamer.clip_queue.qsize()
            logger.info(f"[Status] Queue: {qsize} clips")

            # Generate more if buffer low
            if qsize < MIN_BUFFER_CLIPS:
                logger.info("[Producer] Buffer low, generating more clips...")
                new_texts = [await content.segment(), await content.segment()]
                tasks = [gen.generate(t) for t in new_texts]
                new_clips = [c for c in await asyncio.gather(*tasks) if c]

                for clip in new_clips:
                    streamer.add_clip(clip)

            await asyncio.sleep(10)

        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(5)

    await gen.close()
    logger.info("Stream ended")


if __name__ == "__main__":
    asyncio.run(main())
