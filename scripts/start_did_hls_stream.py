#!/usr/bin/env python3
"""
Farnsworth D-ID Stream - HLS MODE
Writes HLS segments to disk, Twitter pulls from playlist URL.

Better for segmented video workflow:
1. Generate D-ID avatar videos
2. Convert to HLS .ts segments
3. Update m3u8 playlist
4. Twitter pulls from public URL

No RTMP connection issues - just serve files!
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

# HLS Settings
HLS_OUTPUT_DIR = Path("/workspace/Farnsworth/farnsworth/web/static/hls")
HLS_PUBLIC_URL = "https://ai.farnsworth.cloud/static/hls"
HLS_SEGMENT_DURATION = 6  # seconds per segment
HLS_PLAYLIST_SIZE = 30    # segments in playlist - LARGE buffer so Twitter doesn't lose track

# Video settings
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30
KEYFRAME_SEC = 2

# Cache for raw D-ID videos
CACHE_DIR = Path("/workspace/Farnsworth/cache/did_hls")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
# HLS PLAYLIST MANAGER
# ============================================================================

class HLSPlaylistManager:
    """Manages the HLS playlist and segment files"""

    def __init__(self, output_dir: Path, segment_duration: int = 6, max_segments: int = 10):
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.max_segments = max_segments
        self.segments: List[str] = []
        self.sequence = 0

        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clean old segments
        self._clean_old_segments()

    def _clean_old_segments(self):
        """Remove old HLS files"""
        for f in self.output_dir.glob("segment_*.ts"):
            f.unlink()
        playlist = self.output_dir / "stream.m3u8"
        if playlist.exists():
            playlist.unlink()

    def add_segment(self, video_path: str, duration: float) -> Optional[str]:
        """
        Convert a video file to HLS segment(s) and add to playlist.
        Returns the segment filename or None on error.
        """
        seg_name = f"segment_{self.sequence:05d}.ts"
        seg_path = self.output_dir / seg_name

        # Convert video to MPEG-TS segment
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-profile:v", "main",
            "-b:v", "2500k",
            "-maxrate", "3000k",
            "-bufsize", "6000k",
            "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-ac", "2",
            "-f", "mpegts",
            str(seg_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr.decode()[:200]}")
                return None
        except Exception as e:
            logger.error(f"Segment conversion error: {e}")
            return None

        # Get actual segment duration
        actual_duration = self._get_duration(seg_path)
        if actual_duration <= 0:
            actual_duration = duration

        # Add to playlist
        self.segments.append((seg_name, actual_duration))
        self.sequence += 1

        # Trim old segments - keep extra on disk for Twitter buffering
        # Only remove from playlist, keep files for 2x playlist duration
        while len(self.segments) > self.max_segments:
            old_name, _ = self.segments.pop(0)
            # DON'T delete files immediately - Twitter may still need them
            # Files will be cleaned up on next restart

        # Update playlist file
        self._write_playlist()

        logger.info(f"[HLS] Added segment {seg_name} ({actual_duration:.1f}s)")
        return seg_name

    def _get_duration(self, path: Path) -> float:
        """Get video duration using ffprobe"""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return float(result.stdout.strip())
        except:
            return 0.0

    def _write_playlist(self):
        """Write the m3u8 playlist file - EVENT type for stability"""
        playlist_path = self.output_dir / "stream.m3u8"

        # Calculate target duration (max segment duration)
        max_dur = max(d for _, d in self.segments) if self.segments else self.segment_duration

        # Use EVENT playlist type - segments are never removed
        # This is more stable for Twitter pulling
        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{int(max_dur) + 1}",
            "#EXT-X-PLAYLIST-TYPE:EVENT",  # EVENT = segments never removed
            "#EXT-X-MEDIA-SEQUENCE:0",     # Always start at 0
        ]

        for seg_name, duration in self.segments:
            lines.append(f"#EXTINF:{duration:.3f},")
            lines.append(seg_name)

        # Write playlist
        with open(playlist_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        logger.debug(f"[HLS] Playlist updated: {len(self.segments)} segments")

    @property
    def playlist_url(self) -> str:
        """Get the public playlist URL"""
        return f"{HLS_PUBLIC_URL}/stream.m3u8"


# ============================================================================
# D-ID VIDEO GENERATOR
# ============================================================================

@dataclass
class Segment:
    path: str
    duration: float
    text: str


class VideoGen:
    """Generate D-ID avatar videos"""

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
        """Generate a D-ID video from text"""
        await self._session_ok()
        self._n += 1
        n = self._n

        if any(b in text.lower() for b in BLOCKED) or len(text) < 20:
            return None

        try:
            logger.info(f"[{n}] TTS ({len(text)} chars)...")
            audio = await self._tts(text)
            if not audio:
                logger.error(f"[{n}] TTS failed")
                return None

            logger.info(f"[{n}] Uploading audio to D-ID...")
            audio_url = await self._did_upload(audio, n)
            if not audio_url:
                logger.error(f"[{n}] Audio upload failed")
                return None

            logger.info(f"[{n}] Creating D-ID talk...")
            talk_id = await self._did_talk(audio_url)
            if not talk_id:
                logger.error(f"[{n}] Talk creation failed")
                return None

            logger.info(f"[{n}] Waiting for D-ID video...")
            video_url = await self._did_wait(talk_id)
            if not video_url:
                logger.error(f"[{n}] Video generation failed")
                return None

            # Download raw video
            raw = CACHE_DIR / f"raw_{n}.mp4"
            await self._download(video_url, raw)

            # Process to standard format
            logger.info(f"[{n}] Processing video...")
            final = CACHE_DIR / f"video_{n}.mp4"
            if not await self._process(raw, final):
                return None

            # Cleanup raw
            if raw.exists():
                raw.unlink()

            duration = self._get_duration(final)
            logger.info(f"[{n}] Ready: {duration:.1f}s video")

            return Segment(str(final), duration, text)

        except Exception as e:
            logger.error(f"[{n}] Error: {e}")
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
                if r.status == 200:
                    return await r.read()
                else:
                    logger.error(f"TTS error {r.status}: {await r.text()}")
                    return None
        except Exception as e:
            logger.error(f"TTS exception: {e}")
            return None

    async def _did_upload(self, audio_data: bytes, n: int) -> Optional[str]:
        """Upload audio to D-ID"""
        form = aiohttp.FormData()
        form.add_field("audio", audio_data, filename=f"audio_{n}.mp3", content_type="audio/mpeg")

        async with self._session.post(
            "https://api.d-id.com/audios",
            data=form,
            headers={"Authorization": f"Basic {DID_API_KEY}"}
        ) as r:
            if r.status == 201:
                data = await r.json()
                return data.get("url")
            else:
                logger.error(f"D-ID upload error {r.status}: {await r.text()}")
                return None

    async def _did_talk(self, audio_url: str) -> Optional[str]:
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
                data = await r.json()
                return data.get("id")
            else:
                logger.error(f"D-ID talk error {r.status}: {await r.text()}")
                return None

    async def _did_wait(self, talk_id: str, timeout: int = 120) -> Optional[str]:
        """Wait for D-ID video to complete"""
        start = time.time()
        while time.time() - start < timeout:
            async with self._session.get(
                f"https://api.d-id.com/talks/{talk_id}",
                headers={"Authorization": f"Basic {DID_API_KEY}"}
            ) as r:
                data = await r.json()
                status = data.get("status")

                if status == "done":
                    return data.get("result_url")
                elif status == "error":
                    logger.error(f"D-ID error: {data}")
                    return None

            await asyncio.sleep(2)

        logger.error("D-ID timeout")
        return None

    async def _download(self, url: str, path: Path):
        """Download file"""
        async with self._session.get(url) as r:
            with open(path, "wb") as f:
                f.write(await r.read())

    async def _process(self, inp: Path, out: Path) -> bool:
        """Process video to standard format"""
        cmd = [
            "ffmpeg", "-y", "-i", str(inp),
            "-vf", f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a2e,"
                   f"fps={OUTPUT_FPS}",
            "-c:v", "libx264", "-preset", "fast", "-profile:v", "main",
            "-b:v", "2500k", "-maxrate", "3000k", "-bufsize", "6000k",
            "-g", str(OUTPUT_FPS * KEYFRAME_SEC),
            "-pix_fmt", "yuv420p",
            "-ac", "2", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            str(out)
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await proc.wait()
        return proc.returncode == 0

    def _get_duration(self, path: Path) -> float:
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

class Content:
    """Generate research content"""

    def __init__(self):
        self._idx = 0

    def opening(self) -> str:
        return (
            "Good news everyone! Welcome to the Farnsworth AI deep research stream. "
            "I am Professor Farnsworth, investigating documents the establishment "
            "prefers remain hidden. Tonight we dive into the Epstein files. "
            "Drop questions in the chat and I will investigate them live."
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
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Farnsworth D-ID HLS Stream")
    parser.add_argument("--initial-segments", type=int, default=4, help="Segments to pre-generate")
    args = parser.parse_args()

    # Validate config
    if not all([DID_API_KEY, DID_AVATAR_URL, ELEVENLABS_API_KEY]):
        logger.error("Missing API keys in .env")
        logger.error("Required: DID_API_KEY, DID_AVATAR_URL, ELEVENLABS_API_KEY")
        return

    logger.info("=" * 60)
    logger.info("  FARNSWORTH D-ID HLS STREAM")
    logger.info("  Video segments served via HLS")
    logger.info("=" * 60)
    logger.info(f"  HLS Directory: {HLS_OUTPUT_DIR}")
    logger.info(f"  Playlist URL: {HLS_PUBLIC_URL}/stream.m3u8")
    logger.info("=" * 60)

    # Initialize
    hls = HLSPlaylistManager(HLS_OUTPUT_DIR, HLS_SEGMENT_DURATION, HLS_PLAYLIST_SIZE)
    gen = VideoGen()
    content = Content()

    # Generate initial segments
    logger.info(f"Pre-generating {args.initial_segments} segments...")

    texts = [content.opening()]
    for _ in range(args.initial_segments - 1):
        texts.append(await content.segment())

    # Generate videos in parallel
    tasks = [gen.generate(t) for t in texts]
    segments = [s for s in await asyncio.gather(*tasks) if s]

    if not segments:
        logger.error("Failed to generate initial segments")
        return

    # Add to HLS playlist
    for seg in segments:
        hls.add_segment(seg.path, seg.duration)
        # Cleanup processed video
        Path(seg.path).unlink(missing_ok=True)

    logger.info(f"Initial playlist ready: {len(segments)} segments")
    logger.info(f"Playlist URL: {hls.playlist_url}")
    logger.info("")
    logger.info("Point Twitter Media Studio to this URL to start streaming!")
    logger.info("")

    # Production loop
    running = True

    def stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        try:
            # Generate new segments to maintain playlist
            logger.info(f"[HLS] Playlist: {len(hls.segments)} segments")

            # Generate 2 new segments
            new_texts = [await content.segment(), await content.segment()]
            tasks = [gen.generate(t) for t in new_texts]
            new_segs = [s for s in await asyncio.gather(*tasks) if s]

            for seg in new_segs:
                hls.add_segment(seg.path, seg.duration)
                Path(seg.path).unlink(missing_ok=True)

            # Generate continuously - don't wait long
            # The more segments we have buffered, the more stable the stream
            logger.info(f"[HLS] Continuing generation...")
            await asyncio.sleep(5)  # Brief pause between batches

        except Exception as e:
            logger.error(f"Error: {e}")
            await asyncio.sleep(10)

    await gen.close()
    logger.info("Stream ended")


if __name__ == "__main__":
    asyncio.run(main())
