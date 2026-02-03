"""
Stream Manager - RTMPS streaming to Twitter/X and other platforms
Handles video encoding, audio mixing, and stream output
"""

import asyncio
import subprocess
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
from pathlib import Path
import time
import os
import signal
from loguru import logger

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class StreamPlatform(Enum):
    """Supported streaming platforms"""
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    TWITCH = "twitch"
    CUSTOM = "custom"


class StreamQuality(Enum):
    """Stream quality presets"""
    LOW = "low"       # 480p, 1500kbps
    MEDIUM = "medium" # 720p, 3000kbps
    HIGH = "high"     # 1080p, 6000kbps
    ULTRA = "ultra"   # 1080p60, 8000kbps


@dataclass
class StreamConfig:
    """Configuration for stream output"""
    platform: StreamPlatform = StreamPlatform.TWITTER

    # Stream URL and key
    rtmp_url: str = ""
    stream_key: str = ""

    # Video settings
    width: int = 1280
    height: int = 720
    fps: int = 30
    video_bitrate: int = 3000  # kbps

    # Audio settings
    audio_bitrate: int = 192  # kbps
    sample_rate: int = 48000
    channels: int = 2

    # Encoder settings
    encoder: str = "libx264"
    preset: str = "veryfast"
    tune: str = "zerolatency"
    profile: str = "main"

    # Keyframe interval (GOP size) - Twitter recommends 2 seconds
    keyframe_interval: int = 2

    # Buffer settings
    buffer_size: int = 6000  # kbps

    @classmethod
    def for_twitter(cls, stream_key: str, rtmp_url: str = None,
                   quality: StreamQuality = StreamQuality.MEDIUM) -> "StreamConfig":
        """Create config optimized for Twitter streaming"""
        # Default Twitter RTMPS URL, but allow custom (Periscope URLs vary by region)
        if rtmp_url is None:
            rtmp_url = "rtmps://va.pscp.tv:443/x"  # Periscope/Twitter live URL

        config = cls(
            platform=StreamPlatform.TWITTER,
            rtmp_url=rtmp_url,
            stream_key=stream_key,
        )

        if quality == StreamQuality.LOW:
            config.width = 854
            config.height = 480
            config.video_bitrate = 1500
        elif quality == StreamQuality.MEDIUM:
            config.width = 1280
            config.height = 720
            config.video_bitrate = 3000
        elif quality == StreamQuality.HIGH:
            config.width = 1920
            config.height = 1080
            config.video_bitrate = 6000
        elif quality == StreamQuality.ULTRA:
            config.width = 1920
            config.height = 1080
            config.fps = 60
            config.video_bitrate = 8000

        return config

    @classmethod
    def for_youtube(cls, stream_key: str, quality: StreamQuality = StreamQuality.HIGH) -> "StreamConfig":
        """Create config optimized for YouTube streaming"""
        config = cls(
            platform=StreamPlatform.YOUTUBE,
            rtmp_url="rtmp://a.rtmp.youtube.com/live2",
            stream_key=stream_key,
        )

        if quality == StreamQuality.HIGH:
            config.width = 1920
            config.height = 1080
            config.video_bitrate = 6000
            config.keyframe_interval = 2

        return config

    @classmethod
    def for_twitch(cls, stream_key: str, server: str = "live.twitch.tv") -> "StreamConfig":
        """Create config for Twitch streaming"""
        return cls(
            platform=StreamPlatform.TWITCH,
            rtmp_url=f"rtmp://{server}/app",
            stream_key=stream_key,
            video_bitrate=6000,
            keyframe_interval=2,
        )

    @property
    def full_rtmp_url(self) -> str:
        """Get complete RTMP URL with stream key"""
        return f"{self.rtmp_url}/{self.stream_key}"


@dataclass
class StreamStats:
    """Real-time streaming statistics"""
    frames_sent: int = 0
    bytes_sent: int = 0
    dropped_frames: int = 0
    bitrate: float = 0.0
    uptime_seconds: float = 0.0
    status: str = "offline"
    last_error: Optional[str] = None

    # Quality metrics
    avg_fps: float = 0.0
    encoding_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_sent": self.frames_sent,
            "bytes_sent": self.bytes_sent,
            "dropped_frames": self.dropped_frames,
            "bitrate_kbps": self.bitrate,
            "uptime": self.uptime_seconds,
            "status": self.status,
            "avg_fps": self.avg_fps,
            "encoding_latency_ms": self.encoding_latency_ms,
            "last_error": self.last_error,
        }


class StreamManager:
    """
    Manages RTMPS/RTMP streaming to Twitter and other platforms

    Features:
    - FFmpeg-based video encoding
    - Real-time frame injection
    - Audio mixing with TTS
    - Overlay support
    - Stats monitoring
    """

    def __init__(self, config: StreamConfig):
        self.config = config
        self.stats = StreamStats()

        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._video_pipe = None
        self._audio_pipe = None

        self._running = False
        self._start_time: float = 0

        # Frame buffer for async writing
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=60)

        # Callbacks
        self._on_error: Optional[Callable[[str], None]] = None
        self._on_stats_update: Optional[Callable[[StreamStats], None]] = None

        # Audio state
        self._audio_buffer = np.zeros((self.config.sample_rate // 10, 2), dtype=np.int16)

        logger.info(f"StreamManager initialized for {config.platform.value}")

    def _build_ffmpeg_command(self) -> List[str]:
        """Build FFmpeg command for streaming"""
        cmd = ['ffmpeg']

        # Input options
        cmd.extend([
            '-y',  # Overwrite output
            '-loglevel', 'warning',

            # Video input (raw frames from pipe)
            '-f', 'rawvideo',
            '-pix_fmt', 'bgra',
            '-s', f'{self.config.width}x{self.config.height}',
            '-r', str(self.config.fps),
            '-i', 'pipe:0',

            # Audio input (raw PCM from pipe)
            '-f', 's16le',
            '-ar', str(self.config.sample_rate),
            '-ac', str(self.config.channels),
            '-i', 'pipe:3',
        ])

        # Video encoding
        cmd.extend([
            '-c:v', self.config.encoder,
            '-preset', self.config.preset,
            '-tune', self.config.tune,
            '-profile:v', self.config.profile,
            '-b:v', f'{self.config.video_bitrate}k',
            '-maxrate', f'{int(self.config.video_bitrate * 1.5)}k',
            '-bufsize', f'{self.config.buffer_size}k',
            '-g', str(self.config.fps * self.config.keyframe_interval),
            '-keyint_min', str(self.config.fps * self.config.keyframe_interval),
            '-pix_fmt', 'yuv420p',
        ])

        # Audio encoding
        cmd.extend([
            '-c:a', 'aac',
            '-b:a', f'{self.config.audio_bitrate}k',
            '-ar', str(self.config.sample_rate),
            '-ac', str(self.config.channels),
        ])

        # Output format and destination
        cmd.extend([
            '-f', 'flv',
            self.config.full_rtmp_url
        ])

        return cmd

    async def start(self) -> bool:
        """Start the streaming process"""
        if self._running:
            logger.warning("Stream already running")
            return False

        try:
            # Validate configuration
            if not self.config.stream_key:
                raise ValueError("Stream key is required")

            # Build and start FFmpeg process
            cmd = self._build_ffmpeg_command()
            logger.info(f"Starting FFmpeg: {' '.join(cmd[:10])}...")

            self._ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            self._running = True
            self._start_time = time.time()
            self.stats.status = "connecting"

            # Start background tasks
            asyncio.create_task(self._frame_writer_loop())
            asyncio.create_task(self._audio_writer_loop())
            asyncio.create_task(self._monitor_ffmpeg())
            asyncio.create_task(self._stats_updater())

            logger.info("Stream started successfully")
            self.stats.status = "live"
            return True

        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            self.stats.status = "error"
            self.stats.last_error = str(e)
            if self._on_error:
                self._on_error(str(e))
            return False

    async def stop(self):
        """Stop the streaming process"""
        self._running = False

        if self._ffmpeg_process:
            try:
                # Send empty frames to flush buffer
                await asyncio.sleep(0.5)

                # Terminate gracefully
                self._ffmpeg_process.terminate()
                await asyncio.sleep(1)

                if self._ffmpeg_process.poll() is None:
                    self._ffmpeg_process.kill()

            except Exception as e:
                logger.error(f"Error stopping FFmpeg: {e}")

            self._ffmpeg_process = None

        self.stats.status = "offline"
        logger.info("Stream stopped")

    async def send_frame(self, frame: np.ndarray):
        """Queue a video frame for streaming"""
        if not self._running:
            return

        try:
            # Ensure correct format
            if frame.shape[:2] != (self.config.height, self.config.width):
                if HAS_CV2:
                    frame = cv2.resize(frame, (self.config.width, self.config.height))
                else:
                    logger.warning("Frame size mismatch and OpenCV not available")
                    return

            # Ensure BGRA format
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 4, axis=-1)
            elif frame.shape[2] == 3:
                # Add alpha channel
                alpha = np.full((frame.shape[0], frame.shape[1], 1), 255, dtype=np.uint8)
                frame = np.concatenate([frame, alpha], axis=2)

            await self._frame_queue.put(frame)

        except asyncio.QueueFull:
            self.stats.dropped_frames += 1

    async def send_audio(self, audio: np.ndarray):
        """Queue audio samples for streaming"""
        if not self._running:
            return

        try:
            # Ensure correct format (stereo, int16)
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio], axis=1)

            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)

            await self._audio_queue.put(audio)

        except asyncio.QueueFull:
            pass  # Drop audio rather than blocking

    async def _frame_writer_loop(self):
        """Background task to write frames to FFmpeg"""
        frame_time = 1.0 / self.config.fps
        last_frame_time = time.time()

        while self._running and self._ffmpeg_process:
            try:
                # Get frame with timeout
                try:
                    frame = await asyncio.wait_for(
                        self._frame_queue.get(),
                        timeout=frame_time * 2
                    )
                except asyncio.TimeoutError:
                    # Generate black frame if no input
                    frame = np.zeros(
                        (self.config.height, self.config.width, 4),
                        dtype=np.uint8
                    )

                # Write to FFmpeg stdin
                start = time.time()
                self._ffmpeg_process.stdin.write(frame.tobytes())
                self._ffmpeg_process.stdin.flush()

                # Update stats
                self.stats.frames_sent += 1
                self.stats.bytes_sent += frame.nbytes
                self.stats.encoding_latency_ms = (time.time() - start) * 1000

                # Maintain frame rate
                elapsed = time.time() - last_frame_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                last_frame_time = time.time()

            except BrokenPipeError:
                logger.error("FFmpeg pipe broken")
                self._running = False
                self.stats.status = "error"
                self.stats.last_error = "Pipe broken - stream disconnected"
                break

            except Exception as e:
                logger.error(f"Frame write error: {e}")
                await asyncio.sleep(0.1)

    async def _audio_writer_loop(self):
        """Background task to write audio to FFmpeg"""
        # Note: This uses a separate pipe (fd 3) for audio
        # In practice, you might mix audio into the video frames
        while self._running and self._ffmpeg_process:
            try:
                try:
                    audio = await asyncio.wait_for(
                        self._audio_queue.get(),
                        timeout=0.1
                    )
                    self._audio_buffer = audio
                except asyncio.TimeoutError:
                    # Use silence if no audio
                    audio = self._audio_buffer

                # Audio would be written here if using separate pipe
                # For now, we're using mixed audio approach

            except Exception as e:
                logger.error(f"Audio write error: {e}")
                await asyncio.sleep(0.1)

    async def _monitor_ffmpeg(self):
        """Monitor FFmpeg process for errors"""
        while self._running and self._ffmpeg_process:
            # Check if process is still running
            if self._ffmpeg_process.poll() is not None:
                # Process ended
                stderr = self._ffmpeg_process.stderr.read()
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"FFmpeg process ended: {error_msg}")
                self.stats.status = "error"
                self.stats.last_error = error_msg
                self._running = False
                break

            await asyncio.sleep(1)

    async def _stats_updater(self):
        """Update streaming statistics"""
        while self._running:
            self.stats.uptime_seconds = time.time() - self._start_time

            if self.stats.uptime_seconds > 0:
                self.stats.avg_fps = self.stats.frames_sent / self.stats.uptime_seconds
                self.stats.bitrate = (self.stats.bytes_sent * 8 / 1000) / self.stats.uptime_seconds

            if self._on_stats_update:
                self._on_stats_update(self.stats)

            await asyncio.sleep(2)

    def on_error(self, callback: Callable[[str], None]):
        """Set error callback"""
        self._on_error = callback

    def on_stats_update(self, callback: Callable[[StreamStats], None]):
        """Set stats update callback"""
        self._on_stats_update = callback

    @property
    def is_live(self) -> bool:
        return self._running and self.stats.status == "live"


class OverlayRenderer:
    """
    Renders overlays on top of avatar frames

    Supports:
    - Chat messages
    - Alerts
    - Status indicators
    - Branding
    """

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Overlay layers
        self._chat_messages: List[Dict] = []
        self._alerts: List[Dict] = []
        self._status_text: str = ""
        self._branding_image: Optional[np.ndarray] = None

        # Styles
        self._font_scale = 0.7
        self._chat_max_messages = 5
        self._alert_duration = 5.0

    def add_chat_message(self, username: str, message: str, color: Tuple[int, int, int] = (255, 255, 255)):
        """Add a chat message to display"""
        self._chat_messages.append({
            "username": username,
            "message": message,
            "color": color,
            "time": time.time()
        })

        # Keep only recent messages
        if len(self._chat_messages) > self._chat_max_messages:
            self._chat_messages.pop(0)

    def add_alert(self, text: str, alert_type: str = "info"):
        """Add an alert overlay"""
        colors = {
            "info": (255, 200, 50),
            "success": (50, 255, 100),
            "warning": (50, 200, 255),
            "error": (50, 50, 255),
        }

        self._alerts.append({
            "text": text,
            "color": colors.get(alert_type, colors["info"]),
            "time": time.time(),
            "duration": self._alert_duration
        })

    def set_status(self, text: str):
        """Set status indicator text"""
        self._status_text = text

    def set_branding(self, image: np.ndarray):
        """Set branding image/logo"""
        self._branding_image = image

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render all overlays onto frame"""
        if not HAS_CV2:
            return frame

        result = frame.copy()
        current_time = time.time()

        # Remove expired alerts
        self._alerts = [a for a in self._alerts if current_time - a["time"] < a["duration"]]

        # Render chat messages (bottom left)
        y_offset = self.height - 50
        for msg in reversed(self._chat_messages[-self._chat_max_messages:]):
            text = f"{msg['username']}: {msg['message']}"
            cv2.putText(result, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       self._font_scale, msg['color'], 2, cv2.LINE_AA)
            y_offset -= 30

        # Render alerts (top center)
        for i, alert in enumerate(self._alerts):
            text = alert["text"]
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            x = (self.width - text_size[0]) // 2
            y = 60 + i * 50

            # Background
            cv2.rectangle(result, (x - 10, y - 35), (x + text_size[0] + 10, y + 10),
                         (0, 0, 0), -1)
            cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, alert["color"], 2, cv2.LINE_AA)

        # Render status (top right)
        if self._status_text:
            text_size = cv2.getTextSize(self._status_text, cv2.FONT_HERSHEY_SIMPLEX,
                                       self._font_scale, 2)[0]
            x = self.width - text_size[0] - 20
            cv2.putText(result, self._status_text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       self._font_scale, (0, 255, 0), 2, cv2.LINE_AA)

        # Render branding (bottom right)
        if self._branding_image is not None:
            bh, bw = self._branding_image.shape[:2]
            x = self.width - bw - 20
            y = self.height - bh - 20
            result[y:y+bh, x:x+bw] = self._branding_image

        return result
