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
    TWITTER_HLS = "twitter_hls"  # HLS pull-based for Twitter
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

    # HLS output settings
    hls_output_dir: str = "/workspace/Farnsworth/farnsworth/web/static/hls"
    hls_segment_duration: int = 2  # seconds per segment
    hls_playlist_size: int = 6  # number of segments in playlist (12 sec buffer)
    hls_base_url: str = "https://ai.farnsworth.cloud/static/hls"  # Public URL

    # Video settings
    width: int = 854
    height: int = 480
    fps: int = 24
    video_bitrate: int = 1500  # kbps

    # Audio settings
    audio_bitrate: int = 128  # kbps
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
            config.width = 640
            config.height = 360
            config.video_bitrate = 800
        elif quality == StreamQuality.MEDIUM:
            config.width = 854
            config.height = 480
            config.video_bitrate = 1500
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

    @classmethod
    def for_twitter_hls(cls, hls_output_dir: str = None,
                        hls_base_url: str = None) -> "StreamConfig":
        """Create config for Twitter HLS pull-based streaming"""
        config = cls(
            platform=StreamPlatform.TWITTER_HLS,
            width=854,
            height=480,
            fps=24,
            video_bitrate=1500,
            audio_bitrate=128,
        )
        if hls_output_dir:
            config.hls_output_dir = hls_output_dir
        if hls_base_url:
            config.hls_base_url = hls_base_url
        return config

    @property
    def hls_playlist_url(self) -> str:
        """Get the public HLS playlist URL for Twitter to pull"""
        return f"{self.hls_base_url}/stream.m3u8"


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

        # Audio file management for TTS
        self._current_audio_file: Optional[str] = None
        self._audio_queue_files: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._temp_audio_dir = Path(os.environ.get('TEMP', '/tmp')) / 'farnsworth_stream_audio'
        self._temp_audio_dir.mkdir(parents=True, exist_ok=True)
        self._audio_restarting = False  # Flag to prevent monitor conflicts

        # Frame buffer for async writing
        self._frame_queue: asyncio.Queue = asyncio.Queue(maxsize=30)
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=60)

        # Callbacks
        self._on_error: Optional[Callable[[str], None]] = None
        self._on_stats_update: Optional[Callable[[StreamStats], None]] = None

        # Audio state
        self._audio_buffer = np.zeros((self.config.sample_rate // 10, 2), dtype=np.int16)

        logger.info(f"StreamManager initialized for {config.platform.value}")

    def _build_ffmpeg_command(self, audio_file: Optional[str] = None) -> List[str]:
        """Build FFmpeg command for streaming with stability settings"""
        cmd = ['ffmpeg']

        # Input options
        cmd.extend([
            '-y',  # Overwrite output
            '-loglevel', 'warning',
            '-threads', '4',

            # Video input (raw frames from pipe)
            '-f', 'rawvideo',
            '-pix_fmt', 'bgra',
            '-s', f'{self.config.width}x{self.config.height}',
            '-r', str(self.config.fps),
            '-i', 'pipe:0',
        ])

        # Audio input - use file if provided, otherwise generate silence
        if audio_file and os.path.exists(audio_file):
            # Use the audio file (play once, then we'll restart with silence)
            cmd.extend([
                '-i', audio_file,
            ])
        else:
            # Use lavfi to generate silence (anullsrc)
            cmd.extend([
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
            ])

        # Video encoding - Twitter Media Studio specs
        gop_size = self.config.fps * self.config.keyframe_interval  # 3 sec keyframes
        cmd.extend([
            '-c:v', self.config.encoder,
            '-preset', 'medium',  # Better quality
            '-tune', 'zerolatency',
            '-profile:v', 'high',  # High profile for better compression
            '-level', '4.2',
            # CBR-like encoding for consistent bitrate
            '-b:v', f'{self.config.video_bitrate}k',
            '-minrate', f'{self.config.video_bitrate}k',  # Force minimum = target
            '-maxrate', f'{self.config.video_bitrate}k',  # Force maximum = target
            '-bufsize', f'{self.config.video_bitrate * 2}k',  # 2 second buffer for stability
            '-g', str(gop_size),  # Keyframe interval (3 sec = 90 frames at 30fps)
            '-keyint_min', str(gop_size),  # Force exact keyframe interval
            '-sc_threshold', '0',  # Disable scene change detection
            '-pix_fmt', 'yuv420p',
            '-flags', '+cgop',  # Closed GOP for better streaming
            '-x264opts', f'keyint={gop_size}:min-keyint={gop_size}:no-scenecut',
        ])

        # Audio encoding - Twitter optimized with volume boost
        cmd.extend([
            '-af', 'volume=2.5',  # Boost volume 2.5x for better audibility
            '-c:a', 'aac',
            '-b:a', f'{self.config.audio_bitrate}k',
            '-ar', '44100',  # Twitter prefers 44.1kHz
            '-ac', '2',
            '-profile:a', 'aac_low',
        ])

        # Output format
        if self.config.platform == StreamPlatform.TWITTER_HLS:
            # HLS output for pull-based streaming
            hls_dir = Path(self.config.hls_output_dir)
            hls_dir.mkdir(parents=True, exist_ok=True)

            cmd.extend([
                '-f', 'hls',
                '-hls_time', str(self.config.hls_segment_duration),
                '-hls_list_size', str(self.config.hls_playlist_size),
                '-hls_flags', 'delete_segments+append_list',
                '-hls_segment_filename', str(hls_dir / 'segment_%03d.ts'),
                str(hls_dir / 'stream.m3u8')
            ])
        else:
            # RTMP/RTMPS output
            if audio_file and os.path.exists(audio_file):
                # When using audio file, stop when audio ends
                cmd.extend([
                    '-shortest',
                ])

            cmd.extend([
                '-f', 'flv',
                '-flvflags', 'no_duration_filesize',
                self.config.full_rtmp_url
            ])

        return cmd

    async def start(self) -> bool:
        """Start the streaming process"""
        if self._running:
            logger.warning("Stream already running")
            return False

        try:
            # Validate configuration (stream_key not needed for HLS)
            if self.config.platform != StreamPlatform.TWITTER_HLS and not self.config.stream_key:
                raise ValueError("Stream key is required")

            # Build and start FFmpeg process (with silence initially)
            cmd = self._build_ffmpeg_command(self._current_audio_file)
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

    async def queue_audio_file(self, audio_file: str):
        """Queue an audio file (WAV/MP3) to be played in stream"""
        if not self._running:
            return

        if os.path.exists(audio_file):
            await self._audio_queue_files.put(audio_file)
            logger.info(f"Queued audio file: {audio_file}")

    def save_audio_to_file(self, audio: np.ndarray, sample_rate: int = 44100) -> str:
        """Save audio numpy array to temp WAV file and return path"""
        import wave
        import struct

        # Ensure correct format
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=1)

        if audio.dtype != np.int16:
            if audio.dtype == np.float32 or audio.dtype == np.float64:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        # Generate unique filename
        filename = self._temp_audio_dir / f"tts_{int(time.time() * 1000)}.wav"

        # Write WAV file
        with wave.open(str(filename), 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())

        return str(filename)

    async def _frame_writer_loop(self):
        """Background task to write frames to FFmpeg"""
        frame_time = 1.0 / self.config.fps
        last_frame_time = time.time()

        while self._running:
            try:
                # Wait for FFmpeg process to be ready
                if not self._ffmpeg_process or self._audio_restarting:
                    await asyncio.sleep(0.1)
                    continue

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

                # Double-check process is still valid
                if not self._ffmpeg_process or not self._ffmpeg_process.stdin:
                    await asyncio.sleep(0.1)
                    continue

                # Write to FFmpeg stdin
                start = time.time()
                try:
                    self._ffmpeg_process.stdin.write(frame.tobytes())
                    self._ffmpeg_process.stdin.flush()
                except (BrokenPipeError, OSError):
                    # Pipe closed during audio restart, just skip
                    await asyncio.sleep(0.1)
                    continue

                # Update stats
                self.stats.frames_sent += 1
                self.stats.bytes_sent += frame.nbytes
                self.stats.encoding_latency_ms = (time.time() - start) * 1000

                # Maintain frame rate
                elapsed = time.time() - last_frame_time
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                last_frame_time = time.time()

            except Exception as e:
                if "closed file" not in str(e).lower() and "broken pipe" not in str(e).lower():
                    logger.error(f"Frame write error: {e}")
                await asyncio.sleep(0.1)

    async def _audio_writer_loop(self):
        """Background task to handle audio files for TTS playback"""
        # Process queued audio files and restart FFmpeg with them
        while self._running:
            try:
                try:
                    # Check for queued audio files
                    audio_file = await asyncio.wait_for(
                        self._audio_queue_files.get(),
                        timeout=1.0
                    )

                    if audio_file and os.path.exists(audio_file):
                        logger.info(f"Switching to TTS audio: {audio_file}")
                        self._current_audio_file = audio_file

                        # Restart FFmpeg with the new audio file
                        await self._restart_with_audio(audio_file)

                        # Get audio duration and wait for it to finish
                        try:
                            import wave
                            with wave.open(audio_file, 'rb') as wf:
                                frames = wf.getnframes()
                                rate = wf.getframerate()
                                duration = frames / rate
                                logger.info(f"Audio duration: {duration:.1f}s")
                                await asyncio.sleep(duration + 0.5)
                        except Exception as e:
                            logger.debug(f"Could not get audio duration: {e}")
                            await asyncio.sleep(5)  # Default wait

                        # Return to silence after audio finishes
                        self._current_audio_file = None
                        await self._restart_with_audio(None)
                        logger.info("Returned to silence")

                        # Clean up old temp files (keep last 5)
                        try:
                            temp_files = sorted(self._temp_audio_dir.glob("tts_*.wav"))
                            for old_file in temp_files[:-5]:
                                old_file.unlink()
                        except Exception:
                            pass

                except asyncio.TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Audio handler error: {e}")
                await asyncio.sleep(0.1)

    async def _restart_with_audio(self, audio_file: Optional[str]):
        """Restart FFmpeg with new audio source (file or silence)"""
        if not self._running:
            return

        self._audio_restarting = True
        old_process = self._ffmpeg_process
        self._ffmpeg_process = None  # Clear reference so frame writer pauses

        try:
            # Give frame writer time to see the cleared process
            await asyncio.sleep(0.1)

            # Terminate old process gracefully
            if old_process:
                try:
                    old_process.stdin.close()
                except:
                    pass
                old_process.terminate()
                await asyncio.sleep(0.3)
                if old_process.poll() is None:
                    old_process.kill()
                await asyncio.sleep(0.2)

            # Start new FFmpeg with updated audio
            cmd = self._build_ffmpeg_command(audio_file)
            new_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )

            # Set the new process
            self._ffmpeg_process = new_process
            logger.info(f"FFmpeg restarted with audio: {audio_file or 'silence'}")

        except Exception as e:
            logger.error(f"FFmpeg restart failed: {e}")
        finally:
            self._audio_restarting = False

    async def _monitor_ffmpeg(self):
        """Monitor FFmpeg process for errors with auto-reconnect"""
        reconnect_attempts = 0
        max_reconnects = 5

        while self._running:
            # Skip monitoring during audio restarts
            if self._audio_restarting:
                await asyncio.sleep(1)
                continue

            if not self._ffmpeg_process or self._ffmpeg_process.poll() is not None:
                # Skip if audio handler is managing the restart
                if self._current_audio_file:
                    # Audio file playback ended normally, let audio handler restart
                    await asyncio.sleep(1)
                    continue

                # Process ended or not running
                if self._ffmpeg_process:
                    stderr = self._ffmpeg_process.stderr.read() if self._ffmpeg_process.stderr else b""
                    error_msg = stderr.decode() if stderr else "Unknown error"
                    logger.error(f"FFmpeg process ended: {error_msg[:200]}")
                    self.stats.last_error = error_msg[:200]

                # Try to reconnect
                if reconnect_attempts < max_reconnects:
                    reconnect_attempts += 1
                    logger.info(f"Attempting reconnect {reconnect_attempts}/{max_reconnects}...")
                    self.stats.status = "reconnecting"

                    await asyncio.sleep(3)  # Wait before reconnect

                    try:
                        # Restart FFmpeg (with current audio file if any)
                        cmd = self._build_ffmpeg_command(self._current_audio_file)
                        self._ffmpeg_process = subprocess.Popen(
                            cmd,
                            stdin=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=0,
                        )
                        self.stats.status = "live"
                        logger.info("Reconnected successfully")
                        reconnect_attempts = 0  # Reset on success
                    except Exception as e:
                        logger.error(f"Reconnect failed: {e}")
                else:
                    logger.error("Max reconnect attempts reached")
                    self.stats.status = "error"
                    self._running = False
                    break
            else:
                # Process running, reset counter
                reconnect_attempts = 0

            await asyncio.sleep(2)

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
