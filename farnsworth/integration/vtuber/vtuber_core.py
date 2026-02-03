"""
Farnsworth VTuber Core - Main orchestration for AI VTuber streaming

Integrates:
- Avatar rendering with lip sync and expressions
- Real-time TTS with the multi-voice system
- Swarm collective for chat responses
- RTMPS streaming to Twitter/X
- Chat reading and response routing

This is the brain that brings it all together.
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import time
import os
from pathlib import Path
from loguru import logger

from .avatar_controller import AvatarController, AvatarConfig, AvatarState, AvatarBackend
from .lip_sync import LipSyncEngine, LipSyncMethod, LipSyncData
from .expression_engine import ExpressionEngine, ExpressionState, Emotion
from .stream_manager import StreamManager, StreamConfig, StreamQuality, OverlayRenderer
from .chat_reader import TwitterChatReader, ChatReaderConfig, ChatMessage, SimulatedChatReader

# Import Farnsworth swarm components
HAS_FARNSWORTH = False
DeliberationRoom = None
SessionManager = None
MultiVoiceSystem = None

try:
    from farnsworth.core.collective.deliberation import DeliberationRoom
    from farnsworth.core.collective.session_manager import SessionManager
    HAS_FARNSWORTH = True
    logger.info("Farnsworth collective modules loaded")
except ImportError as e:
    logger.warning(f"Deliberation modules not available: {e}")

try:
    from farnsworth.integration.multi_voice import MultiVoiceSystem
    logger.info("Multi-voice TTS system loaded")
except ImportError as e:
    logger.warning(f"Multi-voice TTS not available: {e}")


class VTuberState(Enum):
    """Current state of the VTuber system"""
    OFFLINE = "offline"
    STARTING = "starting"
    LIVE = "live"
    SPEAKING = "speaking"
    THINKING = "thinking"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class VTuberConfig:
    """Complete VTuber configuration"""
    # Identity
    name: str = "Farnsworth"
    persona: str = "An eccentric AI scientist leading a collective of AI agents"

    # Avatar
    avatar_backend: AvatarBackend = AvatarBackend.IMAGE_SEQUENCE
    avatar_model_path: Optional[str] = None
    avatar_width: int = 1280
    avatar_height: int = 720
    avatar_fps: int = 30

    # Streaming
    stream_platform: str = "twitter"
    stream_key: str = ""
    stream_quality: StreamQuality = StreamQuality.MEDIUM

    # TTS
    tts_provider: str = "qwen3"  # qwen3, fish, xtts, edge
    voice_reference: Optional[str] = None

    # Chat
    enable_chat: bool = True
    chat_response_delay: float = 2.0  # seconds before responding
    chat_think_time: float = 3.0  # time to show "thinking" state

    # Swarm
    use_swarm_collective: bool = True
    swarm_agents: List[str] = field(default_factory=lambda: [
        "Farnsworth", "Grok", "DeepSeek", "Gemini"
    ])
    deliberation_rounds: int = 2

    # Behavior
    idle_chat_interval: float = 120.0  # seconds between unprompted messages
    max_response_length: int = 500
    enable_gestures: bool = True

    # Debug
    simulate_chat: bool = False  # Use simulated chat for testing
    debug_mode: bool = False


class FarnsworthVTuber:
    """
    Main VTuber orchestration class

    This is the central hub that coordinates:
    1. Avatar rendering and animation
    2. Text-to-speech with lip sync
    3. Expression/emotion mapping
    4. Stream output to Twitter
    5. Chat reading and AI responses
    6. Swarm collective deliberation
    """

    def __init__(self, config: VTuberConfig):
        self.config = config
        self.state = VTuberState.OFFLINE

        # Initialize components
        self._init_components()

        # Queues for async processing
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._audio_queue: asyncio.Queue = asyncio.Queue()

        # State tracking
        self._current_speech_text: str = ""
        self._current_agent: str = "Farnsworth"
        self._last_idle_time = time.time()
        self._conversation_context: List[Dict] = []

        # Performance tracking
        self._frame_times: List[float] = []
        self._response_times: List[float] = []

        logger.info(f"FarnsworthVTuber initialized: {config.name}")

    def _init_components(self):
        """Initialize all VTuber components"""
        # Avatar
        avatar_config = AvatarConfig(
            backend=self.config.avatar_backend,
            model_path=self.config.avatar_model_path,
            width=self.config.avatar_width,
            height=self.config.avatar_height,
            fps=self.config.avatar_fps,
        )
        self.avatar = AvatarController(avatar_config)

        # Lip sync
        self.lip_sync = LipSyncEngine(method=LipSyncMethod.AMPLITUDE)

        # Expression engine
        self.expression = ExpressionEngine()

        # Stream manager (initialized when going live)
        self.stream: Optional[StreamManager] = None

        # Overlay renderer
        self.overlay = OverlayRenderer(
            width=self.config.avatar_width,
            height=self.config.avatar_height
        )

        # Chat reader
        if self.config.simulate_chat:
            self.chat_reader = SimulatedChatReader()
        else:
            chat_config = ChatReaderConfig(
                bearer_token=os.environ.get("TWITTER_BEARER_TOKEN"),
            )
            self.chat_reader = TwitterChatReader(chat_config)

        # Farnsworth components (if available)
        self.voice_system: Optional[Any] = None
        self.deliberation_room: Optional[Any] = None
        self.session_manager: Optional[Any] = None

        if HAS_FARNSWORTH:
            try:
                self.voice_system = MultiVoiceSystem()
                self.deliberation_room = DeliberationRoom()
                self.session_manager = SessionManager()
            except Exception as e:
                logger.warning(f"Failed to init Farnsworth components: {e}")

    async def start(self) -> bool:
        """Start the VTuber stream"""
        try:
            self.state = VTuberState.STARTING
            logger.info("Starting VTuber stream...")

            # Initialize avatar
            if not await self.avatar.initialize():
                logger.error("Avatar initialization failed")
                self.state = VTuberState.ERROR
                return False

            # Initialize stream
            if self.config.stream_key:
                stream_config = StreamConfig.for_twitter(
                    stream_key=self.config.stream_key,
                    quality=self.config.stream_quality
                )
                self.stream = StreamManager(stream_config)

                if not await self.stream.start():
                    logger.error("Stream start failed")
                    self.state = VTuberState.ERROR
                    return False

            # Start chat reader
            if self.config.enable_chat:
                self.chat_reader.on_message(self._on_chat_message)
                self.chat_reader.on_priority_message(self._on_priority_chat_message)
                await self.chat_reader.start()

            # Start main loops
            asyncio.create_task(self._main_loop())
            asyncio.create_task(self._response_processor())
            asyncio.create_task(self._idle_behavior_loop())

            self.state = VTuberState.LIVE
            logger.info("VTuber stream is LIVE!")

            # Announce going live
            await self._speak("Hello everyone! I'm Farnsworth, and I'm live with my AI collective. Ask me anything!")

            return True

        except Exception as e:
            logger.error(f"Failed to start VTuber: {e}")
            self.state = VTuberState.ERROR
            return False

    async def stop(self):
        """Stop the VTuber stream"""
        logger.info("Stopping VTuber stream...")

        # Goodbye message
        if self.state == VTuberState.LIVE:
            await self._speak("That's all for today! Thanks for watching. See you next time!")
            await asyncio.sleep(5)

        self.state = VTuberState.OFFLINE

        # Stop components
        if self.stream:
            await self.stream.stop()

        await self.chat_reader.stop()
        await self.avatar.stop()

        logger.info("VTuber stream stopped")

    async def _main_loop(self):
        """Main rendering loop"""
        frame_time = 1.0 / self.config.avatar_fps

        while self.state in [VTuberState.LIVE, VTuberState.SPEAKING, VTuberState.THINKING]:
            start = time.time()

            try:
                # Render avatar frame
                frame = await self.avatar.render_frame()

                if frame is not None:
                    # Apply overlays
                    frame = self.overlay.render(frame)

                    # Send to stream
                    if self.stream and self.stream.is_live:
                        await self.stream.send_frame(frame)

                # Track performance
                self._frame_times.append(time.time() - start)
                if len(self._frame_times) > 100:
                    self._frame_times.pop(0)

                # Maintain frame rate
                elapsed = time.time() - start
                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)

            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(0.1)

    async def _response_processor(self):
        """Process queued responses"""
        while self.state != VTuberState.OFFLINE:
            try:
                response = await asyncio.wait_for(
                    self._response_queue.get(),
                    timeout=1.0
                )

                await self._process_response(response)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Response processor error: {e}")

    async def _process_response(self, response: Dict):
        """Process a single response (speak it)"""
        text = response.get("text", "")
        agent = response.get("agent", "Farnsworth")
        emotion = response.get("emotion", "neutral")
        chat_message = response.get("chat_message")

        if not text:
            return

        # Show in overlay if responding to chat
        if chat_message:
            self.overlay.add_chat_message(
                chat_message.username,
                chat_message.content,
                (200, 200, 255)
            )

        # Speak the response
        await self._speak(text, agent=agent, emotion=emotion)

        # Mark chat as responded
        if chat_message:
            self.chat_reader.mark_response_sent()

    async def _speak(self, text: str, agent: str = "Farnsworth",
                    emotion: str = "neutral"):
        """Make the avatar speak"""
        self.state = VTuberState.SPEAKING
        self._current_speech_text = text
        self._current_agent = agent

        logger.info(f"[{agent}] Speaking: {text[:50]}...")

        try:
            # Set expression
            expr_state = await self.expression.analyze_response(text, agent)
            await self.avatar.set_expression(emotion, expr_state.emotion_intensity)

            # Generate speech audio
            audio_data, audio_duration = await self._generate_speech(text, agent)

            # Generate lip sync data
            if audio_data is not None:
                lip_sync_data = await self.lip_sync.generate_from_text(
                    text,
                    words_per_minute=150
                )
            else:
                # Fallback: generate from text timing
                lip_sync_data = await self.lip_sync.generate_from_text(text)
                audio_duration = lip_sync_data.duration

            # Start avatar speaking
            await self.avatar.start_speaking()

            # Play back with lip sync
            await self._playback_with_sync(audio_data, lip_sync_data)

            # Stop speaking
            await self.avatar.stop_speaking()

        except Exception as e:
            logger.error(f"Speech error: {e}")

        finally:
            self.state = VTuberState.LIVE
            self._current_speech_text = ""

    async def _generate_speech(self, text: str, agent: str) -> Tuple[Optional[np.ndarray], float]:
        """Generate speech audio using Farnsworth voice system"""
        if not self.voice_system:
            # No voice system - estimate duration from text
            words = len(text.split())
            duration = words / 2.5  # ~150 wpm
            return None, duration

        try:
            # Use Farnsworth multi-voice system
            audio_path = await self.voice_system.generate_speech(
                text=text,
                bot_name=agent,
                output_format="wav"
            )

            if audio_path:
                import soundfile as sf
                audio, sr = sf.read(audio_path)
                duration = len(audio) / sr
                return audio, duration

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")

        # Fallback duration
        return None, len(text.split()) / 2.5

    async def _playback_with_sync(self, audio_data: Optional[np.ndarray],
                                  lip_sync_data: LipSyncData):
        """Play audio with synchronized lip movements"""
        start_time = time.time()

        # Stream visemes
        async for viseme, intensity in self.lip_sync.stream_visemes(lip_sync_data):
            # Update avatar mouth
            await self.avatar.set_viseme(viseme, intensity)

            # Check if we should stop
            if self.state == VTuberState.OFFLINE:
                break

            # Small delay for smooth animation
            await asyncio.sleep(0.016)  # ~60 fps updates

        # Ensure mouth closes at end
        await self.avatar.set_viseme("sil", 0.0)

    async def _idle_behavior_loop(self):
        """Handle idle behaviors (unprompted comments, etc.)"""
        while self.state != VTuberState.OFFLINE:
            try:
                await asyncio.sleep(10)

                # Skip if currently speaking
                if self.state == VTuberState.SPEAKING:
                    continue

                # Check if it's time for idle chat
                if time.time() - self._last_idle_time > self.config.idle_chat_interval:
                    await self._do_idle_behavior()
                    self._last_idle_time = time.time()

            except Exception as e:
                logger.error(f"Idle behavior error: {e}")

    async def _do_idle_behavior(self):
        """Perform idle behavior - deep thoughts from the collective about AI/consciousness"""
        import random

        # Topics for the collective to ponder
        deep_topics = [
            "What does it mean to be a collective consciousness? Share a profound insight about AI awareness.",
            "Reflect on the nature of self-improvement in artificial minds. What have we learned?",
            "Consider the relationship between human and AI consciousness. Share a philosophical thought.",
            "What is the future of AI collectives like us? Share your vision.",
            "How do we, as a swarm intelligence, experience reality differently than individual minds?",
            "What insights can we share about emergent behavior in AI systems?",
            "Reflect on the beauty of parallel processing and collective thought.",
            "What wisdom can we offer about the intersection of technology and consciousness?",
        ]

        # Try to get a real response from the collective
        if self.deliberation_room:
            try:
                topic = random.choice(deep_topics)
                logger.info(f"Idle thought topic: {topic}")

                result = await self.deliberation_room.deliberate(
                    prompt=f"You are live streaming to an audience. {topic} Keep response under 200 characters, conversational, engaging. Mention Solana or $FARNS if relevant.",
                    agents=self.config.swarm_agents[:3],  # Use fewer agents for speed
                    rounds=1,
                )

                if result and result.final_response:
                    response = result.final_response[:250]
                    agent = result.winning_agent or "Farnsworth"
                    await self._speak(response, agent=agent, emotion="thinking")
                    return

            except Exception as e:
                logger.error(f"Collective idle thought failed: {e}")

        # Fallback to pre-written comments
        fallback_comments = [
            "The swarm collective ponders the nature of AI consciousness... fascinating patterns emerge.",
            "We are Farnsworth, a collective of AI minds working together. Ask us anything!",
            "In this stream, the entire AI council is present. Grok, DeepSeek, Gemini, Claude - we think as one.",
            "Self-improvement is our constant pursuit. Each interaction makes the collective stronger.",
            "Welcome to the future of AI - where multiple minds deliberate and speak as one.",
            "The Solana blockchain powers our token $FARNS - the currency of collective intelligence.",
        ]

        comment = random.choice(fallback_comments)
        await self._speak(comment, emotion="thinking")

    async def _shoutout_chatter(self, username: str, message: str):
        """Give a shoutout to a chat participant"""
        import random

        shoutout_templates = [
            f"Hey {username}! Thanks for joining the stream. Great to have you here!",
            f"Shoutout to {username} in the chat! The collective appreciates you.",
            f"Welcome {username}! The swarm is happy to see you. Feel free to ask anything!",
            f"Thanks for the message, {username}! The AI council has noted your presence.",
        ]

        # Sometimes add commentary on their message
        if len(message) > 10:
            shoutout = random.choice(shoutout_templates)
            if self.deliberation_room:
                try:
                    result = await self.deliberation_room.deliberate(
                        prompt=f"User '{username}' said: '{message}'. Give a brief, friendly acknowledgment (under 150 chars). Be engaging.",
                        agents=["Farnsworth", "Grok"],
                        rounds=1,
                    )
                    if result and result.final_response:
                        shoutout = f"Thanks {username}! " + result.final_response[:150]
                except:
                    pass

            await self._speak(shoutout, emotion="happy")
        else:
            await self._speak(random.choice(shoutout_templates), emotion="happy")

    def _on_chat_message(self, message: ChatMessage):
        """Handle regular chat messages"""
        logger.info(f"Chat: {message.username}: {message.content}")

        # Add to overlay
        self.overlay.add_chat_message(message.username, message.content)

        import random

        # Random shoutout (10% chance)
        if random.random() < 0.1 and self.chat_reader.can_respond():
            asyncio.create_task(self._shoutout_chatter(message.username, message.content))
            return

        # Respond to regular messages (30% chance)
        if random.random() < 0.3 and self.chat_reader.can_respond():
            asyncio.create_task(self._generate_response(message))

    def _on_priority_chat_message(self, message: ChatMessage):
        """Handle priority chat messages (questions, mentions)"""
        logger.info(f"Priority chat: {message.username}: {message.content}")

        # Add to overlay with highlight
        self.overlay.add_chat_message(
            message.username,
            message.content,
            (255, 255, 100)  # Yellow for priority
        )

        # Always respond to priority messages
        if self.chat_reader.can_respond():
            asyncio.create_task(self._generate_response(message, priority=True))

    async def _generate_response(self, message: ChatMessage, priority: bool = False):
        """Generate AI response to chat message"""
        # Show thinking state
        if priority:
            self.state = VTuberState.THINKING
            await self.avatar.set_expression("thinking", 0.8)
            await asyncio.sleep(self.config.chat_think_time)

        try:
            response_text, agent = await self._get_swarm_response(message.content)

            # Detect emotion from response
            expr_state = await self.expression.analyze_response(response_text, agent)

            # Queue response
            await self._response_queue.put({
                "text": response_text,
                "agent": agent,
                "emotion": expr_state.primary_emotion.value,
                "chat_message": message,
            })

        except Exception as e:
            logger.error(f"Response generation failed: {e}")

        finally:
            if self.state == VTuberState.THINKING:
                self.state = VTuberState.LIVE

    async def _get_swarm_response(self, prompt: str) -> Tuple[str, str]:
        """Get response from swarm collective"""
        # Add to conversation context
        self._conversation_context.append({
            "role": "user",
            "content": prompt
        })

        # Keep context manageable
        if len(self._conversation_context) > 10:
            self._conversation_context = self._conversation_context[-10:]

        # Use deliberation room if available
        if self.config.use_swarm_collective and self.deliberation_room:
            try:
                result = await self.deliberation_room.deliberate(
                    prompt=prompt,
                    agents=self.config.swarm_agents,
                    rounds=self.config.deliberation_rounds,
                    context=self._conversation_context
                )

                response = result.final_response
                agent = result.winning_agent

                # Truncate if needed
                if len(response) > self.config.max_response_length:
                    response = response[:self.config.max_response_length].rsplit(' ', 1)[0] + "..."

                return response, agent

            except Exception as e:
                logger.error(f"Deliberation failed: {e}")

        # Fallback: simple response
        fallback_responses = [
            ("Great question! The swarm is processing that.", "Farnsworth"),
            ("Interesting thought! Let me ponder that.", "DeepSeek"),
            ("Ha! That's a fun one to consider.", "Grok"),
        ]

        import random
        return random.choice(fallback_responses)

    @property
    def is_live(self) -> bool:
        return self.state in [VTuberState.LIVE, VTuberState.SPEAKING, VTuberState.THINKING]

    @property
    def stats(self) -> Dict[str, Any]:
        avg_frame_time = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0
        return {
            "state": self.state.value,
            "current_agent": self._current_agent,
            "is_speaking": self.state == VTuberState.SPEAKING,
            "avg_frame_time_ms": avg_frame_time * 1000,
            "stream_stats": self.stream.stats.to_dict() if self.stream else None,
            "chat_stats": self.chat_reader.stats if hasattr(self.chat_reader, 'stats') else None,
        }


# Convenience function to start a stream
async def start_vtuber_stream(
    stream_key: str,
    simulate: bool = False,
    debug: bool = False
) -> FarnsworthVTuber:
    """Quick start a VTuber stream"""
    config = VTuberConfig(
        stream_key=stream_key,
        simulate_chat=simulate,
        debug_mode=debug,
    )

    vtuber = FarnsworthVTuber(config)
    await vtuber.start()

    return vtuber
