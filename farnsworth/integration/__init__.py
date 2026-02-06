"""
Farnsworth Integration Module

External tool and capability integration with:
- Dynamic tool routing and capability matching
- Multimodal input processing (vision, audio)
- Composio integration for 500+ tools
- Vision understanding (CLIP/BLIP)
- Voice interaction (Whisper/TTS)
"""

# Lazy imports to avoid circular/missing dependency crashes
try:
    from farnsworth.integration.tool_router import ToolRouter
except ImportError:
    ToolRouter = None

try:
    from farnsworth.integration.multimodal import MultimodalProcessor
except ImportError:
    MultimodalProcessor = None

# Vision capabilities (lazy)
try:
    from farnsworth.integration.vision import (
        VisionModule,
        VisionTask,
        VisionResult,
        ImageInput,
        SceneGraph,
    )
except ImportError:
    VisionModule = None
    VisionTask = None
    VisionResult = None
    ImageInput = None
    SceneGraph = None

# Voice capabilities (lazy)
try:
    from farnsworth.integration.voice import (
        VoiceModule,
        TranscriptionResult,
        TranscriptionSegment,
        VoiceCommand,
        AudioInput,
    )
except ImportError:
    VoiceModule = None
    TranscriptionResult = None
    TranscriptionSegment = None
    VoiceCommand = None
    AudioInput = None

__all__ = [
    "ToolRouter",
    "MultimodalProcessor",
    # Vision
    "VisionModule",
    "VisionTask",
    "VisionResult",
    "ImageInput",
    "SceneGraph",
    # Voice
    "VoiceModule",
    "TranscriptionResult",
    "TranscriptionSegment",
    "VoiceCommand",
    "AudioInput",
]
