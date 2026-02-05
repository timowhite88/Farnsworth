"""
Farnsworth OpenClaw Compatibility Layer
========================================

Universal adapter for running OpenClaw skills and tools in Farnsworth.

This Shadow Layer provides:
- Skill parser (reads SKILL.md format)
- Tool mapper (translates OpenClaw calls to Farnsworth agents)
- Device nodes (camera, screen, location, notifications)
- Visual canvas (A2UI-compatible live workspace)
- Voice interface (speech-to-text, text-to-speech)
- Session coordination (spawning, messaging, history)

Architecture:
    OpenClaw Skill Call
           ↓
    Shadow Layer (Adapter)
           ↓
    Translate → Map to Farnsworth Agent/Tool
           ↓
    Farnsworth Swarm / Agent Execution
           ↓
    Return result in OpenClaw format

"Two claws are better than one." - The Collective
"""

from .openclaw_adapter import (
    OpenClawAdapter,
    get_openclaw_adapter,
    invoke_openclaw_tool,
    load_openclaw_skill,
    OpenClawToolResult,
    # ClawHub Marketplace
    ClawHubClient,
    get_clawhub_client,
    search_clawhub_skills,
    download_clawhub_skill,
    install_and_load_skill,
)

from .device_node import (
    DeviceNode,
    get_device_node,
    camera_snap,
    camera_clip,
    screen_record,
    get_location,
    send_notification,
)

from .visual_canvas import (
    VisualCanvas,
    get_canvas,
    canvas_push,
    canvas_eval,
    canvas_snapshot,
    canvas_reset,
    A2UIComponent,
)

# Voice interface (optional - requires pyaudio/sounddevice)
try:
    from .voice_interface import (
        VoiceInterface,
        get_voice_interface,
        speech_to_text,
        text_to_speech,
        start_voice_wake,
        stop_voice_wake,
    )
    VOICE_AVAILABLE = True
except ImportError:
    VoiceInterface = None
    get_voice_interface = None
    speech_to_text = None
    text_to_speech = None
    start_voice_wake = None
    stop_voice_wake = None
    VOICE_AVAILABLE = False

from .task_routing import (
    OpenClawTaskType,
    ModelCapability,
    MODEL_REGISTRY,
    TASK_ROUTING,
    CHANNEL_MODEL_ROUTING,
    get_best_model_for_task,
    get_models_for_task,
    get_fallback_chain,
    classify_openclaw_tool,
    route_openclaw_task,
    get_model_for_channel,
    get_routing_summary,
)

from .model_invoker import (
    ModelInvoker,
    ModelResponse,
    get_model_invoker,
    invoke_model,
    invoke_for_tool,
    invoke_for_channel,
)

__all__ = [
    # Main adapter
    "OpenClawAdapter",
    "get_openclaw_adapter",
    "invoke_openclaw_tool",
    "load_openclaw_skill",
    "OpenClawToolResult",
    # ClawHub Marketplace
    "ClawHubClient",
    "get_clawhub_client",
    "search_clawhub_skills",
    "download_clawhub_skill",
    "install_and_load_skill",
    # Device nodes
    "DeviceNode",
    "get_device_node",
    "camera_snap",
    "camera_clip",
    "screen_record",
    "get_location",
    "send_notification",
    # Visual canvas
    "VisualCanvas",
    "get_canvas",
    "canvas_push",
    "canvas_eval",
    "canvas_snapshot",
    "canvas_reset",
    "A2UIComponent",
    # Voice interface
    "VoiceInterface",
    "get_voice_interface",
    "speech_to_text",
    "text_to_speech",
    "start_voice_wake",
    "stop_voice_wake",
    # Task routing
    "OpenClawTaskType",
    "ModelCapability",
    "MODEL_REGISTRY",
    "TASK_ROUTING",
    "CHANNEL_MODEL_ROUTING",
    "get_best_model_for_task",
    "get_models_for_task",
    "get_fallback_chain",
    "classify_openclaw_tool",
    "route_openclaw_task",
    "get_model_for_channel",
    "get_routing_summary",
    # Model invoker
    "ModelInvoker",
    "ModelResponse",
    "get_model_invoker",
    "invoke_model",
    "invoke_for_tool",
    "invoke_for_channel",
]
