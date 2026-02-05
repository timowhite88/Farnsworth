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

from .voice_interface import (
    VoiceInterface,
    get_voice_interface,
    speech_to_text,
    text_to_speech,
    start_voice_wake,
    stop_voice_wake,
)

__all__ = [
    # Main adapter
    "OpenClawAdapter",
    "get_openclaw_adapter",
    "invoke_openclaw_tool",
    "load_openclaw_skill",
    "OpenClawToolResult",
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
]
