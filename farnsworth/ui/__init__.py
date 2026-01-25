"""
Farnsworth UI Module

Streamlit-based interface with:
- Interactive chat with memory awareness
- Knowledge graph visualization
- Evolution dashboard
- Agent activity monitoring
"""

from farnsworth.ui.streamlit_app import FarnsworthUI
from farnsworth.ui.visualizations import MemoryVisualizer, GraphVisualizer, EvolutionVisualizer

__all__ = [
    "FarnsworthUI",
    "MemoryVisualizer",
    "GraphVisualizer",
    "EvolutionVisualizer",
]
