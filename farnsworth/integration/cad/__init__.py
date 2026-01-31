"""
Farnsworth CAD Integration.

Parametric 3D modeling using CadQuery.

Capabilities:
- Create primitives (box, cylinder, sphere)
- Boolean operations (union, difference, intersection)
- Extrusions and sweeps
- Export to STL, STEP, OBJ
- Natural language model generation
"""

from .engine import FarnsworthCAD
from .translator import CADTranslator

__all__ = ['FarnsworthCAD', 'CADTranslator']
