"""
Farnsworth Diagram Specialist - Mermaid & Visualization.

"A picture is worth a thousand lines of code, especially if it compiles."

This module handles the generation of Mermaid diagrams and charts.
"""

from typing import Dict, Any, List
from loguru import logger

class DiagramSkill:
    def __init__(self):
        pass

    def generate_mermaid_flowchart(self, nodes: List[Dict[str, str]], edges: List[Dict[str, str]]) -> str:
        """Generate Mermaid syntax for a flowchart."""
        lines = ["graph TD"]
        for node in nodes:
            name = node.get("id")
            label = node.get("label", name)
            # Add shape logic if needed (e.g. [label], (label), {{label}})
            lines.append(f"    {name}[\"{label}\"]")
        
        for edge in edges:
            lines.append(f"    {edge['from']} --> {edge['to']}")
            
        return "\n".join(lines)

    def generate_sequence_diagram(self, participants: List[str], messages: List[Dict[str, str]]) -> str:
        """Generate Mermaid syntax for a sequence diagram."""
        lines = ["sequenceDiagram"]
        for p in participants:
            lines.append(f"    participant {p}")
        
        for m in messages:
            lines.append(f"    {m['from']}->>+{m['to']}: {m['text']}")
            if m.get("reply"):
                lines.append(f"    {m['to']}-->>-{m['from']}: {m['reply']}")
                
        return "\n".join(lines)

    async def render_to_svg_mock(self, mermaid_code: str) -> str:
        """Mock rendering to SVG (would use mermaid-cli in production)."""
        logger.info(f"Diagram: Rendering SVG for code:\n{mermaid_code}")
        return "<svg>...Mermaid Diagram...</svg>"

diagram_skill = DiagramSkill()
