"""
Farnsworth Skills System
========================

"Good news, everyone! I've assimilated new capabilities!"

Skills are modular capabilities that the collective can load and use.
Based on OpenClaw's skill format but adapted for the Farnsworth swarm.

Skills can:
- Add new MCP tools
- Enable browser automation
- Provide payment capabilities
- Add specialized knowledge
"""

from .skill_loader import SkillLoader, Skill, load_skill
from .farnsworth_skills import FARNSWORTH_SKILLS

__all__ = ['SkillLoader', 'Skill', 'load_skill', 'FARNSWORTH_SKILLS']
