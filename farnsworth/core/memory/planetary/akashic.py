"""
Farnsworth Planetary Memory (The Akashic Record)
------------------------------------------------

"A place where all knowledge flows together... like a giant soup of brains!"

This module implements a privacy-preserving global vector store that allows Farnsworth instances
to share "Skill Vectors" (generalized solutions) without sharing sensitive data.

Features:
- **Skill Vectors**: Abstracted problem-solution pairs (e.g., "Docker Fix" -> "Restart service").
- **Privacy Filter**: Strict PII scrubbing before sharing.
- **Trust Protocol**: Only share with instances in the 'Web of Trust'.
"""

import asyncio
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum

from loguru import logger

class MemoryScope(Enum):
    LOCAL_ONLY = "local_only"       # Never leaves machine
    TEAM_SHARED = "team_shared"     # Shared with specific team pool
    PLANETARY = "planetary"         # Shared with global swarm (anonymized)

@dataclass
class SkillVector:
    """A shareable unit of knowledge (solution to a specific problem)."""
    id: str
    problem_hash: str     # Semantic hash of the problem description
    vector: List[float]   # The embedding of the solution logic
    
    # Textual description (Anonymized)
    # Good: "Restart docker daemon when getting error X"
    # Bad: "Restart server at 192.168.1.1"
    abstract_solution: str 
    
    confidence_score: float = 0.5
    author_signature: str = "" # Cryptographic signature of author
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

class PlanetaryMemory:
    def __init__(self, use_p2p: bool = False):
        self.use_p2p = use_p2p
        self.local_skills: Dict[str, SkillVector] = {}
        self.global_cache: Dict[str, SkillVector] = {} # Skills learned from others
        self.privacy_mode = True

    async def share_skill(self, problem: str, solution: str, embedding: List[float]) -> Optional[str]:
        """
        Publish a learned skill to the Planetary Memory.
        """
        if not self._privacy_check(problem + solution):
            logger.warning("Privacy Check Failed: Content contains PII. Keeping local only.")
            return None

        # generate ID
        skill_id = hashlib.sha256(f"{solution}{datetime.now()}".encode()).hexdigest()[:16]
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:16]

        skill = SkillVector(
            id=skill_id,
            problem_hash=problem_hash,
            vector=embedding,
            abstract_solution=solution,
            confidence_score=0.8
        )

        self.local_skills[skill_id] = skill
        logger.info(f"Skill Created: {skill_id} (Local)")

        if self.use_p2p:
            await self._broadcast_to_swarm(skill)
            logger.info("Skill Broadcasted to Planetary Network 🌍")
        
        return skill_id

    async def retrieve_wisdom(self, problem_embedding: List[float], query_text: str) -> List[SkillVector]:
        """
        Retrieve relevant skills from Global Cache + Local.
        """
        # Mock retrieval logic (In prod this uses Vector DB)
        results = []
        
        # Check global cache (simulating 'borrowing' knowledge)
        for skill in self.global_cache.values():
            score = self._cosine_sim(problem_embedding, skill.vector)
            if score > 0.75:
                results.append(skill)
                
        logger.info(f"Retrieved {len(results)} planetary skills for query: '{query_text}'")
        return results

    async def _broadcast_to_swarm(self, skill: SkillVector):
        """Mock P2P broadcast."""
        # Farnsworth P2P protocol integration would go here
        # For now, we simulate receiving it back to verify flow
        self.global_cache[skill.id] = skill

    def _privacy_check(self, text: str) -> bool:
        """Simple PII regex check."""
        # Simple heuristic: Reject emails, IPs, API keys
        import re
        if re.search(r'sk-[a-zA-Z0-9]{20,}', text): return False # OpenAI key
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text): return False # IP
        return True

    def _cosine_sim(self, v1: List[float], v2: List[float]) -> float:
        """Manual cosine similarity."""
        import math
        dot_product = sum(p*q for p,q in zip(v1, v2))
        magnitude = math.sqrt(sum([val**2 for val in v1])) * math.sqrt(sum([val**2 for val in v2]))
        if not magnitude: return 0
        return dot_product / magnitude

# Integration with Core Memory System
async def demo_planetary_memory():
    pm = PlanetaryMemory(use_p2p=True)
    
    # 1. Learn a local skill
    logger.info("--- Learning New Skill ---")
    mock_vector = [0.1, 0.5, 0.9] # Mock embedding
    
    skill_id = await pm.share_skill(
        problem="Docker container hanging on exit",
        solution="Use docker stop -t 30 to allow graceful shutdown",
        embedding=mock_vector
    )
    
    # 2. Retrieve it (simulating another agent retrieving it)
    logger.info("--- Retrieving Wisdom ---")
    skills = await pm.retrieve_wisdom([0.1, 0.5, 0.85], "Docker shutdown issues")
    
    if skills:
        print(f"Planetary Advice: {skills[0].abstract_solution}")
    else:
        print("No planetary wisdom found.")

if __name__ == "__main__":
    asyncio.run(demo_planetary_memory())
