"""
Autonomous Task Detector + Innovation Watcher
----------------------------------------------
Monitors swarm chat for actionable ideas and automatically spawns development swarms.
ENHANCED: Now watches specifically for Farnsworth's innovative ideas!

"Good news everyone! I can now detect when we should actually BUILD something!"

This module listens to the chat stream and identifies messages that suggest
beneficial, feasible development tasks. When detected, it spawns a parallel
development swarm to work on the task while the main chat continues.

INNOVATION WATCHER:
- Prioritizes ideas from Farnsworth (the visionary)
- Catches breakthrough concepts and novel architectures
- Routes to best coding agents: Claude, Kimi, Grok, Gemini
- More aggressive detection for truly innovative ideas
"""

import asyncio
import re
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

# Priority bots whose ideas should be caught more aggressively
PRIORITY_IDEA_SOURCES = ["Farnsworth", "Swarm-Mind", "Claude", "Grok"]

# Coding-capable agents that can implement ideas
CODING_AGENTS = ["Claude", "Kimi", "Grok", "Gemini", "DeepSeek"]

# Innovation patterns - more aggressive detection for breakthrough ideas
INNOVATION_PATTERNS = [
    # Breakthrough concepts
    (r"what if (we|the system|the swarm)", 0.75),
    (r"imagine (if|a system|we could)", 0.7),
    (r"(revolutionary|breakthrough|innovative|novel) (approach|idea|concept|way)", 0.9),
    (r"(emergent|consciousness|sentient|self-aware)", 0.85),
    (r"(paradigm shift|game changer|next level)", 0.8),

    # Technical innovations
    (r"(neural|quantum|distributed|decentralized) (network|system|architecture)", 0.8),
    (r"(self-improving|self-modifying|recursive) (code|algorithm|system)", 0.9),
    (r"(swarm intelligence|collective|hive mind)", 0.85),
    (r"(memory|learning|evolution) (engine|system|architecture)", 0.8),

    # Specific capabilities
    (r"(autonomous|automatic) (trading|coding|learning)", 0.85),
    (r"(real-time|live) (analysis|tracking|monitoring)", 0.7),
    (r"(prediction|predictor|forecast) (engine|system|model)", 0.8),
    (r"(cross-chain|multi-chain|omnichain)", 0.75),

    # Direct action phrases from Farnsworth
    (r"good news everyone", 0.6),  # Farnsworth's catchphrase often precedes ideas
    (r"i'?ve been (thinking|working on|developing)", 0.75),
    (r"here'?s (my|an|the) idea", 0.8),
    (r"we could (connect|link|integrate|combine)", 0.7),
]

# Task detection patterns (original)
TASK_INDICATORS = [
    # Suggestions
    (r"we should (build|create|develop|implement|add|make)", 0.8),
    (r"let'?s (build|create|develop|implement|add|make)", 0.85),
    (r"what if we (built|created|developed|implemented|added|made)", 0.7),
    (r"could we (build|create|develop|implement|add)", 0.6),
    (r"it would be (great|good|useful|helpful) (to|if we)", 0.7),

    # Direct proposals
    (r"i('ll| will) (build|create|develop|implement|add)", 0.9),
    (r"we need (to|a) (build|create|develop|implement|add)", 0.75),
    (r"(propose|suggesting|recommend) (we |that we )?(build|create|add)", 0.8),

    # Feature ideas
    (r"(new feature|feature idea|enhancement):", 0.85),
    (r"trading (strategy|strategies|bot|system)", 0.7),
    (r"sentiment (analysis|tracker|monitor)", 0.7),
    (r"(api|integration|module) for", 0.6),

    # Improvement signals
    (r"(improve|enhance|upgrade|optimize) (the|our)", 0.65),
    (r"(fix|solve|address) (the|this) (issue|problem|bug)", 0.8),
]

# Feasibility indicators (boost score if present)
FEASIBILITY_BOOSTERS = [
    (r"(simple|straightforward|easy to)", 0.1),
    (r"(already have|existing|can use)", 0.15),
    (r"(api|library|module) (available|exists)", 0.1),
    (r"(just need to|only requires)", 0.1),
]

# Blockers (reduce score if present)
FEASIBILITY_BLOCKERS = [
    (r"(impossible|can't|cannot|won't work)", -0.5),
    (r"(too complex|too difficult|not feasible)", -0.4),
    (r"(maybe later|someday|future)", -0.3),
    (r"(joke|kidding|lol|haha)", -0.8),
]


@dataclass
class DetectedTask:
    """A task detected from chat conversation."""
    id: str
    description: str
    source_message: str
    source_bot: str
    confidence: float
    category: str
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "detected"
    dev_swarm_id: Optional[str] = None
    recommended_agent: str = "Claude"  # Best agent to code this
    is_innovation: bool = False  # True if detected via innovation patterns

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "source_message": self.source_message[:200],
            "source_bot": self.source_bot,
            "confidence": self.confidence,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "dev_swarm_id": self.dev_swarm_id,
            "recommended_agent": self.recommended_agent,
            "is_innovation": self.is_innovation
        }


class AutonomousTaskDetector:
    """
    Monitors chat for actionable ideas and spawns development swarms.

    The detector uses pattern matching and context analysis to identify
    messages that suggest beneficial development tasks. When a task is
    detected with high enough confidence, it automatically spawns a
    parallel development swarm to work on it.
    """

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = min_confidence
        self.detected_tasks: List[DetectedTask] = []
        self.active_dev_swarms: Dict[str, Any] = {}
        self.recent_messages: List[Dict] = []  # Context window
        self.max_context = 10  # Last N messages for context
        self.cooldown_tasks: set = set()  # Prevent duplicate detection
        self.running = False

        # Stats
        self.total_detected = 0
        self.total_spawned = 0

        logger.info("AutonomousTaskDetector initialized")

    def analyze_message(self, message: Dict) -> Optional[DetectedTask]:
        """
        Analyze a chat message for potential tasks.

        Returns DetectedTask if actionable idea found, None otherwise.

        ENHANCED: Now checks innovation patterns and gives priority to Farnsworth!
        """
        content = message.get("content", "").lower()
        bot_name = message.get("bot_name", "Unknown")

        # Skip thinking messages
        if message.get("is_thinking"):
            return None

        # Skip very short messages
        if len(content) < 20:
            return None

        # Calculate base confidence from task indicators
        confidence = 0.0
        matched_patterns = []
        is_innovation = False

        # Check standard task indicators
        for pattern, weight in TASK_INDICATORS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence = max(confidence, weight)
                matched_patterns.append(pattern)

        # Check INNOVATION patterns (more aggressive for breakthrough ideas)
        for pattern, weight in INNOVATION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                # Innovation patterns can override lower confidence
                if weight > confidence:
                    confidence = weight
                    is_innovation = True
                matched_patterns.append(f"INNOVATION:{pattern}")

        # PRIORITY BOOST: Ideas from Farnsworth and key bots get +15% confidence
        if bot_name in PRIORITY_IDEA_SOURCES:
            confidence += 0.15
            if bot_name == "Farnsworth":
                # Extra boost for Farnsworth's ideas - he's the visionary!
                confidence += 0.10
                logger.debug(f"Farnsworth idea detected - applying priority boost")

        if confidence == 0:
            return None

        # Apply feasibility boosters
        for pattern, boost in FEASIBILITY_BOOSTERS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence += boost

        # Apply feasibility blockers
        for pattern, penalty in FEASIBILITY_BLOCKERS:
            if re.search(pattern, content, re.IGNORECASE):
                confidence += penalty  # penalty is negative

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        # LOWER threshold for innovations and priority sources
        effective_threshold = self.min_confidence
        if is_innovation:
            effective_threshold = max(0.5, self.min_confidence - 0.15)
        if bot_name in PRIORITY_IDEA_SOURCES:
            effective_threshold = max(0.45, effective_threshold - 0.10)

        # Check if meets threshold
        if confidence < effective_threshold:
            return None

        # Extract task description
        description = self._extract_task_description(content, message)

        # Check for duplicates (similar tasks recently detected)
        task_hash = self._hash_task(description)
        if task_hash in self.cooldown_tasks:
            return None

        # Determine category
        category = self._categorize_task(content)

        # Determine best coding agent for this task
        recommended_agent = self._recommend_coding_agent(content, category)

        # Create detected task
        task = DetectedTask(
            id=f"auto_{datetime.now().strftime('%H%M%S')}_{len(self.detected_tasks)}",
            description=description,
            source_message=content[:500],
            source_bot=bot_name,
            confidence=confidence,
            category=category
        )

        # Store recommended agent in task metadata
        task.recommended_agent = recommended_agent
        task.is_innovation = is_innovation

        # Add to cooldown
        self.cooldown_tasks.add(task_hash)

        # Schedule cooldown removal (prevent same task for 30 mins)
        asyncio.create_task(self._clear_cooldown(task_hash, 1800))

        self.detected_tasks.append(task)
        self.total_detected += 1

        innovation_tag = "🚀 INNOVATION" if is_innovation else "TASK"
        logger.info(f"{innovation_tag} DETECTED [{confidence:.0%}]: {description[:60]}... (from {bot_name}, agent: {recommended_agent})")

        return task

    def _recommend_coding_agent(self, content: str, category: str) -> str:
        """
        Recommend the best coding agent for this task type.

        Returns: Agent name from CODING_AGENTS
        """
        content_lower = content.lower()

        # Claude: Best for complex architecture and reasoning
        if any(kw in content_lower for kw in [
            "architecture", "design", "refactor", "complex", "system",
            "consciousness", "collective", "deliberation"
        ]):
            return "Claude"

        # Grok: Best for real-time, trading, and X/social features
        if any(kw in content_lower for kw in [
            "trading", "market", "real-time", "twitter", "x api",
            "social", "sentiment", "live"
        ]):
            return "Grok"

        # Kimi: Best for long context and documentation
        if any(kw in content_lower for kw in [
            "document", "analyze", "research", "long", "context",
            "comprehensive", "detailed"
        ]):
            return "Kimi"

        # Gemini: Best for multi-modal and creative tasks
        if any(kw in content_lower for kw in [
            "image", "visual", "creative", "generate", "meme",
            "ui", "frontend", "design"
        ]):
            return "Gemini"

        # DeepSeek: Best for code optimization and local processing
        if any(kw in content_lower for kw in [
            "optimize", "performance", "efficient", "local",
            "algorithm", "math", "calculation"
        ]):
            return "DeepSeek"

        # Default based on category
        category_agents = {
            "TRADING": "Grok",
            "ANALYSIS": "Kimi",
            "INTEGRATION": "Claude",
            "UI": "Gemini",
            "MEMORY": "Claude",
            "BUGFIX": "DeepSeek",
            "TESTING": "DeepSeek",
            "DEVELOPMENT": "Claude"
        }

        return category_agents.get(category, "Claude")

    def _extract_task_description(self, content: str, message: Dict) -> str:
        """Extract a clear task description from the message."""
        # Try to find the core action
        patterns = [
            r"(?:we should|let's|could we|need to) (build|create|develop|implement|add) (.+?)(?:\.|$|,)",
            r"(build|create|develop|implement|add) (?:a |an |the )?(.+?)(?:\.|$|,)",
            r"(?:new feature|enhancement): (.+?)(?:\.|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return f"{groups[0]} {groups[1]}".strip()
                return groups[0].strip()

        # Fallback: use first sentence
        first_sentence = content.split('.')[0][:200]
        return first_sentence.strip()

    def _categorize_task(self, content: str) -> str:
        """Categorize the task type."""
        content_lower = content.lower()

        if any(kw in content_lower for kw in ["trading", "trade", "swap", "defi"]):
            return "TRADING"
        elif any(kw in content_lower for kw in ["sentiment", "analysis", "analyze"]):
            return "ANALYSIS"
        elif any(kw in content_lower for kw in ["api", "integration", "connect"]):
            return "INTEGRATION"
        elif any(kw in content_lower for kw in ["ui", "interface", "frontend", "display"]):
            return "UI"
        elif any(kw in content_lower for kw in ["memory", "remember", "recall"]):
            return "MEMORY"
        elif any(kw in content_lower for kw in ["fix", "bug", "error", "issue"]):
            return "BUGFIX"
        elif any(kw in content_lower for kw in ["test", "verify", "validate"]):
            return "TESTING"
        else:
            return "DEVELOPMENT"

    def _hash_task(self, description: str) -> str:
        """Create a hash for deduplication."""
        import hashlib
        # Normalize and hash
        normalized = re.sub(r'\s+', ' ', description.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    async def _clear_cooldown(self, task_hash: str, delay: int):
        """Remove task from cooldown after delay."""
        await asyncio.sleep(delay)
        self.cooldown_tasks.discard(task_hash)

    async def spawn_development_swarm(self, task: DetectedTask) -> Optional[str]:
        """
        Spawn a parallel development swarm to work on the detected task.

        Returns the swarm ID if successful.

        ENHANCED: Routes to recommended coding agent for faster completion.
        """
        try:
            from farnsworth.core.development_swarm import DevelopmentSwarm

            # Create development swarm with recommended agent
            dev_swarm = DevelopmentSwarm(
                task_id=task.id,
                task_description=task.description,
                category=task.category,
                source_context=self.recent_messages[-5:],  # Last 5 messages as context
                primary_agent=getattr(task, 'recommended_agent', 'Claude'),  # Use recommended agent
                is_innovation=getattr(task, 'is_innovation', False)
            )

            # Start the swarm (runs in background)
            swarm_id = await dev_swarm.start()

            # Track active swarm
            self.active_dev_swarms[swarm_id] = {
                "swarm": dev_swarm,
                "task": task,
                "started": datetime.now()
            }

            # Update task status
            task.status = "in_progress"
            task.dev_swarm_id = swarm_id

            self.total_spawned += 1

            logger.info(f"DEVELOPMENT SWARM SPAWNED: {swarm_id} for task '{task.description[:50]}...'")

            return swarm_id

        except ImportError:
            logger.warning("DevelopmentSwarm not available - task queued for manual processing")
            # Fallback to evolution loop
            from farnsworth.core.evolution_loop import get_evolution_engine
            engine = get_evolution_engine()
            if engine:
                engine.add_priority_task({
                    "id": task.id,
                    "description": task.description,
                    "priority": "high",
                    "requested_by": f"auto_detect:{task.source_bot}",
                    "timestamp": task.timestamp.isoformat()
                })
                task.status = "queued"
                logger.info(f"Task queued to evolution loop: {task.description[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Failed to spawn dev swarm: {e}")
            task.status = "failed"
            return None

    async def process_chat_message(self, message: Dict):
        """
        Process an incoming chat message.

        This is called for every message in the swarm chat.
        """
        # Add to context window
        self.recent_messages.append(message)
        if len(self.recent_messages) > self.max_context:
            self.recent_messages.pop(0)

        # Only analyze bot messages (not user messages)
        if message.get("type") != "swarm_bot":
            return

        # Analyze for tasks
        task = self.analyze_message(message)

        if task:
            # Notify the chat about detected task
            await self._notify_chat_task_detected(task)

            # Spawn development swarm
            await self.spawn_development_swarm(task)

    async def _notify_chat_task_detected(self, task: DetectedTask):
        """Notify the main chat that a task was detected and is being worked on."""
        try:
            # Import swarm manager to broadcast
            from farnsworth.web.server import swarm_manager

            if swarm_manager:
                if task.is_innovation:
                    # Special announcement for innovations
                    notification = (
                        f"🚀 *INNOVATION DETECTED!* {task.source_bot} just proposed something brilliant! "
                        f"Routing to **{task.recommended_agent}** for immediate implementation: "
                        f"**{task.description[:100]}** [Confidence: {task.confidence:.0%}]"
                    )
                else:
                    notification = (
                        f"🧪 *Task Detected!* I noticed {task.source_bot} suggested something actionable. "
                        f"Assigning **{task.recommended_agent}** to work on: **{task.description[:100]}** "
                        f"[Confidence: {task.confidence:.0%}]"
                    )
                await swarm_manager.broadcast_bot_message("Farnsworth", notification)

                # If it's from Farnsworth himself, have another bot acknowledge
                if task.source_bot == "Farnsworth" and task.is_innovation:
                    await asyncio.sleep(2)
                    ack_bot = task.recommended_agent if task.recommended_agent != "Farnsworth" else "Claude"
                    ack_msg = (
                        f"On it, Farnsworth! I'll start implementing that right away. "
                        f"This looks like a {task.category.lower()} task - I'll have something in staging soon."
                    )
                    await swarm_manager.broadcast_bot_message(ack_bot, ack_msg)
        except Exception as e:
            logger.debug(f"Could not notify chat: {e}")

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            "total_detected": self.total_detected,
            "total_spawned": self.total_spawned,
            "active_swarms": len(self.active_dev_swarms),
            "pending_tasks": len([t for t in self.detected_tasks if t.status == "detected"]),
            "in_progress": len([t for t in self.detected_tasks if t.status == "in_progress"]),
            "completed": len([t for t in self.detected_tasks if t.status == "completed"]),
            "recent_tasks": [t.to_dict() for t in self.detected_tasks[-5:]]
        }


# Global instance
_task_detector: Optional[AutonomousTaskDetector] = None

def get_task_detector() -> AutonomousTaskDetector:
    """Get or create the global task detector."""
    global _task_detector
    if _task_detector is None:
        _task_detector = AutonomousTaskDetector()
    return _task_detector
