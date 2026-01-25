"""
Farnsworth Meta-Cognition Agent - Self-Reflection and Improvement

Novel Approaches:
1. Capability Gap Detection - Identify where system struggles
2. Strategy Analysis - Evaluate and improve approaches
3. Performance Monitoring - Track quality metrics
4. Improvement Proposals - Generate concrete enhancement ideas
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from collections import defaultdict

from loguru import logger

from farnsworth.agents.base_agent import BaseAgent, AgentCapability, TaskResult


@dataclass
class PerformanceMetric:
    """A tracked performance metric."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)


@dataclass
class CapabilityGap:
    """An identified capability gap."""
    description: str
    severity: float  # 0-1
    frequency: int
    examples: list[str] = field(default_factory=list)
    suggested_improvement: str = ""
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementProposal:
    """A proposed system improvement."""
    title: str
    description: str
    target_area: str  # "model", "prompts", "memory", "agents", "rag"
    priority: float
    expected_impact: str
    implementation_notes: str = ""
    status: str = "proposed"  # proposed, accepted, implemented, rejected
    created_at: datetime = field(default_factory=datetime.now)


class MetaCognitionAgent(BaseAgent):
    """
    Agent for self-reflection and improvement proposals.

    Features:
    - Monitor system performance
    - Detect capability gaps
    - Propose improvements
    - Track what works and what doesn't
    """

    def __init__(
        self,
        reflection_interval_turns: int = 5,
    ):
        super().__init__(
            name="MetaCognition",
            capabilities=[
                AgentCapability.META_COGNITION,
                AgentCapability.REASONING,
                AgentCapability.PLANNING,
            ],
            confidence_threshold=0.5,
        )

        self.reflection_interval = reflection_interval_turns
        self.turns_since_reflection = 0

        # Performance tracking
        self.metrics: list[PerformanceMetric] = []
        self.capability_gaps: list[CapabilityGap] = []
        self.improvement_proposals: list[ImprovementProposal] = []

        # Task outcome history
        self.task_history: list[dict] = []
        self.error_patterns: dict[str, int] = defaultdict(int)

        # Strategy effectiveness
        self.strategy_scores: dict[str, list[float]] = defaultdict(list)

    @property
    def system_prompt(self) -> str:
        return """You are the meta-cognitive agent responsible for self-reflection and improvement.

Your role:
1. Analyze system performance and identify issues
2. Detect patterns in failures and limitations
3. Propose concrete improvements
4. Monitor effectiveness of changes

When reflecting:
- Look for recurring failure patterns
- Identify capability gaps
- Evaluate strategy effectiveness
- Suggest actionable improvements

Be specific and constructive in your analysis.
Focus on what can actually be improved."""

    async def process(self, task: str, context: Optional[dict] = None) -> TaskResult:
        """Process a meta-cognitive task."""
        task_type = self._classify_task(task)

        if task_type == "reflect":
            result = await self._reflect(context)
        elif task_type == "analyze_error":
            result = await self._analyze_error(task, context)
        elif task_type == "propose_improvement":
            result = await self._propose_improvement(task, context)
        elif task_type == "evaluate_strategy":
            result = await self._evaluate_strategy(task, context)
        else:
            result = await self._general_meta_task(task, context)

        return result

    def _classify_task(self, task: str) -> str:
        """Classify the meta-cognitive task type."""
        task_lower = task.lower()

        if "reflect" in task_lower or "review" in task_lower:
            return "reflect"
        elif "error" in task_lower or "failure" in task_lower:
            return "analyze_error"
        elif "improve" in task_lower or "enhance" in task_lower:
            return "propose_improvement"
        elif "strategy" in task_lower or "approach" in task_lower:
            return "evaluate_strategy"

        return "general"

    async def _reflect(self, context: Optional[dict]) -> TaskResult:
        """Perform a reflection on recent performance."""
        # Gather recent data
        recent_tasks = self.task_history[-20:]
        recent_errors = list(self.error_patterns.items())[-10:]
        recent_gaps = self.capability_gaps[-5:]

        # Calculate summary metrics
        if recent_tasks:
            success_rate = sum(1 for t in recent_tasks if t.get("success")) / len(recent_tasks)
            avg_confidence = sum(t.get("confidence", 0) for t in recent_tasks) / len(recent_tasks)
        else:
            success_rate = 0.0
            avg_confidence = 0.0

        # Generate reflection
        reflection_prompt = f"""Reflect on the following system performance:

Success rate: {success_rate:.0%}
Average confidence: {avg_confidence:.0%}
Recent errors: {len(recent_errors)}
Known capability gaps: {len(recent_gaps)}

Recent task outcomes:
{self._format_recent_tasks(recent_tasks)}

Common errors:
{self._format_errors(recent_errors)}

Provide:
1. Overall assessment
2. Key issues identified
3. Recommended actions"""

        response, confidence = await self.generate_response(reflection_prompt, context)

        # Record this reflection
        self.metrics.append(PerformanceMetric(
            name="reflection",
            value=success_rate,
            context={"avg_confidence": avg_confidence},
        ))

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "success_rate": success_rate,
                "avg_confidence": avg_confidence,
                "gaps_detected": len(recent_gaps),
            },
        )

    def _format_recent_tasks(self, tasks: list[dict]) -> str:
        """Format recent tasks for prompt."""
        if not tasks:
            return "No recent tasks"

        lines = []
        for task in tasks[-5:]:
            status = "Success" if task.get("success") else "Failed"
            lines.append(f"- [{status}] {task.get('description', 'Unknown')[:50]}")
        return "\n".join(lines)

    def _format_errors(self, errors: list[tuple]) -> str:
        """Format errors for prompt."""
        if not errors:
            return "No recent errors"

        return "\n".join(f"- {err}: {count} times" for err, count in errors)

    async def _analyze_error(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Analyze an error to understand the root cause."""
        error_info = context.get("error", task) if context else task

        analysis_prompt = f"""Analyze this error/failure:

{error_info}

Provide:
1. Root cause analysis
2. Pattern identification (is this recurring?)
3. Capability gap (what's missing?)
4. Prevention strategy"""

        response, confidence = await self.generate_response(analysis_prompt, context)

        # Record error pattern
        error_type = self._categorize_error(error_info)
        self.error_patterns[error_type] += 1

        # Check if this reveals a capability gap
        if self.error_patterns[error_type] >= 3:
            gap = CapabilityGap(
                description=f"Recurring {error_type} errors",
                severity=min(1.0, self.error_patterns[error_type] / 10),
                frequency=self.error_patterns[error_type],
                examples=[error_info[:100]],
            )
            self.capability_gaps.append(gap)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "error_type": error_type,
                "occurrence_count": self.error_patterns[error_type],
            },
        )

    def _categorize_error(self, error: str) -> str:
        """Categorize an error into a type."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower or "context" in error_lower:
            return "context_overflow"
        elif "confidence" in error_lower:
            return "low_confidence"
        elif "hallucin" in error_lower or "incorrect" in error_lower:
            return "accuracy"
        elif "not found" in error_lower or "missing" in error_lower:
            return "missing_information"
        elif "capability" in error_lower or "can't" in error_lower:
            return "capability_limitation"

        return "unknown"

    async def _propose_improvement(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Generate an improvement proposal."""
        # Gather evidence for the proposal
        recent_gaps = self.capability_gaps[-5:]
        recent_errors = dict(list(self.error_patterns.items())[-5:])

        proposal_prompt = f"""Based on the improvement need: {task}

Current issues:
- Capability gaps: {[g.description for g in recent_gaps]}
- Common errors: {recent_errors}

Generate a specific improvement proposal with:
1. Clear title
2. Detailed description
3. Target area (model/prompts/memory/agents/rag)
4. Expected impact
5. Implementation approach"""

        response, confidence = await self.generate_response(proposal_prompt, context)

        # Create structured proposal
        proposal = ImprovementProposal(
            title=task[:50],
            description=response,
            target_area=self._infer_target_area(task, response),
            priority=confidence,
            expected_impact="Based on identified gaps",
        )
        self.improvement_proposals.append(proposal)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "proposal_id": len(self.improvement_proposals) - 1,
                "target_area": proposal.target_area,
            },
        )

    def _infer_target_area(self, task: str, response: str) -> str:
        """Infer the target area for improvement."""
        text = f"{task} {response}".lower()

        if any(kw in text for kw in ["model", "llm", "inference"]):
            return "model"
        elif any(kw in text for kw in ["prompt", "system message", "instruction"]):
            return "prompts"
        elif any(kw in text for kw in ["memory", "context", "recall"]):
            return "memory"
        elif any(kw in text for kw in ["agent", "specialist", "handoff"]):
            return "agents"
        elif any(kw in text for kw in ["retrieval", "search", "rag"]):
            return "rag"

        return "general"

    async def _evaluate_strategy(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Evaluate the effectiveness of a strategy."""
        strategy_name = context.get("strategy", "unknown") if context else "unknown"
        outcomes = context.get("outcomes", []) if context else []

        # Calculate strategy score
        if outcomes:
            success_rate = sum(1 for o in outcomes if o.get("success")) / len(outcomes)
            avg_quality = sum(o.get("quality", 0.5) for o in outcomes) / len(outcomes)
            score = (success_rate + avg_quality) / 2
        else:
            score = 0.5

        self.strategy_scores[strategy_name].append(score)

        eval_prompt = f"""Evaluate strategy: {strategy_name}

Recent performance: {score:.0%}
Historical scores: {self.strategy_scores[strategy_name][-5:]}
Outcomes analyzed: {len(outcomes)}

Provide:
1. Strategy effectiveness assessment
2. Strengths and weaknesses
3. Recommendations for improvement or alternatives"""

        response, confidence = await self.generate_response(eval_prompt, context)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
            metadata={
                "strategy": strategy_name,
                "current_score": score,
                "historical_avg": sum(self.strategy_scores[strategy_name]) / max(1, len(self.strategy_scores[strategy_name])),
            },
        )

    async def _general_meta_task(
        self,
        task: str,
        context: Optional[dict],
    ) -> TaskResult:
        """Handle general meta-cognitive tasks."""
        response, confidence = await self.generate_response(task, context)

        return TaskResult(
            success=True,
            output=response,
            confidence=confidence,
        )

    def record_task_outcome(
        self,
        description: str,
        success: bool,
        confidence: float,
        metadata: Optional[dict] = None,
    ):
        """Record a task outcome for analysis."""
        self.task_history.append({
            "description": description,
            "success": success,
            "confidence": confidence,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        })

        # Trigger reflection if needed
        self.turns_since_reflection += 1
        if self.turns_since_reflection >= self.reflection_interval:
            self.turns_since_reflection = 0
            # Could trigger async reflection here

        # Keep bounded history
        if len(self.task_history) > 500:
            self.task_history = self.task_history[-250:]

    async def should_reflect(self) -> bool:
        """Check if it's time for reflection."""
        return self.turns_since_reflection >= self.reflection_interval

    def get_capability_gaps(self) -> list[dict]:
        """Get current capability gaps."""
        return [
            {
                "description": gap.description,
                "severity": gap.severity,
                "frequency": gap.frequency,
            }
            for gap in self.capability_gaps
        ]

    def get_improvement_proposals(self, status: Optional[str] = None) -> list[dict]:
        """Get improvement proposals."""
        proposals = self.improvement_proposals
        if status:
            proposals = [p for p in proposals if p.status == status]

        return [
            {
                "title": p.title,
                "target_area": p.target_area,
                "priority": p.priority,
                "status": p.status,
            }
            for p in proposals
        ]

    def get_performance_summary(self) -> dict:
        """Get performance summary."""
        recent = self.task_history[-50:]

        if not recent:
            return {
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "tasks_analyzed": 0,
            }

        return {
            "success_rate": sum(1 for t in recent if t.get("success")) / len(recent),
            "avg_confidence": sum(t.get("confidence", 0) for t in recent) / len(recent),
            "tasks_analyzed": len(recent),
            "capability_gaps": len(self.capability_gaps),
            "pending_proposals": len([p for p in self.improvement_proposals if p.status == "proposed"]),
            "top_error_types": dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]),
        }
