"""
Farnsworth Model Swarm - Collaborative Multi-Model Inference

Novel Approaches:
1. PSO-Based Model Swarm - Particle Swarm Optimization for model collaboration
2. Ensemble Voting - Multiple models vote on best response
3. Mixture of Experts Routing - Dynamic routing to specialized models
4. Speculative Ensemble - Draft with fast models, verify with strong
5. Confidence-Weighted Fusion - Combine outputs weighted by confidence
6. Adaptive Load Balancing - Distribute work based on model availability

Research Sources:
- Model Swarms: Collaborative Search to Adapt LLM Experts (arXiv:2410.11163)
- LLM Ensemble Survey (arXiv:2502.18036)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Callable
import random
import math
import json
from collections import defaultdict

from loguru import logger


# PSO Position Dimension Semantics
# dim[0-2]: Quality weight, speed weight, efficiency weight (softmax -> sum to 1)
# dim[3]:   Temperature preference (mapped to 0.0 - 2.0)
# dim[4]:   Confidence threshold (mapped to 0.5 - 1.0)
# dim[5]:   Timeout multiplier (mapped to 0.5 - 2.0)
# dim[6-9]: Task-type affinity scores (reasoning, coding, creative, general)
PSO_DIM = 10
PSO_DIM_QUALITY_W = 0
PSO_DIM_SPEED_W = 1
PSO_DIM_EFFICIENCY_W = 2
PSO_DIM_TEMPERATURE = 3
PSO_DIM_CONFIDENCE_THRESH = 4
PSO_DIM_TIMEOUT_MULT = 5
PSO_DIM_AFFINITY_REASONING = 6
PSO_DIM_AFFINITY_CODING = 7
PSO_DIM_AFFINITY_CREATIVE = 8
PSO_DIM_AFFINITY_GENERAL = 9

# Map detected task types to affinity dimension indices
TASK_AFFINITY_MAP = {
    "reasoning": PSO_DIM_AFFINITY_REASONING,
    "coding": PSO_DIM_AFFINITY_CODING,
    "creative": PSO_DIM_AFFINITY_CREATIVE,
    "general": PSO_DIM_AFFINITY_GENERAL,
    "math": PSO_DIM_AFFINITY_REASONING,  # math uses reasoning affinity
}


class SwarmStrategy(Enum):
    """Model swarm inference strategies."""
    FASTEST_FIRST = "fastest_first"  # Start with fastest, escalate if needed
    QUALITY_FIRST = "quality_first"  # Start with best, fall back if slow
    PARALLEL_VOTE = "parallel_vote"  # Run all, vote on best
    MIXTURE_OF_EXPERTS = "moe"  # Route to best expert per query
    SPECULATIVE_ENSEMBLE = "speculative"  # Draft + verify
    CONFIDENCE_FUSION = "fusion"  # Weighted combination
    PSO_COLLABORATIVE = "pso"  # Particle swarm optimization


class ModelRole(Enum):
    """Specialized roles for models in swarm."""
    GENERALIST = "generalist"
    REASONING = "reasoning"
    CODING = "coding"
    CREATIVE = "creative"
    MATH = "math"
    MULTILINGUAL = "multilingual"
    SPEED = "speed"
    VERIFIER = "verifier"


@dataclass
class ModelParticle:
    """
    A model treated as a particle in PSO swarm.

    Each particle has:
    - Position: Current model configuration/weights
    - Velocity: Direction of adaptation
    - Best position: Best configuration found
    """
    model_id: str
    model_name: str
    role: ModelRole

    # Performance metrics
    success_rate: float = 0.5
    avg_latency: float = 1.0
    avg_confidence: float = 0.7

    # PSO state
    position: list[float] = field(default_factory=list)  # Abstract position in solution space
    velocity: list[float] = field(default_factory=list)
    personal_best_position: list[float] = field(default_factory=list)
    personal_best_score: float = 0.0

    # Resource requirements
    vram_gb: float = 2.0
    ram_gb: float = 4.0

    # Statistics
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens: int = 0

    # Per-task-type performance tracking for PSO feedback loop
    task_type_stats: dict = field(default_factory=lambda: defaultdict(lambda: {
        "requests": 0, "successes": 0, "total_latency": 0.0, "total_confidence": 0.0
    }))

    def get_objective_weights(self) -> tuple[float, float, float]:
        """Extract quality/speed/efficiency weights from position (softmax normalized)."""
        if len(self.position) < 3:
            return (0.5, 0.3, 0.2)
        raw = [self.position[PSO_DIM_QUALITY_W], self.position[PSO_DIM_SPEED_W], self.position[PSO_DIM_EFFICIENCY_W]]
        # Softmax to ensure they sum to 1
        max_val = max(raw)
        exps = [math.exp(v - max_val) for v in raw]
        total = sum(exps)
        return (exps[0] / total, exps[1] / total, exps[2] / total)

    def get_temperature(self) -> float:
        """Temperature preference from position dim[3], mapped to 0.0-2.0."""
        if len(self.position) <= PSO_DIM_TEMPERATURE:
            return 1.0
        return self.position[PSO_DIM_TEMPERATURE] * 2.0

    def get_confidence_threshold(self) -> float:
        """Confidence threshold from position dim[4], mapped to 0.5-1.0."""
        if len(self.position) <= PSO_DIM_CONFIDENCE_THRESH:
            return 0.7
        return 0.5 + self.position[PSO_DIM_CONFIDENCE_THRESH] * 0.5

    def get_timeout_multiplier(self) -> float:
        """Timeout multiplier from position dim[5], mapped to 0.5-2.0."""
        if len(self.position) <= PSO_DIM_TIMEOUT_MULT:
            return 1.0
        return 0.5 + self.position[PSO_DIM_TIMEOUT_MULT] * 1.5

    def get_task_affinity(self, task_type: str) -> float:
        """Get affinity score for a task type from position dims [6-9]."""
        dim_idx = TASK_AFFINITY_MAP.get(task_type, PSO_DIM_AFFINITY_GENERAL)
        if len(self.position) <= dim_idx:
            return 0.5
        return self.position[dim_idx]

    def fitness(self) -> float:
        """Calculate particle fitness using PSO-learned objective weights."""
        w_quality, w_speed, w_efficiency = self.get_objective_weights()

        quality = self.success_rate * self.avg_confidence
        speed = 1.0 / max(0.1, self.avg_latency)
        efficiency = 1.0 / max(0.1, self.vram_gb)

        return quality * w_quality + speed * w_speed + efficiency * w_efficiency

    def update_stats(self, success: bool, latency: float, confidence: float, task_type: str = "general"):
        """Update running statistics and per-task-type performance."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1

        # Exponential moving average
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        self.avg_latency = alpha * latency + (1 - alpha) * self.avg_latency
        self.avg_confidence = alpha * confidence + (1 - alpha) * self.avg_confidence

        # Per-task-type tracking for PSO feedback loop
        stats = self.task_type_stats[task_type]
        stats["requests"] += 1
        if success:
            stats["successes"] += 1
        stats["total_latency"] += latency
        stats["total_confidence"] += confidence

        # Update personal best
        current_fitness = self.fitness()
        if current_fitness > self.personal_best_score:
            self.personal_best_score = current_fitness
            self.personal_best_position = self.position.copy()

    def task_type_score(self, task_type: str) -> float:
        """Get historical performance score for a specific task type."""
        stats = self.task_type_stats.get(task_type)
        if not stats or stats["requests"] == 0:
            return 0.5  # neutral default for unknown tasks
        success_rate = stats["successes"] / stats["requests"]
        avg_confidence = stats["total_confidence"] / stats["requests"]
        avg_latency = stats["total_latency"] / stats["requests"]
        return (success_rate * avg_confidence) / max(0.1, avg_latency)


@dataclass
class SwarmResponse:
    """Response from model swarm."""
    text: str
    model_id: str
    model_name: str
    strategy_used: SwarmStrategy

    # Quality metrics
    confidence: float = 0.0
    agreement_score: float = 0.0  # How many models agreed

    # Performance
    latency: float = 0.0
    tokens_generated: int = 0
    tokens_per_second: float = 0.0

    # Ensemble details
    num_models_used: int = 1
    model_contributions: dict = field(default_factory=dict)

    # Verification
    verified: bool = False
    verifier_model: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "strategy": self.strategy_used.value,
            "confidence": self.confidence,
            "latency": self.latency,
            "tokens_per_second": self.tokens_per_second,
            "num_models": self.num_models_used,
            "verified": self.verified,
        }


@dataclass
class QueryAnalysis:
    """Analysis of incoming query for routing."""
    query: str
    detected_task: str = "general"
    complexity: float = 0.5
    requires_reasoning: bool = False
    requires_code: bool = False
    requires_math: bool = False
    requires_creativity: bool = False
    is_multilingual: bool = False
    estimated_tokens: int = 100

    def best_role(self) -> ModelRole:
        """Determine best model role for this query."""
        if self.requires_math:
            return ModelRole.MATH
        elif self.requires_code:
            return ModelRole.CODING
        elif self.requires_reasoning and self.complexity > 0.7:
            return ModelRole.REASONING
        elif self.requires_creativity:
            return ModelRole.CREATIVE
        elif self.is_multilingual:
            return ModelRole.MULTILINGUAL
        elif self.complexity < 0.3:
            return ModelRole.SPEED
        return ModelRole.GENERALIST


class QueryAnalyzer:
    """Analyze queries to determine optimal routing."""

    CODE_PATTERNS = [
        r'\b(code|function|class|implement|debug|fix|program|script)\b',
        r'\b(python|javascript|typescript|java|rust|go|cpp|sql)\b',
        r'\b(api|database|server|client|frontend|backend)\b',
        r'```',
    ]

    MATH_PATTERNS = [
        r'\b(calculate|compute|solve|equation|formula|derivative|integral)\b',
        r'\b(probability|statistics|matrix|vector|algebra)\b',
        r'[\d\+\-\*\/\=\^]',
        r'\b(proof|theorem|lemma)\b',
    ]

    REASONING_PATTERNS = [
        r'\b(why|how|explain|analyze|compare|evaluate)\b',
        r'\b(step by step|think through|reason|logic)\b',
        r'\b(cause|effect|consequence|implication)\b',
    ]

    CREATIVE_PATTERNS = [
        r'\b(write|create|compose|generate|imagine|story|poem)\b',
        r'\b(creative|artistic|narrative|fiction)\b',
    ]

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze query for optimal routing."""
        import re

        analysis = QueryAnalysis(query=query)

        # Check patterns
        for pattern in self.CODE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_code = True
                analysis.detected_task = "coding"
                break

        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_math = True
                analysis.detected_task = "math"
                break

        for pattern in self.REASONING_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_reasoning = True
                if analysis.detected_task == "general":
                    analysis.detected_task = "reasoning"
                break

        for pattern in self.CREATIVE_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                analysis.requires_creativity = True
                analysis.detected_task = "creative"
                break

        # Estimate complexity
        words = len(query.split())
        analysis.complexity = min(1.0, words / 200 + (0.3 if analysis.requires_reasoning else 0))

        # Estimate output tokens
        if analysis.requires_code:
            analysis.estimated_tokens = 500
        elif analysis.requires_creativity:
            analysis.estimated_tokens = 300
        else:
            analysis.estimated_tokens = max(50, words * 2)

        # Check for non-English
        try:
            # Simple heuristic: high ratio of non-ASCII chars
            non_ascii = sum(1 for c in query if ord(c) > 127)
            if non_ascii / max(1, len(query)) > 0.3:
                analysis.is_multilingual = True
        except Exception as e:
            logger.debug(f"Multilingual detection skipped: {e}")

        return analysis


class ModelSwarm:
    """
    Collaborative model swarm for ensemble inference.

    Features:
    - PSO-based collaborative optimization
    - Multiple inference strategies
    - Dynamic model selection
    - Confidence-weighted fusion
    - Verification loops
    """

    def __init__(
        self,
        models: Optional[list[dict]] = None,
        default_strategy: SwarmStrategy = SwarmStrategy.MIXTURE_OF_EXPERTS,
        pso_inertia: float = 0.7,
        pso_cognitive: float = 1.5,
        pso_social: float = 1.5,
        enable_verification: bool = True,
    ):
        self.default_strategy = default_strategy
        self.pso_inertia = pso_inertia
        self.pso_cognitive = pso_cognitive
        self.pso_social = pso_social
        self.enable_verification = enable_verification

        # Model particles
        self.particles: dict[str, ModelParticle] = {}

        # Global best (PSO)
        self.global_best_position: list[float] = []
        self.global_best_score: float = 0.0
        self.global_best_model: Optional[str] = None

        # Query analyzer
        self.analyzer = QueryAnalyzer()

        # Backend references (set by external code)
        self.backends: dict[str, Any] = {}

        # Statistics
        self.total_queries = 0
        self.strategy_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "success": 0, "latency": []})

        # Quantum PSO: run QAOA exploration every N PSO queries
        self._quantum_pso_interval = 10
        self._pso_query_count = 0

        # Initialize with default models if provided
        if models:
            for model_config in models:
                self.register_model(model_config)

    def register_model(
        self,
        config: dict,
        backend: Optional[Any] = None,
    ) -> ModelParticle:
        """Register a model in the swarm."""
        model_id = config.get("id", config.get("name", f"model_{len(self.particles)}"))

        # Determine role from strengths
        strengths = config.get("strengths", [])
        role = ModelRole.GENERALIST
        if "reasoning" in strengths or "math" in strengths:
            role = ModelRole.REASONING
        elif "code" in strengths:
            role = ModelRole.CODING
        elif "creative" in strengths:
            role = ModelRole.CREATIVE
        elif "speed" in strengths or "fast" in strengths:
            role = ModelRole.SPEED
        elif "multilingual" in strengths:
            role = ModelRole.MULTILINGUAL

        # Initialize PSO position with meaningful semantics per role
        # dim[0-2]: quality/speed/efficiency weights (pre-softmax)
        # dim[3]: temperature, dim[4]: confidence threshold, dim[5]: timeout mult
        # dim[6-9]: task-type affinities (reasoning, coding, creative, general)
        position = [0.5] * PSO_DIM  # neutral defaults

        # Set role-specific priors
        if role == ModelRole.REASONING:
            position[PSO_DIM_QUALITY_W] = 0.8       # prefer quality
            position[PSO_DIM_SPEED_W] = 0.3
            position[PSO_DIM_TEMPERATURE] = 0.35     # ~0.7 temperature
            position[PSO_DIM_CONFIDENCE_THRESH] = 0.7  # high threshold
            position[PSO_DIM_TIMEOUT_MULT] = 0.7     # patient
            position[PSO_DIM_AFFINITY_REASONING] = 0.9
            position[PSO_DIM_AFFINITY_CODING] = 0.6
        elif role == ModelRole.CODING:
            position[PSO_DIM_QUALITY_W] = 0.7
            position[PSO_DIM_SPEED_W] = 0.5
            position[PSO_DIM_TEMPERATURE] = 0.15     # ~0.3 low temperature
            position[PSO_DIM_CONFIDENCE_THRESH] = 0.8
            position[PSO_DIM_AFFINITY_CODING] = 0.9
            position[PSO_DIM_AFFINITY_REASONING] = 0.6
        elif role == ModelRole.CREATIVE:
            position[PSO_DIM_QUALITY_W] = 0.6
            position[PSO_DIM_TEMPERATURE] = 0.6      # ~1.2 higher temp
            position[PSO_DIM_CONFIDENCE_THRESH] = 0.3  # more exploratory
            position[PSO_DIM_AFFINITY_CREATIVE] = 0.9
            position[PSO_DIM_AFFINITY_GENERAL] = 0.6
        elif role == ModelRole.SPEED:
            position[PSO_DIM_QUALITY_W] = 0.3
            position[PSO_DIM_SPEED_W] = 0.9          # speed first
            position[PSO_DIM_EFFICIENCY_W] = 0.7
            position[PSO_DIM_TIMEOUT_MULT] = 0.1     # tight timeout
            position[PSO_DIM_AFFINITY_GENERAL] = 0.8
        elif role == ModelRole.MATH:
            position[PSO_DIM_QUALITY_W] = 0.9
            position[PSO_DIM_TEMPERATURE] = 0.1      # ~0.2 very low
            position[PSO_DIM_CONFIDENCE_THRESH] = 0.9
            position[PSO_DIM_TIMEOUT_MULT] = 0.8
            position[PSO_DIM_AFFINITY_REASONING] = 0.9
        else:
            # Generalist / other: balanced with slight randomization
            for i in range(PSO_DIM):
                position[i] = 0.4 + random.random() * 0.2

        velocity = [random.uniform(-0.1, 0.1) for _ in range(PSO_DIM)]

        particle = ModelParticle(
            model_id=model_id,
            model_name=config.get("name", model_id),
            role=role,
            position=position,
            velocity=velocity,
            personal_best_position=position.copy(),
            vram_gb=config.get("vram_gb", 2.0),
            ram_gb=config.get("ram_gb", 4.0),
        )

        self.particles[model_id] = particle

        if backend:
            self.backends[model_id] = backend

        logger.info(f"Registered model {model_id} with role {role.value}")
        return particle

    def set_backend(self, model_id: str, backend: Any):
        """Set backend for a registered model."""
        self.backends[model_id] = backend

    async def infer(
        self,
        prompt: str,
        strategy: Optional[SwarmStrategy] = None,
        max_models: int = 3,
        timeout: float = 30.0,
    ) -> SwarmResponse:
        """
        Run swarm inference with specified strategy.
        """
        strategy = strategy or self.default_strategy
        self.total_queries += 1

        # Analyze query
        analysis = self.analyzer.analyze(prompt)

        # Select strategy-specific method
        if strategy == SwarmStrategy.FASTEST_FIRST:
            response = await self._infer_fastest_first(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.QUALITY_FIRST:
            response = await self._infer_quality_first(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.PARALLEL_VOTE:
            response = await self._infer_parallel_vote(prompt, analysis, max_models, timeout)
        elif strategy == SwarmStrategy.MIXTURE_OF_EXPERTS:
            response = await self._infer_moe(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.SPECULATIVE_ENSEMBLE:
            response = await self._infer_speculative(prompt, analysis, timeout)
        elif strategy == SwarmStrategy.CONFIDENCE_FUSION:
            response = await self._infer_fusion(prompt, analysis, max_models, timeout)
        elif strategy == SwarmStrategy.PSO_COLLABORATIVE:
            response = await self._infer_pso(prompt, analysis, max_models, timeout)
        else:
            response = await self._infer_moe(prompt, analysis, timeout)

        # Update statistics
        self.strategy_stats[strategy.value]["count"] += 1
        self.strategy_stats[strategy.value]["latency"].append(response.latency)
        if response.confidence > 0.7:
            self.strategy_stats[strategy.value]["success"] += 1

        # Verification if enabled
        if self.enable_verification and response.confidence < 0.8:
            response = await self._verify_response(prompt, response)

        return response

    async def _infer_fastest_first(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Start with fastest model, escalate if confidence low."""
        import time

        # Sort by latency
        sorted_particles = sorted(
            self.particles.values(),
            key=lambda p: p.avg_latency
        )

        for particle in sorted_particles:
            if particle.model_id not in self.backends:
                continue

            start = time.time()
            try:
                backend = self.backends[particle.model_id]
                result = await asyncio.wait_for(
                    backend.generate(prompt),
                    timeout=timeout
                )

                latency = time.time() - start
                confidence = result.confidence_score

                particle.update_stats(True, latency, confidence)

                # Check if quality is acceptable
                if confidence >= 0.7 or particle == sorted_particles[-1]:
                    return SwarmResponse(
                        text=result.text,
                        model_id=particle.model_id,
                        model_name=particle.model_name,
                        strategy_used=SwarmStrategy.FASTEST_FIRST,
                        confidence=confidence,
                        latency=latency,
                        tokens_generated=result.tokens_generated,
                        tokens_per_second=result.tokens_per_second,
                    )

                # Low confidence - escalate
                logger.info(f"Escalating from {particle.model_id} (confidence: {confidence:.2f})")

            except asyncio.TimeoutError:
                particle.update_stats(False, timeout, 0.0)
                continue
            except Exception as e:
                logger.warning(f"Model {particle.model_id} failed: {e}")
                particle.update_stats(False, timeout, 0.0)
                continue

        return SwarmResponse(
            text="Error: All models failed",
            model_id="none",
            model_name="none",
            strategy_used=SwarmStrategy.FASTEST_FIRST,
            confidence=0.0,
        )

    async def _infer_quality_first(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Start with highest quality model."""
        import time

        # Sort by fitness (quality)
        sorted_particles = sorted(
            self.particles.values(),
            key=lambda p: p.fitness(),
            reverse=True
        )

        for particle in sorted_particles:
            if particle.model_id not in self.backends:
                continue

            start = time.time()
            try:
                backend = self.backends[particle.model_id]
                result = await asyncio.wait_for(
                    backend.generate(prompt),
                    timeout=timeout
                )

                latency = time.time() - start
                confidence = result.confidence_score

                particle.update_stats(True, latency, confidence)

                return SwarmResponse(
                    text=result.text,
                    model_id=particle.model_id,
                    model_name=particle.model_name,
                    strategy_used=SwarmStrategy.QUALITY_FIRST,
                    confidence=confidence,
                    latency=latency,
                    tokens_generated=result.tokens_generated,
                    tokens_per_second=result.tokens_per_second,
                )

            except asyncio.TimeoutError:
                particle.update_stats(False, timeout, 0.0)
                continue
            except Exception as e:
                logger.warning(f"Model {particle.model_id} failed: {e}")
                continue

        return SwarmResponse(
            text="Error: All models failed",
            model_id="none",
            model_name="none",
            strategy_used=SwarmStrategy.QUALITY_FIRST,
            confidence=0.0,
        )

    async def _infer_parallel_vote(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        max_models: int,
        timeout: float,
    ) -> SwarmResponse:
        """Run multiple models in parallel, vote on best response."""
        import time

        # Select top models
        sorted_particles = sorted(
            self.particles.values(),
            key=lambda p: p.fitness(),
            reverse=True
        )[:max_models]

        available = [p for p in sorted_particles if p.model_id in self.backends]

        if not available:
            return SwarmResponse(
                text="Error: No models available",
                model_id="none",
                model_name="none",
                strategy_used=SwarmStrategy.PARALLEL_VOTE,
                confidence=0.0,
            )

        start = time.time()

        # Run all models in parallel
        async def run_model(particle):
            try:
                backend = self.backends[particle.model_id]
                result = await asyncio.wait_for(
                    backend.generate(prompt),
                    timeout=timeout
                )
                return (particle, result)
            except Exception as e:
                logger.warning(f"Model {particle.model_id} failed: {e}")
                return (particle, None)

        results = await asyncio.gather(*[run_model(p) for p in available])

        # Filter successful results
        successful = [(p, r) for p, r in results if r is not None]

        if not successful:
            return SwarmResponse(
                text="Error: All models failed",
                model_id="none",
                model_name="none",
                strategy_used=SwarmStrategy.PARALLEL_VOTE,
                confidence=0.0,
            )

        # Vote: select response with highest confidence
        best_particle, best_result = max(successful, key=lambda x: x[1].confidence_score)

        # Calculate agreement (simplified: just confidence average)
        avg_confidence = sum(r.confidence_score for _, r in successful) / len(successful)

        latency = time.time() - start

        # Update stats for all participants
        for particle, result in successful:
            particle.update_stats(True, latency, result.confidence_score)

        return SwarmResponse(
            text=best_result.text,
            model_id=best_particle.model_id,
            model_name=best_particle.model_name,
            strategy_used=SwarmStrategy.PARALLEL_VOTE,
            confidence=best_result.confidence_score,
            agreement_score=avg_confidence,
            latency=latency,
            tokens_generated=best_result.tokens_generated,
            tokens_per_second=best_result.tokens_per_second,
            num_models_used=len(successful),
            model_contributions={p.model_id: r.confidence_score for p, r in successful},
        )

    async def _infer_moe(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Mixture of Experts - route to best specialist."""
        import time

        target_role = analysis.best_role()

        # Find best model for role
        candidates = [
            p for p in self.particles.values()
            if p.role == target_role and p.model_id in self.backends
        ]

        # Fallback to generalists
        if not candidates:
            candidates = [
                p for p in self.particles.values()
                if p.model_id in self.backends
            ]

        if not candidates:
            return SwarmResponse(
                text="Error: No models available",
                model_id="none",
                model_name="none",
                strategy_used=SwarmStrategy.MIXTURE_OF_EXPERTS,
                confidence=0.0,
            )

        # Select best candidate by fitness
        particle = max(candidates, key=lambda p: p.fitness())

        start = time.time()
        try:
            backend = self.backends[particle.model_id]
            result = await asyncio.wait_for(
                backend.generate(prompt),
                timeout=timeout
            )

            latency = time.time() - start
            particle.update_stats(True, latency, result.confidence_score)

            return SwarmResponse(
                text=result.text,
                model_id=particle.model_id,
                model_name=particle.model_name,
                strategy_used=SwarmStrategy.MIXTURE_OF_EXPERTS,
                confidence=result.confidence_score,
                latency=latency,
                tokens_generated=result.tokens_generated,
                tokens_per_second=result.tokens_per_second,
            )

        except Exception as e:
            logger.error(f"MoE inference failed: {e}")
            return SwarmResponse(
                text=f"Error: {e}",
                model_id=particle.model_id,
                model_name=particle.model_name,
                strategy_used=SwarmStrategy.MIXTURE_OF_EXPERTS,
                confidence=0.0,
            )

    async def _infer_speculative(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        timeout: float,
    ) -> SwarmResponse:
        """Speculative decoding: draft with fast model, verify with strong."""
        import time

        # Find fastest and strongest models
        available = [p for p in self.particles.values() if p.model_id in self.backends]

        if len(available) < 2:
            return await self._infer_moe(prompt, analysis, timeout)

        fastest = min(available, key=lambda p: p.avg_latency)
        strongest = max(available, key=lambda p: p.fitness())

        if fastest.model_id == strongest.model_id:
            return await self._infer_moe(prompt, analysis, timeout)

        start = time.time()

        # Draft generation
        try:
            draft_backend = self.backends[fastest.model_id]
            draft_result = await asyncio.wait_for(
                draft_backend.generate(prompt),
                timeout=timeout * 0.4
            )
        except Exception as e:
            logger.debug(f"Speculative draft failed, falling back to MoE: {e}")
            return await self._infer_moe(prompt, analysis, timeout)

        # If draft is high confidence, use it
        if draft_result.confidence_score > 0.85:
            latency = time.time() - start
            fastest.update_stats(True, latency, draft_result.confidence_score)

            return SwarmResponse(
                text=draft_result.text,
                model_id=fastest.model_id,
                model_name=fastest.model_name,
                strategy_used=SwarmStrategy.SPECULATIVE_ENSEMBLE,
                confidence=draft_result.confidence_score,
                latency=latency,
                tokens_generated=draft_result.tokens_generated,
                tokens_per_second=draft_result.tokens_per_second,
            )

        # Verify/improve with stronger model
        verify_prompt = f"""Review and improve this response if needed:

Question: {prompt}

Draft Response: {draft_result.text}

Provide the improved response:"""

        try:
            strong_backend = self.backends[strongest.model_id]
            final_result = await asyncio.wait_for(
                strong_backend.generate(verify_prompt),
                timeout=timeout * 0.6
            )

            latency = time.time() - start
            strongest.update_stats(True, latency, final_result.confidence_score)

            return SwarmResponse(
                text=final_result.text,
                model_id=strongest.model_id,
                model_name=strongest.model_name,
                strategy_used=SwarmStrategy.SPECULATIVE_ENSEMBLE,
                confidence=final_result.confidence_score,
                latency=latency,
                tokens_generated=final_result.tokens_generated,
                tokens_per_second=final_result.tokens_per_second,
                verified=True,
                verifier_model=strongest.model_name,
            )

        except Exception as e:
            # Fall back to draft
            logger.debug(f"Verification failed, using draft response: {e}")
            latency = time.time() - start
            return SwarmResponse(
                text=draft_result.text,
                model_id=fastest.model_id,
                model_name=fastest.model_name,
                strategy_used=SwarmStrategy.SPECULATIVE_ENSEMBLE,
                confidence=draft_result.confidence_score,
                latency=latency,
                tokens_generated=draft_result.tokens_generated,
                tokens_per_second=draft_result.tokens_per_second,
            )

    async def _infer_fusion(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        max_models: int,
        timeout: float,
    ) -> SwarmResponse:
        """Confidence-weighted fusion of multiple responses."""
        import time

        # Get parallel responses
        vote_response = await self._infer_parallel_vote(prompt, analysis, max_models, timeout)

        if vote_response.num_models_used <= 1:
            vote_response.strategy_used = SwarmStrategy.CONFIDENCE_FUSION
            return vote_response

        # For true fusion, we'd need to merge responses semantically
        # Simplified: just return the voted best with fusion metadata
        vote_response.strategy_used = SwarmStrategy.CONFIDENCE_FUSION
        return vote_response

    def _pso_score_model(self, particle: ModelParticle, task_type: str) -> float:
        """
        Score a model using PSO position-weighted combination of historical metrics.

        The PSO positions encode learned preferences, and this function translates
        those preferences into a concrete model selection score.
        """
        w_quality, w_speed, w_efficiency = particle.get_objective_weights()

        # Historical success rate weighted by quality preference
        quality_score = particle.success_rate * particle.avg_confidence

        # Inverse latency weighted by speed preference
        speed_score = 1.0 / max(0.1, particle.avg_latency)

        # Resource efficiency weighted by efficiency preference
        efficiency_score = 1.0 / max(0.1, particle.vram_gb)

        base_score = quality_score * w_quality + speed_score * w_speed + efficiency_score * w_efficiency

        # Task-type affinity bonus from PSO-learned position dims [6-9]
        task_affinity = particle.get_task_affinity(task_type)

        # Historical task-type performance bonus
        task_perf = particle.task_type_score(task_type)

        # Combine: base fitness + affinity weighting + task-specific history
        return base_score * (0.5 + 0.5 * task_affinity) + task_perf * 0.3

    async def _infer_pso(
        self,
        prompt: str,
        analysis: QueryAnalysis,
        max_models: int,
        timeout: float,
    ) -> SwarmResponse:
        """
        PSO-based collaborative inference.

        PSO positions drive model selection BEFORE inference:
        1. Score each model using position-weighted combination of metrics
        2. Select top-K models by PSO score
        3. Run inference with selected models (parallel if K>1)
        4. Update fitness and PSO positions based on results (feedback loop)
        5. Optionally use quantum QAOA for landscape exploration
        """
        import time

        task_type = analysis.detected_task
        available = [p for p in self.particles.values() if p.model_id in self.backends]

        if not available:
            return SwarmResponse(
                text="Error: No models available",
                model_id="none",
                model_name="none",
                strategy_used=SwarmStrategy.PSO_COLLABORATIVE,
                confidence=0.0,
            )

        # Step 1: Score all available models using PSO positions
        scored = [(p, self._pso_score_model(p, task_type)) for p in available]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 2: Select top-K models
        top_k = min(max_models, len(scored))
        selected = [p for p, _ in scored[:top_k]]

        # Get confidence threshold from best particle's PSO position
        confidence_threshold = selected[0].get_confidence_threshold()
        timeout_mult = selected[0].get_timeout_multiplier()
        adjusted_timeout = timeout * timeout_mult

        start = time.time()

        if top_k == 1:
            # Single model path - fast
            particle = selected[0]
            try:
                backend = self.backends[particle.model_id]
                result = await asyncio.wait_for(
                    backend.generate(prompt),
                    timeout=adjusted_timeout
                )
                latency = time.time() - start
                confidence = result.confidence_score

                particle.update_stats(True, latency, confidence, task_type)

                # PSO feedback: update global best and step
                self._pso_feedback(particle)

                return SwarmResponse(
                    text=result.text,
                    model_id=particle.model_id,
                    model_name=particle.model_name,
                    strategy_used=SwarmStrategy.PSO_COLLABORATIVE,
                    confidence=confidence,
                    latency=latency,
                    tokens_generated=result.tokens_generated,
                    tokens_per_second=result.tokens_per_second,
                    num_models_used=1,
                )
            except Exception as e:
                particle.update_stats(False, adjusted_timeout, 0.0, task_type)
                logger.warning(f"PSO primary model {particle.model_id} failed: {e}")
                # Fall through to fallback
        else:
            # Multi-model path: run top-K in parallel, pick best
            async def run_model(particle):
                try:
                    backend = self.backends[particle.model_id]
                    result = await asyncio.wait_for(
                        backend.generate(prompt),
                        timeout=adjusted_timeout
                    )
                    return (particle, result)
                except Exception as e:
                    logger.warning(f"PSO model {particle.model_id} failed: {e}")
                    return (particle, None)

            results = await asyncio.gather(*[run_model(p) for p in selected])
            successful = [(p, r) for p, r in results if r is not None]

            latency = time.time() - start

            if successful:
                # Pick best by confidence
                best_particle, best_result = max(successful, key=lambda x: x[1].confidence_score)

                # Update stats for all participants (feedback loop)
                for particle, result in successful:
                    particle.update_stats(True, latency, result.confidence_score, task_type)
                for particle, result in results:
                    if result is None:
                        particle.update_stats(False, adjusted_timeout, 0.0, task_type)

                # PSO feedback with best performer
                self._pso_feedback(best_particle)

                # If best confidence is below threshold and we have more models, escalate
                if best_result.confidence_score < confidence_threshold and len(scored) > top_k:
                    # Try next model in ranking as fallback
                    fallback_particle = scored[top_k][0]
                    if fallback_particle.model_id in self.backends:
                        try:
                            fb_backend = self.backends[fallback_particle.model_id]
                            fb_result = await asyncio.wait_for(
                                fb_backend.generate(prompt),
                                timeout=adjusted_timeout
                            )
                            fb_latency = time.time() - start
                            if fb_result.confidence_score > best_result.confidence_score:
                                fallback_particle.update_stats(True, fb_latency, fb_result.confidence_score, task_type)
                                return SwarmResponse(
                                    text=fb_result.text,
                                    model_id=fallback_particle.model_id,
                                    model_name=fallback_particle.model_name,
                                    strategy_used=SwarmStrategy.PSO_COLLABORATIVE,
                                    confidence=fb_result.confidence_score,
                                    latency=fb_latency,
                                    tokens_generated=fb_result.tokens_generated,
                                    tokens_per_second=fb_result.tokens_per_second,
                                    num_models_used=len(successful) + 1,
                                    model_contributions={p.model_id: r.confidence_score for p, r in successful},
                                )
                        except Exception:
                            pass  # fall through to best result

                return SwarmResponse(
                    text=best_result.text,
                    model_id=best_particle.model_id,
                    model_name=best_particle.model_name,
                    strategy_used=SwarmStrategy.PSO_COLLABORATIVE,
                    confidence=best_result.confidence_score,
                    latency=latency,
                    tokens_generated=best_result.tokens_generated,
                    tokens_per_second=best_result.tokens_per_second,
                    num_models_used=len(successful),
                    model_contributions={p.model_id: r.confidence_score for p, r in successful},
                )

        # Fallback: try any remaining models
        for particle, _ in scored:
            if particle.model_id in self.backends and particle not in selected:
                try:
                    backend = self.backends[particle.model_id]
                    result = await asyncio.wait_for(
                        backend.generate(prompt),
                        timeout=timeout
                    )
                    fb_latency = time.time() - start
                    particle.update_stats(True, fb_latency, result.confidence_score, task_type)
                    self._pso_feedback(particle)

                    return SwarmResponse(
                        text=result.text,
                        model_id=particle.model_id,
                        model_name=particle.model_name,
                        strategy_used=SwarmStrategy.PSO_COLLABORATIVE,
                        confidence=result.confidence_score,
                        latency=fb_latency,
                        tokens_generated=result.tokens_generated,
                        tokens_per_second=result.tokens_per_second,
                    )
                except Exception:
                    continue

        return SwarmResponse(
            text="Error: All models failed",
            model_id="none",
            model_name="none",
            strategy_used=SwarmStrategy.PSO_COLLABORATIVE,
            confidence=0.0,
        )

    def _pso_feedback(self, particle: ModelParticle):
        """
        PSO feedback loop: update global best and run PSO step.
        Called after each inference to close the feedback loop:
        result quality -> fitness update -> PSO position update -> next model selection.

        Every N PSO queries, also runs quantum QAOA exploration.
        """
        current_fitness = particle.fitness()
        if current_fitness > self.global_best_score:
            self.global_best_score = current_fitness
            self.global_best_position = particle.position.copy()
            self.global_best_model = particle.model_id

        # Update all particle velocities and positions
        self._pso_step()

        # Periodically run quantum-enhanced exploration
        self._pso_query_count += 1
        if self._pso_query_count % self._quantum_pso_interval == 0:
            try:
                asyncio.ensure_future(self._quantum_pso_step())
            except Exception:
                pass  # Don't fail inference for quantum exploration

    def _pso_step(self):
        """
        Perform one PSO update step for all particles.

        Positions are dimension-aware: each dimension is clamped to [0, 1]
        and interpreted semantically by ModelParticle helper methods.
        """
        for particle in self.particles.values():
            if not particle.position or not particle.personal_best_position:
                continue

            for i in range(len(particle.velocity)):
                # Random factors
                r1, r2 = random.random(), random.random()

                # Cognitive component (pull toward personal best)
                cognitive = self.pso_cognitive * r1 * (
                    particle.personal_best_position[i] - particle.position[i]
                )

                # Social component (pull toward global best)
                social = 0.0
                if self.global_best_position and i < len(self.global_best_position):
                    social = self.pso_social * r2 * (
                        self.global_best_position[i] - particle.position[i]
                    )

                # Update velocity
                particle.velocity[i] = (
                    self.pso_inertia * particle.velocity[i] +
                    cognitive + social
                )

                # Clamp velocity
                particle.velocity[i] = max(-0.5, min(0.5, particle.velocity[i]))

                # Update position (clamped to [0, 1] for all dims)
                particle.position[i] += particle.velocity[i]
                particle.position[i] = max(0.0, min(1.0, particle.position[i]))

    async def _quantum_pso_step(self):
        """
        Use QAOA for PSO landscape exploration.

        Maps particle positions to a QAOA graph where edges represent
        affinity relationships between models. The quantum result can
        shift the global best to explore regions classical PSO might miss.
        """
        try:
            from farnsworth.integration.quantum.ibm_quantum import get_quantum_provider, QAOAOptimizer, QISKIT_AVAILABLE
            if not QISKIT_AVAILABLE:
                return

            qp = get_quantum_provider()
            if qp is None:
                return

            particles_list = list(self.particles.values())
            num_particles = len(particles_list)
            if num_particles < 2:
                return

            # Build QAOA graph: edges between particles whose positions are close
            # (exploring whether nearby models should collaborate or compete)
            edges = []
            for i in range(num_particles):
                for j in range(i + 1, num_particles):
                    # Distance in PSO space
                    dist = sum(
                        (particles_list[i].position[d] - particles_list[j].position[d]) ** 2
                        for d in range(min(len(particles_list[i].position), len(particles_list[j].position)))
                    ) ** 0.5
                    # Connect nearby particles (distance < 2.0 in 10-dim unit hypercube)
                    if dist < 2.0:
                        edges.append((i, j))

            if not edges:
                return

            # Limit qubits to practical range
            num_qubits = min(num_particles, 8)

            optimizer = QAOAOptimizer(qp)
            result = await optimizer.optimize(
                num_qubits=num_qubits,
                edges=[(i, j) for i, j in edges if i < num_qubits and j < num_qubits],
                p=2,
                shots=512,
                prefer_hardware=False  # use simulator for routine PSO
            )

            if result.success and result.counts:
                # Interpret the best bitstring: bits=1 means "pull toward global best"
                best_bitstring = max(result.counts, key=result.counts.get)
                bs = best_bitstring.replace(" ", "")

                for idx, bit in enumerate(bs):
                    if idx < num_particles and bit == '1':
                        p = particles_list[idx]
                        # Quantum nudge: blend position toward global best
                        if self.global_best_position:
                            for d in range(min(len(p.position), len(self.global_best_position))):
                                p.position[d] = 0.8 * p.position[d] + 0.2 * self.global_best_position[d]
                                p.position[d] = max(0.0, min(1.0, p.position[d]))

                logger.debug(f"Quantum PSO step: nudged {sum(1 for b in bs if b == '1')}/{num_qubits} particles")

        except ImportError:
            pass  # Quantum module not available
        except Exception as e:
            logger.debug(f"Quantum PSO step failed (classical fallback): {e}")

    async def _verify_response(
        self,
        prompt: str,
        response: SwarmResponse,
    ) -> SwarmResponse:
        """Verify response with a different model."""
        # Find a verifier (preferably reasoning-focused)
        verifiers = [
            p for p in self.particles.values()
            if p.model_id != response.model_id and p.model_id in self.backends
        ]

        if not verifiers:
            return response

        # Prefer reasoning models for verification
        verifier = next(
            (p for p in verifiers if p.role == ModelRole.REASONING),
            verifiers[0]
        )

        verify_prompt = f"""Evaluate this response and rate its quality from 0-10:

Question: {prompt}

Response: {response.text}

Rating (just the number):"""

        try:
            backend = self.backends[verifier.model_id]
            result = await asyncio.wait_for(
                backend.generate(verify_prompt),
                timeout=10.0
            )

            # Parse rating
            import re
            match = re.search(r'(\d+)', result.text)
            if match:
                rating = int(match.group(1))
                response.verified = True
                response.verifier_model = verifier.model_name
                # Adjust confidence based on verification
                response.confidence = (response.confidence + rating / 10) / 2

        except Exception as e:
            logger.debug(f"Verification failed: {e}")

        return response

    def get_stats(self) -> dict:
        """Get swarm statistics including PSO state."""
        return {
            "total_queries": self.total_queries,
            "num_models": len(self.particles),
            "num_active": len(self.backends),
            "global_best_model": self.global_best_model,
            "global_best_score": self.global_best_score,
            "pso_queries": self._pso_query_count,
            "quantum_pso_runs": self._pso_query_count // self._quantum_pso_interval,
            "strategy_stats": {
                k: {
                    "count": v["count"],
                    "success_rate": v["success"] / max(1, v["count"]),
                    "avg_latency": sum(v["latency"]) / max(1, len(v["latency"])),
                }
                for k, v in self.strategy_stats.items()
            },
            "model_stats": {
                model_id: {
                    "role": p.role.value,
                    "fitness": p.fitness(),
                    "success_rate": p.success_rate,
                    "avg_latency": p.avg_latency,
                    "total_requests": p.total_requests,
                    "objective_weights": dict(zip(
                        ["quality", "speed", "efficiency"],
                        p.get_objective_weights()
                    )),
                    "temperature": p.get_temperature(),
                    "confidence_threshold": p.get_confidence_threshold(),
                    "task_affinities": {
                        t: p.get_task_affinity(t)
                        for t in ["reasoning", "coding", "creative", "general"]
                    },
                }
                for model_id, p in self.particles.items()
            },
        }

    def recommend_strategy(self, analysis: QueryAnalysis) -> SwarmStrategy:
        """Recommend best strategy based on query and history."""
        # Simple heuristics
        if analysis.complexity > 0.8:
            return SwarmStrategy.QUALITY_FIRST
        elif analysis.complexity < 0.3:
            return SwarmStrategy.FASTEST_FIRST
        elif analysis.requires_reasoning or analysis.requires_math:
            return SwarmStrategy.SPECULATIVE_ENSEMBLE
        elif len(self.backends) >= 3:
            return SwarmStrategy.PARALLEL_VOTE
        else:
            return SwarmStrategy.MIXTURE_OF_EXPERTS


# Default model configurations for swarm
DEFAULT_SWARM_MODELS = [
    {
        "id": "deepseek-r1-1.5b",
        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "strengths": ["reasoning", "math", "code"],
        "vram_gb": 2.0,
        "ram_gb": 4.0,
    },
    {
        "id": "phi-4-mini",
        "name": "Phi-4-mini",
        "strengths": ["reasoning", "math", "code"],
        "vram_gb": 3.0,
        "ram_gb": 6.0,
    },
    {
        "id": "qwen3-0.6b",
        "name": "Qwen3-0.6B",
        "strengths": ["speed", "multilingual", "fast"],
        "vram_gb": 1.0,
        "ram_gb": 2.0,
    },
    {
        "id": "smollm2-1.7b",
        "name": "SmolLM2-1.7B",
        "strengths": ["generalist", "quality"],
        "vram_gb": 1.5,
        "ram_gb": 3.0,
    },
    {
        "id": "tinyllama-1.1b",
        "name": "TinyLlama-1.1B",
        "strengths": ["speed", "fast", "edge"],
        "vram_gb": 1.0,
        "ram_gb": 2.0,
    },
]

# HuggingFace models available for local inference
HUGGINGFACE_SWARM_MODELS = [
    {
        "id": "hf-phi-3-mini",
        "name": "Phi-3-Mini-4K (HuggingFace)",
        "hf_model_id": "microsoft/Phi-3-mini-4k-instruct",
        "strengths": ["reasoning", "code", "fast"],
        "vram_gb": 4.0,
        "ram_gb": 8.0,
        "provider": "huggingface_local",
    },
    {
        "id": "hf-mistral-7b",
        "name": "Mistral-7B-Instruct (HuggingFace)",
        "hf_model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "strengths": ["generalist", "reasoning", "quality"],
        "vram_gb": 14.0,
        "ram_gb": 16.0,
        "provider": "huggingface_local",
    },
    {
        "id": "hf-codellama-7b",
        "name": "CodeLlama-7B-Instruct (HuggingFace)",
        "hf_model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "strengths": ["code", "reasoning"],
        "vram_gb": 14.0,
        "ram_gb": 16.0,
        "provider": "huggingface_local",
    },
    {
        "id": "hf-starcoder2-3b",
        "name": "StarCoder2-3B (HuggingFace)",
        "hf_model_id": "bigcode/starcoder2-3b",
        "strengths": ["code", "fast"],
        "vram_gb": 6.0,
        "ram_gb": 8.0,
        "provider": "huggingface_local",
    },
    {
        "id": "hf-qwen2.5-1.5b",
        "name": "Qwen2.5-1.5B-Instruct (HuggingFace)",
        "hf_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "strengths": ["speed", "multilingual", "reasoning"],
        "vram_gb": 3.0,
        "ram_gb": 4.0,
        "provider": "huggingface_local",
    },
    {
        "id": "hf-llama3-8b",
        "name": "Llama-3-8B-Instruct (HuggingFace)",
        "hf_model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "strengths": ["generalist", "reasoning", "quality"],
        "vram_gb": 16.0,
        "ram_gb": 20.0,
        "provider": "huggingface_local",
    },
]


def register_huggingface_models(swarm: ModelSwarm):
    """
    Register all HuggingFace local models with the swarm.

    This enables HuggingFace models to participate in:
    - PSO collaborative inference
    - Ensemble voting
    - Mixture of Experts routing
    - Speculative ensemble verification
    """
    try:
        from farnsworth.integration.external.huggingface import get_huggingface_provider

        hf_provider = get_huggingface_provider()
        if hf_provider is None:
            logger.warning("HuggingFace provider not available")
            return 0

        registered = 0
        for model_config in HUGGINGFACE_SWARM_MODELS:
            try:
                particle = swarm.register_model(model_config)

                # Create a wrapper backend for swarm inference
                class HFBackendWrapper:
                    def __init__(self, provider, model_id):
                        self.provider = provider
                        self.model_id = model_id

                    async def generate(self, prompt: str):
                        result = await self.provider.chat(
                            prompt=prompt,
                            model=self.model_id,
                            prefer_local=True
                        )
                        # Return result in expected format
                        from dataclasses import dataclass

                        @dataclass
                        class GenerateResult:
                            text: str
                            confidence_score: float = 0.8
                            tokens_generated: int = 0
                            tokens_per_second: float = 0.0

                        return GenerateResult(
                            text=result.get("content", ""),
                            confidence_score=0.8 if result.get("content") else 0.0,
                            tokens_generated=result.get("tokens", 0),
                        )

                backend = HFBackendWrapper(hf_provider, model_config.get("hf_model_id"))
                swarm.set_backend(model_config["id"], backend)

                registered += 1
                logger.info(f"Registered HuggingFace model: {model_config['name']}")

            except Exception as e:
                logger.warning(f"Failed to register HF model {model_config['id']}: {e}")

        logger.info(f"Registered {registered} HuggingFace models with swarm")
        return registered

    except ImportError:
        logger.warning("HuggingFace integration not available")
        return 0
