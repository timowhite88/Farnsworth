"""
Farnsworth Fitness Tracker - Performance Metrics Collection

Tracks and aggregates fitness metrics for evolution:
- Task success rate
- Response quality
- Efficiency (tokens, time)
- User satisfaction
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from collections import defaultdict

from loguru import logger


@dataclass
class FitnessMetric:
    """A single fitness measurement."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict = field(default_factory=dict)


@dataclass
class FitnessSnapshot:
    """Aggregated fitness at a point in time."""
    timestamp: datetime
    metrics: dict[str, float]
    sample_count: int
    genome_id: Optional[str] = None


class FitnessTracker:
    """
    Tracks and aggregates fitness metrics over time.

    Features:
    - Rolling window statistics
    - Multi-dimensional fitness tracking
    - Genome-specific performance
    - Trend detection
    """

    def __init__(
        self,
        window_size: int = 100,
        snapshot_interval_minutes: int = 30,
    ):
        self.window_size = window_size
        self.snapshot_interval = timedelta(minutes=snapshot_interval_minutes)

        # Metric storage (rolling windows)
        self.metrics: dict[str, list[FitnessMetric]] = defaultdict(list)

        # Genome-specific metrics
        self.genome_metrics: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Snapshots
        self.snapshots: list[FitnessSnapshot] = []
        self.last_snapshot: Optional[datetime] = None

        # Fitness weights (importance of each metric)
        self.weights: dict[str, float] = {
            "task_success": 0.4,
            "efficiency": 0.3,
            "user_satisfaction": 0.3,
        }

    def record(
        self,
        metric_name: str,
        value: float,
        genome_id: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """Record a fitness metric."""
        metric = FitnessMetric(
            name=metric_name,
            value=value,
            context=context or {},
        )

        # Add to global metrics
        self.metrics[metric_name].append(metric)

        # Trim to window size
        if len(self.metrics[metric_name]) > self.window_size:
            self.metrics[metric_name] = self.metrics[metric_name][-self.window_size:]

        # Add to genome-specific metrics
        if genome_id:
            self.genome_metrics[genome_id][metric_name].append(value)
            # Trim genome metrics too
            if len(self.genome_metrics[genome_id][metric_name]) > self.window_size:
                self.genome_metrics[genome_id][metric_name] = \
                    self.genome_metrics[genome_id][metric_name][-self.window_size:]

        # Check if snapshot needed
        self._maybe_snapshot()

    def record_task_outcome(
        self,
        success: bool,
        tokens_used: int,
        time_seconds: float,
        user_feedback: Optional[float] = None,
        genome_id: Optional[str] = None,
    ):
        """Convenience method to record task outcome."""
        # Task success
        self.record("task_success", 1.0 if success else 0.0, genome_id)

        # Efficiency (inverse of tokens, normalized)
        efficiency = 1.0 / (1.0 + tokens_used / 1000)
        self.record("efficiency", efficiency, genome_id)

        # Time efficiency
        time_efficiency = 1.0 / (1.0 + time_seconds / 10)
        self.record("time_efficiency", time_efficiency, genome_id)

        # User satisfaction
        if user_feedback is not None:
            self.record("user_satisfaction", user_feedback, genome_id)

    def get_current_fitness(self, genome_id: Optional[str] = None) -> dict[str, float]:
        """Get current fitness scores."""
        if genome_id and genome_id in self.genome_metrics:
            metrics = self.genome_metrics[genome_id]
            return {
                name: sum(values) / len(values) if values else 0.0
                for name, values in metrics.items()
            }

        # Global metrics
        return {
            name: sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
            for name, metrics in self.metrics.items()
        }

    def get_weighted_fitness(self, genome_id: Optional[str] = None) -> float:
        """Get weighted total fitness."""
        scores = self.get_current_fitness(genome_id)

        total = 0.0
        weight_sum = 0.0
        for name, score in scores.items():
            weight = self.weights.get(name, 0.1)
            total += score * weight
            weight_sum += weight

        return total / max(0.001, weight_sum)

    def evaluate_genome(self, genome_id: str) -> dict[str, float]:
        """
        Evaluate fitness for a specific genome.

        Returns fitness scores suitable for genetic optimizer.
        """
        if genome_id not in self.genome_metrics:
            return {}

        metrics = self.genome_metrics[genome_id]
        return {
            name: sum(values) / len(values) if values else 0.0
            for name, values in metrics.items()
        }

    def _maybe_snapshot(self):
        """Create snapshot if interval has passed."""
        now = datetime.now()

        if self.last_snapshot and now - self.last_snapshot < self.snapshot_interval:
            return

        snapshot = FitnessSnapshot(
            timestamp=now,
            metrics=self.get_current_fitness(),
            sample_count=sum(len(m) for m in self.metrics.values()),
        )
        self.snapshots.append(snapshot)
        self.last_snapshot = now

        # Keep bounded snapshots
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-500:]

    def get_trend(self, metric_name: str, periods: int = 5) -> float:
        """
        Calculate trend for a metric.

        Returns positive for improving, negative for declining.
        """
        relevant_snapshots = [
            s for s in self.snapshots
            if metric_name in s.metrics
        ][-periods:]

        if len(relevant_snapshots) < 2:
            return 0.0

        values = [s.metrics[metric_name] for s in relevant_snapshots]

        # Simple linear trend
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def set_weight(self, metric_name: str, weight: float):
        """Set weight for a metric."""
        self.weights[metric_name] = weight

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "metrics_tracked": list(self.metrics.keys()),
            "sample_counts": {name: len(metrics) for name, metrics in self.metrics.items()},
            "genomes_tracked": len(self.genome_metrics),
            "snapshot_count": len(self.snapshots),
            "current_fitness": self.get_current_fitness(),
            "weighted_fitness": self.get_weighted_fitness(),
            "trends": {name: self.get_trend(name) for name in self.metrics.keys()},
        }

    def get_leaderboard(self, top_k: int = 10) -> list[tuple[str, float]]:
        """Get top performing genomes."""
        genome_scores = [
            (gid, self.get_weighted_fitness(gid))
            for gid in self.genome_metrics.keys()
        ]
        return sorted(genome_scores, key=lambda x: x[1], reverse=True)[:top_k]
