"""
Integration tests for PSO-driven model selection.

Tests that PSO particle positions meaningfully drive model
selection and that the feedback loop works correctly.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestModelSwarmImports:
    """Test model swarm module imports correctly."""

    def test_import(self):
        """ModelSwarm should be importable."""
        from farnsworth.core.model_swarm import ModelSwarm, SwarmStrategy, ModelParticle
        assert ModelSwarm is not None

    def test_strategy_enum(self):
        """SwarmStrategy should have all expected strategies."""
        from farnsworth.core.model_swarm import SwarmStrategy

        expected = ["fastest_first", "quality_first", "parallel_vote",
                     "moe", "speculative", "fusion", "pso"]
        for strategy in expected:
            assert hasattr(SwarmStrategy, strategy.upper()) or \
                   any(s.value == strategy for s in SwarmStrategy), \
                   f"Missing strategy: {strategy}"

    def test_model_roles(self):
        """ModelRole should have expected roles."""
        from farnsworth.core.model_swarm import ModelRole

        expected = ["generalist", "reasoning", "coding", "creative"]
        for role in expected:
            assert any(r.value == role for r in ModelRole), f"Missing role: {role}"


class TestModelParticle:
    """Test PSO particle functionality."""

    def test_particle_fitness(self):
        """Particle fitness should be calculable."""
        from farnsworth.core.model_swarm import ModelParticle, ModelRole

        particle = ModelParticle(
            model_id="test",
            model_name="Test Model",
            role=ModelRole.GENERALIST,
            success_rate=0.8,
            avg_latency=0.5,
            avg_confidence=0.9,
        )

        fitness = particle.fitness()
        assert 0 <= fitness <= 10  # Reasonable fitness range
        assert fitness > 0  # Should be positive for good stats

    def test_particle_update_stats(self):
        """Updating stats should use EMA smoothing."""
        from farnsworth.core.model_swarm import ModelParticle, ModelRole

        particle = ModelParticle(
            model_id="test",
            model_name="Test",
            role=ModelRole.GENERALIST,
            success_rate=0.5,
            avg_latency=1.0,
            avg_confidence=0.5,
        )

        initial_rate = particle.success_rate
        particle.update_stats(success=True, latency=0.1, confidence=0.95)

        # Stats should move toward new values
        assert particle.success_rate > initial_rate
        assert particle.avg_latency < 1.0
        assert particle.avg_confidence > 0.5
        assert particle.total_requests == 1
        assert particle.successful_requests == 1

    def test_personal_best_updates(self):
        """Personal best should update when fitness improves."""
        from farnsworth.core.model_swarm import ModelParticle, ModelRole

        particle = ModelParticle(
            model_id="test",
            model_name="Test",
            role=ModelRole.GENERALIST,
            position=[0.5, 0.5, 0.5],
            velocity=[0.0, 0.0, 0.0],
            personal_best_position=[0.5, 0.5, 0.5],
            personal_best_score=0.0,
        )

        # Good result should update personal best
        particle.update_stats(success=True, latency=0.1, confidence=0.99)
        assert particle.personal_best_score > 0


class TestQueryAnalysis:
    """Test query analysis for routing."""

    def test_code_detection(self):
        """Should detect code-related queries."""
        from farnsworth.core.model_swarm import QueryAnalyzer, ModelRole

        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("Write a Python function to sort a list")

        assert analysis.requires_code is True
        assert analysis.best_role() == ModelRole.CODING

    def test_math_detection(self):
        """Should detect math queries."""
        from farnsworth.core.model_swarm import QueryAnalyzer, ModelRole

        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("Calculate the derivative of x^2 + 3x")

        assert analysis.requires_math is True
        assert analysis.best_role() == ModelRole.MATH

    def test_general_query(self):
        """General queries should route to generalist."""
        from farnsworth.core.model_swarm import QueryAnalyzer, ModelRole

        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("Hello, how are you?")

        assert analysis.best_role() == ModelRole.GENERALIST or analysis.best_role() == ModelRole.SPEED


class TestPSOStep:
    """Test PSO update mechanics."""

    def test_pso_velocity_clamping(self):
        """Velocities should be clamped within bounds."""
        from farnsworth.core.model_swarm import ModelSwarm

        swarm = ModelSwarm.__new__(ModelSwarm)
        swarm.particles = {}
        swarm.backends = {}
        swarm.pso_inertia = 0.7
        swarm.pso_cognitive = 1.5
        swarm.pso_social = 1.5
        swarm.global_best_position = [0.5] * 10
        swarm.global_best_score = 0.5
        swarm.global_best_model = None

        from farnsworth.core.model_swarm import ModelParticle, ModelRole

        particle = ModelParticle(
            model_id="test",
            model_name="Test",
            role=ModelRole.GENERALIST,
            position=[0.5] * 10,
            velocity=[0.0] * 10,
            personal_best_position=[0.5] * 10,
            personal_best_score=0.3,
        )
        swarm.particles["test"] = particle

        swarm._pso_step()

        # Check velocities are clamped
        for v in particle.velocity:
            assert -0.5 <= v <= 0.5

        # Check positions are in [0, 1]
        for p in particle.position:
            assert 0 <= p <= 1
