"""
Tests for AGI Memory System Upgrades.

Tests:
- Differential privacy noise calibration
- Semantic merge threshold
- Cross-attention reranking
- Proactive compaction triggers
- Local model routing
- Concept drift detection
- Importance decay formula
"""

import asyncio
import math
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================================
# Test: Differential Privacy Noise Calibration
# ============================================================================

def test_differential_privacy_noise():
    """Verify noise calibration follows Laplace mechanism."""
    from farnsworth.memory.memory_system import MemorySystem

    system = MemorySystem()

    # Test with epsilon=1.0 (moderate privacy)
    embedding = [0.5] * 384  # Standard dimension
    noisy = system._add_differential_noise(embedding, epsilon=1.0, sensitivity=1.0)

    # Verify output is same dimension
    assert len(noisy) == len(embedding)

    # Verify output is normalized (approximately unit length)
    norm = math.sqrt(sum(x**2 for x in noisy))
    assert 0.9 < norm < 1.1, f"Noisy embedding should be normalized, got norm={norm}"

    # Verify noise was actually added (not identical)
    diff = sum(abs(a - b) for a, b in zip(embedding, noisy))
    assert diff > 0.01, "Noise should have been added"


def test_differential_privacy_epsilon_scale():
    """Lower epsilon should add more noise."""
    from farnsworth.memory.memory_system import MemorySystem

    system = MemorySystem()
    embedding = [0.5] * 384

    # High privacy (low epsilon) should have more variance
    diffs_low_epsilon = []
    for _ in range(10):
        noisy = system._add_differential_noise(embedding, epsilon=0.1)
        diff = sum(abs(a - b) for a, b in zip(embedding, noisy))
        diffs_low_epsilon.append(diff)

    # Low privacy (high epsilon) should have less variance
    diffs_high_epsilon = []
    for _ in range(10):
        noisy = system._add_differential_noise(embedding, epsilon=10.0)
        diff = sum(abs(a - b) for a, b in zip(embedding, noisy))
        diffs_high_epsilon.append(diff)

    avg_low = sum(diffs_low_epsilon) / len(diffs_low_epsilon)
    avg_high = sum(diffs_high_epsilon) / len(diffs_high_epsilon)

    # Low epsilon should produce larger average difference
    assert avg_low > avg_high, f"Lower epsilon should mean more noise: {avg_low} vs {avg_high}"


# ============================================================================
# Test: Hybrid Weights Configuration
# ============================================================================

def test_hybrid_weights_normalization():
    """Verify HybridWeights normalizes to sum=1.0."""
    from farnsworth.memory.archival_memory import HybridWeights

    weights = HybridWeights(semantic=2.0, keyword=1.0, temporal=1.0, graph=0.5, attention=0.5)
    normalized = weights.normalize()

    total = (
        normalized.semantic +
        normalized.keyword +
        normalized.temporal +
        normalized.graph +
        normalized.attention
    )
    assert abs(total - 1.0) < 0.001, f"Normalized weights should sum to 1.0, got {total}"


# ============================================================================
# Test: Cost-Sensitive Budget Status
# ============================================================================

@pytest.mark.asyncio
async def test_budget_status_recommendations():
    """Verify budget status generates appropriate recommendations."""
    from farnsworth.memory.working_memory import WorkingMemory, SlotType

    wm = WorkingMemory(max_slots=50)

    # Add some content
    await wm.set("test1", "x" * 1000, SlotType.SCRATCH)
    await wm.set("test2", "y" * 20000, SlotType.SCRATCH)  # Large slot

    status = wm.get_budget_status(token_budget=10000, daily_cost_limit=1.0)

    # Should identify large slot
    assert "test2" in status.slots_over_budget, "Should identify large slot"
    assert len(status.recommendations) > 0, "Should have recommendations"


@pytest.mark.asyncio
async def test_token_estimation():
    """Verify token estimation is approximately correct."""
    from farnsworth.memory.working_memory import WorkingMemory

    wm = WorkingMemory()

    # ~4 chars per token
    tokens = wm._estimate_tokens("Hello world!")  # 12 chars
    assert 2 <= tokens <= 5, f"Expected 2-5 tokens for 'Hello world!', got {tokens}"


# ============================================================================
# Test: Context Compaction
# ============================================================================

def test_compaction_result_ratio():
    """Verify CompactionResult calculates compression ratio correctly."""
    from farnsworth.memory.virtual_context import CompactionResult

    result = CompactionResult(
        blocks_compacted=5,
        tokens_saved=500,
        original_tokens=1000,
        final_tokens=500,
        summaries_created=5,
        archival_links=["a", "b", "c"],
    )

    ratio = result.compression_ratio()
    assert ratio == 0.5, f"Expected 50% compression ratio, got {ratio}"


def test_allocation_plan():
    """Verify token allocation planning."""
    from farnsworth.memory.virtual_context import ContextWindow, MemoryBlock

    window = ContextWindow(max_tokens=1000, reserve_tokens=100)

    # Add a low-importance block
    block = MemoryBlock(id="test", content="x" * 400, importance_score=0.3)
    window.add_block(block)

    # Plan allocation for high-priority content
    plan = window._token_aware_allocation(required_tokens=500, priority=0.8)

    # Should suggest evicting the low-importance block
    assert "test" in plan.slots_to_evict or plan.can_fit


# ============================================================================
# Test: Adaptive Schema Manager
# ============================================================================

@pytest.mark.asyncio
async def test_concept_drift_detection():
    """Verify concept drift detection with EMA centroids."""
    from farnsworth.memory.memory_system import AdaptiveSchemaManager

    manager = AdaptiveSchemaManager(drift_threshold=0.3, min_samples_for_drift=3)

    # First sample - initializes centroid
    embedding1 = [1.0, 0.0, 0.0] + [0.0] * 381
    result1 = await manager.detect_concept_drift("test_concept", embedding1)

    assert result1.samples_analyzed == 1
    assert not result1.is_significant, "First sample shouldn't be significant"

    # Similar samples - no drift
    for _ in range(5):
        embedding_similar = [0.95, 0.1, 0.0] + [0.0] * 381
        result = await manager.detect_concept_drift("test_concept", embedding_similar)

    # Drifted sample - should detect
    embedding_drifted = [0.0, 1.0, 0.0] + [0.0] * 381  # Orthogonal
    result_drifted = await manager.detect_concept_drift("test_concept", embedding_drifted)

    assert result_drifted.drift_magnitude > 0.3, "Should detect drift"


def test_importance_decay():
    """Verify importance decay formula."""
    from farnsworth.memory.memory_system import AdaptiveSchemaManager, MemorySearchResult

    manager = AdaptiveSchemaManager(decay_halflife_hours=24.0)

    # Create test memories with different ages
    memories = [
        MemorySearchResult(
            content="recent",
            source="archival",
            score=1.0,
            metadata={"created_at": datetime.now().isoformat(), "access_count": 5},
        ),
        MemorySearchResult(
            content="old",
            source="archival",
            score=1.0,
            metadata={
                "created_at": (datetime.now() - timedelta(hours=48)).isoformat(),
                "access_count": 1,
            },
        ),
    ]

    decayed = manager.apply_importance_decay(memories)

    # Recent memory should have higher score after decay
    assert decayed[0].content == "recent", "Recent memory should rank higher"
    assert decayed[0].score > decayed[1].score, "Recent memory should have higher score"


@pytest.mark.asyncio
async def test_schema_evolution():
    """Verify schema evolution actions."""
    from farnsworth.memory.memory_system import AdaptiveSchemaManager, DriftResult

    manager = AdaptiveSchemaManager(drift_threshold=0.3)

    # Initialize concept
    embedding = [1.0, 0.0, 0.0] + [0.0] * 381
    await manager.detect_concept_drift("evolving", embedding)

    # Test moderate drift - should update centroid
    moderate_drift = DriftResult(
        concept_name="evolving",
        drift_magnitude=0.35,
        drift_direction=[0.9, 0.3, 0.0] + [0.0] * 381,
        samples_analyzed=10,
        is_significant=True,
        recommended_action="update_centroid",
    )

    evolution = await manager.evolve_schema("evolving", moderate_drift)
    assert evolution.action_taken == "update_centroid"

    # Test high drift - should create branch
    high_drift = DriftResult(
        concept_name="evolving",
        drift_magnitude=0.6,
        drift_direction=[0.0, 1.0, 0.0] + [0.0] * 381,
        samples_analyzed=20,
        is_significant=True,
        recommended_action="create_branch",
    )

    evolution2 = await manager.evolve_schema("evolving", high_drift)
    assert "create_branch" in evolution2.action_taken


# ============================================================================
# Test: AGI Config Loading
# ============================================================================

def test_agi_config_defaults():
    """Verify MemoryAGIConfig has sensible defaults."""
    from farnsworth.memory.memory_system import MemoryAGIConfig

    config = MemoryAGIConfig()

    # All features should be enabled by default
    assert config.sync_enabled is True
    assert config.hybrid_enabled is True
    assert config.proactive_context is True
    assert config.cost_aware is True
    assert config.drift_detection is True

    # Check default values
    assert config.sync_epsilon == 1.0
    assert config.drift_threshold == 0.3
    assert config.decay_halflife == 24.0


def test_agi_config_from_env():
    """Verify MemoryAGIConfig loads from environment."""
    from farnsworth.memory.memory_system import MemoryAGIConfig
    import os

    # Set test environment variables
    os.environ["FARNSWORTH_SYNC_ENABLED"] = "false"
    os.environ["FARNSWORTH_DRIFT_THRESHOLD"] = "0.5"

    config = MemoryAGIConfig.from_env()

    assert config.sync_enabled is False
    assert config.drift_threshold == 0.5

    # Clean up
    del os.environ["FARNSWORTH_SYNC_ENABLED"]
    del os.environ["FARNSWORTH_DRIFT_THRESHOLD"]


# ============================================================================
# Test: Sync Result
# ============================================================================

def test_sync_result_summary():
    """Verify SyncResult generates human-readable summary."""
    from farnsworth.memory.memory_system import SyncResult

    result = SyncResult(
        pushed_count=5,
        pulled_count=3,
        merged_count=2,
        conflicts_resolved=1,
        privacy_budget_used=0.5,
        sync_timestamp=datetime.now(),
        peer_id="test_peer",
    )

    summary = result.summary()
    assert "test_peer" in summary
    assert "pushed=5" in summary
    assert "pulled=3" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
