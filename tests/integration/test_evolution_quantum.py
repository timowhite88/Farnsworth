"""
Integration tests for quantum evolution system.

Tests that the genetic optimizer properly integrates with
QuantumGeneticOptimizer from IBM Quantum, with classical fallback.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestGeneticOptimizer:
    """Test genetic optimizer with quantum crossover integration."""

    def test_import(self):
        """GeneticOptimizer should be importable."""
        from farnsworth.evolution.genetic_optimizer import GeneticOptimizer, Gene, Genome
        assert GeneticOptimizer is not None
        assert Gene is not None
        assert Genome is not None

    def test_gene_mutation(self):
        """Gene mutation should produce valid values within bounds."""
        from farnsworth.evolution.genetic_optimizer import Gene

        gene = Gene(name="test", value=0.5, min_val=0.0, max_val=1.0, mutation_sigma=0.1)
        mutated = gene.mutate()

        assert mutated.name == "test"
        assert mutated.min_val <= mutated.value <= mutated.max_val

    def test_genome_creation(self):
        """Genome should be creatable with multiple genes."""
        from farnsworth.evolution.genetic_optimizer import Gene, Genome

        genes = {
            "exploration": Gene("exploration", 0.5, 0.0, 1.0),
            "learning_rate": Gene("learning_rate", 0.01, 0.001, 0.1),
        }
        genome = Genome(genes=genes)

        assert len(genome.genes) == 2
        assert "exploration" in genome.genes


class TestQuantumEvolutionEngine:
    """Test quantum evolution integration."""

    def test_import(self):
        """QuantumEvolutionEngine should be importable."""
        from farnsworth.evolution.quantum_evolution import QuantumEvolutionEngine
        assert QuantumEvolutionEngine is not None

    @pytest.mark.asyncio
    async def test_classical_fallback(self):
        """When quantum is unavailable, should fall back to classical."""
        from farnsworth.evolution.quantum_evolution import QuantumEvolutionEngine

        engine = QuantumEvolutionEngine()
        # Don't initialize quantum - test classical path
        result = await engine.evolve_agent(
            agent_id="test_agent",
            initial_params={"exploration": 0.5, "learning_rate": 0.01},
            generations=2,
            population_size=4,
            use_quantum=False,
        )

        assert result is not None
        assert result.best_genome is not None or isinstance(result, dict)


class TestEvolutionModuleExports:
    """Test that all evolution submodules are properly exported."""

    def test_behavior_mutation_importable(self):
        """BehaviorMutator should be importable."""
        from farnsworth.evolution.behavior_mutation import BehaviorMutator
        assert BehaviorMutator is not None

    def test_lora_evolver_importable(self):
        """LoRAEvolver should be importable."""
        from farnsworth.evolution.lora_evolver import LoRAEvolver
        assert LoRAEvolver is not None

    def test_federated_population_importable(self):
        """FederatedPopulationManager should be importable."""
        from farnsworth.evolution.federated_population import FederatedPopulationManager
        assert FederatedPopulationManager is not None

    def test_fitness_tracker_importable(self):
        """FitnessTracker should be importable."""
        from farnsworth.evolution.fitness_tracker import FitnessTracker
        assert FitnessTracker is not None

    def test_module_init_exports(self):
        """All expected classes should be in __init__.py exports."""
        from farnsworth.evolution import (
            GeneticOptimizer,
            LoRAEvolver,
            BehaviorMutator,
            FitnessTracker,
            FederatedPopulationManager,
        )
        assert all([GeneticOptimizer, LoRAEvolver, BehaviorMutator,
                     FitnessTracker, FederatedPopulationManager])


class TestQuantumAvailability:
    """Test quantum availability detection."""

    def test_qiskit_availability_flag(self):
        """QISKIT_AVAILABLE flag should be defined."""
        try:
            from farnsworth.integration.quantum.ibm_quantum import QISKIT_AVAILABLE
            assert isinstance(QISKIT_AVAILABLE, bool)
        except ImportError:
            # Module itself may not be importable without qiskit
            pytest.skip("Quantum module not importable")

    def test_quantum_provider_factory(self):
        """get_quantum_provider should return provider or None."""
        try:
            from farnsworth.integration.quantum.ibm_quantum import get_quantum_provider
            provider = get_quantum_provider()
            # May be None if not configured, that's OK
            assert provider is None or hasattr(provider, "run_quantum_circuit")
        except ImportError:
            pytest.skip("Quantum module not importable")
