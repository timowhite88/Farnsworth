"""
Farnsworth Evolution Module

Genetic optimization for self-improvement with:
- NSGA-II multi-objective optimization
- Behavioral genome encoding for swarm evolution
- LoRA adapter breeding and merging
- Hash-chain evolution logging for integrity
- Federated population evolution across P2P (AGI Cohesion)
- Quantum-enhanced evolution via IBM Quantum (AGI v1.8)
"""

from farnsworth.evolution.genetic_optimizer import GeneticOptimizer
from farnsworth.evolution.lora_evolver import LoRAEvolver
from farnsworth.evolution.behavior_mutation import BehaviorMutator
from farnsworth.evolution.fitness_tracker import FitnessTracker
from farnsworth.evolution.federated_population import (
    FederatedPopulationManager,
    FederatedEvolutionConfig,
    setup_federated_evolution,
)
from farnsworth.evolution.quantum_evolution import (
    QuantumEvolutionEngine,
    get_quantum_evolution_engine,
    quantum_evolve_agent_params,
)

__all__ = [
    "GeneticOptimizer",
    "LoRAEvolver",
    "BehaviorMutator",
    "FitnessTracker",
    "FederatedPopulationManager",
    "FederatedEvolutionConfig",
    "setup_federated_evolution",
    "QuantumEvolutionEngine",
    "get_quantum_evolution_engine",
    "quantum_evolve_agent_params",
]
