"""
Farnsworth Evolution Module

Genetic optimization for self-improvement with:
- NSGA-II multi-objective optimization
- Behavioral genome encoding for swarm evolution
- LoRA adapter breeding and merging
- Hash-chain evolution logging for integrity
"""

from farnsworth.evolution.genetic_optimizer import GeneticOptimizer
from farnsworth.evolution.lora_evolver import LoRAEvolver
from farnsworth.evolution.behavior_mutation import BehaviorMutator
from farnsworth.evolution.fitness_tracker import FitnessTracker

__all__ = [
    "GeneticOptimizer",
    "LoRAEvolver",
    "BehaviorMutator",
    "FitnessTracker",
]
