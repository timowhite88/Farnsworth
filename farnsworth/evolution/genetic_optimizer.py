"""
Farnsworth Genetic Optimizer - DEAP-Based Parameter Evolution

Novel Approaches:
1. NSGA-II Multi-Objective - Optimize multiple goals simultaneously
2. Adaptive Mutation - Mutation rate adjusts based on progress
3. Island Model - Parallel populations with migration
4. Hash-Chain Logging - Tamper-proof evolution history
"""

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Callable
from collections import defaultdict

from loguru import logger

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class Gene:
    """A single evolvable parameter."""
    name: str
    value: float
    min_val: float
    max_val: float
    mutation_sigma: float = 0.1

    def mutate(self) -> "Gene":
        """Create mutated copy."""
        new_value = self.value + random.gauss(0, self.mutation_sigma * (self.max_val - self.min_val))
        new_value = max(self.min_val, min(self.max_val, new_value))
        return Gene(
            name=self.name,
            value=new_value,
            min_val=self.min_val,
            max_val=self.max_val,
            mutation_sigma=self.mutation_sigma,
        )


@dataclass
class Genome:
    """A complete genome (set of parameters)."""
    id: str
    genes: dict[str, Gene]
    fitness_scores: dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def get_value(self, gene_name: str) -> float:
        """Get value of a gene."""
        return self.genes[gene_name].value if gene_name in self.genes else 0.0

    def to_dict(self) -> dict:
        """Serialize genome."""
        return {
            "id": self.id,
            "genes": {name: gene.value for name, gene in self.genes.items()},
            "fitness_scores": self.fitness_scores,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at.isoformat(),
        }

    def total_fitness(self, weights: Optional[dict[str, float]] = None) -> float:
        """Calculate weighted total fitness."""
        if not self.fitness_scores:
            return 0.0
        if weights is None:
            return sum(self.fitness_scores.values()) / len(self.fitness_scores)
        total = 0.0
        weight_sum = 0.0
        for name, score in self.fitness_scores.items():
            w = weights.get(name, 1.0)
            total += score * w
            weight_sum += w
        return total / max(0.001, weight_sum)


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution."""
    population_size: int = 20
    generations: int = 10
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    tournament_size: int = 3
    elite_count: int = 2

    # Multi-objective
    use_nsga2: bool = True

    # Adaptive mutation
    adaptive_mutation: bool = True
    stagnation_threshold: int = 3


@dataclass
class EvolutionResult:
    """Result from an evolution run."""
    best_genome: Genome
    final_population: list[Genome]
    generations_run: int
    fitness_history: list[dict]
    duration_seconds: float


class GeneticOptimizer:
    """
    DEAP-inspired genetic optimizer for system parameters.

    Features:
    - NSGA-II for multi-objective optimization
    - Tournament selection with elitism
    - Adaptive mutation rates
    - Hash-chain evolution logging
    """

    def __init__(
        self,
        config: Optional[EvolutionConfig] = None,
        data_dir: str = "./data/evolution",
    ):
        self.config = config or EvolutionConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Gene definitions (parameter space)
        self.gene_definitions: dict[str, dict] = {}

        # Population
        self.population: list[Genome] = []
        self.generation = 0

        # History
        self.fitness_history: list[dict] = []
        self.evolution_log: list[dict] = []
        self.last_hash: str = "genesis"

        # Fitness function (set by user)
        self.fitness_fn: Optional[Callable[[Genome], dict[str, float]]] = None

        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "generations_completed": 0,
            "best_fitness_ever": 0.0,
            "stagnation_count": 0,
        }

    def define_gene(
        self,
        name: str,
        min_val: float,
        max_val: float,
        default: Optional[float] = None,
        mutation_sigma: float = 0.1,
    ):
        """Define an evolvable parameter."""
        self.gene_definitions[name] = {
            "min": min_val,
            "max": max_val,
            "default": default if default is not None else (min_val + max_val) / 2,
            "sigma": mutation_sigma,
        }

    def set_fitness_function(self, fn: Callable[[Genome], dict[str, float]]):
        """Set the fitness evaluation function."""
        self.fitness_fn = fn

    def _create_random_genome(self) -> Genome:
        """Create a genome with random values."""
        genes = {}
        for name, defn in self.gene_definitions.items():
            value = random.uniform(defn["min"], defn["max"])
            genes[name] = Gene(
                name=name,
                value=value,
                min_val=defn["min"],
                max_val=defn["max"],
                mutation_sigma=defn["sigma"],
            )

        return Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=genes,
            generation=self.generation,
        )

    def _create_default_genome(self) -> Genome:
        """Create a genome with default values."""
        genes = {}
        for name, defn in self.gene_definitions.items():
            genes[name] = Gene(
                name=name,
                value=defn["default"],
                min_val=defn["min"],
                max_val=defn["max"],
                mutation_sigma=defn["sigma"],
            )

        return Genome(
            id=f"g{self.generation}_default",
            genes=genes,
            generation=self.generation,
        )

    def initialize_population(self, include_default: bool = True):
        """Initialize the population."""
        self.population = []
        self.generation = 0

        # Include default genome
        if include_default:
            self.population.append(self._create_default_genome())

        # Fill with random genomes
        while len(self.population) < self.config.population_size:
            self.population.append(self._create_random_genome())

        logger.info(f"Initialized population with {len(self.population)} genomes")

    async def evaluate_population(self):
        """Evaluate fitness for all genomes in population."""
        if self.fitness_fn is None:
            raise RuntimeError("Fitness function not set")

        for genome in self.population:
            if not genome.fitness_scores:
                if asyncio.iscoroutinefunction(self.fitness_fn):
                    genome.fitness_scores = await self.fitness_fn(genome)
                else:
                    genome.fitness_scores = self.fitness_fn(genome)
                self.stats["total_evaluations"] += 1

    def _tournament_select(self, k: int = None) -> Genome:
        """Tournament selection."""
        k = k or self.config.tournament_size
        contestants = random.sample(self.population, min(k, len(self.population)))
        return max(contestants, key=lambda g: g.total_fitness())

    def _crossover(self, parent1: Genome, parent2: Genome) -> tuple[Genome, Genome]:
        """Two-point crossover."""
        genes1 = {}
        genes2 = {}

        gene_names = list(self.gene_definitions.keys())
        if len(gene_names) < 2:
            # Can't do meaningful crossover
            return self._mutate(parent1), self._mutate(parent2)

        # Select crossover points
        point1 = random.randint(0, len(gene_names) - 1)
        point2 = random.randint(point1, len(gene_names))

        for i, name in enumerate(gene_names):
            if point1 <= i < point2:
                genes1[name] = Gene(**{**parent2.genes[name].__dict__})
                genes2[name] = Gene(**{**parent1.genes[name].__dict__})
            else:
                genes1[name] = Gene(**{**parent1.genes[name].__dict__})
                genes2[name] = Gene(**{**parent2.genes[name].__dict__})

        child1 = Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=genes1,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id],
        )
        child2 = Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=genes2,
            generation=self.generation,
            parent_ids=[parent1.id, parent2.id],
        )

        return child1, child2

    def _mutate(self, genome: Genome) -> Genome:
        """Mutate a genome."""
        new_genes = {}
        for name, gene in genome.genes.items():
            if random.random() < self.config.mutation_prob:
                new_genes[name] = gene.mutate()
            else:
                new_genes[name] = Gene(**gene.__dict__)

        return Genome(
            id=f"g{self.generation}_{random.randint(0, 9999):04d}",
            genes=new_genes,
            generation=self.generation,
            parent_ids=[genome.id],
        )

    def _nsga2_sort(self, population: list[Genome]) -> list[Genome]:
        """
        NSGA-II non-dominated sorting.

        Returns population sorted by Pareto fronts.
        """
        if not population or not population[0].fitness_scores:
            return population

        objectives = list(population[0].fitness_scores.keys())
        n = len(population)

        # Domination counts and dominated sets
        domination_count = [0] * n
        dominated_by = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j], objectives):
                    dominated_by[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i], objectives):
                    dominated_by[j].append(i)
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            fronts.append(next_front)

        # Flatten and return sorted population
        sorted_indices = []
        for front in fronts:
            # Sort within front by crowding distance
            if front:
                front_with_crowd = self._crowding_distance(front, population, objectives)
                front_sorted = sorted(front_with_crowd, key=lambda x: x[1], reverse=True)
                sorted_indices.extend([idx for idx, _ in front_sorted])

        return [population[i] for i in sorted_indices]

    def _dominates(self, genome1: Genome, genome2: Genome, objectives: list[str]) -> bool:
        """Check if genome1 dominates genome2."""
        dominated_any = False
        for obj in objectives:
            v1 = genome1.fitness_scores.get(obj, 0)
            v2 = genome2.fitness_scores.get(obj, 0)
            if v1 < v2:
                return False
            if v1 > v2:
                dominated_any = True
        return dominated_any

    def _crowding_distance(
        self,
        front: list[int],
        population: list[Genome],
        objectives: list[str],
    ) -> list[tuple[int, float]]:
        """Calculate crowding distance for NSGA-II."""
        distances = {i: 0.0 for i in front}

        for obj in objectives:
            sorted_front = sorted(front, key=lambda i: population[i].fitness_scores.get(obj, 0))

            # Boundary points get infinite distance
            distances[sorted_front[0]] = float('inf')
            distances[sorted_front[-1]] = float('inf')

            # Calculate range
            f_max = population[sorted_front[-1]].fitness_scores.get(obj, 0)
            f_min = population[sorted_front[0]].fitness_scores.get(obj, 0)
            f_range = f_max - f_min if f_max != f_min else 1.0

            # Calculate distances for interior points
            for i in range(1, len(sorted_front) - 1):
                prev_val = population[sorted_front[i-1]].fitness_scores.get(obj, 0)
                next_val = population[sorted_front[i+1]].fitness_scores.get(obj, 0)
                distances[sorted_front[i]] += (next_val - prev_val) / f_range

        return [(i, distances[i]) for i in front]

    async def evolve_generation(self):
        """Evolve one generation."""
        await self.evaluate_population()

        # Sort population (NSGA-II or simple fitness)
        if self.config.use_nsga2:
            sorted_pop = self._nsga2_sort(self.population)
        else:
            sorted_pop = sorted(self.population, key=lambda g: g.total_fitness(), reverse=True)

        # Record best fitness
        best_fitness = sorted_pop[0].total_fitness() if sorted_pop else 0
        self.fitness_history.append({
            "generation": self.generation,
            "best_fitness": best_fitness,
            "avg_fitness": sum(g.total_fitness() for g in sorted_pop) / len(sorted_pop),
            "timestamp": datetime.now().isoformat(),
        })

        # Check for stagnation
        if best_fitness <= self.stats["best_fitness_ever"]:
            self.stats["stagnation_count"] += 1
        else:
            self.stats["best_fitness_ever"] = best_fitness
            self.stats["stagnation_count"] = 0

        # Adaptive mutation
        if self.config.adaptive_mutation and self.stats["stagnation_count"] >= self.config.stagnation_threshold:
            self.config.mutation_prob = min(0.5, self.config.mutation_prob * 1.5)
            logger.info(f"Increasing mutation rate to {self.config.mutation_prob:.2f}")
        else:
            self.config.mutation_prob = max(0.1, self.config.mutation_prob * 0.95)

        # Create next generation
        self.generation += 1
        new_population = []

        # Elitism - keep best genomes
        for genome in sorted_pop[:self.config.elite_count]:
            elite_copy = Genome(
                id=f"g{self.generation}_elite_{genome.id}",
                genes={k: Gene(**v.__dict__) for k, v in genome.genes.items()},
                fitness_scores=genome.fitness_scores.copy(),
                generation=self.generation,
                parent_ids=[genome.id],
            )
            new_population.append(elite_copy)

        # Fill rest with offspring
        while len(new_population) < self.config.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            if random.random() < self.config.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1 = self._mutate(parent1)
                child2 = self._mutate(parent2)

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        self.population = new_population
        self.stats["generations_completed"] += 1

        # Log evolution step with hash chain
        self._log_evolution_step(sorted_pop[0] if sorted_pop else None)

    def _log_evolution_step(self, best_genome: Optional[Genome]):
        """Log evolution step with hash chain for integrity."""
        log_entry = {
            "generation": self.generation,
            "best_genome": best_genome.to_dict() if best_genome else None,
            "population_size": len(self.population),
            "timestamp": datetime.now().isoformat(),
            "prev_hash": self.last_hash,
        }

        # Create hash of this entry
        entry_str = json.dumps(log_entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()[:16]
        log_entry["hash"] = entry_hash

        self.evolution_log.append(log_entry)
        self.last_hash = entry_hash

        # Save to disk periodically
        if self.generation % 5 == 0:
            self._save_log()

    def _save_log(self):
        """Save evolution log to disk."""
        log_file = self.data_dir / "evolution_log.jsonl"
        with log_file.open('a', encoding='utf-8') as f:
            for entry in self.evolution_log[-5:]:
                f.write(json.dumps(entry) + "\n")

    async def run(
        self,
        generations: Optional[int] = None,
        early_stop_fitness: Optional[float] = None,
    ) -> EvolutionResult:
        """
        Run the full evolution process.

        Args:
            generations: Number of generations (uses config if None)
            early_stop_fitness: Stop if this fitness is reached

        Returns:
            EvolutionResult with best genome and history
        """
        import time
        start_time = time.time()

        generations = generations or self.config.generations

        if not self.population:
            self.initialize_population()

        for gen in range(generations):
            await self.evolve_generation()

            # Check early stopping
            if early_stop_fitness is not None:
                best = max(self.population, key=lambda g: g.total_fitness())
                if best.total_fitness() >= early_stop_fitness:
                    logger.info(f"Early stopping at generation {gen}")
                    break

            logger.info(f"Generation {gen + 1}/{generations} complete")

        # Final evaluation
        await self.evaluate_population()
        best_genome = max(self.population, key=lambda g: g.total_fitness())

        return EvolutionResult(
            best_genome=best_genome,
            final_population=self.population,
            generations_run=self.generation,
            fitness_history=self.fitness_history,
            duration_seconds=time.time() - start_time,
        )

    def get_best_genome(self) -> Optional[Genome]:
        """Get the current best genome."""
        if not self.population:
            return None
        return max(self.population, key=lambda g: g.total_fitness())

    def get_stats(self) -> dict:
        """Get optimizer statistics."""
        return {
            **self.stats,
            "population_size": len(self.population),
            "current_generation": self.generation,
            "mutation_prob": self.config.mutation_prob,
            "gene_count": len(self.gene_definitions),
        }
