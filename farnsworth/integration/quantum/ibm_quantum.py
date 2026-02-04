"""
Farnsworth IBM Quantum Integration
===================================

Quantum computing integration for enhanced AI capabilities using IBM Quantum Experience.

Free Tier Limits (as of 2026):
- Hardware: 10 minutes/month (~10-20 jobs)
- Simulators: Unlimited
- Qubits: 5-127 (Falcon/Eagle architectures)

Strategy: 95% simulators (development), 5% hardware (high-value tasks)

Use Cases:
- Quantum Genetic Algorithm (QGA) for agent evolution
- QAOA for multi-objective optimization
- VQE for variational simulations
- Quantum-inspired pattern extraction for memory dreaming
- Probabilistic modeling for financial risk assessment

"When classical optimization hits a wall, we go quantum." - Farnsworth Collective
"""

import os
import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

# Qiskit imports (graceful fallback if not installed)
try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.primitives import Sampler, Estimator
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, Options
    from qiskit_ibm_runtime import SamplerV2, EstimatorV2
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not installed. Run: pip install qiskit qiskit-ibm-runtime qiskit-aer")


class QuantumBackend(Enum):
    """Available quantum backends."""
    SIMULATOR_IDEAL = "aer_simulator"           # Noise-free, unlimited
    SIMULATOR_NOISY = "aer_simulator_noisy"     # With noise model, unlimited
    HARDWARE_SMALL = "ibm_brisbane"             # 127 qubits, limited
    HARDWARE_FAST = "ibm_kyoto"                 # Fast queue, limited
    AUTO = "auto"                               # Auto-select based on task


class QuantumTaskType(Enum):
    """Types of quantum tasks for resource allocation."""
    OPTIMIZATION = "optimization"       # QAOA, VQE - high value
    EVOLUTION = "evolution"             # QGA for agent mutation
    PATTERN = "pattern"                 # Memory pattern extraction
    INFERENCE = "inference"             # Knowledge graph queries
    SAMPLING = "sampling"               # Probabilistic modeling
    BENCHMARK = "benchmark"             # Performance testing


@dataclass
class QuantumUsageStats:
    """Track quantum resource usage against free tier limits."""
    hardware_seconds_used: float = 0.0
    hardware_jobs_count: int = 0
    simulator_jobs_count: int = 0
    last_hardware_run: Optional[datetime] = None
    month_start: datetime = field(default_factory=lambda: datetime.now().replace(day=1, hour=0, minute=0, second=0))

    # Free tier limits
    MONTHLY_HARDWARE_SECONDS: int = 600  # 10 minutes
    MAX_HARDWARE_JOBS_RECOMMENDED: int = 20

    @property
    def hardware_seconds_remaining(self) -> float:
        """Remaining hardware time this month."""
        return max(0, self.MONTHLY_HARDWARE_SECONDS - self.hardware_seconds_used)

    @property
    def hardware_percentage_used(self) -> float:
        """Percentage of monthly hardware quota used."""
        return (self.hardware_seconds_used / self.MONTHLY_HARDWARE_SECONDS) * 100

    def can_use_hardware(self, estimated_seconds: float = 30) -> bool:
        """Check if hardware can be used for estimated job duration."""
        return self.hardware_seconds_remaining >= estimated_seconds

    def record_hardware_job(self, duration_seconds: float):
        """Record a hardware job execution."""
        self.hardware_seconds_used += duration_seconds
        self.hardware_jobs_count += 1
        self.last_hardware_run = datetime.now()
        logger.info(f"Quantum hardware job: {duration_seconds:.1f}s used, {self.hardware_seconds_remaining:.1f}s remaining")

    def record_simulator_job(self):
        """Record a simulator job (unlimited)."""
        self.simulator_jobs_count += 1

    def reset_if_new_month(self):
        """Reset counters if new month started."""
        now = datetime.now()
        if now.month != self.month_start.month or now.year != self.month_start.year:
            logger.info(f"Quantum usage reset for new month. Previous: {self.hardware_seconds_used:.1f}s hardware, {self.simulator_jobs_count} simulator jobs")
            self.hardware_seconds_used = 0.0
            self.hardware_jobs_count = 0
            self.simulator_jobs_count = 0
            self.month_start = now.replace(day=1, hour=0, minute=0, second=0)

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "hardware_seconds_used": self.hardware_seconds_used,
            "hardware_jobs_count": self.hardware_jobs_count,
            "simulator_jobs_count": self.simulator_jobs_count,
            "last_hardware_run": self.last_hardware_run.isoformat() if self.last_hardware_run else None,
            "month_start": self.month_start.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QuantumUsageStats":
        """Deserialize from persistence."""
        stats = cls()
        stats.hardware_seconds_used = data.get("hardware_seconds_used", 0.0)
        stats.hardware_jobs_count = data.get("hardware_jobs_count", 0)
        stats.simulator_jobs_count = data.get("simulator_jobs_count", 0)
        if data.get("last_hardware_run"):
            stats.last_hardware_run = datetime.fromisoformat(data["last_hardware_run"])
        if data.get("month_start"):
            stats.month_start = datetime.fromisoformat(data["month_start"])
        stats.reset_if_new_month()
        return stats


@dataclass
class QuantumJobResult:
    """Result from a quantum job execution."""
    success: bool
    backend_used: str
    execution_time: float
    shots: int
    counts: Optional[Dict[str, int]] = None
    expectation_values: Optional[List[float]] = None
    optimal_parameters: Optional[List[float]] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class IBMQuantumProvider:
    """
    IBM Quantum Experience integration for Farnsworth.

    Manages connections, usage tracking, and intelligent backend selection
    to maximize the free tier while enabling quantum-enhanced AI capabilities.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize IBM Quantum provider.

        Args:
            api_key: IBM Quantum API key (or set IBM_QUANTUM_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("IBM_QUANTUM_API_KEY")
        self.service: Optional[QiskitRuntimeService] = None
        self.simulator: Optional[AerSimulator] = None
        self.usage_stats = QuantumUsageStats()
        self._connected = False
        self._usage_file = Path("data/quantum_usage.json")

        # Load persisted usage stats
        self._load_usage_stats()

    def _load_usage_stats(self):
        """Load usage stats from disk."""
        try:
            if self._usage_file.exists():
                with open(self._usage_file) as f:
                    data = json.load(f)
                self.usage_stats = QuantumUsageStats.from_dict(data)
                logger.debug(f"Loaded quantum usage: {self.usage_stats.hardware_seconds_used:.1f}s used this month")
        except Exception as e:
            logger.warning(f"Could not load quantum usage stats: {e}")

    def _save_usage_stats(self):
        """Persist usage stats to disk."""
        try:
            self._usage_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._usage_file, "w") as f:
                json.dump(self.usage_stats.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save quantum usage stats: {e}")

    async def connect(self) -> bool:
        """
        Connect to IBM Quantum services.

        Returns:
            True if connected successfully
        """
        if not QISKIT_AVAILABLE:
            logger.error("Qiskit not available. Install with: pip install qiskit qiskit-ibm-runtime qiskit-aer")
            return False

        if not self.api_key:
            logger.error("No IBM Quantum API key provided")
            return False

        try:
            # Initialize IBM Runtime service
            self.service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.api_key
            )

            # Initialize local simulator (always available)
            self.simulator = AerSimulator()

            self._connected = True
            logger.info("Connected to IBM Quantum Experience")
            logger.info(f"Hardware quota: {self.usage_stats.hardware_seconds_remaining:.1f}s remaining this month")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBM Quantum: {e}")
            return False

    def get_available_backends(self) -> List[str]:
        """Get list of available quantum backends."""
        if not self._connected or not self.service:
            return ["aer_simulator"]

        try:
            backends = self.service.backends()
            return [b.name for b in backends]
        except Exception as e:
            logger.warning(f"Could not fetch backends: {e}")
            return ["aer_simulator"]

    def _select_backend(
        self,
        task_type: QuantumTaskType,
        num_qubits: int,
        prefer_hardware: bool = False
    ) -> Tuple[str, bool]:
        """
        Intelligently select backend based on task and resources.

        Args:
            task_type: Type of quantum task
            num_qubits: Required qubits
            prefer_hardware: Prefer hardware if available

        Returns:
            (backend_name, is_hardware)
        """
        # Always check usage stats
        self.usage_stats.reset_if_new_month()

        # High-value tasks that benefit from hardware
        high_value_tasks = {
            QuantumTaskType.OPTIMIZATION,
            QuantumTaskType.EVOLUTION,
            QuantumTaskType.BENCHMARK
        }

        # Estimate job duration (simple heuristic)
        estimated_seconds = 30 + (num_qubits * 2)  # Base + per-qubit overhead

        # Decision logic
        use_hardware = (
            prefer_hardware and
            task_type in high_value_tasks and
            self.usage_stats.can_use_hardware(estimated_seconds) and
            self._connected and
            self.service is not None
        )

        if use_hardware:
            # Select appropriate hardware backend
            if num_qubits <= 27:
                return ("ibm_kyoto", True)  # Faster queue for small circuits
            else:
                return ("ibm_brisbane", True)  # 127 qubits for larger

        # Default to simulator
        return ("aer_simulator", False)

    async def run_circuit(
        self,
        circuit: "QuantumCircuit",
        shots: int = 1024,
        task_type: QuantumTaskType = QuantumTaskType.SAMPLING,
        prefer_hardware: bool = False,
        parameters: Optional[Dict] = None
    ) -> QuantumJobResult:
        """
        Execute a quantum circuit.

        Args:
            circuit: Qiskit QuantumCircuit to execute
            shots: Number of measurement shots
            task_type: Type of task for backend selection
            prefer_hardware: Prefer real quantum hardware
            parameters: Optional parameter bindings

        Returns:
            QuantumJobResult with execution results
        """
        if not QISKIT_AVAILABLE:
            return QuantumJobResult(
                success=False,
                backend_used="none",
                execution_time=0,
                shots=0,
                error="Qiskit not installed"
            )

        start_time = datetime.now()
        backend_name, is_hardware = self._select_backend(
            task_type, circuit.num_qubits, prefer_hardware
        )

        try:
            if is_hardware and self.service:
                # Run on real quantum hardware
                backend = self.service.backend(backend_name)

                # Transpile for hardware
                transpiled = transpile(circuit, backend, optimization_level=3)

                with Session(service=self.service, backend=backend) as session:
                    sampler = SamplerV2(session=session)

                    if parameters:
                        job = sampler.run([transpiled.bind_parameters(parameters)], shots=shots)
                    else:
                        job = sampler.run([transpiled], shots=shots)

                    result = job.result()

                execution_time = (datetime.now() - start_time).total_seconds()

                # Record usage
                self.usage_stats.record_hardware_job(execution_time)
                self._save_usage_stats()

                # Extract counts
                counts = result[0].data.meas.get_counts() if hasattr(result[0].data, 'meas') else {}

                return QuantumJobResult(
                    success=True,
                    backend_used=backend_name,
                    execution_time=execution_time,
                    shots=shots,
                    counts=counts,
                    metadata={"is_hardware": True, "transpiled_depth": transpiled.depth()}
                )

            else:
                # Run on simulator (unlimited)
                transpiled = transpile(circuit, self.simulator, optimization_level=1)

                if parameters:
                    transpiled = transpiled.bind_parameters(parameters)

                job = self.simulator.run(transpiled, shots=shots)
                result = job.result()

                execution_time = (datetime.now() - start_time).total_seconds()
                self.usage_stats.record_simulator_job()

                counts = result.get_counts()

                return QuantumJobResult(
                    success=True,
                    backend_used="aer_simulator",
                    execution_time=execution_time,
                    shots=shots,
                    counts=counts,
                    metadata={"is_hardware": False, "circuit_depth": transpiled.depth()}
                )

        except Exception as e:
            logger.error(f"Quantum execution failed: {e}")
            return QuantumJobResult(
                success=False,
                backend_used=backend_name,
                execution_time=(datetime.now() - start_time).total_seconds(),
                shots=shots,
                error=str(e)
            )

    def get_usage_summary(self) -> Dict:
        """Get current usage summary."""
        self.usage_stats.reset_if_new_month()
        return {
            "hardware_seconds_used": self.usage_stats.hardware_seconds_used,
            "hardware_seconds_remaining": self.usage_stats.hardware_seconds_remaining,
            "hardware_percentage_used": self.usage_stats.hardware_percentage_used,
            "hardware_jobs_count": self.usage_stats.hardware_jobs_count,
            "simulator_jobs_count": self.usage_stats.simulator_jobs_count,
            "last_hardware_run": self.usage_stats.last_hardware_run.isoformat() if self.usage_stats.last_hardware_run else None,
            "connected": self._connected
        }


# =============================================================================
# QUANTUM ALGORITHMS FOR FARNSWORTH
# =============================================================================

class QuantumGeneticOptimizer:
    """
    Quantum-enhanced Genetic Algorithm for agent evolution.

    Uses quantum superposition for exploring solution spaces and
    quantum interference for amplifying good solutions.

    Integrates with: evolution/genetic_optimizer.py, population_manager.py
    """

    def __init__(self, provider: IBMQuantumProvider, num_qubits: int = 8):
        """
        Initialize quantum genetic optimizer.

        Args:
            provider: IBM Quantum provider instance
            num_qubits: Number of qubits (determines solution space size)
        """
        self.provider = provider
        self.num_qubits = num_qubits
        self.generation = 0

    def _create_superposition_circuit(self) -> "QuantumCircuit":
        """Create circuit for uniform superposition (all possibilities)."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Hadamard on all qubits creates uniform superposition
        for i in range(self.num_qubits):
            qc.h(i)

        return qc

    def _create_grover_oracle(self, target_fitness: Callable[[str], float]) -> "QuantumCircuit":
        """
        Create Grover oracle that marks good solutions.

        For genetic algorithms, "good" = high fitness score.
        """
        if not QISKIT_AVAILABLE:
            return None

        # Simplified oracle - in practice, this would encode the fitness function
        qc = QuantumCircuit(self.num_qubits)

        # Multi-controlled Z gate marks solutions
        # This is a placeholder - real implementation would encode fitness
        qc.h(self.num_qubits - 1)
        qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
        qc.h(self.num_qubits - 1)

        return qc

    def _create_diffusion_operator(self) -> "QuantumCircuit":
        """Create Grover diffusion operator for amplitude amplification."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(self.num_qubits)

        # Hadamard on all
        for i in range(self.num_qubits):
            qc.h(i)

        # Phase flip about |0>
        for i in range(self.num_qubits):
            qc.x(i)

        # Multi-controlled Z
        qc.h(self.num_qubits - 1)
        qc.mcx(list(range(self.num_qubits - 1)), self.num_qubits - 1)
        qc.h(self.num_qubits - 1)

        # Undo X gates
        for i in range(self.num_qubits):
            qc.x(i)

        # Hadamard on all
        for i in range(self.num_qubits):
            qc.h(i)

        return qc

    async def generate_quantum_population(
        self,
        population_size: int,
        fitness_func: Optional[Callable[[str], float]] = None,
        grover_iterations: int = 1,
        prefer_hardware: bool = False
    ) -> List[Tuple[str, float]]:
        """
        Generate population using quantum sampling.

        Uses Grover's algorithm to bias sampling toward high-fitness solutions.

        Args:
            population_size: Number of individuals to generate
            fitness_func: Function to evaluate fitness of bitstring
            grover_iterations: Number of Grover iterations (amplitude amplification)
            prefer_hardware: Use real quantum hardware if available

        Returns:
            List of (bitstring, fitness_score) tuples
        """
        if not QISKIT_AVAILABLE:
            # Classical fallback
            logger.warning("Qiskit not available, using classical random generation")
            population = []
            for _ in range(population_size):
                bitstring = ''.join(str(np.random.randint(2)) for _ in range(self.num_qubits))
                fitness = fitness_func(bitstring) if fitness_func else np.random.random()
                population.append((bitstring, fitness))
            return sorted(population, key=lambda x: x[1], reverse=True)

        # Build quantum circuit
        qc = self._create_superposition_circuit()

        if grover_iterations > 0 and fitness_func:
            oracle = self._create_grover_oracle(fitness_func)
            diffusion = self._create_diffusion_operator()

            for _ in range(grover_iterations):
                qc.compose(oracle, inplace=True)
                qc.compose(diffusion, inplace=True)

        # Measure all qubits
        qc.measure_all()

        # Run circuit
        result = await self.provider.run_circuit(
            qc,
            shots=population_size * 2,  # Oversample for diversity
            task_type=QuantumTaskType.EVOLUTION,
            prefer_hardware=prefer_hardware
        )

        if not result.success or not result.counts:
            logger.warning(f"Quantum population generation failed: {result.error}")
            # Classical fallback
            population = []
            for _ in range(population_size):
                bitstring = ''.join(str(np.random.randint(2)) for _ in range(self.num_qubits))
                fitness = fitness_func(bitstring) if fitness_func else np.random.random()
                population.append((bitstring, fitness))
            return sorted(population, key=lambda x: x[1], reverse=True)

        # Process results
        population = []
        for bitstring, count in result.counts.items():
            # Normalize bitstring format
            bs = bitstring.replace(" ", "")
            fitness = fitness_func(bs) if fitness_func else count / result.shots
            population.append((bs, fitness))

        # Sort by fitness and return top individuals
        population.sort(key=lambda x: x[1], reverse=True)

        self.generation += 1
        logger.info(f"Quantum GA generation {self.generation}: {len(population)} candidates, best fitness {population[0][1]:.4f}")

        return population[:population_size]

    async def quantum_crossover(
        self,
        parent1: str,
        parent2: str,
        prefer_hardware: bool = False
    ) -> str:
        """
        Quantum-inspired crossover using entanglement.

        Creates offspring that has quantum superposition of both parents' traits.
        """
        if not QISKIT_AVAILABLE or len(parent1) != len(parent2):
            # Classical single-point crossover
            point = np.random.randint(1, len(parent1))
            return parent1[:point] + parent2[point:]

        n = len(parent1)
        qc = QuantumCircuit(n, n)

        # Encode parents into quantum state
        for i in range(n):
            if parent1[i] == '1':
                qc.x(i)  # Set to |1> if parent1 bit is 1

            # Create superposition between parents at each position
            if parent1[i] != parent2[i]:
                qc.h(i)  # Superposition when parents differ

        # Add entanglement for correlated inheritance
        for i in range(n - 1):
            qc.cx(i, i + 1)

        qc.measure_all()

        result = await self.provider.run_circuit(
            qc,
            shots=1,
            task_type=QuantumTaskType.EVOLUTION,
            prefer_hardware=prefer_hardware
        )

        if result.success and result.counts:
            offspring = max(result.counts, key=result.counts.get)
            return offspring.replace(" ", "")

        # Fallback to classical
        point = np.random.randint(1, n)
        return parent1[:point] + parent2[point:]

    async def quantum_mutation(
        self,
        individual: str,
        mutation_rate: float = 0.1,
        prefer_hardware: bool = False
    ) -> str:
        """
        Quantum-inspired mutation using rotation gates.

        Uses parameterized rotation to control mutation probability.
        """
        if not QISKIT_AVAILABLE:
            # Classical mutation
            result = list(individual)
            for i in range(len(result)):
                if np.random.random() < mutation_rate:
                    result[i] = '0' if result[i] == '1' else '1'
            return ''.join(result)

        n = len(individual)
        qc = QuantumCircuit(n, n)

        # Encode individual
        for i in range(n):
            if individual[i] == '1':
                qc.x(i)

        # Apply rotation gates based on mutation rate
        # RY rotation angle controls probability of bit flip
        theta = 2 * np.arcsin(np.sqrt(mutation_rate))
        for i in range(n):
            qc.ry(theta, i)

        qc.measure_all()

        result = await self.provider.run_circuit(
            qc,
            shots=1,
            task_type=QuantumTaskType.EVOLUTION,
            prefer_hardware=prefer_hardware
        )

        if result.success and result.counts:
            mutated = max(result.counts, key=result.counts.get)
            return mutated.replace(" ", "")

        # Classical fallback
        result_list = list(individual)
        for i in range(len(result_list)):
            if np.random.random() < mutation_rate:
                result_list[i] = '0' if result_list[i] == '1' else '1'
        return ''.join(result_list)


class QAOAOptimizer:
    """
    Quantum Approximate Optimization Algorithm (QAOA) for combinatorial problems.

    Useful for:
    - Multi-objective optimization in genetic_optimizer.py
    - Task allocation in swarm orchestration
    - Knowledge graph inference optimization

    Integrates with: core/genetic_optimizer.py, core/collective/
    """

    def __init__(self, provider: IBMQuantumProvider):
        self.provider = provider

    def _create_cost_circuit(self, num_qubits: int, gamma: float, edges: List[Tuple[int, int]]) -> "QuantumCircuit":
        """Create cost unitary for QAOA (e.g., MaxCut problem)."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(num_qubits)

        for i, j in edges:
            if i < num_qubits and j < num_qubits:
                qc.rzz(2 * gamma, i, j)

        return qc

    def _create_mixer_circuit(self, num_qubits: int, beta: float) -> "QuantumCircuit":
        """Create mixer unitary for QAOA."""
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(num_qubits)

        for i in range(num_qubits):
            qc.rx(2 * beta, i)

        return qc

    async def optimize(
        self,
        num_qubits: int,
        edges: List[Tuple[int, int]],
        p: int = 2,
        shots: int = 1024,
        prefer_hardware: bool = False
    ) -> QuantumJobResult:
        """
        Run QAOA optimization.

        Args:
            num_qubits: Problem size
            edges: Graph edges for MaxCut-style problem
            p: QAOA depth (layers)
            shots: Measurement shots
            prefer_hardware: Use real hardware

        Returns:
            QuantumJobResult with optimal bitstring
        """
        if not QISKIT_AVAILABLE:
            return QuantumJobResult(
                success=False,
                backend_used="none",
                execution_time=0,
                shots=0,
                error="Qiskit not installed"
            )

        # Initialize parameters (would be optimized in full implementation)
        gammas = [np.pi / 4] * p
        betas = [np.pi / 8] * p

        # Build QAOA circuit
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Initial superposition
        for i in range(num_qubits):
            qc.h(i)

        # QAOA layers
        for layer in range(p):
            # Cost layer
            cost_circuit = self._create_cost_circuit(num_qubits, gammas[layer], edges)
            qc.compose(cost_circuit, inplace=True)

            # Mixer layer
            mixer_circuit = self._create_mixer_circuit(num_qubits, betas[layer])
            qc.compose(mixer_circuit, inplace=True)

        qc.measure_all()

        # Run optimization
        result = await self.provider.run_circuit(
            qc,
            shots=shots,
            task_type=QuantumTaskType.OPTIMIZATION,
            prefer_hardware=prefer_hardware
        )

        if result.success and result.counts:
            # Find best solution
            best_bitstring = max(result.counts, key=result.counts.get)
            result.optimal_parameters = gammas + betas
            result.metadata["best_solution"] = best_bitstring
            result.metadata["qaoa_depth"] = p

        return result


class QuantumPatternExtractor:
    """
    Quantum-inspired pattern extraction for memory consolidation.

    Uses quantum sampling for exploring pattern spaces in
    high-dimensional memory embeddings.

    Integrates with: memory/memory_dreaming.py, memory/dream_consolidation.py
    """

    def __init__(self, provider: IBMQuantumProvider):
        self.provider = provider

    async def extract_patterns(
        self,
        embedding_matrix: np.ndarray,
        num_patterns: int = 5,
        prefer_hardware: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract patterns from embedding matrix using quantum sampling.

        Args:
            embedding_matrix: Matrix of memory embeddings (n_memories x embedding_dim)
            num_patterns: Number of patterns to extract
            prefer_hardware: Use quantum hardware

        Returns:
            List of pattern dictionaries with indices and weights
        """
        n_memories = embedding_matrix.shape[0]

        # Limit qubits to practical range
        num_qubits = min(int(np.ceil(np.log2(n_memories + 1))), 10)

        if not QISKIT_AVAILABLE:
            # Classical clustering fallback
            logger.info("Using classical pattern extraction")
            patterns = []
            for i in range(num_patterns):
                # Random pattern selection
                indices = np.random.choice(n_memories, size=min(3, n_memories), replace=False)
                patterns.append({
                    "pattern_id": i,
                    "memory_indices": indices.tolist(),
                    "weights": [1.0 / len(indices)] * len(indices),
                    "method": "classical_random"
                })
            return patterns

        # Build quantum circuit for pattern sampling
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Hadamard superposition
        for i in range(num_qubits):
            qc.h(i)

        # Add some entanglement for correlated patterns
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)

        # Parameterized rotations based on embedding statistics
        embedding_var = np.var(embedding_matrix, axis=0).mean()
        rotation_angle = np.arctan(embedding_var) * 2
        for i in range(num_qubits):
            qc.ry(rotation_angle, i)

        qc.measure_all()

        # Run quantum sampling
        result = await self.provider.run_circuit(
            qc,
            shots=num_patterns * 100,
            task_type=QuantumTaskType.PATTERN,
            prefer_hardware=prefer_hardware
        )

        if not result.success or not result.counts:
            logger.warning("Quantum pattern extraction failed, using classical")
            # Fallback
            patterns = []
            for i in range(num_patterns):
                indices = np.random.choice(n_memories, size=min(3, n_memories), replace=False)
                patterns.append({
                    "pattern_id": i,
                    "memory_indices": indices.tolist(),
                    "weights": [1.0 / len(indices)] * len(indices),
                    "method": "classical_fallback"
                })
            return patterns

        # Convert quantum samples to patterns
        patterns = []
        sorted_counts = sorted(result.counts.items(), key=lambda x: x[1], reverse=True)

        for i, (bitstring, count) in enumerate(sorted_counts[:num_patterns]):
            # Convert bitstring to memory indices
            bs = bitstring.replace(" ", "")
            index = int(bs, 2) % n_memories

            # Build pattern around this index
            related_indices = [
                (index + j) % n_memories
                for j in range(-1, 2)
            ]

            patterns.append({
                "pattern_id": i,
                "memory_indices": related_indices,
                "weights": [count / result.shots] * len(related_indices),
                "quantum_amplitude": count / result.shots,
                "method": "quantum_sampling"
            })

        logger.info(f"Extracted {len(patterns)} quantum patterns from {n_memories} memories")
        return patterns


# =============================================================================
# SINGLETON AND UTILITY FUNCTIONS
# =============================================================================

_quantum_provider: Optional[IBMQuantumProvider] = None


def get_quantum_provider() -> Optional[IBMQuantumProvider]:
    """Get or create the quantum provider singleton."""
    global _quantum_provider
    if _quantum_provider is None:
        api_key = os.environ.get("IBM_QUANTUM_API_KEY")
        if api_key:
            _quantum_provider = IBMQuantumProvider(api_key)
    return _quantum_provider


async def initialize_quantum(api_key: str = None) -> bool:
    """
    Initialize quantum computing integration.

    Args:
        api_key: IBM Quantum API key (optional, uses env var if not provided)

    Returns:
        True if initialized successfully
    """
    global _quantum_provider

    if api_key:
        os.environ["IBM_QUANTUM_API_KEY"] = api_key

    _quantum_provider = IBMQuantumProvider(api_key or os.environ.get("IBM_QUANTUM_API_KEY"))
    return await _quantum_provider.connect()


async def quantum_evolve_agent(
    agent_genome: str,
    fitness_func: Callable[[str], float],
    generations: int = 5,
    population_size: int = 20,
    prefer_hardware: bool = False
) -> Tuple[str, float]:
    """
    Evolve an agent genome using quantum genetic algorithm.

    High-level function for integration with evolution/ modules.

    Args:
        agent_genome: Binary string representing agent parameters
        fitness_func: Function to evaluate genome fitness
        generations: Number of evolution generations
        population_size: Population size per generation
        prefer_hardware: Use quantum hardware for key operations

    Returns:
        (best_genome, best_fitness) tuple
    """
    provider = get_quantum_provider()
    if not provider:
        logger.warning("Quantum provider not initialized, using classical evolution")
        # Classical fallback
        best = agent_genome
        best_fitness = fitness_func(agent_genome)
        for _ in range(generations * population_size):
            mutated = list(best)
            for i in range(len(mutated)):
                if np.random.random() < 0.1:
                    mutated[i] = '0' if mutated[i] == '1' else '1'
            mutated = ''.join(mutated)
            fit = fitness_func(mutated)
            if fit > best_fitness:
                best = mutated
                best_fitness = fit
        return best, best_fitness

    qga = QuantumGeneticOptimizer(provider, num_qubits=len(agent_genome))

    # Initialize with quantum population
    population = await qga.generate_quantum_population(
        population_size,
        fitness_func,
        grover_iterations=1,
        prefer_hardware=prefer_hardware
    )

    best_genome, best_fitness = population[0]

    for gen in range(generations):
        # Selection (tournament)
        selected = population[:population_size // 2]

        # Crossover and mutation
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            child = await qga.quantum_crossover(
                selected[i][0],
                selected[i + 1][0],
                prefer_hardware=prefer_hardware and gen == generations - 1  # Hardware on last gen
            )
            child = await qga.quantum_mutation(child, mutation_rate=0.1)
            fitness = fitness_func(child)
            offspring.append((child, fitness))

        # Combine and select best
        combined = population + offspring
        combined.sort(key=lambda x: x[1], reverse=True)
        population = combined[:population_size]

        if population[0][1] > best_fitness:
            best_genome, best_fitness = population[0]

        logger.debug(f"Quantum evolution gen {gen + 1}: best fitness {best_fitness:.4f}")

    return best_genome, best_fitness


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

async def test_quantum_integration():
    """Test IBM Quantum integration."""
    print("Testing Farnsworth Quantum Integration")
    print("=" * 50)

    api_key = os.environ.get("IBM_QUANTUM_API_KEY")
    if not api_key:
        print("No IBM_QUANTUM_API_KEY found in environment")
        print("Set it with: export IBM_QUANTUM_API_KEY=your_key")
        return False

    # Initialize
    success = await initialize_quantum(api_key)
    print(f"Connection: {'Success' if success else 'Failed'}")

    if not success:
        return False

    provider = get_quantum_provider()

    # Check usage
    usage = provider.get_usage_summary()
    print(f"\nUsage Summary:")
    print(f"  Hardware: {usage['hardware_seconds_used']:.1f}s / 600s ({usage['hardware_percentage_used']:.1f}%)")
    print(f"  Simulator jobs: {usage['simulator_jobs_count']}")

    # Test simple circuit
    if QISKIT_AVAILABLE:
        print("\nRunning test circuit on simulator...")
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()

        result = await provider.run_circuit(qc, shots=100)
        print(f"  Success: {result.success}")
        print(f"  Backend: {result.backend_used}")
        print(f"  Counts: {result.counts}")

        # Test quantum GA
        print("\nTesting Quantum Genetic Algorithm...")

        def test_fitness(bitstring: str) -> float:
            return sum(int(b) for b in bitstring) / len(bitstring)

        qga = QuantumGeneticOptimizer(provider, num_qubits=5)
        population = await qga.generate_quantum_population(10, test_fitness)
        print(f"  Generated population of {len(population)}")
        print(f"  Best individual: {population[0][0]} (fitness: {population[0][1]:.2f})")

    print("\nQuantum integration test complete!")
    return True


if __name__ == "__main__":
    asyncio.run(test_quantum_integration())
