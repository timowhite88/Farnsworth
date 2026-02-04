"""
Farnsworth Quantum Computing Integration
=========================================

Provides quantum-enhanced capabilities via IBM Quantum Experience.

Free Tier Strategy:
- 10 minutes/month hardware for high-value tasks (evolution, optimization)
- Unlimited simulators for development and testing

Quick Start:
    from farnsworth.integration.quantum import initialize_quantum, quantum_evolve_agent

    # Initialize (use env var IBM_QUANTUM_API_KEY or pass key)
    await initialize_quantum()

    # Evolve an agent genome
    best_genome, fitness = await quantum_evolve_agent(
        agent_genome="10110010",
        fitness_func=lambda x: sum(int(b) for b in x) / len(x),
        generations=5
    )

Components:
- IBMQuantumProvider: Core connection and execution management
- QuantumGeneticOptimizer: QGA for agent evolution
- QAOAOptimizer: Combinatorial optimization
- QuantumPatternExtractor: Memory pattern discovery
"""

from .ibm_quantum import (
    # Core provider
    IBMQuantumProvider,
    get_quantum_provider,
    initialize_quantum,

    # Enums
    QuantumBackend,
    QuantumTaskType,

    # Data classes
    QuantumUsageStats,
    QuantumJobResult,

    # Algorithms
    QuantumGeneticOptimizer,
    QAOAOptimizer,
    QuantumPatternExtractor,

    # High-level functions
    quantum_evolve_agent,

    # Availability flag
    QISKIT_AVAILABLE,
)

__all__ = [
    "IBMQuantumProvider",
    "get_quantum_provider",
    "initialize_quantum",
    "QuantumBackend",
    "QuantumTaskType",
    "QuantumUsageStats",
    "QuantumJobResult",
    "QuantumGeneticOptimizer",
    "QAOAOptimizer",
    "QuantumPatternExtractor",
    "quantum_evolve_agent",
    "QISKIT_AVAILABLE",
]
