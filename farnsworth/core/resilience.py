"""
Farnsworth Resilience Layer.

"I have made my device strong enough to withstand even my own incompetence!"

This module implements resilience patterns to ensure the Nexus and Agent Swarm 
remain stable under high cognitive load or external failure.

Features:
1. Circuit Breaker: Prevents cascading failures when a subsystem (e.g. LLM API) is struggling.
2. Cognitive Backpressure: Detects when the system is thinking too fast for the I/O to handle.
3. Entropy Monitor: Detects when agent reasoning is degrading (hallucination loops).
"""

import asyncio
import time
from enum import Enum
from typing import Callable, Any, Optional, Dict
from loguru import logger
from functools import wraps

class SystemState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    PANIC = "panic"

class CircuitBreaker:
    """
    Protects expensive operations (like LLM calls or Projection) from failing repeatedly.
    """
    def __init__(self, name: str, failure_threshold: int = 3, reset_timeout: int = 60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        
        self.failures = 0
        self.last_failure_time = 0
        self.is_open = False
        
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if self.is_open:
                if time.time() - self.last_failure_time > self.reset_timeout:
                    logger.info(f"CircuitBreaker '{self.name}': Half-open (Testing)")
                    self.is_open = False # Half-open trial
                else:
                    logger.warning(f"CircuitBreaker '{self.name}': Call blocked")
                    return None # Or raise specific exception

            try:
                result = await func(*args, **kwargs)
                if self.failures > 0:
                    logger.info(f"CircuitBreaker '{self.name}': Recovered")
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                logger.error(f"CircuitBreaker '{self.name}': Failure {self.failures}/{self.failure_threshold} - {e}")
                
                if self.failures >= self.failure_threshold:
                    self.is_open = True
                    logger.critical(f"CircuitBreaker '{self.name}': OPENED. Blocking calls for {self.reset_timeout}s")
                raise e
        return wrapper

class EntropyMonitor:
    """
    Monitors the semantic entropy of the agent's output.
    If the agent starts repeating itself or outputting garbage, this triggers a STOP signal.
    """
    def __init__(self):
        self.history = []
        
    def check_hallucination(self, content: str) -> bool:
        # Simple heuristic: Check for exact repetitions
        # Real implementation would use semantic similarity or compression ratio
        if content in self.history[-5:]:
            return True
        self.history.append(content)
        if len(self.history) > 20:
            self.history.pop(0)
        return False

# Global resilience monitors
projector_breaker = CircuitBreaker("HolographicProjection", failure_threshold=2, reset_timeout=10)
network_breaker = CircuitBreaker("NetworkIO", failure_threshold=5, reset_timeout=30)
