"""
Farnsworth External Integration Interface.

"I can wire anything into anything! I'm the Professor!"

This module defines the standard interface for connecting Farnsworth to 3rd party apps.
It uses the Nexus to emit events from these apps into the swarm.

AGI Upgrade (v1.5):
- Circuit breaker pattern for fault tolerance
- Automatic failure detection and recovery
- Exponential backoff with jitter
- Half-open state for gradual recovery
"""

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

from loguru import logger

from farnsworth.core.nexus import nexus, Signal, SignalType


# =============================================================================
# CIRCUIT BREAKER (AGI Upgrade v1.5)
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation - requests flow through
    OPEN = "open"  # Circuit tripped - requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout_seconds: float = 30.0  # Time before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open
    exponential_backoff: bool = True  # Increase timeout after repeated failures
    max_timeout_seconds: float = 300.0  # Max backoff timeout (5 min)
    jitter_factor: float = 0.1  # Random jitter to prevent thundering herd


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected when open
    state_changes: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    current_state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    time_in_current_state: float = 0.0
    opened_at: Optional[datetime] = None


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Prevents cascading failures by failing fast when a service is unhealthy.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests fail immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed

    Usage:
        breaker = CircuitBreaker("my_service")

        @breaker.protected
        async def call_external_api():
            ...

        # Or manually:
        if breaker.can_execute():
            try:
                result = await call_api()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure(e)
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._state_changed_at = datetime.now()
        self._failure_count = 0
        self._success_count = 0
        self._current_timeout = self.config.timeout_seconds
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        self.stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def is_closed(self) -> bool:
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self._state == CircuitState.HALF_OPEN

    async def can_execute(self) -> bool:
        """Check if a request can be executed."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if timeout has passed
                elapsed = (datetime.now() - self._state_changed_at).total_seconds()
                if elapsed >= self._current_timeout:
                    await self._transition_to(CircuitState.HALF_OPEN)
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                # Allow limited calls in half-open
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

            return False

    async def record_success(self):
        """Record a successful call."""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.successful_calls += 1
            self.stats.last_success = datetime.now()
            self.stats.consecutive_successes += 1
            self.stats.consecutive_failures = 0

            self._failure_count = 0
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)
                    # Reset timeout on successful recovery
                    self._current_timeout = self.config.timeout_seconds

    async def record_failure(self, error: Optional[Exception] = None):
        """Record a failed call."""
        async with self._lock:
            self.stats.total_calls += 1
            self.stats.failed_calls += 1
            self.stats.last_failure = datetime.now()
            self.stats.consecutive_failures += 1
            self.stats.consecutive_successes = 0

            self._failure_count += 1
            self._success_count = 0

            if self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    await self._transition_to(CircuitState.OPEN)

            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open returns to open
                await self._transition_to(CircuitState.OPEN)
                # Exponential backoff
                if self.config.exponential_backoff:
                    self._current_timeout = min(
                        self._current_timeout * 2,
                        self.config.max_timeout_seconds,
                    )

            logger.warning(
                f"Circuit breaker '{self.name}' recorded failure "
                f"(count={self._failure_count}, state={self._state.value})"
            )

    async def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._state_changed_at = datetime.now()

        self.stats.state_changes += 1
        self.stats.current_state = new_state
        self.stats.time_in_current_state = 0.0

        if new_state == CircuitState.OPEN:
            self.stats.opened_at = datetime.now()
            # Add jitter to prevent thundering herd
            if self.config.jitter_factor > 0:
                jitter = random.uniform(0, self.config.jitter_factor * self._current_timeout)
                self._current_timeout += jitter

        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0

        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0

        logger.info(
            f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}"
        )

        # Emit state change to Nexus
        try:
            await nexus.emit(
                SignalType.EXTERNAL_EVENT,
                {
                    "event_type": "circuit_breaker_state_change",
                    "breaker_name": self.name,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "failure_count": self._failure_count,
                    "timeout": self._current_timeout,
                },
                source="circuit_breaker",
                urgency=0.7 if new_state == CircuitState.OPEN else 0.4,
            )
        except Exception:
            pass  # Don't let Nexus errors affect circuit breaker

        # Callback
        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state)
            except Exception:
                pass

    def protected(self, func: Callable) -> Callable:
        """Decorator to protect an async function with this circuit breaker."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not await self.can_execute():
                self.stats.rejected_calls += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open, request rejected"
                )

            try:
                result = await func(*args, **kwargs)
                await self.record_success()
                return result
            except Exception as e:
                await self.record_failure(e)
                raise

        return wrapper

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        elapsed = (datetime.now() - self._state_changed_at).total_seconds()
        self.stats.time_in_current_state = elapsed

        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "rejected_calls": self.stats.rejected_calls,
            "success_rate": (
                self.stats.successful_calls / self.stats.total_calls
                if self.stats.total_calls > 0 else 0
            ),
            "consecutive_failures": self.stats.consecutive_failures,
            "state_changes": self.stats.state_changes,
            "time_in_current_state_seconds": elapsed,
            "current_timeout": self._current_timeout,
            "last_failure": self.stats.last_failure.isoformat() if self.stats.last_failure else None,
            "last_success": self.stats.last_success.isoformat() if self.stats.last_success else None,
        }

    async def reset(self):
        """Manually reset the circuit breaker to closed state."""
        async with self._lock:
            await self._transition_to(CircuitState.CLOSED)
            self._failure_count = 0
            self._success_count = 0
            self._current_timeout = self.config.timeout_seconds
            logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""
    pass


# =============================================================================
# CONNECTION STATUS AND CONFIG
# =============================================================================

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"  # Added for circuit breaker state


@dataclass
class IntegrationConfig:
    name: str
    api_key: Optional[str] = None
    enabled: bool = True
    poll_interval: float = 60.0  # Seconds
    # Circuit breaker config
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    timeout_seconds: float = 30.0


class ExternalProvider(ABC):
    """
    Abstract base class for all external app integrations.

    AGI v1.5: Includes circuit breaker for fault tolerance.
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED

        # AGI v1.5: Initialize circuit breaker
        self._circuit_breaker: Optional[CircuitBreaker] = None
        if config.circuit_breaker_enabled:
            breaker_config = CircuitBreakerConfig(
                failure_threshold=config.failure_threshold,
                timeout_seconds=config.timeout_seconds,
            )
            self._circuit_breaker = CircuitBreaker(
                name=f"provider_{config.name}",
                config=breaker_config,
                on_state_change=self._on_circuit_state_change,
            )

    def _on_circuit_state_change(self, old_state: CircuitState, new_state: CircuitState):
        """Handle circuit breaker state changes."""
        if new_state == CircuitState.OPEN:
            self.status = ConnectionStatus.CIRCUIT_OPEN
            logger.warning(f"Provider {self.config.name}: Circuit opened due to failures")
        elif new_state == CircuitState.CLOSED and self.status == ConnectionStatus.CIRCUIT_OPEN:
            self.status = ConnectionStatus.CONNECTED
            logger.info(f"Provider {self.config.name}: Circuit closed, service recovered")

    @abstractmethod
    async def connect(self) -> bool:
        """Authenticate and establish connection."""
        pass

    @abstractmethod
    async def sync(self):
        """Poll for updates (if webhook not available)."""
        pass

    @abstractmethod
    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Perform an action on the external service."""
        pass

    async def protected_execute(self, action: str, params: Dict[str, Any]) -> Any:
        """
        Execute an action with circuit breaker protection.

        AGI v1.5: Wraps execute_action with fault tolerance.
        """
        if self._circuit_breaker:
            if not await self._circuit_breaker.can_execute():
                raise CircuitBreakerOpenError(
                    f"Provider {self.config.name} circuit is open"
                )

            try:
                result = await self.execute_action(action, params)
                await self._circuit_breaker.record_success()
                return result
            except Exception as e:
                await self._circuit_breaker.record_failure(e)
                raise
        else:
            return await self.execute_action(action, params)

    async def protected_sync(self):
        """
        Sync with circuit breaker protection.

        AGI v1.5: Wraps sync with fault tolerance.
        """
        if self._circuit_breaker:
            if not await self._circuit_breaker.can_execute():
                logger.debug(f"Provider {self.config.name}: sync skipped, circuit open")
                return

            try:
                await self.sync()
                await self._circuit_breaker.record_success()
            except Exception as e:
                await self._circuit_breaker.record_failure(e)
                raise
        else:
            await self.sync()

    async def emit_event(self, event_type: str, payload: Dict[str, Any]):
        """Helper to inject external events into the Nexus."""
        await nexus.emit(
            SignalType.EXTERNAL_ALERT,
            {
                "provider": self.config.name,
                "event": event_type,
                "data": payload
            },
            source=f"ext_{self.config.name}"
        )

    def get_circuit_breaker_stats(self) -> Optional[Dict[str, Any]]:
        """Get circuit breaker statistics for monitoring."""
        if self._circuit_breaker:
            return self._circuit_breaker.get_stats()
        return None

    async def reset_circuit_breaker(self):
        """Manually reset the circuit breaker."""
        if self._circuit_breaker:
            await self._circuit_breaker.reset()

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status including circuit breaker."""
        health = {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "status": self.status.value,
        }

        if self._circuit_breaker:
            health["circuit_breaker"] = self._circuit_breaker.get_stats()

        return health
