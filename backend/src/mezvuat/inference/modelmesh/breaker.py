"""Client-side circuit breaker for ModelMesh (ADR-0011).

Five consecutive fallback-eligible failures open the breaker for 30 s. While it is open, calls skip ModelMesh (and go
to the direct fallback when one is configured). After that a single caller probes `GET /v1/health`: 200 (`ok` or
`degraded`) half-opens the breaker and lets the call through; anything else re-opens it. One success closes it.
Only failures that §3.1 counts as "ModelMesh cannot serve this" are counted: a 4xx or 429 proves it is up.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

State = Literal["closed", "open", "half_open"]


@dataclass
class CircuitBreaker:
    probe: Callable[[], bool]
    threshold: int = 5
    open_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    state: State = field(default="closed", init=False)
    failures: int = field(default=0, init=False)
    opened_at: float = field(default=0.0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def allow(self) -> bool:
        with self._lock:
            if self.state != "open":
                return True
            if self.clock() - self.opened_at < self.open_s:
                return False
            # Only one caller probes: the others still see "open" until the probe returns.
            self.opened_at = self.clock()
        healthy = False
        try:
            healthy = self.probe()
        finally:
            with self._lock:
                if healthy:
                    self.state = "half_open"
                else:
                    self.state, self.opened_at = "open", self.clock()
        return healthy

    def success(self) -> None:
        with self._lock:
            self.state, self.failures = "closed", 0

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.state == "half_open" or self.failures >= self.threshold:
                self.state, self.opened_at = "open", self.clock()
