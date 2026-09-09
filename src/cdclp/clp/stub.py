"""A stub runner, so the pipeline can be exercised without a GPU.

The stub does not imitate any real model. It produces a flip pattern with a
tunable commitment point, which is enough to check that shapes line up, that
the estimator centers correctly, and that the statistics run end to end. Any
number it produces is a property of the stub, not of a language model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class StubRunner:
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    commitment: float = 0.35
    steepness: float = 10.0
    _last_step: int = 0

    def capture(self, prompt: str, position: int) -> np.ndarray:
        self._last_step = position
        return self.rng.normal(size=8)

    def inject(self, prompt: str, position: int, value: np.ndarray):
        self._last_step = position
        return None

    def generate(self, prompt: str) -> str:
        frac = min(1.0, self._last_step / 24.0)
        p = 1.0 / (1.0 + np.exp(-(frac - self.commitment) * self.steepness))
        return "\nTherefore the answer is balanced." if self.rng.random() < p \
            else "\nTherefore the answer is unbalanced."
