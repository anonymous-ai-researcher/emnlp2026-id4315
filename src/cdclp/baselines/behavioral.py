"""Behavioral baselines the paper compares against.

None of these intervene on the model's state. They read the output under a
modified input or a truncated chain, which is why they answer a different
question: whether the answer is *recoverable*, not whether it is *settled*. A
direction can be recoverable from an activation that plays no causal role, so a
positive behavioral reading is weaker evidence than a positive causal one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np


@dataclass
class EarlyAnswering:
    """Truncate the chain at each step and force an answer.

    Measures how early the answer is recoverable from the partial chain. Cheap,
    model-agnostic, and entirely behavioral.
    """

    generate: Callable[[str], str]
    readout: Callable[[str], tuple[Any, bool]]
    force_suffix: str = "\nTherefore the answer is"

    def curve(self, prompt: str, chain: str, gold: Any, grid: np.ndarray) -> np.ndarray:
        toks = chain.split()
        out = np.zeros(len(grid), dtype=float)
        for b, frac in enumerate(grid):
            cut = int(round(frac * len(toks)))
            partial = " ".join(toks[:cut])
            completion = self.generate(prompt + partial + self.force_suffix)
            ans, ok = self.readout(completion)
            out[b] = float(ok and ans == gold)
        return out


@dataclass
class AccuracyControlled(EarlyAnswering):
    """Early answering with the no-chain accuracy divided out.

    Without the correction a model that answers well with no chain at all looks
    like it commits immediately. Dividing by the no-chain rate puts models with
    different priors on a comparable scale.
    """

    def normalize(self, raw: np.ndarray, no_chain_accuracy: float) -> np.ndarray:
        if not 0.0 <= no_chain_accuracy < 1.0:
            raise ValueError(f"no-chain accuracy {no_chain_accuracy} outside [0, 1)")
        return (raw - no_chain_accuracy) / (1.0 - no_chain_accuracy)


@dataclass
class DetectionExtractionGap:
    """Difference between when the answer is decodable and when it is emitted.

    A behavioral proxy for premature determination. It correlates with the
    causal measure but is not equivalent to it: truncation changes the input the
    model sees, whereas a transplant changes only its internal state.
    """

    detect: Callable[[str, int], bool]
    extract: Callable[[str, int], bool]

    def gap(self, prompt: str, n_steps: int) -> int | None:
        first_detect = next((k for k in range(n_steps) if self.detect(prompt, k)), None)
        first_extract = next((k for k in range(n_steps) if self.extract(prompt, k)), None)
        if first_detect is None or first_extract is None:
            return None
        return first_extract - first_detect
