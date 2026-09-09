"""Causal Lookahead Probing: the intervention itself.

The measurement is a single interchange intervention. Run the model on a base
instance; at chain-of-thought step ``k``, overwrite the last-layer residual
stream at that position with the value it took when the model ran on a source
instance; let the chain finish; read the answer.

Two design choices carry the argument:

* **The whole state is transplanted, not a fitted subspace.** No alignment is
  learned, so a positive reading cannot be an artifact of a flexible mapping.
  The cost is that the measurement says *when* the answer was fixed, not *where*
  in the residual stream it lives.
* **The last layer is used** because its residual stream feeds the unembedding
  directly, so a flip there is a flip in the model's own decision, not in an
  intermediate quantity that later layers might overwrite.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class InterventionResult:
    """Outcome of one base-source pair at one step."""

    flipped_to_source: bool
    base_answer: Any
    source_answer: Any
    observed_answer: Any
    parsed: bool


class ResidualStreamPatcher:
    """Backend-agnostic hook that swaps one position's residual stream.

    The class is deliberately thin: it takes callables so the same logic works
    with HuggingFace hooks, an nnsight tracer, or a stub in the tests. Supply a
    ``capture`` that returns the last-layer activation at a position, and an
    ``inject`` that installs a replacement for the duration of a forward pass.
    """

    def __init__(
        self,
        capture: Callable[[str, int], np.ndarray],
        inject: Callable[[str, int, np.ndarray], Any],
        layer: int | str = "last",
    ) -> None:
        self.capture = capture
        self.inject = inject
        self.layer = layer

    @contextmanager
    def patched(self, prompt: str, position: int, value: np.ndarray) -> Iterator[Any]:
        handle = self.inject(prompt, position, value)
        try:
            yield handle
        finally:
            release = getattr(handle, "remove", None)
            if callable(release):
                release()


def run_pair(
    patcher: ResidualStreamPatcher,
    generate: Callable[[str], str],
    readout: Callable[[str], tuple[Any, bool]],
    base_prompt: str,
    source_prompt: str,
    base_answer: Any,
    source_answer: Any,
    step: int,
) -> InterventionResult:
    """One line of Algorithm 1.

    ``generate`` completes the chain from a prompt; ``readout`` maps a completion
    to an answer and a flag saying whether the completion parsed at all.
    Unparseable completions count as non-flips rather than being dropped, so the
    estimator cannot be inflated by silently discarding hard cases.
    """
    cached = patcher.capture(source_prompt, step)
    with patcher.patched(base_prompt, step, cached):
        completion = generate(base_prompt)
    observed, parsed = readout(completion)
    flipped = bool(parsed and observed == source_answer)
    return InterventionResult(
        flipped_to_source=flipped,
        base_answer=base_answer,
        source_answer=source_answer,
        observed_answer=observed,
        parsed=parsed,
    )


def sweep_step(
    patcher: ResidualStreamPatcher,
    generate: Callable[[str], str],
    readout: Callable[[str], tuple[Any, bool]],
    pairs: Sequence[tuple[str, str, Any, Any]],
    step: int,
) -> np.ndarray:
    """Flip indicators for every pair at one step.

    Returns a 0/1 array of length ``len(pairs)``. Keeping the raw indicators,
    rather than only their mean, is what makes paired resampling and permutation
    tests possible downstream; discard them and those analyses are lost.
    """
    out = np.zeros(len(pairs), dtype=np.int8)
    for n, (bp, sp, ba, sa) in enumerate(pairs):
        res = run_pair(patcher, generate, readout, bp, sp, ba, sa, step)
        out[n] = int(res.flipped_to_source)
    return out
