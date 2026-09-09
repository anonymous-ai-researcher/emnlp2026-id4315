"""Turning a completion into an answer.

The readout is fixed in advance and applied identically to every model. It has
to be, because the estimator compares flip rates across models: a readout tuned
per model would let a difference in parsing masquerade as a difference in
commitment.

Completions that do not parse are reported as such rather than dropped. They are
counted as non-flips downstream, which is the conservative choice: discarding
them would remove exactly the cases where the model failed to settle.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Sequence


class Readout:
    """Extract the final answer from a completion.

    Parameters
    ----------
    pattern:
        Regex whose first group holds the answer. Applied to the tail of the
        completion so that intermediate mentions of an answer inside the chain
        do not win over the final one.
    coerce:
        Maps the matched string to the task's answer type.
    tail_chars:
        How much of the end of the completion to search.
    """

    def __init__(
        self,
        pattern: str,
        coerce: Callable[[str], Any],
        tail_chars: int = 200,
    ) -> None:
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.coerce = coerce
        self.tail_chars = tail_chars

    def __call__(self, completion: str) -> tuple[Any, bool]:
        tail = completion[-self.tail_chars :]
        matches = list(self.regex.finditer(tail))
        if not matches:
            return None, False
        try:
            return self.coerce(matches[-1].group(1)), True
        except (ValueError, KeyError, IndexError):
            return None, False


def boolean_readout(true_word: str = "balanced", false_word: str = "unbalanced") -> Readout:
    """Binary answers given as one of two words."""
    pattern = rf"\b({re.escape(false_word)}|{re.escape(true_word)})\b"

    def coerce(s: str) -> bool:
        return s.lower() == true_word.lower()

    return Readout(pattern, coerce)


def integer_readout() -> Readout:
    """A single integer, as used by the automaton task."""
    return Readout(r"\b(\d+)\b", int)


def permutation_readout(degree: int) -> Readout:
    """A permutation written as the images of 1..n, space separated."""
    pattern = r"((?:\d+[ ,]+){%d}\d+)" % (degree - 1)

    def coerce(s: str) -> tuple[int, ...]:
        parts = [int(x) - 1 for x in re.split(r"[ ,]+", s.strip()) if x]
        if sorted(parts) != list(range(degree)):
            raise ValueError(f"not a permutation of 1..{degree}: {s!r}")
        return tuple(parts)

    return Readout(pattern, coerce, tail_chars=300)


def choose_readout(task_name: str, degree: int = 5) -> Readout:
    if task_name == "dyck2":
        return boolean_readout()
    if task_name in {"fsa"}:
        return integer_readout()
    if task_name in {"a4s4", "a5s5"}:
        return permutation_readout(4 if task_name == "a4s4" else 5)
    raise KeyError(f"no default readout for task {task_name!r}")
