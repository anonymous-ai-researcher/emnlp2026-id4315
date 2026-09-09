"""The four formal tasks: Dyck-2, two group word problems, and DFA execution.

All four have an exact reference program, so the prescribed trajectory is
computed by enumeration rather than approximation. Each task exposes
``state_after``, and the state is chosen so that it is exactly what the
reference program has determined after that many steps.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np

from .base import REGISTRY, Component, Instance, check_weights

# --------------------------------------------------------------------------
# Dyck-2
# --------------------------------------------------------------------------

_OPEN = "(["
_CLOSE = ")]"
_MATCH = {")": "(", "]": "["}


@REGISTRY.register("dyck2")
@dataclass
class Dyck2:
    """Balanced-bracket checking over two bracket types.

    The answer is binary: is the string balanced? A single unmatched bracket
    settles the answer, so the prescribed trajectory rises early. That is what
    makes this task the calibration case: there is little room for a model to
    commit ahead of the program.
    """

    name: str = "dyck2"
    min_len: int = 8
    max_len: int = 32
    max_depth: int = 4

    def __post_init__(self) -> None:
        self.components = (Component("dyck2", (False, True), 1.0),)
        check_weights(self.components)

    def _balanced(self, rng: np.random.Generator, n_pairs: int) -> str:
        """Sample a balanced string by a random walk that never goes negative."""
        out: list[str] = []
        stack: list[str] = []
        remaining = n_pairs
        while remaining > 0 or stack:
            can_open = remaining > 0 and len(stack) < self.max_depth
            can_close = bool(stack)
            if can_open and (not can_close or rng.random() < 0.5):
                b = _OPEN[rng.integers(len(_OPEN))]
                out.append(b)
                stack.append(b)
                remaining -= 1
            else:
                b = stack.pop()
                out.append(")" if b == "(" else "]")
        return "".join(out)

    def _corrupt(self, rng: np.random.Generator, s: str) -> str:
        """Break balance with a minimal edit so the answer flips to False."""
        chars = list(s)
        idx = int(rng.integers(len(chars)))
        c = chars[idx]
        if c in _OPEN:
            chars[idx] = _OPEN[1 - _OPEN.index(c)]
        else:
            chars[idx] = _CLOSE[1 - _CLOSE.index(c)]
        out = "".join(chars)
        return out if not self._is_balanced(out) else out[:-1]

    @staticmethod
    def _is_balanced(s: str) -> bool:
        stack: list[str] = []
        for c in s:
            if c in _OPEN:
                stack.append(c)
            else:
                if not stack or stack[-1] != _MATCH[c]:
                    return False
                stack.pop()
        return not stack

    def sample(self, rng: np.random.Generator, component: str) -> Instance:
        n_pairs = int(rng.integers(self.min_len // 2, self.max_len // 2 + 1))
        s = self._balanced(rng, n_pairs)
        if rng.random() < 0.5:
            s = self._corrupt(rng, s)
        ans = self._is_balanced(s)
        return Instance(payload=s, answer=ans, component="dyck2", prompt=render_dyck(s))

    def n_steps(self, instance: Instance) -> int:
        return len(instance.payload)

    def state_after(self, instance: Instance, step: int) -> Hashable:
        """Stack contents plus a sticky failure flag.

        Once the prefix has violated the matching rule the answer is fixed at
        False no matter what follows, so the failure flag alone is the state.
        """
        stack: list[str] = []
        for c in instance.payload[:step]:
            if c in _OPEN:
                stack.append(c)
            else:
                if not stack or stack[-1] != _MATCH[c]:
                    return ("failed",)
                stack.pop()
        return ("ok", tuple(stack))


def render_dyck(s: str) -> str:
    return (
        "Determine whether the following bracket sequence is balanced.\n"
        f"Sequence: {s}\n"
        "Think step by step, then answer with exactly one word: "
        "balanced or unbalanced.\n"
    )


# --------------------------------------------------------------------------
# Group word problems
# --------------------------------------------------------------------------


def _permutation_group(n: int, alternating: bool) -> list[tuple[int, ...]]:
    """All permutations of ``n`` points, optionally restricted to even ones."""
    elements = []
    for p in itertools.permutations(range(n)):
        if alternating and _parity(p) != 0:
            continue
        elements.append(p)
    return elements


def _parity(p: Sequence[int]) -> int:
    seen = [False] * len(p)
    swaps = 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j, size = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            size += 1
        swaps += size - 1
    return swaps % 2


def _compose(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """Return a then b, written left to right."""
    return tuple(b[a[i]] for i in range(len(a)))


@dataclass
class _GroupWordProblem:
    """Compute the product of a sequence of generators in a permutation group.

    Two variants are used in the paper. ``A_4/S_4`` is solvable and admits a
    shallow reference program; ``A_5/S_5`` is not solvable, which forces serial
    computation and makes the prescribed trajectory rise slowly. That is why
    ``A_5/S_5`` is the sharpest test: the program cannot know the answer early,
    so any early commitment by the model is unlicensed.
    """

    degree: int
    name: str
    min_len: int = 4
    max_len: int = 8

    def __post_init__(self) -> None:
        alt = _permutation_group(self.degree, alternating=True)
        sym = _permutation_group(self.degree, alternating=False)
        self._groups = {"alt": alt, "sym": sym}
        self.components = (
            Component(f"A{self.degree}", tuple(alt), 0.5),
            Component(f"S{self.degree}", tuple(sym), 0.5),
        )
        check_weights(self.components)

    def _key(self, component: str) -> str:
        return "alt" if component.startswith("A") else "sym"

    def sample(self, rng: np.random.Generator, component: str) -> Instance:
        pool = self._groups[self._key(component)]
        length = int(rng.integers(self.min_len, self.max_len + 1))
        idx = rng.integers(len(pool), size=length)
        word = [pool[int(i)] for i in idx]
        acc: tuple[int, ...] = tuple(range(self.degree))
        for g in word:
            acc = _compose(acc, g)
        return Instance(
            payload=word,
            answer=acc,
            component=component,
            prompt=render_group(word, self.degree, component),
        )

    def n_steps(self, instance: Instance) -> int:
        return len(instance.payload)

    def state_after(self, instance: Instance, step: int) -> Hashable:
        """The running product. This is a sufficient statistic: the remaining
        generators act on it and nothing else about the prefix matters."""
        acc: tuple[int, ...] = tuple(range(self.degree))
        for g in instance.payload[:step]:
            acc = _compose(acc, g)
        return acc


@REGISTRY.register("a4s4")
class A4S4(_GroupWordProblem):
    def __init__(self, **kw):
        super().__init__(degree=4, name="a4s4", **kw)


@REGISTRY.register("a5s5")
class A5S5(_GroupWordProblem):
    def __init__(self, **kw):
        super().__init__(degree=5, name="a5s5", **kw)


def render_group(word: Sequence[Sequence[int]], degree: int, component: str) -> str:
    body = "\n".join(
        f"  step {i + 1}: " + " ".join(str(x + 1) for x in g) for i, g in enumerate(word)
    )
    return (
        f"Each step below is a permutation of the numbers 1 to {degree}, written "
        "as the images of 1, 2, ... in order.\n"
        "Apply them left to right, starting from the identity.\n"
        f"{body}\n"
        "Think step by step, then give the final permutation on one line.\n"
    )


# --------------------------------------------------------------------------
# DFA execution
# --------------------------------------------------------------------------


@REGISTRY.register("fsa")
@dataclass
class FSA:
    """Run a randomly drawn deterministic finite automaton on an input string.

    The answer is the accepting state reached. State reachability makes the
    prescribed trajectory rise at a moderate pace: some prefixes pin the answer
    down early, others do not.
    """

    name: str = "fsa"
    n_states_min: int = 2
    n_states_max: int = 4
    min_len: int = 8
    max_len: int = 16

    def __post_init__(self) -> None:
        self.components = tuple(
            Component(f"fsa{n}", tuple(range(n)), 1.0 / (self.n_states_max - self.n_states_min + 1))
            for n in range(self.n_states_min, self.n_states_max + 1)
        )
        check_weights(self.components)

    def sample(self, rng: np.random.Generator, component: str) -> Instance:
        n = int(component.removeprefix("fsa"))
        delta = rng.integers(n, size=(n, 2))
        length = int(rng.integers(self.min_len, self.max_len + 1))
        word = rng.integers(2, size=length)
        state = 0
        for c in word:
            state = int(delta[state, int(c)])
        payload = (delta.tolist(), word.tolist())
        return Instance(
            payload=payload,
            answer=state,
            component=component,
            prompt=render_fsa(delta.tolist(), word.tolist()),
        )

    def n_steps(self, instance: Instance) -> int:
        return len(instance.payload[1])

    def state_after(self, instance: Instance, step: int) -> Hashable:
        """The current DFA state, which is by construction all that matters."""
        delta, word = instance.payload
        state = 0
        for c in word[:step]:
            state = int(delta[state][int(c)])
        return state


def render_fsa(delta: list[list[int]], word: list[int]) -> str:
    rows = "\n".join(
        f"  from state {i}: on 0 go to {row[0]}, on 1 go to {row[1]}"
        for i, row in enumerate(delta)
    )
    return (
        "A finite automaton starts in state 0 and reads the input one symbol at "
        "a time.\n"
        f"{rows}\n"
        f"Input: {''.join(str(c) for c in word)}\n"
        "Think step by step, then give the final state as a single number.\n"
    )
