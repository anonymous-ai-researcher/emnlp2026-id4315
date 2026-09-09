"""HuggingFace backend for the residual-stream patch.

Kept separate from ``intervention.py`` so that the measurement logic can be
tested without a model in memory, and so that a different runtime can be
dropped in by supplying the same two callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class HFRunner:
    """Wraps a causal LM and exposes capture/inject for the last layer.

    Notes
    -----
    * ``capture`` runs the source prompt and returns the hidden state at the
      requested chain position. The position is an absolute token index into the
      generated chain, not into the prompt.
    * ``inject`` registers a forward hook that overwrites that position on the
      next forward pass and returns the handle so the caller can remove it.
    * Generation is greedy for the answer and sampled for the chain, matching
      the protocol: the chain is a sample from the model's own distribution,
      while the answer readout should not add noise of its own.
    """

    model: Any
    tokenizer: Any
    device: str = "cuda"
    chain_temperature: float = 0.6
    chain_top_p: float = 0.95
    max_chain_tokens: int = 512
    max_answer_tokens: int = 32

    def __post_init__(self) -> None:
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._pending: np.ndarray | None = None
        self._pending_pos: int | None = None

    # -- capture ---------------------------------------------------------
    def capture(self, prompt: str, position: int) -> np.ndarray:
        key = (prompt, position)
        if key in self._cache:
            return self._cache[key]
        import torch

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.chain_temperature,
                top_p=self.chain_top_p,
                max_new_tokens=self.max_chain_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        # hidden_states is a tuple over generated tokens, each a tuple over
        # layers; [-1] selects the last layer.
        n_prompt = enc["input_ids"].shape[1]
        idx = min(position, len(out.hidden_states) - 1)
        h = out.hidden_states[idx][-1][0, -1, :]
        value = h.detach().float().cpu().numpy()
        self._cache[key] = value
        return value

    # -- inject ----------------------------------------------------------
    def inject(self, prompt: str, position: int, value: np.ndarray) -> Any:
        import torch

        self._pending = value
        self._pending_pos = position
        layer = self.model.model.layers[-1]

        def hook(_module, _inp, output):
            if self._pending is None:
                return output
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[1] > self._pending_pos:
                repl = torch.as_tensor(
                    self._pending, dtype=hidden.dtype, device=hidden.device
                )
                hidden[0, self._pending_pos, :] = repl
                self._pending = None
            return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden

        return layer.register_forward_hook(hook)

    # -- generation ------------------------------------------------------
    def generate(self, prompt: str) -> str:
        import torch

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                do_sample=True,
                temperature=self.chain_temperature,
                top_p=self.chain_top_p,
                max_new_tokens=self.max_chain_tokens + self.max_answer_tokens,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt) :]

    def clear_cache(self) -> None:
        self._cache.clear()
