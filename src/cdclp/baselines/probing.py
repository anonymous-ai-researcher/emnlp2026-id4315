"""Representation-based baselines, and why they are reported separately.

A learned probe or a learned alignment answers whether the answer is *present*
in the residual stream. Presence is not use: a direction can be decodable from
an activation that no downstream computation reads. Worse, a sufficiently
flexible alignment can be fitted to any target on a randomly initialized model,
which makes a positive reading uninformative unless the flexibility is bounded.

CLP avoids the issue by fixing the map to the identity: nothing is fitted, so
there is nothing that could have been fitted to noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LinearProbe:
    """Difference-of-means probe on the residual stream."""

    def fit(self, activations: np.ndarray, labels: np.ndarray) -> "LinearProbe":
        classes = np.unique(labels)
        if len(classes) != 2:
            raise ValueError(f"binary probe needs 2 classes, got {len(classes)}")
        mu = [activations[labels == c].mean(axis=0) for c in classes]
        self.w_ = mu[1] - mu[0]
        self.b_ = -0.5 * float(self.w_ @ (mu[0] + mu[1]))
        self.classes_ = classes
        return self

    def predict(self, activations: np.ndarray) -> np.ndarray:
        scores = activations @ self.w_ + self.b_
        return np.where(scores > 0, self.classes_[1], self.classes_[0])

    def accuracy(self, activations: np.ndarray, labels: np.ndarray) -> float:
        return float((self.predict(activations) == labels).mean())


@dataclass
class LayerwiseMediation:
    """Mediated effect of a layer, estimated by patching that layer alone.

    Reported at depth rather than at step, so it answers "which layer carries
    the behavior" rather than "when was the answer fixed". The two are
    complementary; neither substitutes for the other.
    """

    n_layers: int

    def effect(self, patched_rate: np.ndarray, control_rate: np.ndarray) -> np.ndarray:
        if patched_rate.shape != control_rate.shape:
            raise ValueError("patched and control rates must have the same shape")
        return np.asarray(patched_rate, float) - np.asarray(control_rate, float)
