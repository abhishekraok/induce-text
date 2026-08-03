"""Baseline models wearing the Model interface.

These are the honest opponents any synthesized predictor has to beat, ordered
by what they can see:

- ``Uniform``      — sees nothing; exactly 8 bpc always.  Harness self-test.
- ``AdaptiveIID``  — sees byte frequencies (order-0); the calibration rung's
  natural opponent.
- ``ContextK``     — sees the last k bytes (order-k Markov); the context
  rung's natural opponent.  Additive smoothing within each context; an
  unseen context predicts uniform.

Deliberately absent: any mixing/backoff across orders (PPM-style).  A mixer
of models is the deferred composition decision (CLAUDE.md decision 5) and a
research choice, not a baseline.  The gap between ContextK and a real PPM
is part of what that future work has to earn.

Smoothing default is alpha = 0.5, the Krichevsky–Trofimov estimator — the
add-half rule that is minimax-optimal for this "counts + prior" family.
"""

from __future__ import annotations

import numpy as np

from induce_text.model import Distribution

ALPHABET = 256


class Uniform:
    """Predicts 1/256 for every byte, forever.  Exactly 8 bpc on anything."""

    def init(self) -> None:
        return None

    def predict(self, state: None) -> Distribution:
        return np.full(ALPHABET, 1.0 / ALPHABET)

    def absorb(self, state: None, byte: int) -> None:
        return None


class AdaptiveIID:
    """Order-0 adaptive model: smoothed byte frequencies, learned online.

    P(b) = (count[b] + alpha) / (total + 256 * alpha).  Before any data this
    is exactly uniform, so the first byte always costs 8 bits.
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def init(self) -> np.ndarray:
        return np.zeros(ALPHABET)

    def predict(self, counts: np.ndarray) -> Distribution:
        return (counts + self.alpha) / (counts.sum() + ALPHABET * self.alpha)

    def absorb(self, counts: np.ndarray, byte: int) -> np.ndarray:
        counts[byte] += 1
        return counts


class ContextK:
    """Order-k adaptive model: smoothed byte frequencies per k-byte context.

    Each context owns its own count table with additive smoothing, so a
    never-seen context predicts uniform (8 bits) — the cold-start price of
    more context.  Expect the expressivity/tractability tension in the
    numbers: larger k wins asymptotically and loses on small data.

    State: (table: dict[bytes, counts], context: last min(k, t) bytes).
    """

    def __init__(self, k: int, alpha: float = 0.5):
        if k < 1:
            raise ValueError("k must be >= 1; use AdaptiveIID for order 0")
        self.k = k
        self.alpha = alpha

    def init(self) -> tuple[dict[bytes, np.ndarray], bytes]:
        return ({}, b"")

    def predict(self, state: tuple[dict[bytes, np.ndarray], bytes]) -> Distribution:
        table, context = state
        counts = table.get(context)
        if counts is None:
            return np.full(ALPHABET, 1.0 / ALPHABET)
        return (counts + self.alpha) / (counts.sum() + ALPHABET * self.alpha)

    def absorb(
        self, state: tuple[dict[bytes, np.ndarray], bytes], byte: int
    ) -> tuple[dict[bytes, np.ndarray], bytes]:
        table, context = state
        counts = table.get(context)
        if counts is None:
            counts = np.zeros(ALPHABET)
            table[context] = counts
        counts[byte] += 1
        context = (context + bytes([byte]))[-self.k :]
        return (table, context)


def default_models() -> dict[str, object]:
    """The standard baseline lineup for the benchmark matrix."""
    return {
        "uniform": Uniform(),
        "iid": AdaptiveIID(),
        "ctx1": ContextK(1),
        "ctx2": ContextK(2),
        "ctx3": ContextK(3),
    }
