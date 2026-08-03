"""The model interface and the scoring scan.

This encodes settled design decisions 1-4 from CLAUDE.md:

- A language model is a compressor.  The cost of the actual next byte is
  ``-log2 P(byte)``; the sum over a stream is the number of bits an ideal
  arithmetic coder would emit.  No coder is needed to *measure*, but the
  interface stays coder-ready: ``predict`` returns the full distribution.
- Symbol = byte (256-way).
- State is explicit and threaded: ``predict(state) -> distribution``,
  ``absorb(state, byte) -> state'``.  The online loop is a scan.  Invariant:
  the prediction at position t depends only on bytes before t.

Purity note (to ratify or reject when this is rewritten by hand): ``absorb``
is *allowed to mutate its argument* and return it.  The signature is the pure
JAX-style step, so a pure implementation drops in unchanged, but the baseline
implementations mutate for speed.  Anything that needs snapshot/branch
semantics (search will) must deep-copy the state first — or replace the
offending model with a persistent-state one.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

# A distribution over the next byte: shape (256,), non-negative, sums to 1.
# Models should keep every entry strictly positive; a zero on the byte that
# actually occurs scores +inf bits, which is the honest price of certainty.
Distribution = np.ndarray

State = Any


class Model(Protocol):
    """A next-byte predictor with explicit, threaded state."""

    def init(self) -> State: ...

    def predict(self, state: State) -> Distribution: ...

    def absorb(self, state: State, byte: int) -> State: ...


def score_bits(model: Model, data: bytes) -> np.ndarray:
    """Run the online scan over ``data``; return per-byte costs in bits.

    ``bits[t] = -log2 P(data[t] | data[:t])`` under the model's own online
    beliefs.  ``bits.sum()`` is the length of the ideal arithmetic code;
    ``bits.mean()`` is bpc.  The model is charged for every byte including
    the first: there is no free warm-up.
    """
    state = model.init()
    bits = np.empty(len(data))
    for t, byte in enumerate(data):
        dist = model.predict(state)
        p = dist[byte]
        bits[t] = -np.log2(p) if p > 0 else np.inf
        state = model.absorb(state, byte)
    return bits


def bpc(model: Model, data: bytes) -> float:
    """Bits per character (byte): mean per-byte cost over the stream."""
    return float(score_bits(model, data).mean())
