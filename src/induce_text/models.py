"""Next-byte models (the "language models").

A model is an *online, adaptive* predictor over the byte stream.  At each
position it must answer "how probable is the next byte?", and then it is told
what the byte actually was so it can adapt.  This mirrors how an adaptive
arithmetic coder works: the decoder re-derives the model's state from the bytes
it has already decoded, so the model itself costs ~0 bits to ship.

Minimal interface (see :class:`ByteModel`):

    p = model.prob(byte)   # P(next == byte | history so far), in (0, 1]
    model.update(byte)     # observe the true byte and adapt

Only ``prob`` of the *actual* byte is needed to score cross-entropy, so models
need not materialise the full 256-way distribution — but each model must behave
*as if* it defines a proper normalised distribution, otherwise the bits it
reports would not be achievable by a real coder.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

ALPHABET = 256


@runtime_checkable
class ByteModel(Protocol):
    """An online adaptive predictor over a byte stream."""

    def prob(self, byte: int) -> float:
        """Return P(next byte == ``byte``) given everything observed so far.

        Must be strictly positive (a zero would cost infinite bits) and, summed
        over all 256 possible ``byte`` values, equal 1.0.
        """
        ...

    def update(self, byte: int) -> None:
        """Observe the true next byte and adapt internal state."""
        ...

    def reset(self) -> None:
        """Forget all observed history (return to the initial state)."""
        ...


class Uniform:
    """Assigns every byte probability 1/256 — the no-model baseline (8.0 bpc)."""

    name = "uniform"

    def prob(self, byte: int) -> float:
        return 1.0 / ALPHABET

    def update(self, byte: int) -> None:
        pass

    def reset(self) -> None:
        pass


class Order0:
    """Adaptive order-0 model: byte frequencies with add-alpha smoothing.

    No context at all — just "how often have I seen this byte?".  Converges to
    the empirical byte entropy of the stream (~5 bpc on English text).
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.name = "order0"
        self.reset()

    def reset(self) -> None:
        self._counts = [0] * ALPHABET
        self._total = 0

    def prob(self, byte: int) -> float:
        denom = self._total + self.alpha * ALPHABET
        return (self._counts[byte] + self.alpha) / denom

    def update(self, byte: int) -> None:
        self._counts[byte] += 1
        self._total += 1


class ContextModel:
    """Adaptive order-``k`` model: P(next | previous ``k`` bytes), add-alpha.

    Counts how often each byte follows each length-``k`` context.  Unseen
    contexts fall back to uniform (every byte 1/256), which keeps probabilities
    valid before any evidence is gathered.  This is a plain n-gram; it has no
    backoff between orders (see :class:`Interpolated` for that).
    """

    def __init__(self, order: int, alpha: float = 1.0):
        if order < 0:
            raise ValueError("order must be >= 0")
        self.order = order
        self.alpha = alpha
        self.name = f"order{order}"
        self.reset()

    def reset(self) -> None:
        # context (bytes) -> [per-byte counts, total]
        self._counts: dict[bytes, list[int]] = defaultdict(lambda: [0] * ALPHABET)
        self._totals: dict[bytes, int] = defaultdict(int)
        self._history = bytearray()

    def _context(self) -> bytes:
        if self.order == 0:
            return b""
        return bytes(self._history[-self.order :])

    def prob(self, byte: int) -> float:
        ctx = self._context()
        total = self._totals.get(ctx, 0)
        denom = total + self.alpha * ALPHABET
        count = self._counts[ctx][byte] if ctx in self._counts else 0
        return (count + self.alpha) / denom

    def update(self, byte: int) -> None:
        ctx = self._context()
        self._counts[ctx][byte] += 1
        self._totals[ctx] += 1
        self._history.append(byte)
        if self.order and len(self._history) > self.order:
            # Keep history bounded to what we actually need for the context.
            del self._history[:-self.order]


class Interpolated:
    """Linear interpolation of order-0..``max_order`` context models.

    A simple, robust step up from a single fixed order: the prediction is a
    weighted blend of every order from 0 up to ``max_order``.  Higher orders are
    more specific (and usually sharper once they have evidence); lower orders
    provide a safety net when the high-order context has never been seen.  By
    default weights grow geometrically with the order so the longest context
    dominates when it is informative, while still always mixing in the shorter
    ones (a poor man's PPM backoff — no escape mechanism, just a fixed blend).
    """

    def __init__(self, max_order: int = 3, alpha: float = 1.0, decay: float = 4.0):
        if max_order < 0:
            raise ValueError("max_order must be >= 0")
        self.max_order = max_order
        self.name = f"interp{max_order}"
        self._models = [ContextModel(o, alpha=alpha) for o in range(max_order + 1)]
        # weight(order) proportional to decay**order, normalised to sum to 1.
        raw = [decay ** o for o in range(max_order + 1)]
        s = sum(raw)
        self._weights = [w / s for w in raw]

    def reset(self) -> None:
        for m in self._models:
            m.reset()

    def prob(self, byte: int) -> float:
        return sum(w * m.prob(byte) for w, m in zip(self._weights, self._models))

    def update(self, byte: int) -> None:
        for m in self._models:
            m.update(byte)


# Registry of constructible baseline models, keyed by CLI name.
REGISTRY: dict[str, Callable[[], ByteModel]] = {
    "uniform": Uniform,
    "order0": Order0,
    "order1": lambda: ContextModel(1),
    "order2": lambda: ContextModel(2),
    "order3": lambda: ContextModel(3),
    "order4": lambda: ContextModel(4),
    "interp3": lambda: Interpolated(3),
    "interp4": lambda: Interpolated(4),
}


def build(name: str) -> ByteModel:
    """Construct a model by its registry name."""
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; choices: {sorted(REGISTRY)}")
    return REGISTRY[name]()
