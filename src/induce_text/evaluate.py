"""Evaluation harness: score next-byte models and reference compressors.

Two kinds of thing get scored into the same :class:`Result` currency (bpc):

1. **Adaptive next-byte models** (:func:`evaluate_model`) — run the online
   prediction loop, accumulating ``-log2 P(actual byte)`` per position.  This is
   the cost an ideal arithmetic coder driven by the model would pay.

2. **Off-the-shelf compressors** (:func:`evaluate_external`) — gzip / bz2 / lzma.
   We just compress the bytes and read off the output size.  These are the
   reference bars our models must clear.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
from math import log2

from induce_text.metrics import Result
from induce_text.models import ByteModel, build

# Reference compressors available by CLI name.
EXTERNAL = {
    "gzip": lambda b: gzip.compress(b, compresslevel=9),
    "bz2": lambda b: bz2.compress(b, compresslevel=9),
    "lzma": lambda b: lzma.compress(b, preset=9),
}


def evaluate_model(model: ByteModel, data: bytes, *, name: str | None = None) -> Result:
    """Run the online prediction loop and return the model's coding cost.

    For each byte: read the model's probability for the true byte, add
    ``-log2 p`` bits, then let the model adapt.
    """
    total_bits = 0.0
    for byte in data:
        p = model.prob(byte)
        if p <= 0.0:
            raise ValueError(
                f"model assigned non-positive probability {p} to byte {byte}; "
                "a valid model must keep every byte strictly positive"
            )
        total_bits += -log2(p)
        model.update(byte)
    return Result(
        name=name or getattr(model, "name", type(model).__name__),
        data_bytes=len(data),
        total_bits=total_bits,
    )


def evaluate_external(name: str, data: bytes) -> Result:
    """Compress ``data`` with reference compressor ``name`` and report its size.

    The reported bits include the compressor's own header/framing overhead,
    which is negligible at benchmark sizes but inflates bpc on tiny inputs.
    """
    if name not in EXTERNAL:
        raise KeyError(f"unknown compressor {name!r}; choices: {sorted(EXTERNAL)}")
    compressed = EXTERNAL[name](data)
    return Result(name=name, data_bytes=len(data), total_bits=len(compressed) * 8.0)


def evaluate(name: str, data: bytes) -> Result:
    """Evaluate by name, dispatching to a model or an external compressor."""
    if name in EXTERNAL:
        return evaluate_external(name, data)
    return evaluate_model(build(name), data, name=name)


def evaluate_all(names: list[str], data: bytes) -> list[Result]:
    """Evaluate several models/compressors on the same data, preserving order."""
    return [evaluate(n, data) for n in names]


def format_table(results: list[Result]) -> str:
    """Render results as an aligned text table sorted by bpc (best first)."""
    rows = sorted(results, key=lambda r: r.bits_per_byte)
    header = f"{'model':<12} {'bpc':>8} {'ratio':>8} {'saving':>8} {'comp.bytes':>12}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.name:<12} {r.bits_per_byte:>8.4f} {r.compression_ratio:>8.4f} "
            f"{r.space_saving:>8.2%} {r.compressed_bytes:>12,.0f}"
        )
    return "\n".join(lines)
