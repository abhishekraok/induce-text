"""induce-text: program synthesis for language modelling, scored as compression.

The unifying view: a language model *is* a compressor.  At each position the
model emits a probability distribution over the next byte; the coding cost of
the byte that actually occurs is ``-log2 P(byte)`` bits.  Summed over a stream,
that is the number of bits an (ideal) arithmetic coder would emit.  We therefore
measure any next-byte model directly in **bits per byte (bpc)** without needing
to build a real coder yet.
"""

from __future__ import annotations

__version__ = "0.1.0"
