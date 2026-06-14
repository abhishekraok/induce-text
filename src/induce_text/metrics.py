"""Compression / language-modelling metrics.

The benchmark currency is **bits per byte** (bpc, sometimes written bpb): the
average number of bits used to encode each byte of the original data.  Reference
points on enwik:

    - no compression .... 8.00 bpc
    - gzip .............. ~3.1 bpc
    - bzip2 ............. ~2.3 bpc
    - PPM / lzma ........ ~2.0 bpc
    - state of the art .. ~1.0 bpc (cmix / nncp)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    """Outcome of evaluating one model/compressor on one chunk of data."""

    name: str
    data_bytes: int
    total_bits: float

    @property
    def bits_per_byte(self) -> float:
        """Average bits used per byte of original data (the headline number)."""
        if self.data_bytes == 0:
            return 0.0
        return self.total_bits / self.data_bytes

    @property
    def compressed_bytes(self) -> float:
        """Total bits expressed as bytes (the size an ideal coder would emit)."""
        return self.total_bits / 8.0

    @property
    def compression_ratio(self) -> float:
        """compressed / original size (lower is better; 1.0 means no gain)."""
        if self.data_bytes == 0:
            return 1.0
        return self.compressed_bytes / self.data_bytes

    @property
    def space_saving(self) -> float:
        """Fraction of size removed, ``1 - compression_ratio`` (higher better)."""
        return 1.0 - self.compression_ratio


def bits_per_byte(total_bits: float, data_bytes: int) -> float:
    """Return ``total_bits / data_bytes`` (0.0 for empty input)."""
    if data_bytes == 0:
        return 0.0
    return total_bits / data_bytes
