"""The benchmark matrix: models x sources -> bpc, vs oracle and reference.

Runs every baseline over every requested source, reports bpc against the
source's oracle MDL (when known) and against off-the-shelf compressors
(gzip/bz2/xz) as external reference points.  Persists a JSON record and,
optionally, learning-curve plots (running bpc vs position — where slow
warm-up and silent collapse become visible).

Reference-compressor caveat: their numbers include container headers, which
inflates small inputs; treat them as landmarks, not opponents.  They also see
the whole input at once (two passes over blocks), while our models are
strictly online — charged for every byte including the first.
"""

from __future__ import annotations

import bz2
import gzip
import json
import lzma
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from induce_text import data as data_mod
from induce_text.baselines import default_models
from induce_text.model import Model, score_bits
from induce_text.sources import SOURCES


@dataclass
class Row:
    source: str
    n: int
    oracle_bpc: float | None
    model_bpc: dict[str, float] = field(default_factory=dict)
    ref_bpc: dict[str, float] = field(default_factory=dict)
    # per-model per-byte bits, kept for plotting; not serialized.
    curves: dict[str, np.ndarray] = field(default_factory=dict)


def resolve_source(spec: str, n: int, seed: int) -> tuple[str, bytes, float | None]:
    """Resolve a source spec to (name, data, oracle_bits).

    ``spec`` is a synthetic source name from SOURCES or a corpus name
    (e.g. ``enwik8``), optionally with a byte count: ``enwik8:1000000``.
    Corpora have no oracle.
    """
    name, _, size = spec.partition(":")
    length = int(size) if size else n
    if name in SOURCES:
        data, oracle_bits = SOURCES[name](length, seed)
        return name, data, oracle_bits
    data = data_mod.load(name, n_bytes=length)
    return name, data, None


def reference_bpc(data: bytes) -> dict[str, float]:
    n = len(data)
    return {
        "gzip": len(gzip.compress(data, 9)) * 8 / n,
        "bz2": len(bz2.compress(data, 9)) * 8 / n,
        "xz": len(lzma.compress(data, preset=9)) * 8 / n,
    }


def run(
    source_specs: list[str],
    models: dict[str, Model] | None = None,
    *,
    n: int = 30_000,
    seed: int = 0,
    verbose: bool = True,
) -> list[Row]:
    models = models if models is not None else default_models()
    rows = []
    for spec in source_specs:
        name, data, oracle_bits = resolve_source(spec, n, seed)
        row = Row(
            source=name,
            n=len(data),
            oracle_bpc=None if oracle_bits is None else oracle_bits / len(data),
        )
        row.ref_bpc = reference_bpc(data)
        for model_name, model in models.items():
            t0 = time.perf_counter()
            bits = score_bits(model, data)
            dt = time.perf_counter() - t0
            row.model_bpc[model_name] = float(bits.mean())
            row.curves[model_name] = bits
            if verbose:
                print(
                    f"  {name:>15} x {model_name:<8} "
                    f"{row.model_bpc[model_name]:7.4f} bpc  ({dt:.1f}s)"
                )
        rows.append(row)
    return rows


def format_table(rows: list[Row]) -> str:
    model_names = list(rows[0].model_bpc)
    ref_names = list(rows[0].ref_bpc)
    header = ["source", "n", "oracle"] + model_names + ref_names
    lines = []
    widths = [16, 9, 7] + [8] * (len(model_names) + len(ref_names))
    lines.append("  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for row in rows:
        cells = [
            row.source,
            f"{row.n:,}",
            "-" if row.oracle_bpc is None else f"{row.oracle_bpc:.4f}",
        ]
        cells += [f"{row.model_bpc[m]:.4f}" for m in model_names]
        cells += [f"{row.ref_bpc[r]:.4f}" for r in ref_names]
        lines.append("  ".join(c.rjust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def save_json(rows: list[Row], out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"results-{stamp}.json"
    payload = {
        "meta": meta | {"timestamp": stamp},
        "rows": [
            {
                "source": r.source,
                "n": r.n,
                "oracle_bpc": r.oracle_bpc,
                "model_bpc": r.model_bpc,
                "ref_bpc": r.ref_bpc,
            }
            for r in rows
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_plots(rows: list[Row], out_dir: Path) -> list[Path]:
    """One learning-curve figure per source: running bpc vs position."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(exist_ok=True)
    paths = []
    for row in rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, bits in row.curves.items():
            positions = np.arange(1, len(bits) + 1)
            ax.plot(positions, np.cumsum(bits) / positions, label=model_name)
        if row.oracle_bpc is not None:
            ax.axhline(
                row.oracle_bpc, linestyle="--", color="black", label="oracle"
            )
        ax.set(
            title=f"{row.source} (n={row.n:,})",
            xlabel="position",
            ylabel="running bpc",
            xscale="log",
        )
        ax.legend()
        path = out_dir / f"curves-{row.source}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
