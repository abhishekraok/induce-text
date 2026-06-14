"""Benchmark data: the enwik corpora (Wikipedia XML dumps).

enwik8 (100 MB) and enwik9 (1 GB) are the standard Hutter Prize / large-text
compression benchmarks.  They are *not* vendored in the repo (too large); this
module downloads them on demand into a local ``data/`` cache and reads arbitrary
slices for fast iteration.

Source: Matt Mahoney's site, http://mattmahoney.net/dc/
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

CORPORA = {
    "enwik8": {
        "url": "https://mattmahoney.net/dc/enwik8.zip",
        "member": "enwik8",
        "size": 100_000_000,
    },
    "enwik9": {
        "url": "https://mattmahoney.net/dc/enwik9.zip",
        "member": "enwik9",
        "size": 1_000_000_000,
    },
}


def default_data_dir() -> Path:
    """Return the repo-local ``data/`` directory (created if missing)."""
    # src/induce_text/data.py -> repo root is three parents up.
    root = Path(__file__).resolve().parents[2]
    d = root / "data"
    d.mkdir(exist_ok=True)
    return d


def corpus_path(name: str, data_dir: Path | None = None) -> Path:
    """Return the local path where corpus ``name`` is (or would be) stored."""
    if name not in CORPORA:
        raise KeyError(f"unknown corpus {name!r}; choices: {sorted(CORPORA)}")
    base = data_dir or default_data_dir()
    return base / name


def download(name: str, data_dir: Path | None = None, *, force: bool = False) -> Path:
    """Download and unzip corpus ``name`` into the cache; return its path.

    Idempotent: if the file already exists and ``force`` is False, the existing
    copy is returned untouched.
    """
    spec = CORPORA[name]
    dest = corpus_path(name, data_dir)
    if dest.exists() and not force:
        return dest

    base = dest.parent
    zip_path = base / f"{name}.zip"
    print(f"downloading {spec['url']} -> {zip_path}")
    urllib.request.urlretrieve(spec["url"], zip_path)

    print(f"extracting {spec['member']} -> {dest}")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(spec["member"]) as src, open(dest, "wb") as out:
            while chunk := src.read(1 << 20):
                out.write(chunk)
    zip_path.unlink(missing_ok=True)
    return dest


def load(
    name_or_path: str | Path,
    *,
    n_bytes: int | None = None,
    offset: int = 0,
    data_dir: Path | None = None,
) -> bytes:
    """Load up to ``n_bytes`` bytes from a corpus, starting at ``offset``.

    ``name_or_path`` may be a known corpus name (downloaded if absent) or a
    direct path to any file.  ``n_bytes=None`` reads to end of file.
    """
    if isinstance(name_or_path, str) and name_or_path in CORPORA:
        path = download(name_or_path, data_dir)
    else:
        path = Path(name_or_path)
        if not path.exists():
            raise FileNotFoundError(path)

    with open(path, "rb") as f:
        f.seek(offset)
        return f.read() if n_bytes is None else f.read(n_bytes)
