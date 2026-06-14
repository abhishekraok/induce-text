from pathlib import Path

import pytest

from induce_text import data as data_mod


def test_corpus_path_unknown_raises():
    with pytest.raises(KeyError):
        data_mod.corpus_path("not-a-corpus")


def test_load_from_arbitrary_file(tmp_path: Path):
    f = tmp_path / "blob.bin"
    f.write_bytes(bytes(range(256)))
    assert data_mod.load(f, n_bytes=10) == bytes(range(10))
    assert data_mod.load(f, n_bytes=5, offset=2) == bytes(range(2, 7))
    assert data_mod.load(f) == bytes(range(256))


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        data_mod.load(tmp_path / "nope.bin")


def test_known_corpora_have_specs():
    for name in ("enwik8", "enwik9"):
        spec = data_mod.CORPORA[name]
        assert spec["url"].endswith(".zip")
        assert spec["size"] > 0
