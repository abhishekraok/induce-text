import math

from induce_text.evaluate import (
    evaluate,
    evaluate_all,
    evaluate_external,
    evaluate_model,
    format_table,
)
from induce_text.models import Uniform


def test_uniform_costs_eight_bpc():
    data = bytes(range(256)) * 4
    r = evaluate_model(Uniform(), data)
    assert math.isclose(r.bits_per_byte, 8.0)
    assert r.total_bits == len(data) * 8.0


def test_adaptive_model_beats_uniform_on_repetitive_data():
    data = b"the quick brown fox " * 50
    uni = evaluate("uniform", data)
    o3 = evaluate("order3", data)
    assert o3.bits_per_byte < uni.bits_per_byte


def test_external_compressor_runs():
    data = b"aaaaaaaaaa" * 1000  # highly compressible
    r = evaluate_external("gzip", data)
    assert r.bits_per_byte < 1.0


def test_evaluate_all_preserves_names_and_order():
    data = b"hello world" * 20
    names = ["uniform", "order0", "gzip"]
    results = evaluate_all(names, data)
    assert [r.name for r in results] == names


def test_format_table_sorts_by_bpc():
    data = b"abcabcabc" * 100
    table = format_table(evaluate_all(["uniform", "order2", "lzma"], data))
    assert "model" in table and "bpc" in table
    # uniform (8 bpc) must be the worst -> last data row.
    lines = [ln for ln in table.splitlines() if ln and not ln.startswith("-")]
    assert lines[-1].startswith("uniform")
