import math

import pytest

from induce_text.sources import (
    MARKOV_P,
    SKEW,
    SOURCES,
    long_range_copy,
    markov,
    pcfg,
    periodic,
    skewed_iid,
)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_deterministic_given_seed(name):
    gen = SOURCES[name]
    assert gen(500, 3) == gen(500, 3)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_generates_at_least_n(name):
    data, oracle_bits = SOURCES[name](500, 0)
    assert len(data) >= 500
    assert oracle_bits >= 0.0


def test_periodic_oracle_is_zero():
    data, oracle_bits = periodic(10, pattern=b"\x01\x02")
    assert oracle_bits == 0.0
    assert data == b"\x01\x02" * 5


def test_skewed_iid_oracle_matches_recount():
    data, oracle_bits = skewed_iid(5000, seed=2)
    assert set(data) <= set(SKEW)
    recount = sum(-math.log2(SKEW[b]) for b in data)
    assert oracle_bits == pytest.approx(recount)
    # Empirical mean cost should be near the source entropy, 1.875 bpc.
    assert oracle_bits / len(data) == pytest.approx(1.875, abs=0.1)


def test_markov_oracle_matches_recount():
    data, oracle_bits = markov(5000, seed=2)
    state = 0
    recount = 0.0
    for byte in data:
        recount += -math.log2(MARKOV_P[state][byte])
        state = byte
    assert oracle_bits == pytest.approx(recount)


def test_long_range_copy_alphabet_and_cost():
    data, oracle_bits = long_range_copy(2000, seed=0)
    assert set(data) <= set(range(16))
    # Long cheap copies should pull the average well below the 5 bits/byte
    # all-literal cost (choice + literal).
    assert 0 < oracle_bits < 4 * len(data)


def test_pcfg_stream_uses_grammar_alphabet():
    data, oracle_bits = pcfg(500, seed=0)
    assert set(data) <= {0, 1, 2, 3, 4, 5}
    assert oracle_bits > 0
