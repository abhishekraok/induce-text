import numpy as np
import pytest

from induce_text.baselines import AdaptiveIID, Uniform
from induce_text.model import bpc, score_bits


def test_uniform_is_exactly_8_bpc():
    data = bytes(range(256)) + b"hello world" * 10
    assert bpc(Uniform(), data) == pytest.approx(8.0, abs=1e-12)


def test_score_bits_shape_and_first_byte():
    # Any counts-based model with no data yet predicts uniform: the first
    # byte always costs exactly 8 bits.
    data = b"aaab"
    bits = score_bits(AdaptiveIID(), data)
    assert bits.shape == (4,)
    assert bits[0] == pytest.approx(8.0, abs=1e-12)
    # Second byte, after one 'a': p = (1 + 0.5) / (1 + 128).
    assert bits[1] == pytest.approx(-np.log2(1.5 / 129), abs=1e-12)


def test_prediction_precedes_absorption():
    # The cost of byte t must not depend on byte t itself: scoring "ab" and
    # "aa" must charge the same amount for position 0 and 1's prediction
    # context, differing only via the actual byte's probability.
    bits_ab = score_bits(AdaptiveIID(), b"ab")
    bits_aa = score_bits(AdaptiveIID(), b"aa")
    assert bits_ab[0] == bits_aa[0] == pytest.approx(8.0, abs=1e-12)
    # After absorbing 'a', 'a' must be more probable than 'b'.
    assert bits_aa[1] < bits_ab[1]


def test_zero_probability_scores_inf():
    class Certain:
        """Puts all mass on byte 0."""

        def init(self):
            return None

        def predict(self, state):
            dist = np.zeros(256)
            dist[0] = 1.0
            return dist

        def absorb(self, state, byte):
            return None

    bits = score_bits(Certain(), b"\x00\x01")
    assert bits[0] == 0.0
    assert bits[1] == np.inf
