import math

import pytest

from induce_text.models import (
    ALPHABET,
    ContextModel,
    Interpolated,
    Order0,
    Uniform,
    build,
)


def _distribution_sums_to_one(model):
    return math.isclose(sum(model.prob(b) for b in range(ALPHABET)), 1.0, rel_tol=1e-9)


def test_uniform_is_eight_bits():
    m = Uniform()
    assert m.prob(0) == 1.0 / 256
    assert -math.log2(m.prob(123)) == 8.0
    assert _distribution_sums_to_one(m)


@pytest.mark.parametrize("model_factory", [Order0, lambda: ContextModel(2), lambda: Interpolated(3)])
def test_models_define_valid_distribution_before_and_after_updates(model_factory):
    m = model_factory()
    assert _distribution_sums_to_one(m)
    for b in b"hello hello hello":
        m.update(b)
    assert _distribution_sums_to_one(m)


def test_order0_learns_frequencies():
    m = Order0(alpha=1.0)
    for _ in range(100):
        m.update(ord("a"))
    # 'a' should now be far more probable than an unseen byte.
    assert m.prob(ord("a")) > m.prob(ord("z")) * 50


def test_context_model_uses_context():
    # Teach it that 'a' is always followed by 'b', then present a fresh 'a'.
    m = ContextModel(order=1, alpha=1.0)
    for _ in range(50):
        m.update(ord("a"))
        m.update(ord("b"))
    m.update(ord("a"))
    # In context 'a', the seen successor 'b' must dominate an unseen byte.
    # (Absolute value stays modest because alpha=1 smooths over 256 symbols.)
    assert m.prob(ord("b")) == max(m.prob(c) for c in range(256))
    assert m.prob(ord("b")) > 50 * m.prob(ord("z"))


def test_smaller_alpha_sharpens_predictions():
    # With light smoothing the same evidence yields a near-certain prediction.
    m = ContextModel(order=1, alpha=0.01)
    for _ in range(50):
        m.update(ord("a"))
        m.update(ord("b"))
    m.update(ord("a"))
    assert m.prob(ord("b")) > 0.9


def test_higher_order_beats_lower_on_structured_data():
    data = b"abc" * 200
    o0, o2 = Order0(), ContextModel(2)
    bits0 = bits2 = 0.0
    for b in data:
        bits0 += -math.log2(o0.prob(b)); o0.update(b)
        bits2 += -math.log2(o2.prob(b)); o2.update(b)
    assert bits2 < bits0


def test_build_unknown_raises():
    with pytest.raises(KeyError):
        build("does-not-exist")
