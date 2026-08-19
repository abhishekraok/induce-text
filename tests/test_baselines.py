import pytest

from induce_text.baselines import AdaptiveIID, ContextK, Uniform
from induce_text.model import bpc
from induce_text.sources import markov, periodic, skewed_iid


def test_adaptive_iid_learns_constant_data():
    data = b"\x07" * 2000
    assert bpc(AdaptiveIID(), data) < 0.5
    assert bpc(Uniform(), data) == pytest.approx(8.0)


def test_adaptive_iid_approaches_iid_cost():
    data, process_bits = skewed_iid(20_000, seed=1)
    process_bpc = process_bits / len(data)
    model_bpc = bpc(AdaptiveIID(), data)
    # Online learning pays a redundancy above the process cost, but it shrinks:
    # at 20k bytes the gap should be small.
    assert model_bpc == pytest.approx(process_bpc, abs=0.25)


def test_context_beats_iid_on_markov():
    data, process_bits = markov(20_000, seed=1)
    process_bpc = process_bits / len(data)
    iid_bpc = bpc(AdaptiveIID(), data)
    ctx1_bpc = bpc(ContextK(1), data)
    # The chain's stationary distribution is uniform over 4 symbols: no
    # i.i.d. model can beat 2 bpc.  One symbol of context reaches toward
    # the conditional entropy (~0.61 bpc).
    assert iid_bpc > 1.7
    assert ctx1_bpc < 1.2
    assert ctx1_bpc > process_bpc  # can't beat the source, only approach it


def test_context_nails_periodic():
    data, _ = periodic(4000)
    # The source has zero entropy but the model pays a warm-up tax:
    # add-half smoothing over a 256 alphabet costs ~0.75 bpc at this n.
    assert bpc(ContextK(1), data) < 1.0
    # i.i.d. sees a uniform 4-letter alphabet: stuck near 2 bpc.
    assert bpc(AdaptiveIID(), data) > 1.8


def test_context_k_validation():
    with pytest.raises(ValueError):
        ContextK(0)
