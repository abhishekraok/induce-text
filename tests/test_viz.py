import numpy as np
import pytest

from induce_text.baselines import AdaptiveIID, ContextK
from induce_text.pcfg_gen import RecordingChoice, sample
from induce_text.sources import markov, pcfg, pcfg_main_grammar, pcfg_test_grammar
from induce_text.viz import (
    calibration_data,
    delta_page,
    derivation_tree,
    heat_data,
    heat_page,
    subtree_bits,
    table_growth,
    tree_page,
)


@pytest.mark.parametrize("grammar", [pcfg_test_grammar, pcfg_main_grammar])
@pytest.mark.parametrize("seed", range(10))
def test_derivation_tree_matches_sample(grammar, seed):
    # derivation_tree self-checks: it raises if its walk of the grammar
    # disagrees with the author's interpreter or leaves bits unconsumed.
    rule, env = grammar()
    cs = RecordingChoice(seed=seed)
    out = sample(rule=rule, env=env, choicesource=cs)
    root, leaves = derivation_tree(rule, env, cs.choices)
    assert leaves == out
    # Every transcript bit appears exactly once in the tree as a choice.
    assert subtree_bits(root) == len(cs.choices)


def test_tree_page_renders():
    rule, env = pcfg_test_grammar()
    cs = RecordingChoice(seed=1)
    sample(rule=rule, env=env, choicesource=cs)
    page = tree_page(rule, env, cs.choices, title="t")
    assert page.count("bit #") == len(cs.choices)


def test_heat_data_agrees_with_score_bits():
    # heat_data raises internally if its bits diverge from score_bits.
    data, _ = markov(500, seed=0)
    bits, top = heat_data(ContextK(1), data)
    assert bits.shape == (len(data),)
    assert len(top) == len(data)
    # Top guesses are sorted by probability.
    for guesses in top:
        probs = [p for _, p in guesses]
        assert probs == sorted(probs, reverse=True)


def test_heat_page_has_one_span_per_byte():
    data, _ = pcfg(300, seed=0)
    page = heat_page(AdaptiveIID(), data, title="t")
    assert page.count('title="pos ') == len(data)


def test_delta_page_has_one_span_per_byte():
    data, _ = markov(300, seed=0)
    page = delta_page(
        AdaptiveIID(), ContextK(1), data, name_a="iid", name_b="ctx1", title="t"
    )
    assert page.count('title="pos ') == len(data)


def test_calibration_counts_every_slot():
    data, _ = markov(400, seed=0)
    edges = np.logspace(-8, 0, 33)
    counts, mean_p, freq = calibration_data(AdaptiveIID(), data, edges)
    # Every position states 256 probabilities; all land in some bin.
    assert counts.sum() == len(data) * 256
    # Exactly one hit per position.
    assert np.nansum(freq * counts) == len(data)
    valid = counts > 0
    assert np.all((freq[valid] >= 0) & (freq[valid] <= 1))
    assert np.all((mean_p[valid] > 0) & (mean_p[valid] <= 1))


def test_table_growth_is_monotone():
    data, _ = markov(2000, seed=0)
    xs, sizes = table_growth(ContextK(2), data, every=100)
    assert np.all(np.diff(xs) > 0)
    assert np.all(np.diff(sizes) >= 0)
    # A 4-symbol order-2 source can mint at most 17 contexts (16 + the
    # short warm-up prefixes of length < 2, of which "" and one 1-byte).
    assert sizes[-1] <= 16 + 2
