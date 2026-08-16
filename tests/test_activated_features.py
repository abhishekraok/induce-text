import copy
import math

import numpy as np
import pytest

from induce_text.activated_features import ActivatedSuffixTree
from induce_text.baselines import AdaptiveIID, ContextK
from induce_text.model import score_bits


def test_configuration_validation():
    with pytest.raises(ValueError):
        ActivatedSuffixTree(-1)
    with pytest.raises(ValueError):
        ActivatedSuffixTree(alpha=0)
    with pytest.raises(ValueError):
        ActivatedSuffixTree(split_prior=0)
    with pytest.raises(ValueError):
        ActivatedSuffixTree(split_prior=1)


def test_predictions_are_coder_ready_and_predict_is_read_only():
    model = ActivatedSuffixTree(4)
    state = model.init()
    for byte in b"abracadabra":
        before = copy.deepcopy(state)
        first = model.predict(state)
        second = model.predict(state)
        assert first.shape == (256,)
        assert np.all(np.isfinite(first))
        assert np.all(first > 0)
        assert first.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.array_equal(first, second)
        assert state == before
        state = model.absorb(state, byte)


def test_depth_zero_is_exactly_adaptive_iid():
    tree = ActivatedSuffixTree(0, alpha=0.5)
    iid = AdaptiveIID(alpha=0.5)
    tree_state = tree.init()
    iid_state = iid.init()
    for byte in bytes(range(256)) + b"mississippi":
        assert np.array_equal(tree.predict(tree_state), iid.predict(iid_state))
        tree_state = tree.absorb(tree_state, byte)
        iid_state = iid.absorb(iid_state, byte)


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_deepest_local_feature_is_exactly_context_k(depth):
    tree = ActivatedSuffixTree(depth)
    context_model = ContextK(depth)
    tree_state = tree.init()
    context_state = context_model.init()
    data = b"abracadabra abracadabra"
    for byte in data:
        deepest_local = tree.explain(tree_state)[-1].local
        assert np.array_equal(deepest_local, context_model.predict(context_state))
        tree_state = tree.absorb(tree_state, byte)
        context_state = context_model.absorb(context_state, byte)


def test_incremental_activation_matches_slow_raw_history_interpreter():
    model = ActivatedSuffixTree(5)
    state = model.init()
    history = b""
    for byte in b"the quick brown fox":
        expected = [b""] + [
            history[-depth:]
            for depth in range(1, min(model.max_depth, len(history)) + 1)
        ]
        assert [row.suffix for row in model.explain(state)] == expected
        state = model.absorb(state, byte)
        history += bytes([byte])


def test_online_bits_equal_final_recursive_evidence():
    model = ActivatedSuffixTree(6, split_prior=0.37)
    state = model.init()
    online_bits = 0.0
    for byte in b"abcabcabcabxabcabc":
        online_bits -= math.log2(float(model.predict(state)[byte]))
        state = model.absorb(state, byte)
    assert model.evidence_bits(state) == pytest.approx(online_bits, abs=1e-10)
    assert model.rederived_evidence_bits(state) == pytest.approx(
        online_bits, abs=1e-10
    )


def test_split_prior_has_its_declared_direction():
    model = ActivatedSuffixTree(1, split_prior=0.8)
    state = model.init()
    state = model.absorb(state, ord("a"))
    root = state.nodes[b""]
    # Creation clones the two evidences, so the posterior equals the prior.
    assert model.explain(state)[0].stop_probability == pytest.approx(0.2)
    assert root.log_local_evidence == root.log_split_evidence


def test_depth_zero_evidence_matches_dirichlet_multinomial_closed_form():
    alpha = 0.5
    data = b"banana bandana"
    model = ActivatedSuffixTree(0, alpha=alpha)
    state = model.init()
    for byte in data:
        state = model.absorb(state, byte)

    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    log_evidence = math.lgamma(256 * alpha) - math.lgamma(
        len(data) + 256 * alpha
    )
    log_evidence += sum(
        math.lgamma(int(count) + alpha) - math.lgamma(alpha)
        for count in counts
    )
    assert model.evidence_bits(state) == pytest.approx(
        -log_evidence / math.log(2), abs=1e-10
    )
    assert model.rederived_evidence_bits(state) == pytest.approx(
        -log_evidence / math.log(2), abs=1e-10
    )


def test_only_pre_byte_active_features_are_updated():
    model = ActivatedSuffixTree(3)
    state = model.init()
    for byte in b"abacaba":
        state = model.absorb(state, byte)

    active = {row.suffix for row in model.explain(state)}
    before = {suffix: node.activations for suffix, node in state.nodes.items()}
    state = model.absorb(state, ord("x"))
    for suffix, activations in before.items():
        expected_increment = 1 if suffix in active else 0
        assert state.nodes[suffix].activations == activations + expected_increment


def test_selection_mass_is_one_predictor_per_byte():
    model = ActivatedSuffixTree(5)
    state = model.init()
    data = b"abcabcabcxyz"
    for byte in data:
        state = model.absorb(state, byte)
    assert sum(node.selection_mass for node in state.nodes.values()) == pytest.approx(
        len(data), abs=1e-10
    )
    assert sum(node.activations for node in state.nodes.values()) > len(data)


def test_replay_and_model_reuse_are_deterministic():
    model = ActivatedSuffixTree(5)
    data = b"to be or not to be"
    first = score_bits(model, data)
    second = score_bits(model, data)
    assert np.array_equal(first, second)


def test_snapshot_branches_do_not_interfere():
    model = ActivatedSuffixTree(3)
    state = model.init()
    for byte in b"prefix":
        state = model.absorb(state, byte)
    left = copy.deepcopy(state)
    right = copy.deepcopy(state)
    left = model.absorb(left, ord("a"))
    right = model.absorb(right, ord("z"))
    assert left.nodes[b""].counts[ord("a")] == 1
    assert ord("a") not in right.nodes[b""].counts
    assert right.nodes[b""].counts[ord("z")] == 1
    assert ord("z") not in left.nodes[b""].counts


def test_absorb_rejects_non_bytes():
    model = ActivatedSuffixTree()
    with pytest.raises(ValueError):
        model.absorb(model.init(), -1)
    with pytest.raises(ValueError):
        model.absorb(model.init(), 256)
