"""Reproduce the activated-suffix-tree experiments from the 2026-08-15 report."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from induce_text.activated_features import (
    ActivatedSuffixState,
    ActivatedSuffixTree,
)
from induce_text.baselines import AdaptiveIID, ContextK
from induce_text.model import score_bits
from induce_text.sources import MARKOV_P, SKEW, markov, periodic, skewed_iid

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "results" / "activated-functions"
REPORTS = ROOT / "results" / "reports"
ALPHA = 0.5


def _dirichlet_multinomial_bits(counts: dict[int, int], alpha: float) -> float:
    total = sum(counts.values())
    log_probability = math.lgamma(256 * alpha) - math.lgamma(
        total + 256 * alpha
    )
    log_probability += sum(
        math.lgamma(count + alpha) - math.lgamma(alpha)
        for count in counts.values()
    )
    return -log_probability / math.log(2)


def independent_context_bits(data: bytes, order: int, alpha: float = ALPHA) -> float:
    """Offline Dirichlet-multinomial factorization of a ContextK code."""
    grouped: dict[bytes, dict[int, int]] = defaultdict(dict)
    for position, byte in enumerate(data):
        context = data[max(0, position - order) : position] if order else b""
        counts = grouped[context]
        counts[byte] = counts.get(byte, 0) + 1
    return sum(
        _dirichlet_multinomial_bits(counts, alpha) for counts in grouped.values()
    )


def independent_activated_bits(
    data: bytes,
    max_depth: int,
    *,
    alpha: float = ALPHA,
    split_prior: float = 0.5,
) -> float:
    """Reconstruct the complete observed suffix tree from raw history.

    This implementation shares no learned state or sequential prediction code
    with ``ActivatedSuffixTree``.  KT likelihoods come from final count tables;
    the tree mixture is evaluated bottom-up with an explicit beginning-of-
    stream boundary factor.
    """
    counts: dict[bytes, dict[int, int]] = {b"": {}}
    for position in range(len(data) + 1):
        active = [b""] + [
            data[position - depth : position]
            for depth in range(1, min(max_depth, position) + 1)
        ]
        for suffix in active:
            counts.setdefault(suffix, {})
        if position < len(data):
            byte = data[position]
            for suffix in active:
                table = counts[suffix]
                table[byte] = table.get(byte, 0) + 1

    children: dict[bytes, list[bytes]] = {suffix: [] for suffix in counts}
    for suffix in counts:
        if suffix:
            children[suffix[1:]].append(suffix)

    log_stop_prior = math.log1p(-split_prior)
    log_split_prior = math.log(split_prior)
    cache: dict[bytes, float] = {}

    def weighted(suffix: bytes) -> float:
        if suffix in cache:
            return cache[suffix]
        local = -_dirichlet_multinomial_bits(counts[suffix], alpha) * math.log(2)
        if len(suffix) == max_depth:
            result = local
        else:
            boundary = (
                -math.log(256)
                if len(suffix) < len(data)
                and suffix == data[: len(suffix)]
                else 0.0
            )
            split = boundary + sum(weighted(child) for child in children[suffix])
            result = float(
                np.logaddexp(
                    log_stop_prior + local,
                    log_split_prior + split,
                )
            )
        cache[suffix] = result
        return result

    return -weighted(b"") / math.log(2)


def independent_oracle(source: str, data: bytes) -> float:
    if source == "periodic":
        return 0.0
    if source == "skewed_iid":
        return sum(-math.log2(SKEW[byte]) for byte in data)
    if source == "markov":
        state = 0
        bits = 0.0
        for byte in data:
            bits -= math.log2(MARKOV_P[state][byte])
            state = byte
        return bits
    raise ValueError(source)


def _expected_depth(stop_probabilities: list[float]) -> float:
    remaining = 1.0
    expected = 0.0
    for depth, stop in enumerate(stop_probabilities[:-1]):
        mass = remaining * stop
        expected += depth * mass
        remaining *= 1.0 - stop
    expected += (len(stop_probabilities) - 1) * remaining
    return expected


def score_activated(
    data: bytes, max_depth: int = 8
) -> tuple[float, dict[str, object], ActivatedSuffixState]:
    """Score twice: public scan, then evidence-accounted instrumented replay."""
    model = ActivatedSuffixTree(max_depth)
    harness_costs = score_bits(model, data)

    state = model.init()
    replay_costs = np.empty(len(data))
    depths = np.empty(len(data))
    for position, byte in enumerate(data):
        prediction = model.predict(state)
        replay_costs[position] = -math.log2(float(prediction[byte]))
        trace = model.explain(state)
        depths[position] = _expected_depth(
            [activation.stop_probability for activation in trace]
        )
        state = model.absorb(state, byte)

    if not np.allclose(harness_costs, replay_costs, rtol=0, atol=1e-12):
        raise AssertionError("harness and instrumented replay costs disagree")
    evidence_bits = model.evidence_bits(state)
    if not math.isclose(
        float(replay_costs.sum()), evidence_bits, rel_tol=0, abs_tol=1e-8
    ):
        raise AssertionError("sequential costs and final tree evidence disagree")
    closed_form_bits = independent_activated_bits(data, max_depth)
    if not math.isclose(
        float(replay_costs.sum()), closed_form_bits, rel_tol=0, abs_tol=1e-7
    ):
        raise AssertionError(
            "sequential costs and raw-history tree reconstruction disagree: "
            f"{replay_costs.sum()} != {closed_form_bits}"
        )
    state_rederived_bits = model.rederived_evidence_bits(state)
    if not math.isclose(
        closed_form_bits, state_rederived_bits, rel_tol=0, abs_tol=1e-7
    ):
        raise AssertionError("raw-history and state-count reconstructions disagree")
    total_selection_mass = math.fsum(
        node.selection_mass for node in state.nodes.values()
    )
    if not math.isclose(
        total_selection_mass, len(data), rel_tol=0, abs_tol=1e-7
    ):
        raise AssertionError("feature selection mass does not sum to byte count")

    trace = model.explain(state)
    remaining = 1.0
    final_path = []
    for depth, activation in enumerate(trace):
        if depth + 1 == len(trace):
            selection_mass = remaining
        else:
            selection_mass = remaining * activation.stop_probability
            remaining *= 1.0 - activation.stop_probability
        final_path.append(
            {
                "depth": depth,
                "suffix_hex": activation.suffix.hex(),
                "activations": activation.activations,
                "cumulative_selection_mass": activation.cumulative_selection_mass,
                "stop_probability": activation.stop_probability,
                "selection_mass": selection_mass,
            }
        )
    diagnostics: dict[str, object] = {
        "nodes": len(state.nodes),
        "nodes_selected_at_least_once": sum(
            node.selection_mass >= 1.0 for node in state.nodes.values()
        ),
        "nodes_selected_at_least_0_01": sum(
            node.selection_mass >= 0.01 for node in state.nodes.values()
        ),
        "selection_mass": total_selection_mass,
        "mean_effective_depth": float(depths.mean()),
        "tail_mean_effective_depth": float(depths[-min(1_000, len(depths)) :].mean()),
        "final_effective_depth": float(depths[-1]),
        "final_path": final_path,
        "evidence_bits": evidence_bits,
    }
    return float(replay_costs.sum()), diagnostics, state


def score_baseline(data: bytes, order: int) -> float:
    model = AdaptiveIID() if order == 0 else ContextK(order)
    bits = float(score_bits(model, data).sum())
    independent = independent_context_bits(data, order)
    if not math.isclose(bits, independent, rel_tol=0, abs_tol=1e-8):
        raise AssertionError(
            f"order-{order} online and offline KT codes disagree: "
            f"{bits} != {independent}"
        )
    return bits


def score_models(data: bytes) -> tuple[dict[str, float], dict[str, object]]:
    bits = {
        "iid": score_baseline(data, 0),
        "ctx1": score_baseline(data, 1),
        "ctx2": score_baseline(data, 2),
        "ctx3": score_baseline(data, 3),
    }
    tree_bits, diagnostics, _ = score_activated(data)
    bits["activated_d8"] = tree_bits
    return bits, diagnostics


def run_size_sweep() -> list[dict[str, object]]:
    generators = {
        "periodic": periodic,
        "skewed_iid": skewed_iid,
        "markov": markov,
    }
    records = []
    for source, generator in generators.items():
        seeds = [0] if source == "periodic" else list(range(5))
        for n in (1_000, 10_000, 100_000):
            for seed in seeds:
                print(f"E1 {source:>10} n={n:>7,} seed={seed}", flush=True)
                data, oracle_bits = generator(n, seed)
                recounted = independent_oracle(source, data)
                if not math.isclose(
                    oracle_bits, recounted, rel_tol=0, abs_tol=1e-9
                ):
                    raise AssertionError(f"{source} oracle recount disagrees")
                model_bits, diagnostics = score_models(data)
                for model, bits in model_bits.items():
                    record: dict[str, object] = {
                        "source": source,
                        "n": len(data),
                        "seed": seed,
                        "oracle_bpc": oracle_bits / len(data),
                        "model": model,
                        "bpc": bits / len(data),
                        "gap_bpc": (bits - oracle_bits) / len(data),
                    }
                    if model == "activated_d8":
                        record |= diagnostics
                    records.append(record)
    return records


def order2_exception(n: int, seed: int) -> tuple[bytes, float]:
    """Mostly P(1)=1/8, except P(1|00)=7/8; fixed prefix is free."""
    if n < 2:
        raise ValueError("n must be >= 2")
    rng = random.Random(seed)
    data = bytearray((0, 0))
    bits = 0.0
    while len(data) < n:
        p_one = 7 / 8 if data[-2:] == b"\x00\x00" else 1 / 8
        byte = int(rng.random() < p_one)
        bits -= math.log2(p_one if byte else 1.0 - p_one)
        data.append(byte)
    return bytes(data), bits


def recount_order2_exception(data: bytes) -> float:
    if data[:2] != b"\x00\x00":
        raise AssertionError("exception source has the wrong fixed prefix")
    bits = 0.0
    for position in range(2, len(data)):
        p_one = 7 / 8 if data[position - 2 : position] == b"\x00\x00" else 1 / 8
        byte = data[position]
        bits -= math.log2(p_one if byte else 1.0 - p_one)
    return bits


def _predict_at(
    model: ActivatedSuffixTree, state: ActivatedSuffixState, context: bytes
) -> np.ndarray:
    query = ActivatedSuffixState(nodes=state.nodes, context=context[-model.max_depth :])
    return model.predict(query)


def _js_bits(left: np.ndarray, right: np.ndarray) -> float:
    middle = 0.5 * (left + right)
    value = float(
        0.5 * np.sum(left * np.log2(left / middle))
        + 0.5 * np.sum(right * np.log2(right / middle))
    )
    return max(0.0, value)


def run_locality_probe() -> dict[str, object]:
    data, oracle_bits = order2_exception(40_000, seed=0)
    recounted = recount_order2_exception(data)
    if not math.isclose(oracle_bits, recounted, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("order-2 exception oracle recount disagrees")

    contexts = [b"\x00\x00", b"\x00\x01", b"\x01\x00", b"\x01\x01"]
    target_context = contexts[0]
    target_byte = 1

    tree = ActivatedSuffixTree(2)
    tree_state = tree.init()
    iid = AdaptiveIID()
    iid_state = iid.init()
    hard = ContextK(2)
    hard_state = hard.init()
    def probe_tree(state):
        return {context: _predict_at(tree, state, context) for context in contexts}

    def probe_iid(state):
        return {context: iid.predict(state) for context in contexts}

    def probe_hard(state):
        table, _ = state
        return {
            context: hard.predict((table, context))
            for context in contexts
        }

    def measure_controls():
        before = {
            "iid": probe_iid(iid_state),
            "context2": probe_hard(hard_state),
            "activated_d2": probe_tree(tree_state),
        }
        iid_updated = iid.absorb(copy.deepcopy(iid_state), target_byte)
        hard_table, _ = copy.deepcopy(hard_state)
        hard_updated = hard.absorb((hard_table, target_context), target_byte)
        tree_updated = copy.deepcopy(tree_state)
        tree_updated.context = target_context
        tree_updated = tree.absorb(tree_updated, target_byte)
        after = {
            "iid": probe_iid(iid_updated),
            "context2": probe_hard(hard_updated),
            "activated_d2": probe_tree(tree_updated),
        }

        controls = {}
        for name in before:
            shifts = {
                context.hex(): _js_bits(
                    before[name][context], after[name][context]
                )
                for context in contexts
            }
            other = [shifts[context.hex()] for context in contexts[1:]]
            other_mean = float(np.mean(other))
            controls[name] = {
                "target_probability_before": float(
                    before[name][target_context][target_byte]
                ),
                "target_probability_after": float(
                    after[name][target_context][target_byte]
                ),
                "target_improvement_bits": math.log2(
                    float(
                        after[name][target_context][target_byte]
                        / before[name][target_context][target_byte]
                    )
                ),
                "js_bits_by_context": shifts,
                "target_to_other_mean_js_ratio": (
                    None
                    if other_mean == 0
                    else shifts[target_context.hex()] / other_mean
                ),
            }
        tree_query = ActivatedSuffixState(
            nodes=tree_state.nodes, context=target_context
        )
        controls["activated_d2"]["stop_probabilities"] = [
            activation.stop_probability for activation in tree.explain(tree_query)
        ]
        return controls

    trajectory = []
    checkpoints = {100, 1_000, 2_000, 3_000, 5_000, 7_000, 10_000, 30_000}
    for position, byte in enumerate(data[:30_000], start=1):
        tree_state = tree.absorb(tree_state, byte)
        iid_state = iid.absorb(iid_state, byte)
        hard_state = hard.absorb(hard_state, byte)
        if position in checkpoints:
            trajectory.append(
                {"train_n": position, "controls": measure_controls()}
            )

    score, diagnostics, _ = score_activated(data, max_depth=2)
    return {
        "n": len(data),
        "oracle_bpc": oracle_bits / len(data),
        "model_bpc": score / len(data),
        "target_context_hex": target_context.hex(),
        "target_byte": target_byte,
        "trajectory": trajectory,
        "nodes": diagnostics["nodes"],
    }


def noisy_lag_copy(
    n: int, seed: int, *, lag: int = 32, flip_probability: float = 1 / 32
) -> tuple[bytes, float]:
    rng = random.Random(seed)
    data = bytearray(rng.randrange(2) for _ in range(lag))
    bits = float(lag)
    while len(data) < n:
        flip = rng.random() < flip_probability
        bits -= math.log2(flip_probability if flip else 1.0 - flip_probability)
        data.append(data[-lag] ^ int(flip))
    return bytes(data), bits


def recount_noisy_lag(
    data: bytes, *, lag: int = 32, flip_probability: float = 1 / 32
) -> float:
    bits = float(lag)
    for position in range(lag, len(data)):
        flip = data[position] != data[position - lag]
        bits -= math.log2(flip_probability if flip else 1.0 - flip_probability)
    return bits


def run_failure_probe() -> list[dict[str, object]]:
    records = []
    for seed in range(5):
        print(f"E3 noisy_lag32 n=10,000 seed={seed}", flush=True)
        data, oracle_bits = noisy_lag_copy(10_000, seed)
        recounted = recount_noisy_lag(data)
        if not math.isclose(oracle_bits, recounted, rel_tol=0, abs_tol=1e-12):
            raise AssertionError("noisy-lag oracle recount disagrees")
        model_bits, diagnostics = score_models(data)
        for model, bits in model_bits.items():
            record: dict[str, object] = {
                "source": "noisy_lag32",
                "n": len(data),
                "seed": seed,
                "oracle_bpc": oracle_bits / len(data),
                "model": model,
                "bpc": bits / len(data),
                "gap_bpc": (bits - oracle_bits) / len(data),
            }
            if model == "activated_d8":
                record |= diagnostics
            records.append(record)
    return records


def _mean_by(records, keys, value):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[key] for key in keys)].append(float(record[value]))
    return {
        key: (float(np.mean(values)), float(np.std(values)))
        for key, values in grouped.items()
    }


def plot_size_sweep(records: list[dict[str, object]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["iid", "ctx1", "ctx2", "ctx3", "activated_d8"]
    labels = {
        "iid": "IID",
        "ctx1": "context 1",
        "ctx2": "context 2",
        "ctx3": "context 3",
        "activated_d8": "activated tree (≤8)",
    }
    sources = ["periodic", "skewed_iid", "markov"]
    summary = _mean_by(records, ("source", "n", "model"), "gap_bpc")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharex=True)
    for axis, source in zip(axes, sources):
        for model in models:
            xs = sorted(
                n
                for (candidate_source, n, candidate_model) in summary
                if candidate_source == source and candidate_model == model
            )
            means = [summary[(source, n, model)][0] for n in xs]
            stds = [summary[(source, n, model)][1] for n in xs]
            axis.plot(xs, means, marker="o", label=labels[model])
            if any(stds):
                axis.fill_between(
                    xs,
                    np.asarray(means) - np.asarray(stds),
                    np.asarray(means) + np.asarray(stds),
                    alpha=0.12,
                )
        axis.set_xscale("log")
        axis.set_title(source.replace("_", " "))
        axis.set_xlabel("bytes")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("bpc above oracle")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("One hierarchy adapts its effective context to the source")
    fig.tight_layout()
    path = REPORTS / "2026-08-15-activated-functions-size-sweep.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tree_diagnostics(records: list[dict[str, object]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tree = [record for record in records if record["model"] == "activated_d8"]
    depth = _mean_by(tree, ("source", "n"), "mean_effective_depth")
    nodes = _mean_by(tree, ("source", "n"), "nodes")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    for source in ("periodic", "skewed_iid", "markov"):
        xs = sorted(n for candidate, n in depth if candidate == source)
        axes[0].plot(
            xs,
            [depth[(source, n)][0] for n in xs],
            marker="o",
            label=source.replace("_", " "),
        )
        axes[1].plot(
            xs,
            [nodes[(source, n)][0] for n in xs],
            marker="o",
            label=source.replace("_", " "),
        )
    axes[0].set_ylabel("posterior expected suffix depth")
    axes[1].set_ylabel("feature nodes")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("bytes")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Specificity is evidence-weighted; candidate storage is not")
    fig.tight_layout()
    path = REPORTS / "2026-08-15-activated-functions-depth-growth.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--locality-only",
        action="store_true",
        help="rerun E2 and replace it in an existing experiment.json",
    )
    args = parser.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = SCRATCH / "experiment.json"
    if args.locality_only:
        payload = json.loads(json_path.read_text())
        payload["locality"] = run_locality_probe()
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"updated: {json_path}")
        return
    size_sweep = run_size_sweep()
    print("E2 locality", flush=True)
    locality = run_locality_probe()
    failure = run_failure_probe()
    payload = {
        "config": {
            "max_depth": 8,
            "alpha": ALPHA,
            "split_prior": 0.5,
            "sizes": [1_000, 10_000, 100_000],
            "stochastic_seeds": list(range(5)),
        },
        "size_sweep": size_sweep,
        "locality": locality,
        "failure_probe": failure,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    figures = [plot_size_sweep(size_sweep), plot_tree_diagnostics(size_sweep)]
    print(f"results: {json_path}")
    for figure in figures:
        print(f"figure:  {figure}")


if __name__ == "__main__":
    main()
