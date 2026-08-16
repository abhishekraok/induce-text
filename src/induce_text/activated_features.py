"""A first atom for treating activated functions as predictive data.

The feature language is deliberately tiny: every observed suffix ``s`` names
the inert predicate ``history.endswith(s)``.  At a byte boundary all suffixes
up to ``max_depth`` are active.  Each owns a multinomial KT predictor, and a
recursive evidence mixture chooses whether to stop at a general suffix or use
the next, more-specific active suffix.

This is a CTW-inspired baseline, not a general feature synthesizer.  Nesting
makes credit and suppression unambiguous; arbitrary overlapping predicates are
the next unknown.  State contains only a bounded suffix and replayable online
statistics.  No raw history or learned field lives on the model object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from induce_text.model import Distribution

ALPHABET = 256
_LOG_2 = math.log(2.0)


@dataclass
class FeatureNode:
    """Online evidence owned by one concrete suffix predicate."""

    counts: dict[int, int] = field(default_factory=dict)
    total: int = 0
    activations: int = 0
    # Expected number of predictions for which the recursive mixture selected
    # this node.  Unlike activations, this measures downstream predictive use.
    selection_mass: float = 0.0
    log_local_evidence: float = 0.0
    # None means that no more-specific child has existed yet.  On first child
    # creation this is initialized to local evidence, preserving all probability
    # mass while giving the new split model a fair prospective comparison.
    log_split_evidence: float | None = None
    # The immutable factor copied at split creation.  Keeping it separate lets
    # us re-derive split evidence offline from child counts rather than trusting
    # the recursively accumulated online predictions.
    log_split_boundary: float | None = None


@dataclass
class ActivatedSuffixState:
    """Threaded state: learned nodes plus the bounded sufficient context."""

    nodes: dict[bytes, FeatureNode]
    context: bytes


@dataclass(frozen=True)
class Activation:
    """One row of an inspectable prediction trace."""

    suffix: bytes
    activations: int
    cumulative_selection_mass: float
    stop_probability: float
    local: Distribution
    prediction: Distribution
    local_evidence_bits: float
    split_evidence_bits: float | None


class ActivatedSuffixTree:
    """Evidence-weighted hierarchy of activated suffix functions.

    For an active path ``empty, s1, ..., sd``, node ``si`` compares:

    - stop: its own KT next-byte distribution;
    - split: the recursively mixed distribution at ``s{i+1}``.

    Their posterior mixture weight comes from cumulative prequential evidence
    and ``split_prior``.  Thus a specific feature suppresses its parent only
    after saving enough bits.  Every active node still observes every outcome;
    inference-time competition is intentionally separate from train-on-errors
    residualization.
    """

    def __init__(
        self,
        max_depth: int = 8,
        *,
        alpha: float = 0.5,
        split_prior: float = 0.5,
    ):
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not 0 < split_prior < 1:
            raise ValueError("split_prior must be strictly between 0 and 1")
        self.max_depth = max_depth
        self.alpha = alpha
        self.split_prior = split_prior
        self._log_stop_prior = math.log1p(-split_prior)
        self._log_split_prior = math.log(split_prior)

    def init(self) -> ActivatedSuffixState:
        return ActivatedSuffixState(nodes={b"": FeatureNode()}, context=b"")

    def _active_suffixes(self, context: bytes) -> list[bytes]:
        depth = min(self.max_depth, len(context))
        return [b""] + [context[-d:] for d in range(1, depth + 1)]

    def _local_distribution(self, node: FeatureNode) -> Distribution:
        denominator = node.total + ALPHABET * self.alpha
        distribution = np.full(ALPHABET, self.alpha / denominator)
        for byte, count in node.counts.items():
            distribution[byte] = (count + self.alpha) / denominator
        return distribution

    def _stop_probability(self, node: FeatureNode) -> float:
        if node.log_split_evidence is None:
            return 1.0
        stop = self._log_stop_prior + node.log_local_evidence
        split = self._log_split_prior + node.log_split_evidence
        normalizer = float(np.logaddexp(stop, split))
        return math.exp(stop - normalizer)

    def _prediction_parts(
        self, state: ActivatedSuffixState
    ) -> tuple[list[bytes], list[Distribution], list[Distribution], list[float]]:
        suffixes = self._active_suffixes(state.context)
        nodes = [state.nodes[suffix] for suffix in suffixes]
        local = [self._local_distribution(node) for node in nodes]

        predictions: list[Distribution] = [np.empty(ALPHABET) for _ in nodes]
        stop_probabilities = [1.0] * len(nodes)
        predictions[-1] = local[-1]
        # Only nodes with a concrete active child can exercise their split
        # model.  This matters solely for diagnostic queries with short context;
        # on the real monotone stream, a node never loses available history.
        for index in range(len(nodes) - 2, -1, -1):
            stop_probability = self._stop_probability(nodes[index])
            stop_probabilities[index] = stop_probability
            predictions[index] = (
                stop_probability * local[index]
                + (1.0 - stop_probability) * predictions[index + 1]
            )
        return suffixes, local, predictions, stop_probabilities

    def predict(self, state: ActivatedSuffixState) -> Distribution:
        _, _, predictions, _ = self._prediction_parts(state)
        return predictions[0]

    def explain(self, state: ActivatedSuffixState) -> list[Activation]:
        """Expose all simultaneously active functions and their contribution."""
        suffixes, local, predictions, stop_probabilities = self._prediction_parts(
            state
        )
        rows = []
        for suffix, local_dist, prediction, stop_probability in zip(
            suffixes, local, predictions, stop_probabilities
        ):
            node = state.nodes[suffix]
            rows.append(
                Activation(
                    suffix=suffix,
                    activations=node.activations,
                    cumulative_selection_mass=node.selection_mass,
                    stop_probability=stop_probability,
                    local=local_dist,
                    prediction=prediction,
                    local_evidence_bits=-node.log_local_evidence / _LOG_2,
                    split_evidence_bits=(
                        None
                        if node.log_split_evidence is None
                        else -node.log_split_evidence / _LOG_2
                    ),
                )
            )
        return rows

    def _root_log_evidence(self, state: ActivatedSuffixState) -> float:
        root = state.nodes[b""]
        if root.log_split_evidence is None:
            return root.log_local_evidence
        return float(
            np.logaddexp(
                self._log_stop_prior + root.log_local_evidence,
                self._log_split_prior + root.log_split_evidence,
            )
        )

    def evidence_bits(self, state: ActivatedSuffixState) -> float:
        """Cumulative code length stored by the online evidence recurrence."""
        return -self._root_log_evidence(state) / _LOG_2

    def rederived_evidence_bits(self, state: ActivatedSuffixState) -> float:
        """Rebuild root evidence from final counts and the tree topology.

        Local KT likelihoods are recomputed with the Dirichlet-multinomial
        closed form.  Split likelihoods are then rebuilt bottom-up as their
        creation boundary times the weighted likelihood of every child.  This
        does not use ``log_local_evidence`` or ``log_split_evidence``.
        """
        children: dict[bytes, list[bytes]] = {suffix: [] for suffix in state.nodes}
        for suffix in state.nodes:
            if suffix:
                children[suffix[1:]].append(suffix)

        cache: dict[bytes, float] = {}

        def weighted(suffix: bytes) -> float:
            if suffix in cache:
                return cache[suffix]
            node = state.nodes[suffix]
            local = math.lgamma(ALPHABET * self.alpha) - math.lgamma(
                node.total + ALPHABET * self.alpha
            )
            local += sum(
                math.lgamma(count + self.alpha) - math.lgamma(self.alpha)
                for count in node.counts.values()
            )
            descendants = children[suffix]
            if not descendants:
                result = local
            else:
                if node.log_split_boundary is None:
                    raise RuntimeError("node with children has no split boundary")
                split = node.log_split_boundary + sum(
                    weighted(child) for child in descendants
                )
                result = float(
                    np.logaddexp(
                        self._log_stop_prior + local,
                        self._log_split_prior + split,
                    )
                )
            cache[suffix] = result
            return result

        return -weighted(b"") / _LOG_2

    def _ensure_next_path(self, state: ActivatedSuffixState) -> None:
        for suffix in self._active_suffixes(state.context)[1:]:
            if suffix in state.nodes:
                continue
            state.nodes[suffix] = FeatureNode()
            parent = state.nodes[suffix[1:]]
            if parent.log_split_evidence is None:
                # Before this instant only the stop model existed.  Copying its
                # evidence into the new alternative makes creation probability-
                # neutral; comparison starts with future observations.
                parent.log_split_evidence = parent.log_local_evidence
                parent.log_split_boundary = parent.log_local_evidence

    def absorb(
        self, state: ActivatedSuffixState, byte: int
    ) -> ActivatedSuffixState:
        if not 0 <= byte < ALPHABET:
            raise ValueError("byte must be in [0, 255]")

        evidence_before = self._root_log_evidence(state)
        suffixes, local, predictions, stop_probabilities = self._prediction_parts(
            state
        )
        probability = float(predictions[0][byte])

        remaining_selection = 1.0
        selection_masses = []
        for index, stop_probability in enumerate(stop_probabilities):
            if index + 1 == len(stop_probabilities):
                selection_mass = remaining_selection
            else:
                selection_mass = remaining_selection * stop_probability
                remaining_selection *= 1.0 - stop_probability
            selection_masses.append(selection_mass)

        for index, suffix in enumerate(suffixes):
            node = state.nodes[suffix]
            node.selection_mass += selection_masses[index]
            node.log_local_evidence += math.log(float(local[index][byte]))
            if index + 1 < len(suffixes):
                if node.log_split_evidence is None:
                    raise RuntimeError("active child has no split evidence")
                node.log_split_evidence += math.log(
                    float(predictions[index + 1][byte])
                )
            node.counts[byte] = node.counts.get(byte, 0) + 1
            node.total += 1
            node.activations += 1

        if self.max_depth:
            state.context = (state.context + bytes([byte]))[-self.max_depth :]
        self._ensure_next_path(state)

        # Two independently accumulated views must telescope exactly: the
        # probability just emitted and the root's recursively mixed evidence.
        evidence_after = self._root_log_evidence(state)
        expected_after = evidence_before + math.log(probability)
        if not math.isclose(
            evidence_after, expected_after, rel_tol=1e-12, abs_tol=1e-9
        ):
            raise RuntimeError(
                "sequential probability and mixture evidence disagree: "
                f"{evidence_after=} {expected_after=}"
            )
        return state
