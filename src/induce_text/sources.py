"""Synthetic data sources with a known generative cost (CLAUDE.md decision 7).

Each source returns ``(data, process_bits)`` where ``process_bits`` is the exact
number of bits the generating process spent on random choices — the sum of
``-log2 P(choice)`` over every choice made.  Deterministic structure is free.

Two-part-code honesty: ``process_bits`` is ``bits(data | process)`` only.  The
description of the process itself (the "model" half of the two-part code) is
not charged, because here the process is *known*.  An online model that has to
*learn* the process pays a learning overhead above it; watching that overhead
shrink is the game.

**The invariant, and it is not automatic.**  ``process_bits`` is
``-log2 P(choices)``.  The ideal code length is ``-log2 P(data)``, and
``P(data) = sum P(choices)`` over *every* choice sequence yielding those bytes.
So ``process_bits`` equals the ideal code length **only when the generator is
injective** — when the output determines the choices that made it.  For a
many-to-one generator it is an *upper* bound, and a good compressor can and
should beat it.  Treating it as a floor there is a category error.

The curriculum, each rung targeting a known failure mode:

===================== ============ =========================================
rung                  injective?   ``process_bits`` is
===================== ============ =========================================
``periodic``          yes (det.)   exact; zero entropy, so 0
``skewed_iid``        yes          exact; H = 1.875 bpc
``markov``            yes          exact; order-1, structure one symbol back
``long_range_copy``   **no**       upper bound only — see below
``pcfg``              yes          exact; verified unambiguous
===================== ============ =========================================

``long_range_copy`` is many-to-one: the same bytes are reachable from many
different offsets, so the offset choice carries bits the output does not
determine.  Measured at n=20000: an identical 8-byte segment has on average
3.88 distinct earlier source offsets (max 25), about 1.96 bits per copy op
that this function charges and the data does not carry — roughly 0.18 bpc,
which accounts for essentially all of the observed 0.21 bpc by which lzma
beats it.  **On this rung compare against lzma, not against process_bits.**
Recovering a true lower bound here means marginalising over parses with a
forward DP, which is not implemented.

``pcfg`` was checked by enumerating every complete episode up to 16 choice
bits (510 of them): no two distinct transcripts produce the same output, so
its transcript length is exact.

Sources generate whole steps/episodes until at least ``n`` bytes exist and
return everything generated (possibly slightly more than ``n``): truncating
mid-step would misprice the tail.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from induce_text.pcfg_gen import RecordingChoice, Rule, sample

# --- rung 1: periodic -------------------------------------------------------


def periodic(n: int, seed: int = 0, pattern: bytes = b"\x00\x01\x02\x03") -> tuple[bytes, float]:
    """Repeat ``pattern`` to at least ``n`` bytes.  No choices: process_bits = 0."""
    reps = -(-n // len(pattern))
    return pattern * reps, 0.0


# --- rung 2: skewed i.i.d. --------------------------------------------------

# Dyadic by design so the cost is hand-checkable: byte b costs exactly
# SKEW_BITS[b] bits.  H = 1/2*1 + 1/4*2 + 1/8*3 + 2*(1/16*4) = 1.875 bpc.
SKEW = {0: 1 / 2, 1: 1 / 4, 2: 1 / 8, 3: 1 / 16, 4: 1 / 16}


def skewed_iid(n: int, seed: int = 0) -> tuple[bytes, float]:
    rng = random.Random(seed)
    symbols = list(SKEW)
    weights = list(SKEW.values())
    data = rng.choices(symbols, weights=weights, k=n)
    bits = sum(-math.log2(SKEW[b]) for b in data)
    return bytes(data), bits


# --- rung 3: order-1 Markov -------------------------------------------------

# A legible 4-state chain: strongly tends to cycle 0 -> 1 -> 2 -> 3 -> 0.
# Conditional entropy = H(0.9, 0.1/3, 0.1/3, 0.1/3) ~ 0.6098 bpc, which is
# what a context model can reach and no i.i.d. model can (stationary
# distribution is uniform: i.i.d. floor = 2 bpc).
MARKOV_P: dict[int, dict[int, float]] = {
    s: {t: (0.9 if t == (s + 1) % 4 else 0.1 / 3) for t in range(4)} for s in range(4)
}


def markov(n: int, seed: int = 0) -> tuple[bytes, float]:
    rng = random.Random(seed)
    state = 0  # fixed start state: not a choice, costs nothing
    data = []
    bits = 0.0
    for _ in range(n):
        row = MARKOV_P[state]
        nxt = rng.choices(list(row), weights=list(row.values()))[0]
        bits += -math.log2(row[nxt])
        data.append(nxt)
        state = nxt
    return bytes(data), bits


# --- rung 4: long-range copy ------------------------------------------------


COPY_CONTINUE = 0.9  # copies are long (E[len] = 10) so copying is cheap


def long_range_copy(n: int, seed: int = 0) -> tuple[bytes, float]:
    """LZ77-ish process over 16 hex byte values.

    Each step (when history is non-empty): a 50:50 choice (1 bit) between
    emitting one uniform literal (4 bits) and a copy op: offset uniform over
    the history (log2(len) bits) plus a run of continue/stop choices with
    P(continue) = 0.9, each charged its true cost.  A typical copy moves
    ~10 bytes for ~16 bits — far cheaper than literals — so the compressible
    structure really is in the long-range repeats.  Copies may overlap their
    own output, LZ-style.  The very first step is a forced literal: no
    choice, no choice-bit.
    """
    rng = random.Random(seed)
    data: list[int] = []
    bits = 0.0
    while len(data) < n:
        do_copy = len(data) > 0 and rng.random() < 0.5
        if len(data) > 0:
            bits += 1.0  # the literal-vs-copy choice
        if not do_copy:
            data.append(rng.randrange(16))
            bits += 4.0
        else:
            offset = rng.randrange(1, len(data) + 1)
            bits += math.log2(len(data))
            length = 1
            while rng.random() < COPY_CONTINUE:
                length += 1
                bits += -math.log2(COPY_CONTINUE)
            bits += -math.log2(1 - COPY_CONTINUE)
            start = len(data) - offset
            for i in range(length):  # byte-by-byte so overlap repeats
                data.append(data[start + i])
    return bytes(data), bits


# --- rung 5: nested grammar (the author's PCFG generator) -------------------

# The grammar used by tests/test_pcfg_gen.py.  All choices are 50:50, so an
# episode's process_bits = its transcript length, exactly.


def pcfg_test_grammar() -> tuple[Rule, dict[str, int | Rule]]:
    rule = Rule(symbols=["a", "b", ["c", "d"], "e", "f", ["x", "a"]])
    env = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "x": rule}
    return rule, env


def pcfg_main_grammar() -> tuple[Rule, dict[str, int | Rule]]:
    """The older two-rule grammar from ``pcfg_gen.py``'s ``__main__``."""
    x = Rule(symbols=["a", "b", ["y", "d"], "e", "f", ["x", "a"]])
    y = Rule(symbols=["a", "b", ["c", "d"]])
    env = {"a": 10, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "x": x, "y": y}
    return x, env


def pcfg(n: int, seed: int = 0) -> tuple[bytes, float]:
    """Concatenated episodes from the test grammar; process_bits = total choice bits."""
    rule, env = pcfg_test_grammar()
    rng = random.Random(seed)
    data: list[int] = []
    bits = 0.0
    while len(data) < n:
        cs = RecordingChoice(seed=rng.getrandbits(32))
        data.extend(sample(rule=rule, env=env, choicesource=cs))
        bits += cs.count
    return bytes(data), bits


SOURCES: dict[str, Callable[[int, int], tuple[bytes, float]]] = {
    "periodic": periodic,
    "skewed_iid": skewed_iid,
    "markov": markov,
    "long_range_copy": long_range_copy,
    "pcfg": pcfg,
}
