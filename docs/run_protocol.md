# Run protocol

On-demand detail behind CLAUDE.md § The daily loop, which carries the rules.
This file carries the templates.

## Pre-registration

Before running anything, write down for each experiment:

- what you will run, and on what;
- what you expect to happen, and why;
- what each possible outcome would mean.

If no outcome would change what we do next, drop the experiment. Prefer a few
decisive experiments over many shallow ones — 23 hours defaults to volume, and
volume is what the author's hour cannot absorb.

The point of writing the prediction down first is that it makes *surprise*
visible. Surprise is the signal the author reads; without a prediction on
record, a result is just a number.

## Self-verification

No number is reportable without both:

1. its gap to the rung's reference, and
2. an independent re-derivation that raises on disagreement.

The synthetic curriculum replaces the author's code review — an agent cannot
fake a gap against a known generative cost.

**Which reference.** `process_bits` is a floor only on rungs whose generator is
injective; the per-rung table in `sources.py` says which. On a many-to-one rung
(`long_range_copy`) it is an upper bound and a good compressor beats it —
compare against lzma there and say which reference you used. Quoting a
`process_bits` gap as though it were a floor on those rungs is a category
error, not a small imprecision.

## Report

One file per run: `results/reports/<date>-<slug>.md`. Sections in this order.

**Surprises.** Predicted vs. actual, largest divergence first. This is what the
author's hour is for. If nothing surprised you, say so in one line — then check
that it is not because the predictions were hedged.

**Numbers.** bpc vs. the rung's reference (named per rung), with deltas and what moved. Mark which are
new and which are carried forward.

**Artifacts.** Links. Caption each figure with the claim it supports, not with
what it depicts.

**What I did not do, and where I might be fooling you.** Dead ends, negative
results, and the weakest link in the chain above. Required, not optional; write
it even when it is dull. Silence here is this loop's failure mode — the section
exists because agents suppress negative results by default. Naming the weakest
link is not optional humility, it is the deliverable: the author has no other
way to find it.

**Proposed next.** Ranked, with the reason each is next.
