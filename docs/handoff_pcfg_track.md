# Handoff: PCFG track parked (August 2026)

The author is exploring other approaches to the same problem (language
modelling via program synthesis). Nothing here is broken; this is a clean
pause. These notes are what a future session needs to resume the track.

## What stays live regardless of approach

The evaluation stack is **approach-agnostic** and remains the benchmark for
whatever comes next:

- `model.py` — `Model` protocol + `score_bits` scan (settled decisions 1–4).
- `baselines.py` — uniform / iid / ctx1..3: the opponents to beat.
- `sources.py` — 5 curriculum rungs with exact oracle bits.
- `benchmark.py`, CLI `eval` / `calibrate` / `viz`. 56 tests green.

Any new predictor wearing `predict`/`absorb` plugs in and gets an honest
bpc-vs-oracle number immediately.

## What is parked

- `pcfg_gen.py` (author-written, round-trip tested; transcript = compressed
  form; grammar+transcript = literal two-part code).
- The grammar-aware predictor design (was open item: close 0.58 → 0.36 on
  the pcfg rung).
- `docs/learning_transition_table.md` sketch and its sparring conclusions
  (inside algorithm; one joint inference over partial parses).

## Unfinished business on this track

1. **Calibration win condition — not done.** Hand-derive E[length], E[bits]
   for the test grammar via one-step expansion. Empirics (10k episodes):
   test grammar 11.13 / 4.05; `__main__` grammar 13.09 / 5.04 — the latter
   matches the previously recorded 13 / 5, so the harness itself is
   validated. The agent holds a sealed closed-form answer for the test
   grammar; ask for it only after deriving.
2. **Sprint code never read to fluency.** `model.py` first (every reported
   number flows through it), then `baselines.py` / `sources.py`. Harness
   numbers carry provisional trust until then. Unratified: `absorb` may
   mutate its argument.
3. **Copy rung failure stands.** All context models ~3.6 bpc vs oracle 2.19
   while gzip/xz ≈ 2.0 — long-range reachability needs a mechanism no
   fixed-context model has.
4. **Visualizations did not land.** The viz toolkit (heat/tree/delta/
   calibration/growth) is built and self-checking, but the author looked and
   did not feel he understood it. Before building more views: walk through
   one view together and check comprehension; the gallery-dump format
   failed.

## Re-entry path

`uv run pytest` → `uv run induce-text eval` → read `model.py` → do the
derivation. Then the predictor design, starting from the transition-table
sparring outcomes.
