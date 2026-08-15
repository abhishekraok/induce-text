# Journal

One entry per run, newest first: what was done, and a pointer to where it lives.
Descriptions and links only — full results stay in the linked report. Fold old
entries into a single line as this grows; this file is auto-loaded every session,
so it pays context rent.

## Standing numbers

The bar a new predictor has to clear. Regenerate with `induce-text eval`.
Synthetic n=30k seed 0, enwik8 first 100 KB; measured 2026-08-02.

| rung | oracle | best baseline | gap |
|---|---|---|---|
| periodic | 0.000 | ctx1 0.125 | +0.125 |
| skewed_iid | 1.877 | iid 1.916 | +0.039 |
| markov | 0.628 | ctx1 0.753 | +0.125 |
| long_range_copy | 2.192 | ctx1 3.570 | +1.377 |
| pcfg | 0.364 | ctx2 0.577 | +0.213 |
| enwik8:100k | — | ctx1 4.050 | — |

Open: xz beats the *oracle* on long_range_copy (1.96 vs. 2.19) — that column is
not a lower bound until this is explained. On enwik8, ctx3 (4.97) is worse than
iid (4.89). PCFG calibration awaits the author's hand derivation; the agent holds
a sealed closed-form answer, not to be revealed before it exists.

## Entries

- **2026-08-15** — regime change: artifact-driven daily loop; CLAUDE.md cut to
  invariants; agents may not edit it without consent. PCFG track parked →
  `docs/handoff_pcfg_track.md`.
- **2026-08-02** — evaluation stack and viz toolkit, agent-written: `model.py`,
  `baselines.py`, `sources.py`, `benchmark.py`, `viz.py`, CLI
  `eval`/`calibrate`/`viz`. Produced the numbers above. 56 tests.
- **2026-07** — PCFG generator = decompressor, author-written: the choice
  transcript *is* the compressed form, round-trip tested. Parked; see the
  handoff.
