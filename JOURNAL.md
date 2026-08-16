# Journal

One entry per run, newest first: what was done, and a pointer to where it lives.
Descriptions and links only — numbers live in the linked report, and
`induce-text eval` regenerates the current bar. Fold old entries into a single
line as this grows; this file is auto-loaded every session, so it pays rent.

## Open doubts

Unresolved questions about the harness itself, not about any one result. These
matter more than results because the loop's trust rests on them.

- **The `long_range_copy` oracle may not be a lower bound.** xz compresses it
  below the "oracle" (1.96 vs. 2.19 bpc). Until that is explained, oracle gaps on
  that rung mean nothing — and the whole verification scheme leans on oracles.
- **PCFG calibration is unconfirmed.** Awaits the author's hand derivation of
  E[length] and E[bits]; the agent holds a sealed closed-form answer, not to be
  revealed before that derivation exists.

## Entries

- **2026-08-15** — activated-function first atom: a verified CTW-inspired
  suffix hierarchy establishes the nested-feature control, exposes sharp
  global-to-local gating and candidate-growth waste, and specifies the next
  prospective residual specialist →
  `results/reports/2026-08-15-activated-functions.md`.
- **2026-08-15** — regime change: artifact-driven daily loop; CLAUDE.md cut to
  invariants; agents may not edit it without consent; interactive explainers
  required when a new algorithm enters. PCFG track parked →
  `docs/handoff_pcfg_track.md`.
- **2026-08-02** — evaluation stack and viz toolkit, agent-written: `model.py`,
  `baselines.py`, `sources.py`, `benchmark.py`, `viz.py`, CLI
  `eval`/`calibrate`/`viz`. Never absorbed by the author — treat its numbers as
  provisional.
- **2026-07** — PCFG generator = decompressor, author-written: the choice
  transcript *is* the compressed form, round-trip tested. Parked; see the
  handoff.
