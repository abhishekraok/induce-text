# Journal

One entry per run, newest first: what was done, and a pointer to where it lives.
Descriptions and links only — numbers live in the linked report, and
`induce-text eval` regenerates the current bar. Fold old entries into a single
line as this grows; this file is auto-loaded every session, so it pays rent.

## Open doubts

Unresolved questions about the harness itself, not about any one result. These
matter more than results because the loop's trust rests on them.

- **PCFG calibration is unconfirmed.** Awaits the author's hand derivation of
  E[length] and E[bits]; the agent holds a sealed closed-form answer, not to be
  revealed before that derivation exists.

## Resolved doubts

- **2026-08-18 — the `long_range_copy` oracle was never a lower bound, and the
  scheme is sound elsewhere.** `process_bits` (formerly `oracle_bits`) is
  `-log2 P(choices)`; the ideal code length is `-log2 P(data)`, summed over
  *every* choice sequence yielding those bytes. The two agree only when the
  generator is injective. `long_range_copy` is many-to-one — the same bytes are
  reachable from many offsets — so it overcharges, and lzma beating it (1.94 vs.
  2.15 bpc at n=20000) is expected rather than a defect. Measured: an identical
  8-byte segment has on average 3.88 distinct earlier source offsets (max 25),
  ≈1.96 bits per copy op that the data does not carry, ≈0.18 bpc over ~1800
  copies — essentially the whole 0.21 bpc gap. Fix applied: renamed to
  `process_bits`, per-rung injectivity table in `sources.py`, and the run
  protocol now requires naming which reference a gap is against. The rung stays:
  it is the only one testing non-context-free long-range reachability, and it was
  the only place this was visible. A true floor there needs marginalising over
  parses with a forward DP; not implemented.
- **PCFG ambiguity is not a problem** (checked while resolving the above, and
  distinct from the open calibration question below). Enumerating every complete
  episode up to 16 choice bits — 510 of them — found no two transcripts producing
  the same output, so the grammar is unambiguous and its transcript length is
  exact.

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
