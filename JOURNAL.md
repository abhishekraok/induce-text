# Journal

Running summary of the project. Read before starting a run; append before
finishing. Standing numbers first, then one terse entry per run, newest first.
Entries stay ~5 lines; fold old ones into a single line as this grows. Link only
to tracked reports under `results/reports/`, never to scratch in `results/`.

## Standing numbers

Best model bpc vs. known oracle. Synthetic n=30k, seed 0; enwik8 first 100 KB.
Source: `induce-text eval`, 2026-08-02. No predictor beyond the baselines exists
yet, so "best" here means best *baseline* — these are the gaps to close.

| rung | oracle | best model | gap | best ref |
|---|---|---|---|---|
| periodic | 0.000 | ctx1 0.125 | +0.125 | bz2 0.013 |
| skewed_iid | 1.877 | iid 1.916 | **+0.039** | xz 2.287 |
| markov | 0.628 | ctx1 0.753 | +0.125 | xz 0.845 |
| long_range_copy | 2.192 | ctx1 3.570 | **+1.377** | xz 1.956 |
| pcfg | 0.364 | ctx2 0.577 | +0.213 | bz2 0.471 |
| enwik8:100k | — | ctx1 4.050 | — | bz2 2.510 |

Calibration (`induce-text calibrate`, 10k episodes): test grammar E[len] 11.13,
E[bits] 4.05; `__main__` grammar 13.09 / 5.04 — the latter matches the
independently recorded E[len]=13, E[bits]=5, which is what validates the harness.

Suite: 56 tests green (2026-08-15).

### Open puzzles carried forward

- **long_range_copy: gzip/xz beat the oracle** (1.96–2.02 vs. 2.192) while every
  context model loses badly (3.57+). Both halves need explaining — the second is
  the reachability failure the rung was built to expose; the first says the
  "oracle" is not a lower bound the way the other rungs' are. Resolve before
  trusting that column.
- **enwik8 shows the expressivity/tractability tension in the wild**: ctx3 (4.97)
  is *worse* than iid (4.89). More context, fewer bits, only if the model can pay
  for it.
- **PCFG calibration**: the author derives E[length] and E[bits] for the test
  grammar by hand (one-step-expansion equations). A sealed closed-form answer is
  held by the agent — do not reveal it before his derivation exists; then compare
  for three-way agreement with the empirical column.

## Entries

### 2026-08-15 — regime change: artifact-driven daily loop

Collaboration model replaced. The author no longer reviews diffs; runs are judged
by artifacts. CLAUDE.md trimmed to invariants only (goals, theses, settled
decisions, the loop); moving state moved here. Report discipline, pre-registration
and the oracle self-check rule are now in CLAUDE.md § The daily loop.
PCFG track parked the same week — `docs/handoff_pcfg_track.md`.

### 2026-08-02 — evaluation stack and viz toolkit (agent)

`model.py` (`Model` protocol + `score_bits` scan), `baselines.py`
(uniform/iid/ctx1–3, no cross-order mixing — that's the deferred composition
decision), `sources.py` (5 curriculum rungs with exact oracle bits),
`benchmark.py` + CLI `eval`/`calibrate`/`viz`. Produced the standing numbers
above. Viz: `heat`, `--vs` delta, PCFG `tree`, `calibration`, `growth` — each
re-derives what it shows and raises if it disagrees with `score_bits`.
Artifacts were regenerable scratch, not archived; rerun `induce-text eval`.

### 2026-07 — PCFG generator = decompressor (author)

`pcfg_gen.py`: grammar as inert data, one interpreter `sample(rule, env,
choicesource)`, `RecordingChoice`/`ReplayChoice`. The choice transcript *is* the
compressed form; `(grammar, transcript)` is a literal two-part code; valid
transcripts form a prefix-free code. Round-trip test pins it: replay reproduces
the output, consumes *all* bits, and truncation raises. Now parked.
