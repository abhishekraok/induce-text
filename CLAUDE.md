# induce-text — project & collaboration agreement

This file is the canonical shared context for **all** agents and tools (Claude,
Codex, etc.) working in this repo. It carries everything an agent needs that
the code itself does not say. Keep it current; when a decision here changes,
change it here.

## What this is

A long-term research program: **language modelling via program synthesis**,
scored as compression on the enwik / Hutter Prize benchmark. This is the
author's life task, not a product to ship fast. The ideas *are* the output; the
code is the lab notebook in which the ideas are thought.

## North star (this judges every decision below)

The goal is **poiesis, not poema** — the making, not the made thing. To create
something **beautiful, creative, innovative, powerful, efficient, and fun**, and
to play the **infinite game** (Carse): play to keep playing and keep the
creation alive and growing, not to win and stop.

The **Hutter Prize is a proxy, an instrument** — it tells us where we are, it is
never the objective. We explicitly do **not** game the metric: a specific, ugly,
useless thing that wins the prize is a failure. Prefer scoring *poorly* on Hutter
while building something elegant over the reverse.

Practical bearing on design: weigh choices by elegance, generality, and
generativity (does this open up moves or close them down?), treating "would win
more bpc" as *evidence*, not verdict. When elegance and score conflict, surface
the tension honestly — and lean elegant.

## The author

- **Fun and learning outrank speed.** This is not a professional/ASAP setting.
  The author delegates implementation, but not understanding: what he wants back
  from a run is insight, so reports must *teach* the mechanism, not just post the
  number. Argue the other side of a design choice when there is one.
- The author's Obsidian Zettelkasten notes — i.e. his mindset — are at
  `~/repos/obsidian`.

## Core theses

- **Verification.** Generating candidate programs is easy; *verifying* them is
  the bottleneck. A text corpus is a free, ungameable verifier: every position's
  true next byte labels itself and the score is bits, so verification = loss =
  MDL, all one number — and it is *graded*, not the binary pass/fail of PBE
  exact-match. Corollary, in the author's words: **idea generation is cheap,
  validation is the hard part.** Spend the effort there.
- **Expressivity vs. tractability** (the tension the author has battled for
  years). General software: maximally expressive, unlearnable. wandering-light:
  learnable, inexpressive. The attack: don't pick a point on the curve — *climb
  it under MDL*, buying expressivity only where the data pays for it in bits.
  Compression's dense signal may make expressivity learnable where sparse PBE
  reward never could.
- **World-model prediction** (mid-level idea to explore). JEPA-style: recognize
  the data into a higher-level abstraction, then predict in that space. Kept
  lossless by the two-part code, `bits(model) + bits(data | model)` — the
  abstraction doesn't replace byte prediction, it makes bytes *cheap*. Because
  compression grounds it, JEPA's representational collapse can't happen.

## Prior work (the author's, converging here)

- **StreamPredictor** (`~/repos/StreamPredictor`) — hierarchical
  patterns-of-patterns text predictor. Its plateau left scars now treated as
  design constraints: never key knowledge by exact match (reachability; query
  by partial/prefix match), watch effective context length (silent bigram
  collapse), measure *calibration* not just accuracy, votes must sum not
  overwrite, and charge description length for the model (two-part code) so
  more data always helps.
- **wandering-light** (`~/repos/wandering-light`) — PBE via self-play
  (induction + proposal tasks). Learnable but inexpressive; its binary
  exact-match reward is the sparsity this project's bit-signal replaces.
- **code-map** (`~/repos/code-map`) — Racket image-based function library
  (function-per-file, REPL, persistence). Candidate substrate for the eventual
  library of synthesized predictor functions.

## Settled design decisions

1. **A language model is a compressor.** Measure any next-byte model as
   `sum of -log2 P(actual byte)` = the bits an ideal arithmetic coder would
   emit; report **bits per byte (bpc)**. No coder needed to *measure*; the
   interface must stay coder-ready (full distribution) so one can be added.
2. **Symbol = byte** (256-way). Universal alphabet, no tokenizer to ship. Toy
   hex data is just 16 of the 256 byte values; structure-aware *features* can
   live inside the model.
3. **The model returns a full distribution** over 256 next-byte values — not
   just `p(true byte)` (that shape cannot decode). Later enrich with a "why"
   trace for credit assignment.
4. **State is explicit and threaded (JAX-style pure step, in plain Python).**
   `predict(state) -> distribution`; `absorb(state, byte) -> state'`; the
   online loop is a scan accumulating `-log2 p`. State is an incremental
   sufficient statistic, not raw history. Invariant: the prediction at t
   depends only on bytes < t, and encoder/decoder recompute identical beliefs
   by replaying identical updates. Explicit state gives snapshot/branch for
   search and is the synthesis-friendly target.
5. **Composition deferred.** A mixer of models wears the same
   predict/absorb interface (its state = children's states + weights); a
   mixer that learns its weights online *is* credit assignment. The combine
   rule (average vs product vs gating) is deliberately undecided until a
   working atom exists.
6. **Language: Python now, Racket later.** Keep every step single-unknown:
   design in the fluent language, learn Racket by porting *settled* designs.
   Racket remains the long-term home for the object language / shipped
   artifact (homoiconicity dissolves the data<->code tension felt in Python).
   Represent grammars/programs as **inert data walked by one interpreter**
   (the metacircular-evaluator pattern), never as live object graphs.
7. **Synthetic data with a known-MDL oracle before real data.** The
   generator logs the exact bits spent on each choice, so ideal compression is
   *known* and verification becomes absolute, not merely relative to gzip.
   Curriculum, each rung targeting a known failure mode: periodic -> skewed
   i.i.d. (calibration) -> order-k Markov (context) -> long-range copy
   (reachability; NOT context-free — will need an extension) -> nested
   grammar (abstraction). Anti-circularity: the predictor must not share the
   generator's primitives, and must be a generalizable basis, or hitting ideal
   MDL is trivial and tests nothing. After the synthetic rungs: small prefixes
   of enwik8 (100KB–1MB). Skip Calgary/Canterbury.


## The daily loop (read this before writing code)

One cycle: the author spends ~1 hour on results and direction; the agent then
works up to 23 hours and comes back with artifacts. **The author does not read
the diff.** Artifacts, not code, are the deliverable — so they must be
trustworthy *without* a code review. Communication is the bottleneck; produce
lots of visual artifacts.

A run:

1. **Pre-register.** Before running anything, write down each experiment, what
   you expect, and what each outcome would mean. An experiment whose outcomes
   would not change what we do next is not worth running. Prefer a few decisive
   experiments over many shallow ones — 23 hours defaults to volume, and volume
   is what the author's hour cannot absorb.
2. **Work.** Tests, probes and verification everywhere they are cheap.
3. **Self-verify.** No number is reportable without (a) its oracle gap and (b) an
   independent re-derivation that raises on disagreement. The oracles are what
   replaces the author's code review — an agent cannot fake an oracle gap, which
   is exactly why the synthetic curriculum exists.
4. **Report** to `results/reports/<date>-<slug>.md`, in this order:
   - **Surprises** — predicted vs. actual, largest divergence first. This is what
     the author's hour is for.
   - **Numbers** — bpc vs. oracle per rung, with deltas and what moved.
   - **Artifacts** — links, each figure captioned with the claim it supports.
   - **What I did not do, and where I might be fooling you** — dead ends,
     negative results, and the weakest link above. Required, not optional; write
     it even when it is dull. Silence here is this loop's failure mode.
   - **Proposed next** — ranked, with the reason each is next.
5. **Log.** Append one entry to `JOURNAL.md` linking the report, and commit.

## Where state lives (this file holds only invariants)

CLAUDE.md is for what does not change: goals, theses, settled decisions, how we
work. Anything that moves lives elsewhere, so it can be rewritten without
touching the agreement — and so nothing here goes quietly stale.

- **`JOURNAL.md`** — the running summary: standing numbers (best bpc vs. oracle
  per rung) plus one terse entry per run, each linking its full report. Read it
  before starting; append before finishing. Keep entries to ~5 lines and fold
  old ones into single lines as it grows — like a professor who carries the
  student's results and opens the write-up only when the details matter.
- **`results/reports/`** — the full report for each run, with the figures worth
  keeping. Tracked in git.
- **`results/`** (everything else) — scratch: regenerable eval output, probes,
  throwaway plots. Gitignored, and never linked from `JOURNAL.md`.
- **`docs/`** — durable design notes and handoffs, e.g.
  `docs/handoff_pcfg_track.md` (the parked PCFG track).

The journal is auto-loaded for Claude Code by the import below; other agents
should read it explicitly.

@JOURNAL.md

## Conventions

- Python via `uv`. `uv sync --extra dev`; `uv run pytest`.
- Commit changes with clear, descriptive messages (standing instruction).

