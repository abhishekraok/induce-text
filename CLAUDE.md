# induce-text — project & collaboration agreement

This file is the canonical shared context for **all** agents and tools (Claude,
Codex, etc.) working in this repo. It carries everything an agent needs that
the code itself does not say. It is kept current by proposal, not by fiat — see
§ Changing this file.

## What this is

A long-term research program: **language modelling via program synthesis**,
scored as compression on the enwik / Hutter Prize benchmark. Not a product to
ship fast: the ideas *are* the output, and the code is the lab notebook in
which the ideas are thought.

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
- The author's Obsidian Zettelkasten notes are at `~/repos/obsidian` — local
  and private, not part of this repo.

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
- **Building world model.** A world model is useful for generalization and
  compression. Hopefully a decent world model emerges out of the compression
  task. Lean into concepts like JEPA.

## Prior work (the author's, converging here)

- **[StreamPredictor](https://github.com/abhishekraok/StreamPredictor)** — hierarchical
  patterns-of-patterns text predictor. Its plateau left scars now treated as
  design constraints: never key knowledge by exact match (reachability; query
  by partial/prefix match), watch effective context length (silent bigram
  collapse), measure *calibration* not just accuracy, votes must sum not
  overwrite, and charge description length for the model (two-part code) so
  more data always helps.
- **[wandering-light](https://github.com/abhishekraok/wandering-light)** — PBE via self-play
  (induction + proposal tasks). Learnable but inexpressive; its binary
  exact-match reward is the sparsity this project's bit-signal replaces.
- **[code-map](https://github.com/abhishekraok/code-map)** — Racket image-based function library
  (function-per-file, REPL, persistence). Candidate substrate for the eventual
  library of synthesized predictor functions.

All three are checked out locally under `~/repos/`.

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

One cycle: the author spends ~1 hour on results and direction; the agent works
up to 23 hours and returns artifacts. The author reviews results, not diffs, so
artifacts must be trustworthy without a code review; step 3 below is what takes
its place. Communication is the bottleneck; produce lots of visual artifacts.

1. **Pre-register** each experiment, what you expect, and what each outcome
   would mean, before running. Drop any whose outcomes change nothing. Few
   decisive experiments: 23 hours defaults to volume, and volume is what the
   author's hour cannot absorb.
2. **Work.** Tests and probes wherever they are cheap.
3. **Self-verify.** No number is reportable without its oracle gap and an
   independent re-derivation that raises on disagreement. An agent cannot fake
   an oracle gap; that is what replaces the code review.
4. **Report** to `results/reports/<date>-<slug>.md`: surprises first, then
   numbers, artifacts, **what you did not do and where you might be fooling the
   reader**, then proposed next.
5. **Log** one entry in `JOURNAL.md` linking the report, and commit.
6. **Propose** durable learnings for this file. "Nothing durable" is the
   expected answer most days.

Templates: `docs/run_protocol.md`.

## Explaining an algorithm

When a new algorithm enters — to adopt, to port, or to decide about — build an
interactive artifact rather than a summary: steppable, internal state visible,
and at least one input the algorithm *fails* on. The author learns mechanisms by
driving them. Craft and the bar to match: `docs/explainers.md`.

## Where state lives (this file holds only invariants)

Anything that moves lives outside this file, so it can be rewritten without
touching the agreement.

- **`JOURNAL.md`** — one terse entry per run linking its report, plus open
  doubts about the harness. Descriptions and pointers, *not* results. Read
  before starting, append before finishing; entries ~3 lines, fold old ones
  into one.
- **`results/reports/`** — each run's report and the figures worth keeping.
  Tracked.
- **`results/`** otherwise — scratch, gitignored, never linked from the journal.
- **`docs/`** — design notes, handoffs, and the on-demand protocol files above.

Claude Code auto-loads the journal via the import below; other agents read it
explicitly.

@JOURNAL.md

## Changing this file

This file dictates agent behaviour, so it is the one file an agent may not
change on its own initiative.

- **Never edit it without the author's consent.**
- **Edit, then ask, then commit — in that order.** Propose by making the edit so
  he reviews a real diff, then stop. This inverts the standing rule for ordinary
  work, which commits freely.
- **At the end of every task, look for durable learnings** — anything that
  changed a settled decision, invalidated a thesis, or established a convention
  a future agent would otherwise rediscover. If none, say so; inventing one to
  look thorough is worse than silence.
- **Durable means invariant.** Results and status go in `JOURNAL.md`, which
  needs no permission. Only what is still true in six months belongs here. This
  file grows by exception; it should mostly shrink.

## Conventions

- Python via `uv`. `uv sync --extra dev`; `uv run pytest`.
- Commit with clear, descriptive messages — except this file, per § Changing
  this file.

