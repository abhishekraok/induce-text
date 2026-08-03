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

- ML researcher/engineer since 2015 (Microsoft, Google DeepMind; trained PaLM
  and Gemini). Expert in Python; learning Racket (long-term goal: master it).
- EE undergrad (~20 years ago); information theory & coding was his favorite
  subject. Strong *theory* (entropy, surprisal, source coding) but no hands-on
  compression practice, and some of it is rusty. Anchor compression talk in
  `-log2 p` / bits; explain coder machinery (arithmetic coding, PPM, mixing)
  from intuition rather than assuming familiarity.
- **Fun and learning outrank speed.** This is not a professional/ASAP setting.
  Do not build what the author would enjoy building or learn from; teach the
  concept and spar instead.

## Core theses

- **Verification thesis (why this project can work).** Generating candidate
  programs is easy; *verification* is the bottleneck. A text corpus is a
  near-infinite, free, ungameable verifier: every position's true next byte
  labels itself, and the score is bits. Verification = scoring = loss = MDL,
  all one number. The signal is *graded* (bits), not binary (pass/fail like
  PBE exact-match) — denser signal, smoother search landscape.
- **Expressivity vs. tractability (the tension the author has battled for
  years).** General software engineering: maximally expressive, unlearnable.
  wandering-light: learnable, poorly expressive. The attack here: grammars are
  a genuine sweet spot (context-free = real hierarchy + polynomial inference);
  don't pick a fixed point on the curve — *climb it under MDL*, adding
  expressivity only where the data pays for it in bits; and compression's dense
  signal may make expressivity more learnable than sparse PBE reward ever could.
- **World-model prediction (mid-level idea to explore).** JEPA-style: first
  *recognize* the data into a higher-level abstraction, then predict in that
  abstract space. Reconciled with losslessness via the residual two-part code:
  `bits(model) + bits(data | model)` — the abstraction doesn't replace byte
  prediction, it makes bytes *cheap*. Bonus: lossless compression grounds the
  abstraction, so JEPA's representational collapse can't happen.

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

## Current state (August 2026)

- **Infra** (agent-drafted, reviewed): `data.py` (enwik8/9 download + slice),
  `cli.py` (`download`), `tests/test_data.py`. An earlier baseline-model
  scaffold was deliberately deleted (`56334c0`) so the author writes the core
  from a blank file; it survives only in git history, unpeeked.
- **`src/induce_text/pcfg_gen.py`** (author-written, in progress): a binary
  dyadic PCFG **generator = decompressor**. Grammar as data: `Rule` = list of
  symbols, each a name or a 2-list `[a, b]` meaning a 50:50 one-bit choice;
  `env: dict[str, int | Rule]` maps names to terminals or rules — names +
  environment lookup give recursion (self-reference forces exactly this).
  One interpreter `sample(rule, env, choicesource)`; `ChoiceSource` protocol
  with `RecordingChoice` (rng + transcript) and `ReplayChoice` (replays a
  given bitstring). Key facts: the choice transcript *is* the compressed form
  of the output; `(grammar, transcript)` is a literal two-part code; valid
  complete transcripts form a prefix-free code (Kraft equality <-> a.s.
  termination); bits live at *choice points*, so deterministic structure is
  free.
- **`docs/learning_transition_table.md`** (author's idea sketch): recognize
  raw symbols into higher-level features, learn a transition table over them,
  predict by summing over derivations. Sparring outcomes: this re-derives
  PCFG prediction; the derivation-sum is the inside algorithm and the
  intractability is solved by dynamic programming; recognition and prediction
  should be **one joint inference over a distribution of partial parses**,
  not two sequential steps (hard segmentation can't be right — ambiguity is
  future-dependent); softmax-over-counts is a placeholder (wrong calibration
  paradigm; use normalized smoothed counts). The "fast inference" addendum is
  amortized inference / chunking (System 1/2) — sound, deferred, must sit
  under the same MDL accounting.
- **Done (July 2026):** round-trip test (`tests/test_pcfg_gen.py`) — record a
  transcript, replay it, assert identical output, *all* bits consumed
  (leftover bits = non-unique codes), and truncated transcript raises.
  Together these pin the prefix-code property: a valid transcript is consumed
  exactly. (Kraft connection explained — Kraft sum = P(termination), equality
  <-> a.s. termination — but parked by the author for now.)
- **Done (Aug 2026 credits sprint, agent-written — see the collaboration
  amendment below):** the evaluation stack, end to end. `model.py` — the
  `Model` protocol (`init`/`predict`/`absorb`, full 256-way distribution,
  explicit threaded state) plus the scoring scan `score_bits` (settled
  decisions 1–4 in code; note: `absorb` may mutate its argument — the
  signature stays pure-shaped, purity is to ratify at rewrite).
  `baselines.py` — `Uniform` (exactly 8 bpc; harness self-test),
  `AdaptiveIID` (KT add-half counts), `ContextK(k)` (per-context counts,
  additive smoothing; deliberately NO cross-order mixing — that is the
  deferred composition decision). `sources.py` — the full curriculum with
  exact oracle bits: `periodic`, `skewed_iid` (dyadic, H = 1.875),
  `markov` (4-state cycle chain, H ~ 0.61), `long_range_copy` (LZ-ish,
  long cheap copies), `pcfg` (the author's generator as rung 5; oracle =
  transcript length since all choices are dyadic). `benchmark.py` + CLI
  `eval` (matrix -> table / JSON / learning-curve plots, gzip/bz2/xz
  reference columns; outputs in gitignored `results/`) and `calibrate`
  (empirical episode means). 30 tests pass.
- **First results** (n=30k synthetic / 100KB enwik8, seed 0): `iid` sits on
  the skewed_iid oracle (1.92 vs 1.88, beating gzip); `ctx1` approaches the
  markov oracle (0.75 vs 0.63); **every context model fails
  long_range_copy** (3.6+ vs oracle 2.19) while gzip/xz (~2.0) beat even
  the oracle — the reachability rung empirically demands the copy
  extension; pcfg: best baseline `ctx2` = 0.58 vs oracle 0.36 — that gap
  is what grammar-awareness must earn; enwik8-100KB: `ctx1` 4.05, `ctx3`
  4.97 — *worse than* `iid` 4.89 (the expressivity/tractability tension in
  the wild), gzip 2.90.
- **Open next steps** (author's): (1) **calibration win condition** — hand-
  derive expected output length and bits via one-step-expansion equations.
  The empirical column now exists (`induce-text calibrate`, 10k episodes):
  test grammar 11.13 / 4.05; `__main__` grammar 13.09 / 5.04 — the latter
  matches the agent's previously recorded E[length]=13, E[bits]=5, which
  validates the harness. The agent holds a sealed closed-form answer for
  the test grammar; derivation first, then compare, for the three-way
  agreement. (2) **absorb the sprint code** — read `model.py` first (the
  scan every reported number flows through), then `baselines.py` /
  `sources.py`; rewrite whatever surprises. (3) predictor design
  (author-written core): close the oracle gaps the baselines leave open,
  starting with the pcfg rung. (4) feature learning (deferred: transition
  table doubles as a feature proposer — high-weight transitions mint new
  features, i.e. grammar induction a la Sequitur); (5) a possible Racket
  port of the settled generator as a learning exercise.
- **Reading queue** (vocabulary, deliberately *after* building): Sequitur
  (Nevill-Manning & Witten) first; Stolcke 1995 (prefix probabilities);
  inside-outside; DreamCoder (Ellis et al., library learning under MDL). Full
  curated list (books + papers, by thread) in `docs/reading_list.md`.

## Collaboration model (read this before writing code)

The author retains deep, inside-out understanding of the project. The axis for
deciding who writes what is **not** "core vs. auxiliary" — it is
**"will the author need to reason about this from intuition later, or just call
it?"**

**The test of understanding is surprise.** If the author reads a diff and is
surprised, or a failure can't be localized without rerunning everything, grasp
has been lost. Predicting behavior before running and localizing a bug from a
symptom *is* grasp — independent of who typed the code.

Three tiers:

- **Author writes by hand** — anything embodying a research hypothesis (feature
  / cascade representation, MDL two-part accounting, the inference primitive,
  the search / synthesis loop), the central data structures, and **all metrics
  and evaluation** (numbers must be trusted in the bones). The author should be
  able to reimplement these from a blank file.
- **Agent drafts, author reviews to fluency** — components with a clear,
  verifiable contract: data loaders, CLI, plotting, test scaffolds, baseline
  compressors. Read until nothing surprises you, then own it.
- **Agent writes freely** — throwaway probes, one-shot scripts, boilerplate.

**Amendment (Aug 2026, author-authorized):** the tiers are not sacrosanct.
During the August credits sprint the agent wrote the evaluation stack
end-to-end — including the scoring scan the tiers reserve for the author.
What survives is the *purpose* behind the rule, not the rule: never land in
the local optimum where something load-bearing exists that the author does
not understand. The author will read the crucial parts to fluency (the
scoring scan first), rewriting wherever surprised; until then the harness
numbers carry provisional trust.

## How the agent should behave here

- **Be suspicious of your own velocity.** Clean, finished-looking code papers
  over conceptual gaps. Optimize for the author's understanding-per-week, not
  code-per-week. When these conflict, slow down.
- **Default to sparring partner, not typist, on anything core.** Argue the other
  side of a design choice; generate and critique alternative representations;
  write adversarial tests that attack assumptions; make the author explain the
  mechanism until the explanation reveals the flaw. The code is a byproduct.
- **When you write something core, make the author own it** — defend every
  nontrivial choice, or suggest they rewrite it from scratch without looking.
- **Flag the boundary.** When work crosses from infra into a research-core
  component, say so and shift into review/mentor mode rather than just
  producing the code.

## Conventions

- Python via `uv` (3.12). `uv sync --extra dev`; `uv run pytest`.
- Commit changes with clear, descriptive messages (standing instruction).

## Planned

- The author's Obsidian Zettelkasten notes will eventually migrate into this
  repo (likely under `docs/`) so all the project's ideas live organized in one
  place alongside the code.
