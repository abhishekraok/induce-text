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
  Do not build what the author would enjoy building or learn from; teach the
  concept and spar instead.
- The author's Obsidian Zettelkasten notes, i.e.  mindset can be seen at ~/repos/obsidian


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


## Collaboration model (read this before writing code)
Whenever given a non trivial task, try to produce lots of visual artifacts.
The author will not have time to review all the code changes and hence relies
on experiment results and visualizations. Communication is the key bottleneck.
Try to understand the author's mindset. Try to run lots of tests, experiments 
verifications when possible. 


## Conventions

- Python via `uv`. `uv sync --extra dev`; `uv run pytest`.
- Commit changes with clear, descriptive messages (standing instruction).

