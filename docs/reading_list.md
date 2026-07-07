# Reading list

Curated, not exhaustive — a long list would betray the north star. Organized by
thread. Status tags reflect the author's history where known.

**Reading discipline:** don't let this become read-first. The project lives or
dies on building; papers/books pay off most *after* the questions have been felt
in code (Sequitur reads best *after* the parse blowup; DreamCoder *after* feature
induction hurts). The exception is anything that *refreshes* rather than
*anchors* — MacKay is safe to enjoy anytime.

---

## Expressivity vs. tractability in learning (the central tension)

The tension has two distinct formal homes:

- **Kearns & Vazirani, _An Introduction to Computational Learning Theory_.**
  The PAC canon: sample & computational complexity as a function of
  hypothesis-class richness; VC dimension; the rigorous bias-variance tradeoff.
  Where "richer class = harder to learn" is made precise. Dated and dry, but
  foundational.
- **Luc De Raedt, _Logical and Relational Learning_ (Springer, 2008).** The
  ILP textbook. Closer to this project because it's about learning
  *programs/logic*, not fitting functions. Key concept: **language bias** —
  the field's name for exactly our dial (you *choose* the hypothesis
  language's expressivity, and that choice is the whole game); also
  refinement operators (search moves through hypothesis space) and
  generality ordering. The body of work most directly about "climb the
  expressivity curve under MDL," and the one the author is least likely to
  have absorbed from mainstream ML. Free shorter entry point: **Muggleton &
  De Raedt, "Inductive Logic Programming: Theory and Methods" (J. Logic
  Programming, 1994)** — the classic survey, where declarative/language bias
  is laid out.

## MDL / compression / algorithmic information (the theoretical bedrock)

- **Li & Vitányi, _An Introduction to Kolmogorov Complexity and Its
  Applications_.** *The* foundations book: Solomonoff induction, algorithmic
  information, why "shortest program = best explanation," and why it's
  uncomputable (hence we approximate with grammars + MDL). Live in chapters 4–5;
  it's the theory *under* the north star. If only one book, this.
- **Grünwald, _The Minimum Description Length Principle_.** The practical
  counterpart: two-part codes, model-cost-vs-data-cost accounting (baked into
  this project from line one), refined MDL, the Bayesian model-selection
  connection. The rigorous version of the bit-accounting done by hand in
  `pcfg_gen.py`. Read alongside Li & Vitányi: one gives the ideal, the other the
  achievable.
- **MacKay, _Information Theory, Inference, and Learning Algorithms_.**
  _[author: partially read — a favorite textbook]_ The bridge from EE
  information theory to ML, and a pleasure to read (free online). Arithmetic
  coding, inference, the compression–prediction identity. The best place to
  knock the rust off the coding theory and re-see it through a learning lens.

## Program synthesis / grammar induction

No single canonical *book*; the field lives in papers (see below). Closest:

- **Gulwani, Polozov & Singh, _Program Synthesis_** (short, free monograph). A
  solid landscape survey, though more the search/SAT/deductive tradition than
  this project's learning-under-compression angle.
- **Jurafsky & Martin, _Speech and Language Processing_** (free online). Cleanest
  textbook treatment of PCFGs, CYK, and inside–outside — the machinery the
  transition-table sketch re-derived. Reference it, don't read it through.

## Philosophical frame

- **Carse, _Finite and Infinite Games_.** _[author: read]_ The source of
  "poiesis not poema" / the infinite game.
- **Hofstadter, _Gödel, Escher, Bach_.** _[author: half-read ~20 years ago —
  worth revisiting]_ Self-reference, hierarchy, meaning-from-structure — the
  aesthetic under the recognize-then-predict and metacircular-interpreter
  threads. Not technical, but the right spirit for an infinite-game project.

## Papers (deliberately *after* building; vocabulary, not solutions)

- **Sequitur** (Nevill-Manning & Witten) — linear-time grammar induction by
  replacing repeated digrams; almost literally the "frequent pair -> new rule"
  idea, and it's a compressor. Read first (skimmable, constructive).
- **Stolcke 1995** — efficient probabilistic CFG parsing that computes prefix
  probabilities; the tractable version of "P(next symbol | prefix)" the
  transition-table sketch reaches for.
- **Inside–outside** (Baker; Lari & Young) — the sum-over-parses + EM learning
  for PCFGs.
- **DreamCoder** (Ellis et al.) — grows a DSL/library under a description-length
  objective (wake-sleep). Closest existing system to the vision; induce-text
  differs (compression/LM target, byte-level, online-adaptive, no LLM at
  inference).

---

## Suggested sequence

Given deep ML + rusty-but-real info theory + learning Racket, the path that
compounds fastest:

1. **MacKay** — refresh coding theory; see compression=inference done
   beautifully. Safe to start now (refreshes, doesn't anchor).
2. **Grünwald (MDL)** — make the two-part accounting rigorous, when moving from
   the toy oracle to real model selection.
3. **Li & Vitányi** — the deepest "why," in chunks; ongoing reference.
4. **De Raedt / ILP** — the one most directly on the years-long tension, and the
   least likely already absorbed.

Revisit **GEB** in parallel for spirit; the papers slot in as their questions
arise in code.
