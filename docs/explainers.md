# Interactive explainers

On-demand detail behind CLAUDE.md § Explaining an algorithm. Load this when
building one; ignore it otherwise.

## Why

The author learns a mechanism by driving it, not by reading about it. An
explainer that lets him step an algorithm over an input he chose beats the paper
it came from, and it is the fastest way to decide whether something is worth
adopting or porting.

This is not a report. A report answers *what happened when we ran this?* An
explainer answers *what does this thing actually do?* and usually contains no
experimental result at all.

## When to build one

- A new algorithm is a candidate to adopt, port, or discard.
- Something on `docs/reading_list.md` is about to be read — build first, read
  after. That ordering is the reading list's own discipline.
- An existing component's behaviour has become hard to reason about.

## What makes one work

**Steppable execution.** Step / play / run-to-end over a real input, with the
internal state redrawn every step. The algorithm runs; it is not described. Add
a speed control and keyboard bindings — the author will scrub back and forth.

**State visible at once.** Synchronized views of one state, not separate pages:
the structure being built, the working data, and a log of what just happened.
The insight usually lives in the correspondence between them.

**Inputs that attack it.** This is what separates teaching from demoing, and it
is the property that gets dropped under time pressure — six presets that all
work is easier to build and looks better. Choose each preset to probe a
different claim, and include at least one input the algorithm *fails* on. The
limit teaches more than the success. Always let the author type his own input.

**Honest instrumentation.** If a readout is a proxy for the thing you care
about, say so on the page at the point of use. A compression ratio that ignores
the cost of naming rules is an upper bound, not a score — the page should say
so, not just the caption.

## The bar

Sequitur, one symbol at a time:
https://claude.ai/code/artifact/219e871a-a08e-49f6-b621-3780bbb67790

Its preset list is the pattern worth copying — periodic (best case), doubling
(self-similar), Fibonacci word (structured but never periodic, which kills "it
only finds repeats"), English prose, near-repeats (cat/bat/mat, showing Sequitur
*cannot* form equivalence classes), and random (what does it build when there is
nothing to find?). Two of the six exist purely to expose a limit.
