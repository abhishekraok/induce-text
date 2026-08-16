# Activated functions as data points: first atom

## Surprises

1. **Locality snapped; it did not interpolate.** I expected the evidence mixture
   to sit between a global update and a hard context-2 update. On the order-2
   exception source it was exactly global through 2k bytes, then essentially
   hard-local by 3k. A CTW gate chooses structure; it is not the smooth local
   plasticity mechanism sought in the updated LNN note.
2. **The combiner's regret was simpler than expected.** Against the matching
   fixed KT model, the tree paid exactly 1 total bit on IID data and 5 total
   bits on order-1 Markov data, at every tested size. Its bpc excess shrinks
   only because these structural prior costs are constant. There is little
   reason to tune the nested combiner further.
3. **Prediction was sparse; proposal was not.** At 100k skewed-IID bytes, the
   posterior used only the root for at least one byte-equivalent, yet the eager
   suffix proposal rule materialized 71,699 nodes. Suppression solved scoring,
   not memory or compute.
4. **Deterministic data did not identify a unique explanation.** The period-4
   stream retained posterior mean depth 1.41 because several deeper suffixes
   make equivalent deterministic predictions. Compression can select an
   equivalence class without recovering the smallest program unless the model
   prior distinguishes them more strongly.
5. **The updated LNN note changes the label, not the result.** This model keeps
   feature identities immutable, but every active KT predictor learns on every
   byte. It is therefore a nested activated-feature control, not yet a faithful
   add-only LNN or gated local-update algorithm.

The explicit basis failure was not a surprise: on noisy lag-32 data the tree
collapsed to IID, exactly as pre-registered. That clean failure is useful.

## Numbers

The implemented prediction at suffix `s` is

`q_s = P(stop | evidence) KT_s + P(split | evidence) q_child`.

Every suffix predicate `history.endswith(s)` activates and updates; only its
posterior *use* is suppressed. `P(split)=1/2` was fixed before the run. All
symbols are bytes and every prediction is a positive 256-way distribution.

E1 results below are mean bpc over five paired seeds for stochastic sources;
periodic is deterministic. “Fixed” is IID for skewed IID and context-1 for the
other two. Oracle is re-priced from the realized bytes.

| Source | Bytes | Oracle | Activated ≤8 | Gap | Matching fixed | Tree − fixed, total bits |
|---|---:|---:|---:|---:|---:|---:|
| Periodic | 1k | 0 | 1.401293 | 1.401293 | 1.398605 | 2.688 |
| Periodic | 10k | 0 | 0.295246 | 0.295246 | 0.294977 | 2.692 |
| Periodic | 100k | 0 | 0.046301 | 0.046301 | 0.046274 | 2.693 |
| Skewed IID | 1k | 1.894200 | 2.464434 | 0.570234 | 2.463434 | 1.000 |
| Skewed IID | 10k | 1.872460 | 1.970846 | 0.098386 | 1.970746 | 1.000 |
| Skewed IID | 100k | 1.874908 | 1.888968 | 0.014060 | 1.888958 | 1.000 |
| Markov-1 | 1k | 0.623688 | 2.013139 | 1.389451 | 2.008139 | 5.000 |
| Markov-1 | 10k | 0.628633 | 0.922741 | 0.294108 | 0.922241 | 5.000 |
| Markov-1 | 100k | 0.627787 | 0.673972 | 0.046185 | 0.673922 | 5.000 |

At 100k, stochastic SD of tree bpc was 0.00260 (IID) and 0.00393
(Markov). The posterior/storage split was:

| Source | Materialized nodes | Nodes with ≥1 byte-equivalent use | Nodes with ≥0.01 use | Tail expected depth |
|---|---:|---:|---:|---:|
| Periodic | 33 | 24 | 28 | 1.413 |
| Skewed IID | 71,699 | 1 | 10.8 | 0 |
| Markov-1 | 6,512 | 5 | 17.6 | 1 |

E2 used a source with `P(1|00)=7/8` and `P(1|other)=1/8`. At 2k
training bytes, the suffix-0 node already preferred splitting
(`P(stop)≈2.6e-45`) but the root still stopped with probability 1, so the
observable update remained global. At 3k, root `P(stop)≈4.7e-47` and the tree
became exactly context-2-local. After 30k bytes, one `00 → 1` update improved
the target log score by `2.42e-5` bits; target JS movement was `3.13e-10` bits
and movement at `01`, `10`, and `11` was zero. The full order-2 run scored
0.627855 bpc versus a 0.549520 oracle.

E3 noisy lag-32 (five seeds, 10k bytes): oracle 0.203032 bpc; activated tree
1.095779 ± 0.00549; gap 0.892748. IID scored 1.095679, exactly one total bit
better. The tree created about 511 candidates and assigned ≥1 byte-equivalent
of use only to the root.

Reportability checks raised on disagreement:

- public scoring scan versus an instrumented replay, per byte;
- online cumulative score versus final recursive evidence;
- an independent raw-history reconstruction using closed-form KT likelihoods
  and a bottom-up context tree;
- a second reconstruction from final state counts;
- each baseline versus its offline Dirichlet-multinomial likelihood;
- every source oracle versus an independent realized-data walker.

Tolerances were `1e-8` bits for online evidence and `1e-7` bits for the fully
independent tree reconstruction. Selection mass also summed to exactly one
predictor per byte within `1e-7`.

## Artifacts

- [Size sweep](2026-08-15-activated-functions-size-sweep.png) — one hierarchy
  reaches the appropriate fixed-order code, with a constant structural cost.
- [Depth and node growth](2026-08-15-activated-functions-depth-growth.png) —
  evidence controls effective depth but not eager candidate allocation.
- [Model](../../src/induce_text/activated_features.py),
  [tests](../../tests/test_activated_features.py), and
  [reproduction script](../../scripts/run_activated_functions.py).
- A steppable conversation explainer exposes active functions, local and mixed
  distributions, stop/use weights, updates, and the noisy-lag failure preset.

## What I did not do, and where I might be fooling you

- **Weakest link:** the run does not test the note's central novelty—arbitrary
  learned function identities used as symbolic inputs. Suffix predicates are
  nested, so their credit problem reduces to a generalized context tree.
- I did not synthesize features, charge a transmitted program description, run
  enwik, or benchmark wall time/RAM. Node count is only a warning proxy.
- Deterministic first-occurrence suffix creation makes the lazy tree equivalent
  to a static 256-ary CTW-style recurrence. A function proposed later is only a
  prospective/sleeping expert; copying old evidence cannot give it hindsight.
- `P(split)=1/2` is a valid fixed prior, not a uniquely canonical byte-tree
  prior. I did not tune it after seeing results.
- E2 is a one-seed diagnostic on prompted two-byte contexts, not a naturally
  occurring stream average. Its sharp transition demonstrates structural
  selection, not desirable local learning.
- “Selected” is posterior byte-equivalent mass, not a sampled discrete routing
  decision. Tiny transient mass can spread over more nodes than the ≥1 count.
- The current tree updates all active predictors. It tests inference-time
  competition only; learning-time residualization remains untested.
- Nothing here is algorithmically novel in the suffix-only case. The relevant
  prior art is [context-tree weighting](https://pure.tue.nl/ws/portalfiles/portal/1383848/Metis122608.pdf)
  and [prediction suffix trees](https://arxiv.org/abs/cmp-lg/9607016).

## Proposed next

1. **Implement one prospective residual specialist, faithful to the updated
   LNN note.** On a high-log-loss event, propose an inert correction predicate
   slightly more specific than its parent. It cannot repair the byte already
   encoded. On future activations, maintain:
   `support`, prospective bit gain `Σ log2(q_candidate(y)/p_parent(y))`, and
   posterior output use as three separate quantities. Promote only when gain
   exceeds feature-description cost plus a safety margin. Initialize promotion
   as a no-op; freeze the old chain and update only the new specialist. Include
   nearby counterexamples so a one-example exception cannot become a broad
   complement rule.
2. **Ablate the two ideas that the notes distinguish.** Compare (a)
   inference-time parent/specialist competition, (b) train-only-on-residual
   updates, and (c) both. E2 should report target log-score gain and collateral
   JS separately. A global proximal log-loss update is the disturbance control;
   the present tree and hard ContextK are structural controls.
3. **Only then allow overlapping functions.** Nested predicates admit the CTW
   recurrence; arbitrary active functions need an additive-logit or
   [specialist-expert](https://www.schapire.net/papers/FreundScSiWa97.pdf)
   combiner. [Maximum-entropy feature induction](https://aclanthology.org/J96-1002.pdf)
   is the closest direct precedent for adding a correction under log loss;
   [Passive-Aggressive updates](https://www.jmlr.org/papers/volume7/crammer06a/crammer06a.pdf)
   and [AdaGrad](https://www.jmlr.org/papers/v12/duchi11a.html) supply useful
   minimum-change/rare-feature controls, not the objective by themselves.
4. **Make proposal sparse before broadening the language.** Keep candidates in
   shadow state, prune when their upper bound on future bit gain cannot repay
   description cost, and measure peak bytes and update time. The observed
   71,699-to-1 proposal/use ratio is the next concrete efficiency target.
5. **Defer consolidation, but preserve the seam.** Add-only learning cannot
   retrospectively discover the shared basis that hindsight would choose. The
   updated note is right to reserve a later consolidation/replay phase; it
   should not be conflated with online local correction now.

The literature narrows the claim: suffix activation plus suppression is the
CTW/PST family; frozen additive units also resemble
[cascade-correlation](https://www.manoonpong.com/EmbodiedAILecture/CH3/reading%20materials/The%20Cascade-Correlation%20learning%20architecture.pdf).
The research opening is the combination of program-valued feature identity,
prospective MDL verification, and local residual specialists—not the nested
mixture itself.

## Pre-registration (written before run)

### Model under test

Use the observed suffix `s` as the inert identity of the activated predicate
`history.endswith(s)`. Every active suffix owns a 256-way KT predictor. A
recursive evidence mixture chooses between the current node and its more
specific active child. Child candidates are created deterministically from
past bytes; evidence, not one classification error, decides whether they
suppress the parent. Maximum suffix depth is 8.

This deliberately restricts the first feature language to nested predicates.
It tests activation, prediction, stability, and specificity without also
making overlapping-feature credit assignment or synthesis unknowns.

### E0 — internal identities

- Run: depth 0 against `AdaptiveIID`, online score against final mixture
  evidence, and optimized activation against a slow raw-history interpreter.
- Expect: equality to floating-point tolerance, valid 256-way distributions,
  deterministic replay, and no lookahead.
- Meaning: any disagreement invalidates every later number; stop and fix it.

### E1 — specificity without choosing an order

- Run: periodic, skewed IID, and order-1 Markov at 1k, 10k, and 100k bytes;
  five seeds for stochastic sources. Compare IID, fixed contexts 1–3, and the
  depth-8 activated tree. Report bpc, oracle gap, and node count.
- Expect: the tree approaches context-1 on periodic/Markov and IID on skewed
  data, while avoiding the cold-start failure of an unnecessarily deep fixed
  context. Its excess over the best hindsight order should shrink with data.
- Meaning: a persistent large excess rejects this combiner; a small shrinking
  excess licenses arbitrary feature families as the next unknown.

### E2 — local correction versus collateral disturbance

- Run: an order-2 source whose `00` context is an exception to an order-0
  tendency. After training, apply one `00 -> 1` update to a copied state and
  measure the true-byte improvement at `00` versus Jensen–Shannon disturbance
  on disjoint two-byte contexts.
- Expect: target improvement exceeds mean disjoint disturbance, but is not
  perfectly local because root and suffix-1 evidence also update.
- Meaning: similar target and collateral changes falsify the minimum-
  disturbance story; exact zero collateral would reveal an accidentally hard
  gate rather than evidence mixing.

### E3 — explicit basis failure

Pre-execution amendment: pure lag-32 copy becomes a period-32 stream, so an
order-8 model may infer phase from a locally unique suffix. That would not test
reachability. Instead, use 32 fair seed bits followed by lag-32 copy with an
independent 1/32 flip probability; recount flips from the output to verify the
oracle.

- Run: noisy lag-32 copy at 10k bytes. Compare the same models.
- Expect: the depth-8 tree remains near 1 bpc, far above the roughly 0.20 bpc
  oracle, because no active function can see the copied byte.
- Meaning: a small gap would indicate leakage or an unintended short-context
  cue; failure cleanly locates the next work in feature invention, not mixer
  tuning.

### E4 — inference-time competition versus learning-time residualization

- Run: inspect the tree's posterior stop/split weights and cumulative bits.
  Do not add a separate train-only-on-errors rule in this run.
- Expect: specific nodes win by likelihood evidence on structured sources;
  every active node still updates on every byte.
- Meaning: this isolates inference competition. A later residual-learning
  experiment must beat this score without harming calibration.
