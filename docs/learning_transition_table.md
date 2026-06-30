# Learning

Sketching an idea on how we can learn, using a transition table and higher-level symbols.

Let us assume our vocabulary consists of raw symbols `'(a b c d)`.

## Recognition

Assume we currently have a bunch of existing features that can take us from raw,
low-level features to higher-level features.

```lisp
(define (three-as)
  '(a a a))
```

First we replace the low-level symbols with higher-level symbols. Suppose the raw data is

```lisp
'(a a a b a a a b a a a b a a a)
```

This can be compressed as

```lisp
`(,@(three-as) b ,@(three-as) b ,@(three-as) b ,@(three-as))
```

## Transition table

We will learn a table that looks something like this.

| b | Weight |
| --- | --- |
| three-as | 3 |
| a | 0 |
| b | 0 |
| c | 0 |
| d | 0 |

| three-as | Weight |
| --- | --- |
| b | 3 |
| c | 0 |

| a | Weight |
| --- | --- |
| three-as | 0 |
| b | 0 |

Anything not written down in the table is assumed to have 0 weight. Note that
`a`'s transition table is empty, as all the `a`'s have been consumed by `three-as`.

## Inference

For inference we want to upweight paths that lead to the current symbol based on the
transition table weights and computational simplicity. We want to use Occam's razor,
but here using the learnt tables and features. We can use a softmax to turn weights
into probabilities.

Let us take a few examples.

If the input is `'(a a a)`, then `three-as` $\to$ `b` has the highest weight, so if we
take the softmax we get

$$P(b \mid \text{three-as}) = \frac{e^3}{e^3 + 4 e^0} \approx 0.83$$

Now if the input is `'(a a a b)`, then `a` can be derived in two ways, either as the
start of `three-as` or as a raw `a`:

$$P(a \mid b) = P(b \to \text{three-as}) + P(b \to a)$$

Similarly, if the input is `'(a a a b a)`, then getting `a` has two ways of occurring,
either as a continuation of `three-as` or as the raw symbol `a` after `a`:

$$P(a \mid a) = P(b \to \text{three-as, a symbol ago}) + P(a \to a)$$

The issue is that this is not computationally tractable with a growing number of ways
to reach a given symbol.

## Fast inference
30 Jun 2026

The above method is suitable for slow inference, similar to making a conscious decision. But most of the times we want a quick decision.
I feel we should be able to use some sort of dictionary look up for a fast inference.
The goal is to come up with a find the simplest mapping that gives the correct answer. 
e.g. In the above example, create an inference rule such as

if input is `aaaba` then output is `a`. 

Of course these need to be periodically recomputed using the slow inference rules.