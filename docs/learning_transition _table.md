# Learning
Sketching idea on how we can learn, using transtion table and higher level symbols.
Let us assume our vocabulary consists of raw symbols `'(a, b, c, d)`


## Recognition
Assume currently we have bunch of existing features that can take us from raw, low level feature to higher level features.

```lisp
(define (three-as)
  '(a a a))
```

First we replace the low level symbols into higher level symbols.
Suppose the raw data

```lisp
'(a a a b a a a b a a a b a a a)
```
This can be compressed as
```lisp
`(,@(three-as) b ,@(three-as) b ,@(three-as) b ,@(three-as))
```

## Transition table
We will learn a table that looks something like this


| b | Weight | 
| -------- | -------- 
| three-as  | 3
| a  | 0
| b  | 0
| c  | 0
| d  | 0

| three-as | Weight | 
| -------- | -------- 
| b  | 3
| c  | 0

| a | Weight | 
| -------- | -------- 
| three-as  | 0
| b  | 0

Anything not written down on the table is assumed to have 0 weight.
Note that a's transition table is empty as all the a's have been consumed by three-as.

## Inference
For inference we want to upweight paths that lead to the current symbol based on the transition table weights and computation simplicity. 
We want to use Ocam's razor, but here using the learnt tables and features.
We can use Softmax to turn weights into probabilities 

Let us take few examples.


If the input is `'(a a a)`, then three-a to b has highest weight, so if we take 
softmax we'll get `$P(b/three-as) = e^3/(e^3 + 4*e^0)$` which is about 83% 
Now if the input is `'(a a a b)` then a can be derived in two ways, either as the start of three-as or as raw a. `$P(a/b) = P(b->three-a) + P(b->a)$`
Similarly if the input is `'(a a a b a)` then getting a has two ways of occuring, either as a continuation of three-a or as the raw symbol a after a.
`$P(a/a) = P(b->three-a a symbol ago) + P(a->a)$`

The issue is that this is not computationally tractable with growing number of ways to reach a given symbol.