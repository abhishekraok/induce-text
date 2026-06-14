# induce-text

Program synthesis for language modelling, scored as **compression**.

The thesis (Hutter's): *compression = intelligence*. A language model is a
next-byte predictor, and a next-byte predictor is a compressor — the coding cost
of a byte is `-log2 P(byte)`. So "model the text well" and "make the file small"
are the same objective, measured in **bits per byte (bpc)** on the
[enwik](https://mattmahoney.net/dc/textdata.html) benchmark (Wikipedia XML).

This repo starts with the *measurement infrastructure and dumb baselines*. The
program-synthesis models come later; first we make the benchmark trustworthy.

## Setup

Uses [uv](https://docs.astral.sh/uv/) with Python 3.12.

```bash
uv sync --extra dev      # create .venv and install dev deps
uv run pytest            # run the test suite
```

## Usage

```bash
# Download the 100 MB enwik8 corpus into ./data (gitignored).
uv run induce-text download enwik8

# Evaluate baselines on the first 1 MB.
uv run induce-text eval --data enwik8 --bytes 1000000

# List available models / reference compressors.
uv run induce-text list
```

Example output:

```
model              bpc    ratio   saving   comp.bytes
--------------------------------------------------------
lzma            2.xxxx   0.xxxx   xx.xx%      xxx,xxx
...
uniform         8.0000   1.0000    0.00%    1,000,000
```

## What's here

| Module | Role |
|---|---|
| `models.py` | Online adaptive next-byte predictors: `uniform`, `order0`, `orderN` (n-gram with add-α smoothing), `interpN` (interpolated orders). The `ByteModel` interface: `prob(byte)` + `update(byte)`. |
| `evaluate.py` | Scoring loop (`-log2 P` per byte) and off-the-shelf reference compressors (`gzip`, `bz2`, `lzma`). Everything reports into a common `bpc` `Result`. |
| `metrics.py` | `Result` and bpc / ratio / space-saving definitions. |
| `data.py` | On-demand download + slicing of enwik8 / enwik9. |
| `cli.py` | `download` / `eval` / `list` commands. |

## Why this framing

Measuring a model as `-log2 P(actual byte)` gives the exact bit cost an ideal
arithmetic coder would pay, **without** needing to build the coder yet. Models
adapt online, so (as in cmix/PPM) the model itself ships for ~0 bits: the decoder
re-derives its state from the bytes already decoded. The reference compressors
are the bars to clear — gzip ~3.1, bzip2 ~2.3, lzma ~2.0 bpc on enwik.
