"""Command-line entry point: download data and run the benchmark.

Examples
--------
Download enwik8::

    induce-text download enwik8

Evaluate baselines on the first 1 MB of enwik8::

    induce-text eval --data enwik8 --bytes 1000000 \\
        --models gzip,bz2,lzma,order0,order1,order3,interp3

List the available models/compressors::

    induce-text list
"""

from __future__ import annotations

import argparse
import sys

from induce_text import data as data_mod
from induce_text.evaluate import EXTERNAL, evaluate_all, format_table
from induce_text.models import REGISTRY

DEFAULT_MODELS = [
    "gzip",
    "bz2",
    "lzma",
    "uniform",
    "order0",
    "order1",
    "order2",
    "order3",
    "interp3",
]


def _cmd_list(_: argparse.Namespace) -> int:
    print("models (adaptive next-byte predictors):")
    for name in REGISTRY:
        print(f"  {name}")
    print("\nexternal compressors (reference baselines):")
    for name in EXTERNAL:
        print(f"  {name}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    path = data_mod.download(args.corpus, force=args.force)
    print(f"ready: {path} ({path.stat().st_size:,} bytes)")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    blob = data_mod.load(args.data, n_bytes=args.bytes, offset=args.offset)
    if not blob:
        print("error: loaded 0 bytes", file=sys.stderr)
        return 1
    print(
        f"evaluating {len(names)} model(s) on {len(blob):,} bytes "
        f"of {args.data} (offset {args.offset:,})\n"
    )
    results = evaluate_all(names, blob)
    print(format_table(results))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="induce-text", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available models and compressors")
    p_list.set_defaults(func=_cmd_list)

    p_dl = sub.add_parser("download", help="download a benchmark corpus")
    p_dl.add_argument("corpus", choices=sorted(data_mod.CORPORA))
    p_dl.add_argument("--force", action="store_true", help="re-download if present")
    p_dl.set_defaults(func=_cmd_download)

    p_eval = sub.add_parser("eval", help="evaluate models/compressors on data")
    p_eval.add_argument(
        "--data",
        default="enwik8",
        help="corpus name (enwik8/enwik9) or path to a file (default: enwik8)",
    )
    p_eval.add_argument(
        "--bytes",
        type=int,
        default=1_000_000,
        help="number of bytes to evaluate (default: 1,000,000; 0/None = all)",
    )
    p_eval.add_argument(
        "--offset", type=int, default=0, help="byte offset to start from"
    )
    p_eval.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated model/compressor names (see `list`)",
    )
    p_eval.set_defaults(func=_cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # --bytes 0 means "read everything".
    if getattr(args, "bytes", None) == 0:
        args.bytes = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
