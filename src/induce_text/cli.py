"""Command-line entry point.

Currently infra only — downloading benchmark corpora. The evaluation commands
will return once the core (models, metrics, scoring loop) is written by hand.

Examples
--------
Download enwik8::

    induce-text download enwik8
"""

from __future__ import annotations

import argparse

from induce_text import data as data_mod


def _cmd_download(args: argparse.Namespace) -> int:
    path = data_mod.download(args.corpus, force=args.force)
    print(f"ready: {path} ({path.stat().st_size:,} bytes)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="induce-text", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download a benchmark corpus")
    p_dl.add_argument("corpus", choices=sorted(data_mod.CORPORA))
    p_dl.add_argument("--force", action="store_true", help="re-download if present")
    p_dl.set_defaults(func=_cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
