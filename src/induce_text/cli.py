"""Command-line entry point.

Examples
--------
Download enwik8::

    induce-text download enwik8

Run the benchmark matrix (baselines x synthetic curriculum + enwik8 slice)::

    induce-text eval
    induce-text eval --sources markov pcfg enwik8:1000000 --plot

Empirical means for the PCFG calibration win condition::

    induce-text calibrate --episodes 10000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from induce_text import data as data_mod


def _cmd_download(args: argparse.Namespace) -> int:
    path = data_mod.download(args.corpus, force=args.force)
    print(f"ready: {path} ({path.stat().st_size:,} bytes)")
    return 0


DEFAULT_SOURCES = [
    "periodic",
    "skewed_iid",
    "markov",
    "long_range_copy",
    "pcfg",
    "enwik8:100000",
]


def _cmd_eval(args: argparse.Namespace) -> int:
    from induce_text import benchmark

    rows = benchmark.run(args.sources, n=args.n, seed=args.seed)
    print()
    print(benchmark.format_table(rows))
    out_dir = Path(args.out_dir)
    meta = {"sources": args.sources, "n": args.n, "seed": args.seed}
    json_path = benchmark.save_json(rows, out_dir, meta)
    print(f"\nresults: {json_path}")
    if args.plot:
        for p in benchmark.save_plots(rows, out_dir):
            print(f"plot:    {p}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    """Empirical side of the calibration win condition (CLAUDE.md open item 1).

    Reports mean episode length and mean transcript bits over many seeds, for
    both known grammars.  The hand-derivation (one-step-expansion equations)
    is the author's; this only supplies the empirical column.
    """
    import statistics

    from induce_text.pcfg_gen import RecordingChoice, Rule, sample
    from induce_text.sources import pcfg_test_grammar

    x = Rule(symbols=["a", "b", ["y", "d"], "e", "f", ["x", "a"]])
    y = Rule(symbols=["a", "b", ["c", "d"]])
    main_env = {"a": 10, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "x": x, "y": y}
    grammars = {
        "test grammar (y inlined)": pcfg_test_grammar(),
        "__main__ grammar (with y)": (x, main_env),
    }
    for name, (rule, env) in grammars.items():
        lengths, bits = [], []
        for seed in range(args.episodes):
            cs = RecordingChoice(seed=seed)
            out = sample(rule=rule, env=env, choicesource=cs)
            lengths.append(len(out))
            bits.append(cs.count)
        print(f"{name}  ({args.episodes:,} episodes)")
        print(f"  mean length {statistics.mean(lengths):8.4f}   "
              f"stdev {statistics.stdev(lengths):.4f}")
        print(f"  mean bits   {statistics.mean(bits):8.4f}   "
              f"stdev {statistics.stdev(bits):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="induce-text", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="download a benchmark corpus")
    p_dl.add_argument("corpus", choices=sorted(data_mod.CORPORA))
    p_dl.add_argument("--force", action="store_true", help="re-download if present")
    p_dl.set_defaults(func=_cmd_download)

    p_ev = sub.add_parser("eval", help="run the benchmark matrix")
    p_ev.add_argument(
        "--sources",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="synthetic source names and/or corpus[:bytes] specs",
    )
    p_ev.add_argument("--n", type=int, default=30_000, help="bytes per source")
    p_ev.add_argument("--seed", type=int, default=0)
    p_ev.add_argument("--plot", action="store_true", help="save learning curves")
    p_ev.add_argument("--out-dir", default="results")
    p_ev.set_defaults(func=_cmd_eval)

    p_cal = sub.add_parser(
        "calibrate", help="empirical PCFG episode statistics (mean length/bits)"
    )
    p_cal.add_argument("--episodes", type=int, default=10_000)
    p_cal.set_defaults(func=_cmd_calibrate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
