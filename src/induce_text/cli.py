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

Visualizations (HTML/PNG written to results/)::

    induce-text viz heat --source enwik8:10000 --model ctx2
    induce-text viz heat --source enwik8:10000 --model ctx2 --vs iid
    induce-text viz tree --grammar test --seed 3
    induce-text viz calibration --source markov
    induce-text viz growth --source enwik8:100000
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

    from induce_text.pcfg_gen import RecordingChoice, sample
    from induce_text.sources import pcfg_main_grammar, pcfg_test_grammar

    grammars = {
        "test grammar (y inlined)": pcfg_test_grammar(),
        "__main__ grammar (with y)": pcfg_main_grammar(),
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


def _viz_model(name: str):
    from induce_text.baselines import default_models

    models = default_models()
    if name not in models:
        raise SystemExit(f"unknown model {name!r}; choices: {sorted(models)}")
    return models[name]


def _cmd_viz(args: argparse.Namespace) -> int:
    from induce_text import benchmark, viz
    from induce_text.baselines import ContextK, default_models

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    if args.what == "tree":
        from induce_text.pcfg_gen import RecordingChoice, sample
        from induce_text.sources import pcfg_main_grammar, pcfg_test_grammar

        rule, env = (
            pcfg_test_grammar() if args.grammar == "test" else pcfg_main_grammar()
        )
        cs = RecordingChoice(seed=args.seed)
        sample(rule=rule, env=env, choicesource=cs)
        page = viz.tree_page(
            rule, env, cs.choices,
            title=f"PCFG derivation — {args.grammar} grammar, seed {args.seed}",
        )
        path = out_dir / f"tree-{args.grammar}-{args.seed}.html"
        path.write_text(page)
        print(f"wrote: {path}")
        return 0

    name, data, _ = benchmark.resolve_source(args.source, args.n, args.seed)

    if args.what == "heat":
        model = _viz_model(args.model)
        if args.vs is None:
            page = viz.heat_page(
                model, data,
                title=f"{name} under {args.model}",
                meta=f"Background = cost of that byte in bits under {args.model}; "
                "hover for the model's guesses.",
            )
            path = out_dir / f"heat-{name}-{args.model}.html"
        else:
            other = _viz_model(args.vs)
            page = viz.delta_page(
                model, other, data,
                name_a=args.model, name_b=args.vs,
                title=f"{name}: {args.model} vs {args.vs}",
            )
            path = out_dir / f"delta-{name}-{args.model}-vs-{args.vs}.html"
        path.write_text(page)
        print(f"wrote: {path}")
    elif args.what == "calibration":
        wanted = default_models()
        models = {m: wanted[m] for m in args.models}
        path = viz.calibration_plot(
            models, data, out_dir / f"calibration-{name}.png",
            title=f"reliability on {name} (n={len(data):,})",
        )
        print(f"wrote: {path}")
    elif args.what == "growth":
        models = {f"ctx{k}": ContextK(k) for k in args.ks}
        path = viz.growth_plot(
            models, data, out_dir / f"growth-{name}.png",
            title=f"context tables on {name}",
        )
        print(f"wrote: {path}")
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

    p_viz = sub.add_parser("viz", help="visualizations (HTML/PNG into results/)")
    vsub = p_viz.add_subparsers(dest="what", required=True)

    def _common(p: argparse.ArgumentParser, n: int) -> None:
        p.add_argument("--source", default="pcfg", help="source name or corpus[:bytes]")
        p.add_argument("--n", type=int, default=n, help="bytes to visualize")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--out-dir", default="results")

    p_heat = vsub.add_parser("heat", help="text colored by per-byte cost in bits")
    _common(p_heat, 10_000)
    p_heat.add_argument("--model", default="ctx2")
    p_heat.add_argument("--vs", default=None, help="second model: color by A-B delta")
    p_heat.set_defaults(func=_cmd_viz)

    p_tree = vsub.add_parser("tree", help="PCFG derivation tree with its bits")
    p_tree.add_argument("--grammar", choices=["test", "main"], default="test")
    p_tree.add_argument("--seed", type=int, default=0)
    p_tree.add_argument("--out-dir", default="results")
    p_tree.set_defaults(func=_cmd_viz)

    p_rel = vsub.add_parser("calibration", help="reliability diagram (predicted p vs freq)")
    _common(p_rel, 30_000)
    p_rel.add_argument("--models", nargs="+", default=["iid", "ctx1", "ctx2"])
    p_rel.set_defaults(func=_cmd_viz)

    p_gr = vsub.add_parser("growth", help="context-table size vs position")
    _common(p_gr, 100_000)
    p_gr.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    p_gr.set_defaults(func=_cmd_viz)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
