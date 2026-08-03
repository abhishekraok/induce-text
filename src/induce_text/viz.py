"""Visualization tools: make the bits visible.

Every view here answers one question by eye:

- ``heat_page``        — WHERE is the model surprised?  The data itself,
  each byte colored by its cost in bits, with the model's top guesses in
  the tooltip.
- ``delta_page``       — WHERE do two models differ?  Same text, colored by
  bits(A) - bits(B).
- ``tree_page``        — WHERE do the generator's bits live?  A PCFG episode
  as a derivation tree, each 50:50 choice pinned to the transcript bit it
  consumed; deterministic structure has nothing pinned to it (it is free).
- ``calibration_plot`` — are the probabilities HONEST?  Predicted p vs
  empirical frequency over every byte slot (the StreamPredictor scar:
  measure calibration, not accuracy).
- ``growth_plot``      — how fast does the model GROW?  Contexts minted vs
  position: the description length the baselines never pay — the missing
  half of the two-part code.

Trust discipline: these views re-derive what they show, so each one
self-checks against the instrument (``score_bits`` for costs, the author's
``sample`` for derivations) and raises on disagreement.  A picture that can
silently drift from the numbers is worse than no picture.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from induce_text.model import Model, score_bits
from induce_text.pcfg_gen import ReplayChoice, Rule, sample

# --- shared scan ------------------------------------------------------------


def _scan(model: Model, data: bytes):
    """Yield (t, byte, distribution) for each position, threading state."""
    state = model.init()
    for t, byte in enumerate(data):
        yield t, byte, model.predict(state)
        state = model.absorb(state, byte)


def heat_data(
    model: Model, data: bytes, k: int = 5
) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    """Per-byte bits plus the model's top-k guesses at each position.

    Self-check: the bits recomputed here must match ``score_bits`` exactly.
    """
    bits = np.empty(len(data))
    top: list[list[tuple[int, float]]] = []
    for t, byte, dist in _scan(model, data):
        p = dist[byte]
        bits[t] = -np.log2(p) if p > 0 else np.inf
        order = np.argsort(dist)[::-1][:k]
        top.append([(int(i), float(dist[i])) for i in order])
    if not np.array_equal(bits, score_bits(model, data)):
        raise AssertionError("viz scan disagrees with score_bits")
    return bits, top


# --- HTML machinery ---------------------------------------------------------

N_CLASSES = 17  # q0..q16 = 0.0 .. 8.0 bits in half-bit steps; qX beyond


def _heat_css() -> str:
    rules = []
    for i in range(N_CLASSES):
        b = i / 2  # bits
        lightness = 100 - 6.25 * b  # white at 0 bits -> mid red at 8
        fg = "#fff" if lightness < 62 else "#000"
        rules.append(f".q{i}{{background:hsl(6,85%,{lightness:.0f}%);color:{fg}}}")
    rules.append(".qX{background:#3a0505;color:#fff}")
    return "\n".join(rules)


def _delta_css() -> str:
    # d0..d16: -4 .. +4 bits in half-bit steps.  Blue = A cheaper, red = B.
    rules = []
    for i in range(N_CLASSES):
        d = (i - 8) / 2
        hue = 217 if d < 0 else 6
        lightness = 100 - 11 * min(abs(d), 4)
        fg = "#fff" if lightness < 62 else "#000"
        rules.append(
            f".d{i}{{background:hsl({hue},80%,{lightness:.0f}%);color:{fg}}}"
        )
    return "\n".join(rules)


_BASE_CSS = """
body{font-family:sans-serif;margin:1.5em;background:#fff;color:#000}
h1{font-size:1.2em}
p.meta{color:#444;max-width:60em}
.text{font-family:monospace;font-size:14px;line-height:1.45;
      white-space:pre-wrap;overflow-wrap:anywhere;max-width:96ch;
      border:1px solid #ccc;padding:8px;background:#fff}
.legend span{padding:2px 8px;margin-right:4px;font-family:monospace}
"""


def _page(title: str, meta: str, body: str, extra_css: str) -> str:
    return (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        f"<style>{_BASE_CSS}{extra_css}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p class='meta'>{html.escape(meta)}</p>"
        f"{body}</body></html>\n"
    )


def _bits_class(b: float) -> str:
    if not np.isfinite(b) or b > 8:
        return "qX"
    return f"q{int(round(b * 2))}"


def _delta_class(d: float) -> str:
    if not np.isfinite(d):
        d = 4.0 if d > 0 else -4.0
    i = int(round(d * 2)) + 8
    return f"d{max(0, min(N_CLASSES - 1, i))}"


def _display(byte: int, hex_mode: bool) -> tuple[str, bool]:
    """Return (display string, is_newline)."""
    if hex_mode:
        return f"{byte:x}", False
    if byte == 0x0A:
        return "¶", True
    if byte == 0x09:
        return "→", False
    if byte == 0x0D:
        return "␍", False
    if 32 <= byte < 127:
        return chr(byte), False
    return "·", False


def _char_name(byte: int) -> str:
    return chr(byte) if 32 <= byte < 127 else f"0x{byte:02x}"


def _guess_str(guesses: list[tuple[int, float]]) -> str:
    return " ".join(f"{_char_name(b)}:{p:.2f}" for b, p in guesses)


def _spans(
    data: bytes,
    classes: list[str],
    tips: list[str],
) -> str:
    hex_mode = max(data) < 16
    out = []
    for byte, cls, tip in zip(data, classes, tips):
        shown, is_newline = _display(byte, hex_mode)
        out.append(
            f'<span class="{cls}" title="{html.escape(tip, quote=True)}">'
            f"{html.escape(shown)}</span>"
        )
        if is_newline:
            out.append("\n")
    return f'<div class="text">{"".join(out)}</div>'


def _heat_legend() -> str:
    items = "".join(
        f'<span class="{_bits_class(b)}">{label}</span>'
        for b, label in [(0, "0"), (2, "2"), (4, "4"), (6, "6"), (8, "8"),
                         (9, "&gt;8")]
    )
    return f'<p class="legend">cost in bits: {items}</p>'


def heat_page(model: Model, data: bytes, *, title: str, meta: str = "") -> str:
    """The data itself, colored by per-byte cost; guesses in tooltips."""
    bits, top = heat_data(model, data)
    classes = [_bits_class(b) for b in bits]
    tips = [
        f"pos {t} {_char_name(byte)}: {bits[t]:.2f} bits | guessed: "
        f"{_guess_str(top[t])}"
        for t, byte in enumerate(data)
    ]
    meta = f"{meta}  mean {bits.mean():.4f} bpc over {len(data):,} bytes."
    return _page(title, meta, _heat_legend() + _spans(data, classes, tips),
                 _heat_css())


def delta_page(
    model_a: Model, model_b: Model, data: bytes, *,
    name_a: str, name_b: str, title: str,
) -> str:
    """Same text, colored by bits(A) - bits(B).  Blue: A cheaper; red: B."""
    bits_a, _ = heat_data(model_a, data)
    bits_b, _ = heat_data(model_b, data)
    delta = bits_a - bits_b
    classes = [_delta_class(d) for d in delta]
    tips = [
        f"pos {t} {_char_name(byte)}: {name_a} {bits_a[t]:.2f} vs "
        f"{name_b} {bits_b[t]:.2f} ({delta[t]:+.2f})"
        for t, byte in enumerate(data)
    ]
    legend = (
        f'<p class="legend"><span class="{_delta_class(-3)}">{name_a} '
        f'cheaper</span> <span class="{_delta_class(0)}">tie</span> '
        f'<span class="{_delta_class(3)}">{name_b} cheaper</span></p>'
    )
    meta = (
        f"mean {name_a} {bits_a.mean():.4f} bpc, {name_b} "
        f"{bits_b.mean():.4f} bpc over {len(data):,} bytes."
    )
    return _page(title, meta, legend + _spans(data, classes, tips),
                 _delta_css())


# --- PCFG derivation tree ---------------------------------------------------


@dataclass
class Node:
    kind: str  # "rule" | "choice" | "terminal"
    label: str
    children: list["Node"] = field(default_factory=list)
    value: int | None = None  # terminals only
    bit_index: int | None = None  # choices only
    bit_value: bool | None = None


def _resolve(name: str, env: dict[str, int | Rule], replay: ReplayChoice) -> Node:
    val = env[name]
    if isinstance(val, Rule):
        return _derive(val, name, env, replay)
    return Node(kind="terminal", label=name, value=val)


def _derive(
    rule: Rule, name: str, env: dict[str, int | Rule], replay: ReplayChoice
) -> Node:
    node = Node(kind="rule", label=name)
    for s in rule.symbols:
        if isinstance(s, list):
            bit_index = replay.index
            bit = replay.choice()
            chosen = s[0] if bit else s[1]
            choice = Node(
                kind="choice",
                label=f"{s[0]} | {s[1]}",
                bit_index=bit_index,
                bit_value=bit,
                children=[_resolve(chosen, env, replay)],
            )
            node.children.append(choice)
        else:
            node.children.append(_resolve(s, env, replay))
    return node


def _leaves(node: Node) -> list[int]:
    if node.kind == "terminal":
        assert node.value is not None
        return [node.value]
    return [v for c in node.children for v in _leaves(c)]


def subtree_bits(node: Node) -> int:
    own = 1 if node.kind == "choice" else 0
    return own + sum(subtree_bits(c) for c in node.children)


def derivation_tree(
    rule: Rule, env: dict[str, int | Rule], transcript: list[bool]
) -> tuple[Node, list[int]]:
    """Rebuild the derivation tree a transcript encodes.

    Self-check: this walker is a *second* interpretation of the grammar, so
    its leaves are verified against the author's ``sample`` on the same
    transcript, and the transcript must be exactly consumed.
    """
    replay = ReplayChoice(transcript)
    root = _derive(rule, "start", env, replay)
    if replay.index != len(transcript):
        raise AssertionError("derivation tree did not consume the transcript")
    leaves = _leaves(root)
    if leaves != sample(rule=rule, env=env, choicesource=ReplayChoice(transcript)):
        raise AssertionError("derivation tree disagrees with sample()")
    return root, leaves


_TREE_CSS = """
details{margin-left:1.2em;border-left:1px dotted #bbb;padding-left:.5em}
summary{cursor:pointer;font-family:monospace}
.leaf{margin-left:1.2em;font-family:monospace;color:#000}
.chip{padding:1px 6px;border-radius:3px;font-family:monospace}
.rule{background:#e8eefc}
.bit0{background:#fde3c8}
.bit1{background:#d3ecd3}
.term{background:#f2f2f2}
.free{color:#777;font-style:italic}
"""


def _tree_html(node: Node) -> str:
    if node.kind == "terminal":
        return (
            f'<div class="leaf"><span class="chip term">{html.escape(node.label)}'
            f" = {node.value:x}</span> <span class='free'>0 bits</span></div>"
        )
    if node.kind == "choice":
        bit = int(bool(node.bit_value))
        head = (
            f'<span class="chip bit{bit}">bit #{node.bit_index} = {bit} '
            f"→ [{html.escape(node.label)}]</span>"
        )
        body = "".join(_tree_html(c) for c in node.children)
        return f"<details open><summary>{head}</summary>{body}</details>"
    nbits = subtree_bits(node)
    head = (
        f'<span class="chip rule">{html.escape(node.label)}</span> '
        f"<span class='free'>{nbits} bit{'s' if nbits != 1 else ''} below</span>"
    )
    body = "".join(_tree_html(c) for c in node.children)
    return f"<details open><summary>{head}</summary>{body}</details>"


def tree_page(
    rule: Rule, env: dict[str, int | Rule], transcript: list[bool], *, title: str
) -> str:
    root, leaves = derivation_tree(rule, env, transcript)
    bitstring = "".join(str(int(b)) for b in transcript)
    output = "".join(f"{v:x}" for v in leaves)
    meta = (
        "Choices carry bits; everything else is free. "
        f"{len(transcript)} transcript bits decode {len(leaves)} symbols."
    )
    footer = (
        f'<p class="text">transcript: {bitstring}<br>output:&nbsp;&nbsp;&nbsp;&nbsp;'
        f"{output}</p>"
    )
    return _page(title, meta, _tree_html(root) + footer, _TREE_CSS)


# --- calibration ------------------------------------------------------------


def calibration_data(
    model: Model, data: bytes, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned reliability stats over every (position, byte-value) slot.

    For each position the model states 256 probabilities; each is a
    prediction of a binary event ("the next byte is b") whose outcome we
    know.  Returns (count, mean predicted p, empirical frequency) per bin.
    A calibrated model puts the mean-p and frequency columns on a diagonal.
    """
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins)
    psums = np.zeros(n_bins)
    hits = np.zeros(n_bins)
    for _, byte, dist in _scan(model, data):
        idx = np.clip(np.searchsorted(edges, dist, side="right") - 1, 0, n_bins - 1)
        np.add.at(counts, idx, 1)
        np.add.at(psums, idx, dist)
        hits[idx[byte]] += 1
    with np.errstate(invalid="ignore"):
        return counts, psums / counts, hits / counts


def calibration_plot(
    models: dict[str, Model], data: bytes, path: Path, *,
    title: str, min_count: int = 50,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.logspace(-8, 0, 33)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    lo, hi = 1e-8, 1.0
    ax.plot([lo, hi], [lo, hi], "--", color="gray", label="calibrated")
    for name, model in models.items():
        counts, mean_p, freq = calibration_data(model, data, edges)
        mask = (counts >= min_count) & (freq > 0)
        ax.plot(mean_p[mask], freq[mask], "o-", label=name, markersize=4)
    ax.set(
        xscale="log", yscale="log", title=title,
        xlabel="predicted probability",
        ylabel="empirical frequency (zero-frequency bins hidden)",
    )
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# --- context-table growth ---------------------------------------------------


def table_growth(model, data: bytes, every: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """(positions, table sizes) as a ContextK model absorbs the stream."""
    state = model.init()
    xs, sizes = [], []
    for t, byte in enumerate(data):
        state = model.absorb(state, byte)
        if t % every == 0 or t == len(data) - 1:
            xs.append(t + 1)
            sizes.append(len(state[0]))
    return np.array(xs), np.array(sizes)


def growth_plot(models: dict[str, Model], data: bytes, path: Path, *, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        xs, sizes = table_growth(model, data)
        ax.loglog(xs, sizes, label=name)
    xs_ref = np.array([1, len(data)])
    ax.loglog(xs_ref, xs_ref, "--", color="gray", label="1 context/byte")
    ax.set(
        title=f"{title} — description length the model never pays",
        xlabel="position", ylabel="contexts in table",
    )
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
