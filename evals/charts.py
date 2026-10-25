"""REPORT.md's charts, as SVG in a light and a dark version.

- ``ndcg-<dataset>``: each variant's nDCG@10 difference from the baseline,
  with its paired bootstrap 95% CI. Blue where the CI excludes zero, gray
  where it doesn't. A difference too large for the scale is drawn at its
  edge and labelled.
- ``latency-<dataset>``: nDCG@10 against latency, a dot at p50 and a line
  to p95, on a log scale.

The colors are a validated palette's first categorical slot and its ink and
surface tokens, stepped separately for each theme. REPORT.md shows the
version that matches the reader's theme. Output is reproducible: no dates,
fixed SVG ids.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "accent": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "accent": "#3987e5"},
}
OFF_SCALE = 0.4         # |difference| beyond which a variant is drawn at the scale's edge
ROW = 0.34              # inches per variant in the difference chart
WIDTH = 7.0


def _style(theme: dict) -> None:
    plt.rcParams.update({
        "svg.hashsalt": "alexandria-evals",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": theme["axis"],
        "axes.labelcolor": theme["ink2"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["ink2"],
        "xtick.labelcolor": theme["ink2"],
        "ytick.labelcolor": theme["ink2"],
        "axes.facecolor": theme["surface"],
        "figure.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
    })


def _frame(ax, theme: dict, grid_axis: str) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _titles(fig, title: str, subtitle: str, theme: dict) -> None:
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=11, weight="bold", color=theme["ink"])
    fig.text(0.012, 0.985 - 0.32 / fig.get_figheight(), subtitle, ha="left", va="top",
             fontsize=9, color=theme["ink2"])


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", "−")


def difference_chart(rows: list[tuple[str, float, float, float]], *, baseline: str, title: str,
                     subtitle: str, theme: dict, path: Path) -> None:
    """``rows``: (variant, difference, CI low, CI high), top to bottom."""
    _style(theme)
    height = 1.5 + ROW * len(rows)
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    fig.subplots_adjust(left=0.2, right=0.97, top=1 - 1.1 / height, bottom=0.45 / height)
    _frame(ax, theme, "x")

    on_scale = [r for r in rows if abs(r[1]) <= OFF_SCALE] or rows
    lo = min(0.0, *(r[2] for r in on_scale))
    hi = max(0.0, *(r[3] for r in on_scale))
    pad = (hi - lo) * 0.08 or 0.01
    xmin, xmax = lo - pad, hi + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.axvline(0, color=theme["axis"], linewidth=1, zorder=1)
    at = (0 - xmin) / (xmax - xmin)     # keep the label inside the plot near either edge
    ax.text(0, -0.75, f"{baseline} (baseline)", va="bottom", fontsize=8, color=theme["muted"],
            ha="right" if at > 0.85 else "left" if at < 0.15 else "center")

    for y, (name, diff, ci_lo, ci_hi) in enumerate(rows):
        color = theme["accent"] if ci_lo > 0 or ci_hi < 0 else theme["muted"]
        if abs(diff) > OFF_SCALE:
            edge = xmin if diff < 0 else xmax
            ax.plot([edge], [y], marker="<" if diff < 0 else ">", markersize=8, color=color,
                    markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3, clip_on=False)
            ax.text(edge + (pad * 0.6 if diff < 0 else -pad * 0.6), y,
                    f"{_signed(diff)} [{_signed(ci_lo)}, {_signed(ci_hi)}], off the scale",
                    ha="left" if diff < 0 else "right", va="center", fontsize=8, color=theme["ink2"])
            continue
        ax.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([diff], [y], marker="o", markersize=8, color=color,
                markeredgecolor=theme["surface"], markeredgewidth=2, zorder=3)

    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Δ nDCG@10")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:+g}".replace("-", "−") if abs(x) > 1e-12 else "0")
    handles = [Line2D([], [], color=theme["accent"], marker="o", linewidth=2, markersize=6,
                      label="95% CI excludes 0"),
               Line2D([], [], color=theme["muted"], marker="o", linewidth=2, markersize=6,
                      label="95% CI includes 0")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.004, 1 - 0.58 / height), ncols=2,
               frameon=False, fontsize=8, labelcolor=theme["ink2"], handlelength=1.6)
    _titles(fig, title, subtitle, theme)
    _save(fig, path)


def place_labels(anchors: list[tuple[float, float]], gap: float, width: float,
                 offset: float = 10) -> list[tuple[float, float]]:
    """Label positions (display px) for points at ``anchors``. Labels that
    would overlap, directly or through a neighbour, form a group stacked in
    a column right of the group's rightmost point, in the points' vertical
    order, so their leader lines don't cross."""
    group = list(range(len(anchors)))

    def root(i: int) -> int:
        while group[i] != i:
            i = group[i]
        return i

    for i, (xi, yi) in enumerate(anchors):
        for j, (xj, yj) in enumerate(anchors[:i]):
            if abs(xi - xj) < width and abs(yi - yj) < gap:
                group[root(i)] = root(j)
    members: dict[int, list[int]] = {}
    for i in range(len(anchors)):
        members.setdefault(root(i), []).append(i)
    out = [(x + offset, y) for x, y in anchors]
    for g in members.values():
        if len(g) == 1:
            continue
        x = max(anchors[i][0] for i in g) + offset * 2
        mid = sum(anchors[i][1] for i in g) / len(g)
        for k, i in enumerate(sorted(g, key=lambda i: anchors[i][1])):
            out[i] = (x, mid + gap * (k - (len(g) - 1) / 2))
    return out


def latency_chart(points: list[tuple[str, float, float, float]], *, baseline: str, title: str,
                  subtitle: str, theme: dict, path: Path) -> None:
    """``points``: (variant, p50 ms, p95 ms, nDCG@10). A variant more than
    OFF_SCALE below the baseline is drawn at the bottom edge and labelled."""
    _style(theme)
    height = 4.6
    fig, ax = plt.subplots(figsize=(WIDTH, height))
    fig.subplots_adjust(left=0.09, right=0.8, top=1 - 0.85 / height, bottom=0.5 / height)
    _frame(ax, theme, "both")
    ax.set_xscale("log")
    base = next(ndcg for name, _, _, ndcg in points if name == baseline)
    off = {name for name, _, _, ndcg in points if ndcg < base - OFF_SCALE}
    lo = min(ndcg for name, _, _, ndcg in points if name not in off)
    hi = max(ndcg for name, _, _, ndcg in points if name not in off)
    pad = (hi - lo) * 0.12 or 0.01
    floor = lo - pad                    # where off-scale variants are drawn
    ax.set_ylim(floor - pad, hi + pad)
    ys = [floor if name in off else ndcg for name, _, _, ndcg in points]
    for (name, p50, p95, _), y in zip(points, ys):
        ax.plot([p50, p95], [y, y], color=theme["accent"], linewidth=2,
                solid_capstyle="round", zorder=2)
        ax.plot([p50], [y], marker="v" if name in off else "o", markersize=8,
                color=theme["accent"], markeredgecolor=theme["surface"], markeredgewidth=2,
                zorder=3)
    ax.margins(x=0.08)
    ax.set_xlabel("Latency per query, ms (log scale)")
    ax.set_ylabel("nDCG@10")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:g}")

    fig.canvas.draw()
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    anchors = [tuple(to_px((p95, y))) for (_, _, p95, _), y in zip(points, ys)]
    labels = place_labels(anchors, 13, 95)
    for (name, _, p95, ndcg), y, anchor, label in zip(points, ys, anchors, labels):
        text = f"{name}: {ndcg:.3f}, below the scale" if name in off else name
        moved = abs(label[1] - anchor[1]) > 2 or label[0] - anchor[0] > 12
        ax.annotate(text, xy=(p95, y), xytext=to_data(label), ha="left", va="center",
                    fontsize=8, color=theme["ink2"], annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": theme["muted"], "linewidth": 0.8,
                                "shrinkA": 2, "shrinkB": 5} if moved else None)
    _titles(fig, title, subtitle, theme)
    _save(fig, path)
