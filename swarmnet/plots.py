"""Charts for the study (matplotlib, saved as PNG and shown in Colab)."""
import os

import matplotlib.pyplot as plt
import numpy as np

# fixed colour per method (identity never depends on rank); our central plan is a grey reference
COLORS = {"auction": "#2a78d6", "nearest": "#eb6834", "learned": "#1baf7a", "central": "#6b6a64"}
LABELS = {"auction": "Auction swarm (CBBA)", "nearest": "Nearest-task rule",
          "learned": "Learned swarm (GNN + MAPPO, with relay)", "central": "Central planner (real radio)"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _lines(ax, df, x, y, methods):
    for m in methods:
        d = df[df["method"] == m].groupby(x)[y].agg(["mean", "std"]).reset_index()
        if d.empty:
            continue
        xs = np.arange(len(d)) if x == "radio_range" else d[x].values
        ax.errorbar(xs, d["mean"], yerr=d["std"].fillna(0), color=COLORS.get(m, INK), lw=2,
                    marker="o", ms=7, capsize=3, label=LABELS.get(m, m))
        ax.annotate(LABELS.get(m, m).split(" (")[0], (xs[-1], d["mean"].values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=MUTED)
    return ax


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def gap_vs_range(df, fail_count=0, methods=("central", "nearest", "learned"), path=None):
    d = df[df["fail_count"] == fail_count]
    methods = [m for m in methods if m in set(d["method"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _lines(ax, d, "radio_range", "gap_pct", methods)
    ranges = sorted(d["radio_range"].unique())
    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels(["unlimited" if r >= 1e5 else f"{int(r)} m" for r in ranges])
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.text(0, 0, " our central plan (unlimited radio)", va="bottom", fontsize=9, color=MUTED)
    _style(ax, "How much slower than our central plan, as the radio range changes",
           "Radio range", "Extra time vs our central plan (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return _save(fig, path)


def gap_vs_failures(df, radio_range=150, methods=("central", "nearest", "learned"), path=None):
    d = df[df["radio_range"] == radio_range]
    methods = [m for m in methods if m in set(d["method"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _lines(ax, d, "fail_count", "gap_pct", methods)
    ax.set_xticks(sorted(d["fail_count"].unique()))
    _style(ax, f"Cost of drone failures (radio range {int(radio_range)} m)",
           "Drones that fail mid-mission", "Extra time vs our central plan (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, path)


def connectivity_vs_range(df, fail_count=0, methods=("central", "nearest", "learned"), path=None):
    d = df[(df["fail_count"] == fail_count) & (df["radio_range"] < 1e5)]
    methods = [m for m in methods if m in set(d["method"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _lines(ax, d, "radio_range", "mean_lambda2", methods)
    ranges = sorted(d["radio_range"].unique())
    ax.set_xticks(range(len(ranges)))
    ax.set_xticklabels([f"{int(r)} m" for r in ranges])
    _style(ax, "Network health during the mission (algebraic connectivity)",
           "Radio range", "Mean algebraic connectivity  (0 = network broken)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, path)


def scaling_plot(df, methods=("nearest", "learned"), path=None):
    methods = [m for m in methods if m in set(df["method"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _lines(ax, df, "n_drones", "gap_pct", methods)
    ax.set_xticks(sorted(df["n_drones"].unique()))
    _style(ax, "Does it still work as the swarm grows? (same policy, no retraining)",
           "Number of drones", "Extra time vs our central plan (%)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    return _save(fig, path)


def training_curve(history, path=None):
    it = [h["iter"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(it, [h["return"] for h in history], color=COLORS["learned"], lw=2)
    for n in sorted({h["n_drones"] for h in history}):
        first = next(h["iter"] for h in history if h["n_drones"] == n)
        ax.axvline(first, color=GRID, lw=1)
        ax.text(first, ax.get_ylim()[1], f" {n} drones", va="top", fontsize=9, color=MUTED)
    _style(ax, "Learning curve (team reward per mission)", "Training iteration", "Average return")
    return _save(fig, path)


def eval_curve(history, ref, path=None):
    """Validation score of the policy during training, against the nearest-task rule (dashed)."""
    pts = [(h["iter"], h["eval_score"]) for h in history if h.get("eval_score") is not None]
    fig, ax = plt.subplots(figsize=(8, 4))
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS["learned"], lw=2, marker="o", ms=5,
                label="learned policy")
    ax.axhline(ref, color=COLORS["nearest"], lw=1.5, ls="--", label="nearest-task rule")
    _style(ax, "Validation score during training (higher is better)", "Training iteration", "Score")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, path)
