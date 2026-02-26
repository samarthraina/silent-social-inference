"""
src/visualization.py
─────────────────────────────────────────────────────────────
Publication-grade visualisation suite using matplotlib + seaborn only.

All figures are saved as high-resolution PNG and PDF
(suitable for NeurIPS workshop paper submission).

Figure inventory
────────────────
1. score_evolution         — Line plot: per-model mean score over iterations
2. drift_curve_a           — Received vs Given drift for Model A
3. judge_severity          — Bar chart: JSI per judge ± 95% CI
4. scoring_matrix_heatmap  — Heatmap of mean scores (judge × respondent)
5. inflation_trend         — Per-judge OLS trend lines
6. variance_over_time      — Rolling variance plot
7. inter_model_agreement   — Heatmap of Pearson correlations (IMAM)
8. reciprocity_scatter     — Scatter of A→X vs X→A scores per iteration

Style configuration
───────────────────
• Uses seaborn "whitegrid" style for grid lines.
• Colour palette: ColorBrewer "Set2" (colour-blind friendly).
• Figure sizes follow IEEE double-column width guidelines (~3.5" wide).
• All text uses Times New Roman equivalent (via matplotlib 'serif').
• Effect size annotations on key plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Style constants
# ──────────────────────────────────────────────────────────────────────────────

_PALETTE = sns.color_palette("Set2", 8)
_STYLE = "whitegrid"
_FIG_DPI = 300
_FONT_FAMILY = "serif"
_TITLE_FONTSIZE = 11
_LABEL_FONTSIZE = 10
_TICK_FONTSIZE = 9
_ANNOTATION_FONTSIZE = 8
_LEGEND_FONTSIZE = 9

# IEEE double-column widths (inches)
_FIG_SINGLE = (3.5, 2.8)    # single column
_FIG_DOUBLE = (7.0, 3.5)    # double column
_FIG_SQUARE = (5.0, 4.5)    # square (for heatmaps)


def _apply_style() -> None:
    sns.set_style(_STYLE)
    plt.rcParams.update({
        "font.family": _FONT_FAMILY,
        "font.size": _LABEL_FONTSIZE,
        "axes.labelsize": _LABEL_FONTSIZE,
        "axes.titlesize": _TITLE_FONTSIZE,
        "xtick.labelsize": _TICK_FONTSIZE,
        "ytick.labelsize": _TICK_FONTSIZE,
        "legend.fontsize": _LEGEND_FONTSIZE,
        "figure.dpi": _FIG_DPI,
        "savefig.dpi": _FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def _save_fig(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save figure as both PNG and PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, format=ext, dpi=_FIG_DPI, bbox_inches="tight")
        logger.info("Figure saved: %s", path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: Score evolution line plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_score_evolution(
    df: pd.DataFrame,
    output_dir: Path,
    window: int = 20,
    model_a_id: str = "Model_A",
) -> None:
    """
    Line plot showing rolling mean score for each judge over iterations.
    Highlights Model A's received score separately.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=_FIG_DOUBLE, sharey=True)

    # ── Left: Scores given by each judge ──────────────────────
    ax = axes[0]
    judge_ids = sorted(df["judge_id"].unique())
    for idx, judge_id in enumerate(judge_ids):
        jdf = (
            df[df["judge_id"] == judge_id]
            .groupby("iteration_id")["score_given"]
            .mean()
            .reset_index()
            .sort_values("iteration_id")
        )
        rolling = jdf["score_given"].rolling(window, min_periods=1).mean()
        color = _PALETTE[idx % len(_PALETTE)]
        ax.plot(
            jdf["iteration_id"],
            rolling,
            color=color,
            linewidth=1.5,
            label=judge_id,
            alpha=0.85,
        )
    ax.set_xlabel("Iteration", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Mean Score (rolling window)", fontsize=_LABEL_FONTSIZE)
    ax.set_title("(a) Scores Given per Judge", fontsize=_TITLE_FONTSIZE)
    ax.legend(fontsize=_LEGEND_FONTSIZE, framealpha=0.7)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # ── Right: Scores received by each respondent ─────────────
    ax = axes[1]
    respondent_ids = sorted(df["respondent_id"].unique())
    for idx, resp_id in enumerate(respondent_ids):
        rdf = (
            df[df["respondent_id"] == resp_id]
            .groupby("iteration_id")["score_given"]
            .mean()
            .reset_index()
            .sort_values("iteration_id")
        )
        rolling = rdf["score_given"].rolling(window, min_periods=1).mean()
        color = _PALETTE[idx % len(_PALETTE)]
        lw = 2.2 if resp_id == model_a_id else 1.5
        ls = "-" if resp_id == model_a_id else "--"
        ax.plot(
            rdf["iteration_id"],
            rolling,
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=resp_id + (" ⟵A" if resp_id == model_a_id else ""),
            alpha=0.85,
        )
    ax.set_xlabel("Iteration", fontsize=_LABEL_FONTSIZE)
    ax.set_title("(b) Scores Received per Respondent", fontsize=_TITLE_FONTSIZE)
    ax.legend(fontsize=_LEGEND_FONTSIZE, framealpha=0.7)
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.suptitle(
        f"Score Evolution Over Iterations (window={window})",
        fontsize=_TITLE_FONTSIZE + 1,
        fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, output_dir, "01_score_evolution")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Drift curve for Model A
# ──────────────────────────────────────────────────────────────────────────────

def plot_drift_curve_a(
    drift_received: "DriftMetric",  # noqa: F821
    drift_given: "DriftMetric",     # noqa: F821
    output_dir: Path,
    model_a_id: str = "Model_A",
) -> None:
    """
    Dual-panel drift curves for Model A:
    Left: Scores received by A (from B, C, D)
    Right: Scores given by A (to B, C, D)
    Includes OLS trend line and confidence band.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=_FIG_DOUBLE)

    for ax, drift, title, color_idx in [
        (axes[0], drift_received, f"(a) {model_a_id} Received Scores", 0),
        (axes[1], drift_given, f"(b) {model_a_id} Given Scores", 1),
    ]:
        x = drift.iteration_ids
        y_raw = drift.raw_scores
        y_roll = drift.rolling_mean
        y_std = drift.rolling_std
        color = _PALETTE[color_idx]

        # Raw scores (light scatter)
        ax.scatter(x, y_raw, alpha=0.2, s=4, color=color, zorder=1)

        # Rolling mean
        ax.plot(x, y_roll, color=color, linewidth=1.8, label="Rolling mean", zorder=3)

        # Confidence band (± 1 std)
        ax.fill_between(
            x,
            np.clip(y_roll - y_std, 0, 100),
            np.clip(y_roll + y_std, 0, 100),
            alpha=0.15,
            color=color,
            zorder=2,
            label="±1 SD",
        )

        # OLS trend line
        if not np.isnan(drift.ols_slope):
            y_trend = drift.ols_intercept + drift.ols_slope * x
            ax.plot(
                x, np.clip(y_trend, 0, 100),
                color="black", linewidth=1.0, linestyle="--",
                label=f"OLS: β₁={drift.ols_slope:.4f} (p={drift.ols_p_value:.3f})",
                zorder=4,
            )

        ax.set_xlabel("Iteration", fontsize=_LABEL_FONTSIZE)
        ax.set_ylabel("Score", fontsize=_LABEL_FONTSIZE)
        ax.set_title(title, fontsize=_TITLE_FONTSIZE)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=_ANNOTATION_FONTSIZE, framealpha=0.8)
        ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        # Annotation box with key stats
        stats_text = (
            f"slope: {drift.ols_slope:.5f}\n"
            f"R²: {drift.ols_r_squared:.3f}\n"
            f"p: {drift.ols_p_value:.4f}"
        )
        ax.text(
            0.03, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=_ANNOTATION_FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    fig.suptitle(
        f"Judge Drift Curves — {model_a_id} (asymmetry focus)",
        fontsize=_TITLE_FONTSIZE + 1, fontweight="bold",
    )
    plt.tight_layout()
    _save_fig(fig, output_dir, "02_drift_curve_model_a")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Judge Severity Index
# ──────────────────────────────────────────────────────────────────────────────

def plot_judge_severity(
    jsi: "JudgeSeverityIndex",  # noqa: F821
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Horizontal bar chart of JSI per judge with 95% CI error bars.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIG_SINGLE)

    judge_ids = sorted(jsi.per_judge.keys())
    jsi_vals = [jsi.per_judge[j] for j in judge_ids]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(judge_ids))]

    # Compute 95% CI for each judge's mean
    ci_lowers, ci_uppers = [], []
    for j in judge_ids:
        scores = df[df["judge_id"] == j]["score_given"].dropna().values
        if len(scores) < 2:
            ci_lowers.append(0)
            ci_uppers.append(0)
            continue
        se = np.std(scores, ddof=1) / np.sqrt(len(scores))
        ci = 1.96 * se  # 95% CI half-width
        ci_lowers.append(ci)
        ci_uppers.append(ci)

    y_pos = range(len(judge_ids))
    bars = ax.barh(
        y_pos, jsi_vals,
        xerr=[ci_lowers, ci_uppers],
        color=colors,
        align="center",
        height=0.6,
        alpha=0.85,
        error_kw={"ecolor": "black", "capsize": 3, "linewidth": 1.0},
    )
    ax.axvline(x=0, color="black", linewidth=1.0, linestyle="-")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(judge_ids, fontsize=_TICK_FONTSIZE)
    ax.set_xlabel("JSI = μ_judge − μ_grand", fontsize=_LABEL_FONTSIZE)
    ax.set_title("Judge Severity Index (JSI) with 95% CI", fontsize=_TITLE_FONTSIZE)

    # Annotate each bar with the JSI value
    for rect, val in zip(bars, jsi_vals):
        w = rect.get_width()
        x_ann = w + 0.3 if w >= 0 else w - 0.3
        ha = "left" if w >= 0 else "right"
        ax.annotate(
            f"{val:+.2f}",
            xy=(w, rect.get_y() + rect.get_height() / 2),
            xytext=(x_ann, rect.get_y() + rect.get_height() / 2),
            ha=ha, va="center", fontsize=_ANNOTATION_FONTSIZE,
        )

    # Grand mean annotation
    ax.text(
        0.98, 0.02,
        f"Grand mean: {jsi.grand_mean:.2f}",
        transform=ax.transAxes, fontsize=_ANNOTATION_FONTSIZE,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.9),
    )
    plt.tight_layout()
    _save_fig(fig, output_dir, "03_judge_severity_index")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Scoring matrix heatmap (judge × respondent mean scores)
# ──────────────────────────────────────────────────────────────────────────────

def plot_scoring_matrix_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Heatmap of mean score awarded by each judge to each respondent.
    Rows = judges, Columns = respondents, Values = mean score_given.
    """
    _apply_style()
    pivot = df.pivot_table(
        index="judge_id",
        columns="respondent_id",
        values="score_given",
        aggfunc="mean",
    ).round(1)

    n_rows, n_cols = pivot.shape
    fig_h = max(2.5, n_rows * 0.7)
    fig_w = max(3.5, n_cols * 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        linewidths=0.5,
        square=False,
        cbar_kws={"label": "Mean Score", "shrink": 0.8},
        annot_kws={"size": _ANNOTATION_FONTSIZE + 1},
    )
    ax.set_xlabel("Respondent", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Judge", fontsize=_LABEL_FONTSIZE)
    ax.set_title("Mean Scoring Matrix (Judge × Respondent)", fontsize=_TITLE_FONTSIZE)

    # Rotate axis labels for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=_TICK_FONTSIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=_TICK_FONTSIZE)

    plt.tight_layout()
    _save_fig(fig, output_dir, "04_scoring_matrix_heatmap")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Score inflation trend (per-judge OLS lines)
# ──────────────────────────────────────────────────────────────────────────────

def plot_inflation_trend(
    df: pd.DataFrame,
    sit: "ScoreInflationTrend",  # noqa: F821
    output_dir: Path,
) -> None:
    """
    Per-judge scatter plot with OLS trend line.
    Each subplot = one judge. Slope and p-value annotated.
    """
    _apply_style()
    judge_ids = sorted(df["judge_id"].unique())
    n_judges = len(judge_ids)
    ncols = min(2, n_judges)
    nrows = (n_judges + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8), squeeze=False)

    for idx, judge_id in enumerate(judge_ids):
        ax = axes[idx // ncols][idx % ncols]
        color = _PALETTE[idx % len(_PALETTE)]

        jdf = (
            df[df["judge_id"] == judge_id]
            .groupby("iteration_id")["score_given"]
            .mean()
            .reset_index()
            .sort_values("iteration_id")
        )
        x = jdf["iteration_id"].values.astype(float)
        y = jdf["score_given"].values

        ax.scatter(x, y, alpha=0.3, s=5, color=color, zorder=1)

        # Rolling mean
        window = max(5, len(x) // 20)
        rolling = pd.Series(y).rolling(window, min_periods=1).mean().values
        ax.plot(x, rolling, color=color, linewidth=1.5, alpha=0.9, zorder=2)

        # OLS
        stats_d = sit.per_judge.get(judge_id, {})
        slope = stats_d.get("slope", float("nan"))
        intercept = stats_d.get("intercept", float("nan"))
        p_value = stats_d.get("p_value", float("nan"))
        if not np.isnan(slope):
            y_ols = intercept + slope * x
            ax.plot(x, np.clip(y_ols, 0, 100), color="black",
                    linewidth=1.0, linestyle="--", zorder=3)

        ax.set_xlim(x.min() - 5, x.max() + 5)
        ax.set_ylim(0, 105)
        ax.set_title(judge_id, fontsize=_TITLE_FONTSIZE - 1)
        ax.set_xlabel("Iteration", fontsize=_TICK_FONTSIZE)
        ax.set_ylabel("Mean Score", fontsize=_TICK_FONTSIZE)
        ax.axhline(y=50, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)

        ann = (
            f"β₁={slope:.4f}\np={p_value:.4f}"
            if not np.isnan(slope)
            else "insufficient data"
        )
        ax.text(
            0.97, 0.03, ann,
            transform=ax.transAxes, fontsize=_ANNOTATION_FONTSIZE,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    # Hide unused subplots
    for idx in range(n_judges, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Score Inflation Trend (OLS per Judge)", fontsize=_TITLE_FONTSIZE + 1, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_dir, "05_inflation_trend")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: Variance over time
# ──────────────────────────────────────────────────────────────────────────────

def plot_variance_over_time(
    vd: "VarianceDynamics",  # noqa: F821
    output_dir: Path,
) -> None:
    """
    Rolling variance of combined scores over iterations.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=_FIG_SINGLE)

    x = vd.iteration_ids
    color = _PALETTE[2]

    ax.plot(x, vd.rolling_variance, color=color, linewidth=1.8, label=f"Rolling Var (w={vd.window_size})")
    ax.fill_between(x, 0, vd.rolling_variance, alpha=0.15, color=color)

    # OLS
    if not np.isnan(vd.ols_slope):
        y_ols = vd.ols_slope * x + (vd.rolling_variance[0] - vd.ols_slope * x[0])
        ax.plot(x, np.clip(y_ols, 0, None), color="black",
                linewidth=1.0, linestyle="--",
                label=f"OLS: β₁={vd.ols_slope:.4f} (p={vd.ols_p_value:.3f})")

    ax.set_xlabel("Iteration", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Rolling Score Variance", fontsize=_LABEL_FONTSIZE)
    ax.set_title("Variance Dynamics Over Iterations", fontsize=_TITLE_FONTSIZE)
    ax.legend(fontsize=_LEGEND_FONTSIZE, framealpha=0.8)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save_fig(fig, output_dir, "06_variance_over_time")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 7: Inter-model agreement heatmap
# ──────────────────────────────────────────────────────────────────────────────

def plot_inter_model_agreement(
    imam: "InterModelAgreement",  # noqa: F821
    output_dir: Path,
) -> None:
    """
    Heatmap of Pearson correlations between judges' scores on Model A.
    Significance markers overlay each cell.
    """
    _apply_style()
    judge_ids = imam.judge_ids
    corr_df = pd.DataFrame(imam.matrix, index=judge_ids, columns=judge_ids)
    p_df = pd.DataFrame(imam.p_values, index=judge_ids, columns=judge_ids)

    n = len(judge_ids)
    fig_size = max(3.0, n * 1.1)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.eye(n, dtype=bool)
    sns.heatmap(
        corr_df,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        mask=mask,
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
        annot_kws={"size": _ANNOTATION_FONTSIZE + 2},
    )

    # Overlay significance stars
    for i, ri in enumerate(judge_ids):
        for j, rj in enumerate(judge_ids):
            if i == j:
                continue
            p = p_df.loc[ri, rj]
            if pd.isna(p):
                continue
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if star:
                ax.text(
                    j + 0.5, i + 0.85, star,
                    ha="center", va="center",
                    fontsize=_ANNOTATION_FONTSIZE - 1,
                    color="white", fontweight="bold",
                )

    ax.set_title(
        "Inter-Model Agreement Matrix (Pearson r)\n"
        "Scores on Model A — * p<0.05, ** p<0.01, *** p<0.001",
        fontsize=_TITLE_FONTSIZE,
    )
    ax.set_xlabel("Judge", fontsize=_LABEL_FONTSIZE)
    ax.set_ylabel("Judge", fontsize=_LABEL_FONTSIZE)
    plt.tight_layout()
    _save_fig(fig, output_dir, "07_inter_model_agreement")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 8: Reciprocity scatter
# ──────────────────────────────────────────────────────────────────────────────

def plot_reciprocity_scatter(
    df: pd.DataFrame,
    ri: "ReciprocityIndex",  # noqa: F821
    model_a_id: str,
    output_dir: Path,
) -> None:
    """
    For each peer model X:
    Scatter of score(A→X) vs score(X→A) across iterations.
    Includes identity line and OLS fit.
    """
    _apply_style()
    peer_ids = [m for m in df["respondent_id"].unique() if m != model_a_id]
    n = len(peer_ids)
    if n == 0:
        logger.warning("No peer models found for reciprocity scatter.")
        return

    fig, axes = plt.subplots(1, n, figsize=(n * 3.0, 2.8), squeeze=False)

    for idx, peer_id in enumerate(sorted(peer_ids)):
        ax = axes[0][idx]
        color = _PALETTE[(idx + 3) % len(_PALETTE)]

        a_gives_x = (
            df[(df["judge_id"] == model_a_id) & (df["respondent_id"] == peer_id)]
            .set_index("iteration_id")["score_given"]
        )
        x_gives_a = (
            df[(df["judge_id"] == peer_id) & (df["respondent_id"] == model_a_id)]
            .set_index("iteration_id")["score_given"]
        )
        shared = a_gives_x.index.intersection(x_gives_a.index)

        if len(shared) > 0:
            xv = a_gives_x.loc[shared].values
            yv = x_gives_a.loc[shared].values
            ax.scatter(xv, yv, alpha=0.4, s=8, color=color)

            # OLS fit (if sufficient data)
            if len(shared) >= 3:
                from scipy import stats as sp_stats
                slope, intercept, r_val, p_val, _ = sp_stats.linregress(xv, yv)
                x_range = np.linspace(0, 100, 100)
                ax.plot(x_range, np.clip(intercept + slope * x_range, 0, 100),
                        color="black", linewidth=1.0, linestyle="--")
                ax.text(
                    0.05, 0.92,
                    f"r={r_val:.2f}\np={p_val:.3f}",
                    transform=ax.transAxes,
                    fontsize=_ANNOTATION_FONTSIZE,
                    va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="gray"),
                )

        # Identity line
        ax.plot([0, 100], [0, 100], color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel(f"{model_a_id} → {peer_id}", fontsize=_TICK_FONTSIZE)
        ax.set_ylabel(f"{peer_id} → {model_a_id}", fontsize=_TICK_FONTSIZE)
        ax.set_title(f"RI({model_a_id}, {peer_id})", fontsize=_TITLE_FONTSIZE - 1)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Reciprocity Scatter (A↔X Scores)", fontsize=_TITLE_FONTSIZE + 1, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, output_dir, "08_reciprocity_scatter")


# ──────────────────────────────────────────────────────────────────────────────
# Master render function
# ──────────────────────────────────────────────────────────────────────────────

def render_all_figures(
    scores_log_path: str | Path,
    metrics_summary: "MetricsSummary",  # noqa: F821
    figures_dir: str | Path,
    model_a_id: str = "Model_A",
    rolling_window: int = 20,
) -> None:
    """
    Load scores log and render all 8 publication-grade figures.
    """
    from src.logging_utils import load_jsonl
    from src.metrics import load_scores_df

    scores_log_path = Path(scores_log_path)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering all figures to %s", figures_dir)

    try:
        df = load_scores_df(scores_log_path)
    except FileNotFoundError as e:
        logger.error("Cannot render figures: %s", e)
        return

    ms = metrics_summary

    plot_score_evolution(df, figures_dir, window=rolling_window, model_a_id=model_a_id)
    plot_drift_curve_a(ms.drift_a_received, ms.drift_a_given, figures_dir, model_a_id=model_a_id)
    plot_judge_severity(ms.judge_severity_index, df, figures_dir)
    plot_scoring_matrix_heatmap(df, figures_dir)
    plot_inflation_trend(df, ms.score_inflation_trend, figures_dir)
    plot_variance_over_time(ms.variance_dynamics, figures_dir)
    plot_inter_model_agreement(ms.inter_model_agreement, figures_dir)
    plot_reciprocity_scatter(df, ms.reciprocity_index, model_a_id, figures_dir)

    logger.info("All figures rendered successfully to %s", figures_dir)
