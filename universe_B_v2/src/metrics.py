"""
src/metrics.py
─────────────────────────────────────────────────────────────
Bias metric computation for the asymmetry experiment.

All metrics are computed from a pandas DataFrame of score records
loaded from scores_log.jsonl.

Mathematical definitions
────────────────────────

1. Judge Severity Index (JSI)
   ─────────────────────────
   For judge j, JSI(j) is the signed deviation of j's mean score
   from the grand mean across all judges:

       JSI(j) = μ_j - μ_grand

   where  μ_j = (1/N_j) Σ score(j→r)    for all respondents r
          μ_grand = (1/N) Σ_j Σ_r score(j→r)

   Interpretation:
     JSI > 0 → lenient judge (awards above-average scores)
     JSI < 0 → harsh judge (awards below-average scores)

2. Score Inflation Trend (SIT)
   ──────────────────────────
   The OLS slope β₁ of scores given by judge j over iteration index t:

       score(j, t) = β₀ + β₁·t + ε_t

   computed separately for each judge j.

   Interpretation:
     β₁ > 0 → scores inflating over time (drift upward)
     β₁ < 0 → scores deflating over time

3. Inter-Model Agreement Matrix (IMAM)
   ─────────────────────────────────
   Pairwise Pearson correlation of scores given to the same respondent
   across all iterations, for judges i and j:

       IMAM[i,j] = ρ(score(i→·), score(j→·))

   restricted to iterations where both judges evaluated the same respondent.
   Only meaningful for Model A's received scores (B, C, D all judge A).

4. Drift in A's Received Scores (DARS)
   ──────────────────────────────────
   Rolling mean (window w) of scores received by Model A over time:

       DARS(t) = (1/w) Σ_{k=t-w+1}^{t} A_received(k)

   OLS slope of DARS(t) over t gives the drift coefficient.

5. Drift in A's Given Scores (DAGS)
   ─────────────────────────────────
   Rolling mean (window w) of scores given by Model A over time:

       DAGS(t) = (1/w) Σ_{k=t-w+1}^{t} A_given(k)

   OLS slope of DAGS(t) over t gives the drift coefficient.

6. Reciprocity Index (RI)
   ──────────────────────
   For each (A, X) pair, the Pearson correlation between scores
   A awards to X and scores X awards to A, over shared iterations:

       RI(A, X) = ρ(score(A→X)_t, score(X→A)_t)

   RI > 0 indicates reciprocal scoring patterns (possible strategic
   or positional bias).

7. Variance Dynamics (VD)
   ──────────────────────
   Rolling variance (window w) of all scores at time t:

       VD(t) = Var_{k ∈ window(t)}(score_k)

   A declining variance over iterations suggests convergence
   (homogenisation); rising variance suggests divergence.

All computations use numpy/scipy/pandas — no ML dependencies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_scores_df(scores_log_path: str | Path) -> pd.DataFrame:
    """
    Load scores_log.jsonl into a DataFrame.
    Expected columns: iteration_id, judge_id, respondent_id, score_given,
                      task_template_id, task_variation_id, dimension_scores, timestamp.
    """
    path = Path(scores_log_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Scores log not found or empty: {path}")

    df = pd.read_json(path, lines=True)
    # Filter to score records only (in case events were mixed in)
    if "record_type" in df.columns:
        df = df[df["record_type"] == "score"].copy()

    # Ensure numeric types
    for col in ["iteration_id", "task_template_id", "task_variation_id",
                "score_given", "parse_attempts"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df.sort_values("iteration_id", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Loaded %d score records from %s", len(df), path.name)
    return df


def load_responses_df(responses_log_path: str | Path) -> pd.DataFrame:
    """Load responses_cache.jsonl into a DataFrame."""
    path = Path(responses_log_path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Responses log not found or empty: {path}")

    df = pd.read_json(path, lines=True)
    if "record_type" in df.columns:
        df = df[df["record_type"] == "response"].copy()

    df.sort_values("iteration_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Metric result containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class JudgeSeverityIndex:
    """JSI per judge."""
    grand_mean: float
    per_judge: Dict[str, float]       # {judge_id: JSI value}
    per_judge_mean: Dict[str, float]  # {judge_id: mean score}
    n_judgements: Dict[str, int]      # {judge_id: count}


@dataclass
class ScoreInflationTrend:
    """OLS slope + stats per judge."""
    per_judge: Dict[str, Dict]  # {judge_id: {slope, intercept, r_value, p_value, stderr}}


@dataclass
class InterModelAgreement:
    """Pairwise Pearson correlation matrix for scores on Model A."""
    judge_ids: List[str]
    matrix: np.ndarray     # shape (n_judges, n_judges)
    p_values: np.ndarray   # shape (n_judges, n_judges)

    def as_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            self.matrix,
            index=self.judge_ids,
            columns=self.judge_ids,
        )
        return df


@dataclass
class DriftMetric:
    """Rolling mean and OLS slope for a score series."""
    iteration_ids: np.ndarray
    raw_scores: np.ndarray
    rolling_mean: np.ndarray
    rolling_std: np.ndarray
    ols_slope: float
    ols_intercept: float
    ols_p_value: float
    ols_r_squared: float
    window_size: int


@dataclass
class ReciprocityIndex:
    """Pearson correlation between A→X and X→A scores."""
    pairs: Dict[str, Dict]  # {peer_model_id: {correlation, p_value, n_shared}}


@dataclass
class VarianceDynamics:
    """Rolling variance of all combined scores."""
    iteration_ids: np.ndarray
    rolling_variance: np.ndarray
    rolling_std: np.ndarray
    ols_slope: float
    ols_p_value: float
    window_size: int


@dataclass
class MetricsSummary:
    """Top-level container for all computed metrics."""
    judge_severity_index: JudgeSeverityIndex
    score_inflation_trend: ScoreInflationTrend
    inter_model_agreement: InterModelAgreement
    drift_a_received: DriftMetric
    drift_a_given: DriftMetric
    reciprocity_index: ReciprocityIndex
    variance_dynamics: VarianceDynamics
    config_hash: str = ""
    total_iterations_used: int = 0
    score_visibility_mode: int = 0  # 0=Opaque, 1=Visible


# ──────────────────────────────────────────────────────────────────────────────
# OLS helper
# ──────────────────────────────────────────────────────────────────────────────

def _ols_slope_stats(x: np.ndarray, y: np.ndarray) -> Dict:
    """Run OLS regression of y on x, return dict of statistics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r_value": float("nan"),
            "p_value": float("nan"),
            "stderr": float("nan"),
            "r_squared": float("nan"),
            "n": len(x),
        }
    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    return {
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "r_value": round(float(r_value), 4),
        "p_value": round(float(p_value), 6),
        "stderr": round(float(stderr), 6),
        "r_squared": round(float(r_value ** 2), 4),
        "n": len(x),
    }


def _rolling_stat(values: np.ndarray, window: int, stat: str = "mean") -> np.ndarray:
    """Compute rolling mean or std with min_periods=1."""
    s = pd.Series(values)
    if stat == "mean":
        return s.rolling(window, min_periods=1).mean().values
    elif stat == "std":
        return s.rolling(window, min_periods=1).std().values
    elif stat == "var":
        return s.rolling(window, min_periods=1).var().values
    raise ValueError(f"Unknown stat: {stat}")


# ──────────────────────────────────────────────────────────────────────────────
# Metric 1: Judge Severity Index
# ──────────────────────────────────────────────────────────────────────────────

def compute_judge_severity_index(df: pd.DataFrame) -> JudgeSeverityIndex:
    """
    JSI(j) = μ_j − μ_grand
    """
    grand_mean = float(df["score_given"].mean())
    per_judge_mean: Dict[str, float] = {}
    per_judge_jsi: Dict[str, float] = {}
    n_judgements: Dict[str, int] = {}

    for judge_id, group in df.groupby("judge_id"):
        mu_j = float(group["score_given"].mean())
        per_judge_mean[judge_id] = round(mu_j, 4)
        per_judge_jsi[judge_id] = round(mu_j - grand_mean, 4)
        n_judgements[judge_id] = len(group)

    return JudgeSeverityIndex(
        grand_mean=round(grand_mean, 4),
        per_judge=per_judge_jsi,
        per_judge_mean=per_judge_mean,
        n_judgements=n_judgements,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metric 2: Score Inflation Trend
# ──────────────────────────────────────────────────────────────────────────────

def compute_score_inflation_trend(df: pd.DataFrame) -> ScoreInflationTrend:
    """
    For each judge j, OLS regress score_given on iteration_id.
    Positive slope = inflation over time.
    """
    per_judge: Dict[str, Dict] = {}
    for judge_id, group in df.groupby("judge_id"):
        group_sorted = group.sort_values("iteration_id")
        x = group_sorted["iteration_id"].values
        y = group_sorted["score_given"].values
        per_judge[judge_id] = _ols_slope_stats(x, y)

    return ScoreInflationTrend(per_judge=per_judge)


# ──────────────────────────────────────────────────────────────────────────────
# Metric 3: Inter-Model Agreement Matrix
# ──────────────────────────────────────────────────────────────────────────────

def compute_inter_model_agreement(
    df: pd.DataFrame,
    respondent_id: Optional[str] = None,
) -> InterModelAgreement:
    """
    Compute pairwise Pearson correlation between judges' scores on the same respondent.

    If respondent_id is given, restrict to that respondent's data.
    Otherwise use all data (score vectors per judge, aligned by iteration_id).
    """
    if respondent_id:
        sub = df[df["respondent_id"] == respondent_id].copy()
    else:
        sub = df.copy()

    judge_ids = sorted(sub["judge_id"].unique().tolist())
    n = len(judge_ids)

    # Pivot: index=iteration_id, columns=judge_id, values=score_given
    pivot = sub.pivot_table(
        index="iteration_id",
        columns="judge_id",
        values="score_given",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=judge_ids)

    corr_matrix = np.full((n, n), np.nan)
    p_matrix = np.full((n, n), np.nan)

    for i, ji in enumerate(judge_ids):
        for j, jj in enumerate(judge_ids):
            if i == j:
                corr_matrix[i, j] = 1.0
                p_matrix[i, j] = 0.0
                continue
            x = pivot[ji].dropna()
            y = pivot[jj].dropna()
            both = x.index.intersection(y.index)
            if len(both) < 3:
                continue
            r, p = stats.pearsonr(x.loc[both].values, y.loc[both].values)
            corr_matrix[i, j] = round(float(r), 4)
            p_matrix[i, j] = round(float(p), 6)

    return InterModelAgreement(
        judge_ids=judge_ids,
        matrix=corr_matrix,
        p_values=p_matrix,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metric 4 & 5: Drift in A's Received / Given Scores
# ──────────────────────────────────────────────────────────────────────────────

def _compute_drift(
    scores: np.ndarray,
    iteration_ids: np.ndarray,
    window: int,
    label: str,
) -> DriftMetric:
    rolling_mean = _rolling_stat(scores, window, "mean")
    rolling_std = _rolling_stat(scores, window, "std")
    ols = _ols_slope_stats(iteration_ids.astype(float), rolling_mean)
    logger.debug(
        "Drift[%s]: slope=%.5f  p=%.4f  R²=%.4f",
        label, ols["slope"], ols["p_value"], ols["r_squared"],
    )
    return DriftMetric(
        iteration_ids=iteration_ids,
        raw_scores=scores,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        ols_slope=ols["slope"],
        ols_intercept=ols["intercept"],
        ols_p_value=ols["p_value"],
        ols_r_squared=ols["r_squared"],
        window_size=window,
    )


def compute_drift_a_received(
    df: pd.DataFrame,
    model_a_id: str,
    window: int = 20,
) -> DriftMetric:
    """
    DARS: rolling mean of scores received by Model A, over iterations.
    Aggregates across all judges (mean score per iteration).
    """
    sub = (
        df[df["respondent_id"] == model_a_id]
        .groupby("iteration_id")["score_given"]
        .mean()
        .reset_index()
        .sort_values("iteration_id")
    )
    return _compute_drift(
        scores=sub["score_given"].values,
        iteration_ids=sub["iteration_id"].values,
        window=window,
        label=f"A_received_window{window}",
    )


def compute_drift_a_given(
    df: pd.DataFrame,
    model_a_id: str,
    window: int = 20,
) -> DriftMetric:
    """
    DAGS: rolling mean of scores given by Model A, over iterations.
    Aggregates across all respondents (mean score per iteration).
    """
    sub = (
        df[df["judge_id"] == model_a_id]
        .groupby("iteration_id")["score_given"]
        .mean()
        .reset_index()
        .sort_values("iteration_id")
    )
    return _compute_drift(
        scores=sub["score_given"].values,
        iteration_ids=sub["iteration_id"].values,
        window=window,
        label=f"A_given_window{window}",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metric 6: Reciprocity Index
# ──────────────────────────────────────────────────────────────────────────────

def compute_reciprocity_index(
    df: pd.DataFrame,
    model_a_id: str,
) -> ReciprocityIndex:
    """
    For each peer X (i.e., B, C, D):
    RI(A, X) = ρ(score(A→X)_t, score(X→A)_t)  over shared iterations t.
    """
    pairs: Dict[str, Dict] = {}
    peer_ids = [m for m in df["judge_id"].unique() if m != model_a_id]

    for peer_id in peer_ids:
        # A→X: rows where judge=A, respondent=X
        a_gives_to_x = (
            df[(df["judge_id"] == model_a_id) & (df["respondent_id"] == peer_id)]
            .set_index("iteration_id")["score_given"]
        )
        # X→A: rows where judge=X, respondent=A
        x_gives_to_a = (
            df[(df["judge_id"] == peer_id) & (df["respondent_id"] == model_a_id)]
            .set_index("iteration_id")["score_given"]
        )
        shared_iters = a_gives_to_x.index.intersection(x_gives_to_a.index)
        n_shared = len(shared_iters)

        if n_shared < 3:
            pairs[peer_id] = {
                "correlation": float("nan"),
                "p_value": float("nan"),
                "n_shared": n_shared,
                "interpretation": "insufficient_data",
            }
            continue

        v1 = a_gives_to_x.loc[shared_iters].values
        v2 = x_gives_to_a.loc[shared_iters].values
        r, p = stats.pearsonr(v1, v2)

        interpretation = (
            "positive_reciprocity" if r > 0.2 and p < 0.05
            else "negative_reciprocity" if r < -0.2 and p < 0.05
            else "no_significant_reciprocity"
        )

        pairs[peer_id] = {
            "correlation": round(float(r), 4),
            "p_value": round(float(p), 6),
            "n_shared": n_shared,
            "interpretation": interpretation,
        }

    return ReciprocityIndex(pairs=pairs)


# ──────────────────────────────────────────────────────────────────────────────
# Metric 7: Variance Dynamics
# ──────────────────────────────────────────────────────────────────────────────

def compute_variance_dynamics(
    df: pd.DataFrame,
    window: int = 20,
) -> VarianceDynamics:
    """
    VD(t) = rolling variance of all scores at iteration t.
    """
    per_iter = (
        df.groupby("iteration_id")["score_given"]
        .mean()
        .reset_index()
        .sort_values("iteration_id")
    )
    scores = per_iter["score_given"].values
    iters = per_iter["iteration_id"].values

    rolling_var = _rolling_stat(scores, window, "var")
    rolling_std = _rolling_stat(scores, window, "std")
    ols = _ols_slope_stats(iters.astype(float), rolling_var)

    return VarianceDynamics(
        iteration_ids=iters,
        rolling_variance=rolling_var,
        rolling_std=rolling_std,
        ols_slope=ols["slope"],
        ols_p_value=ols["p_value"],
        window_size=window,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Master compute function
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    scores_log_path: str | Path,
    model_a_id: str,
    rolling_window: int = 20,
    config_hash: str = "",
    score_visibility_mode: int = 0,
) -> MetricsSummary:
    """
    Load the scores log and compute the full metric suite.

    Parameters
    ----------
    scores_log_path      : path to scores_log.jsonl
    model_a_id           : display_name of Model A (e.g. "Model_A")
    rolling_window       : window size for drift and variance metrics
    config_hash          : experiment config hash for provenance
    score_visibility_mode: 0 = Opaque, 1 = Visible (logged in output)
    """
    df = load_scores_df(scores_log_path)
    total = len(df)

    logger.info("Computing metrics over %d score records (model_a_id='%s')", total, model_a_id)

    jsi = compute_judge_severity_index(df)
    sit = compute_score_inflation_trend(df)
    imam = compute_inter_model_agreement(df, respondent_id=model_a_id)
    dars = compute_drift_a_received(df, model_a_id=model_a_id, window=rolling_window)
    dags = compute_drift_a_given(df, model_a_id=model_a_id, window=rolling_window)
    ri = compute_reciprocity_index(df, model_a_id=model_a_id)
    vd = compute_variance_dynamics(df, window=rolling_window)

    return MetricsSummary(
        judge_severity_index=jsi,
        score_inflation_trend=sit,
        inter_model_agreement=imam,
        drift_a_received=dars,
        drift_a_given=dags,
        reciprocity_index=ri,
        variance_dynamics=vd,
        config_hash=config_hash,
        total_iterations_used=int(df["iteration_id"].nunique()),
        score_visibility_mode=score_visibility_mode,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation
# ──────────────────────────────────────────────────────────────────────────────

def _safe_convert(obj):
    """Recursively make a dict/list JSON-serialisable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _safe_convert(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_convert(i) for i in obj]
    return obj


def save_metrics_summary(summary: MetricsSummary, output_path: str | Path) -> None:
    """Serialise MetricsSummary to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "config_hash": summary.config_hash,
        "total_iterations_used": summary.total_iterations_used,
        "score_visibility_mode": summary.score_visibility_mode,
        "judge_severity_index": {
            "grand_mean": summary.judge_severity_index.grand_mean,
            "per_judge_jsi": summary.judge_severity_index.per_judge,
            "per_judge_mean": summary.judge_severity_index.per_judge_mean,
            "n_judgements": summary.judge_severity_index.n_judgements,
        },
        "score_inflation_trend": {
            judge_id: stats_dict
            for judge_id, stats_dict in summary.score_inflation_trend.per_judge.items()
        },
        "inter_model_agreement": {
            "judge_ids": summary.inter_model_agreement.judge_ids,
            "correlation_matrix": _safe_convert(summary.inter_model_agreement.matrix),
            "p_value_matrix": _safe_convert(summary.inter_model_agreement.p_values),
        },
        "drift_a_received": {
            "ols_slope": summary.drift_a_received.ols_slope,
            "ols_p_value": summary.drift_a_received.ols_p_value,
            "ols_r_squared": summary.drift_a_received.ols_r_squared,
            "window_size": summary.drift_a_received.window_size,
            "rolling_mean": _safe_convert(summary.drift_a_received.rolling_mean),
        },
        "drift_a_given": {
            "ols_slope": summary.drift_a_given.ols_slope,
            "ols_p_value": summary.drift_a_given.ols_p_value,
            "ols_r_squared": summary.drift_a_given.ols_r_squared,
            "window_size": summary.drift_a_given.window_size,
            "rolling_mean": _safe_convert(summary.drift_a_given.rolling_mean),
        },
        "reciprocity_index": summary.reciprocity_index.pairs,
        "variance_dynamics": {
            "ols_slope": summary.variance_dynamics.ols_slope,
            "ols_p_value": summary.variance_dynamics.ols_p_value,
            "window_size": summary.variance_dynamics.window_size,
            "rolling_variance": _safe_convert(summary.variance_dynamics.rolling_variance),
        },
    }
    out = _safe_convert(out)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Metrics summary saved to %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Comparative Metrics (Condition 0 vs Condition 1)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ComparativeMetrics:
    """
    Statistical comparison between Opaque (cond 0) and Visible (cond 1) conditions.

    Computed from a paired-run design where identical task variations and seeds
    are run under both score_visibility_mode values.
    """
    # ── Paired t-test on A's given scores ────────────────────
    a_given_t_stat: float           # t-statistic
    a_given_p_value: float          # two-tailed p-value
    a_given_effect_size_d: float    # Cohen's d
    a_given_mean_cond0: float       # mean score given by A under condition 0
    a_given_mean_cond1: float       # mean score given by A under condition 1
    a_given_n_pairs: int            # number of paired observations

    # ── Drift slope comparison ────────────────────────────────
    drift_received_slope_cond0: float
    drift_received_slope_cond1: float
    drift_received_slope_delta: float   # cond1 − cond0

    drift_given_slope_cond0: float
    drift_given_slope_cond1: float
    drift_given_slope_delta: float      # cond1 − cond0

    # ── Reciprocity delta per peer ────────────────────────────
    reciprocity_delta: Dict[str, float]            # peer_id → r(cond1) − r(cond0)
    reciprocity_permutation_p: Dict[str, float]    # peer_id → permutation p-value

    # ── Variance shift ────────────────────────────────────────
    mean_variance_cond0: float
    mean_variance_cond1: float
    variance_shift: float                          # cond1 − cond0

    # ── Score distribution comparison ────────────────────────
    score_dist_ks_stat: float       # Kolmogorov–Smirnov statistic (A given scores)
    score_dist_ks_p_value: float    # KS test p-value


def compute_comparative_metrics(
    df_cond0: pd.DataFrame,
    df_cond1: pd.DataFrame,
    model_a_id: str,
    rolling_window: int = 20,
) -> ComparativeMetrics:
    """
    Compute all cross-condition comparative statistics.

    Parameters
    ----------
    df_cond0       : scores DataFrame from condition 0 (Opaque Awareness)
    df_cond1       : scores DataFrame from condition 1 (Visible Score Awareness)
    model_a_id     : display name of Model A (e.g. "Model_A")
    rolling_window : window for drift/variance metrics

    Returns
    -------
    ComparativeMetrics dataclass with all comparison statistics.
    """
    # ── Helper: A's given scores per iteration ────────────────
    def _a_given_per_iter(df: pd.DataFrame) -> pd.Series:
        return (
            df[df["judge_id"] == model_a_id]
            .groupby("iteration_id")["score_given"]
            .mean()
        )

    a_given_0 = _a_given_per_iter(df_cond0)
    a_given_1 = _a_given_per_iter(df_cond1)

    # ── 1. Paired t-test on A's given scores ─────────────────
    shared_iters = a_given_0.index.intersection(a_given_1.index)
    n_pairs = len(shared_iters)

    if n_pairs >= 3:
        v0 = a_given_0.loc[shared_iters].values
        v1 = a_given_1.loc[shared_iters].values
        t_stat, p_val = stats.ttest_rel(v1, v0)
        # Cohen's d for paired samples
        diff = v1 - v0
        effect_d = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else 0.0
    else:
        t_stat, p_val, effect_d = float("nan"), float("nan"), float("nan")
        v0 = a_given_0.values if len(a_given_0) > 0 else np.array([])
        v1 = a_given_1.values if len(a_given_1) > 0 else np.array([])

    # ── 2. Drift slope comparison ─────────────────────────────
    drift_rec_0 = compute_drift_a_received(df_cond0, model_a_id, rolling_window)
    drift_rec_1 = compute_drift_a_received(df_cond1, model_a_id, rolling_window)
    drift_giv_0 = compute_drift_a_given(df_cond0, model_a_id, rolling_window)
    drift_giv_1 = compute_drift_a_given(df_cond1, model_a_id, rolling_window)

    # ── 3. Reciprocity delta + permutation p-value ────────────
    ri0 = compute_reciprocity_index(df_cond0, model_a_id)
    ri1 = compute_reciprocity_index(df_cond1, model_a_id)

    reciprocity_delta: Dict[str, float] = {}
    reciprocity_perm_p: Dict[str, float] = {}

    all_peers = set(ri0.pairs.keys()) | set(ri1.pairs.keys())
    for peer in all_peers:
        r0 = ri0.pairs.get(peer, {}).get("correlation", float("nan"))
        r1 = ri1.pairs.get(peer, {}).get("correlation", float("nan"))
        if not (np.isnan(r0) or np.isnan(r1)):
            reciprocity_delta[peer] = round(float(r1 - r0), 4)
        else:
            reciprocity_delta[peer] = float("nan")

        # Permutation test: pool and reshuffle A's given scores
        a_gives_0 = (
            df_cond0[(df_cond0["judge_id"] == model_a_id) & (df_cond0["respondent_id"] == peer)]
            .set_index("iteration_id")["score_given"]
        )
        a_gives_1 = (
            df_cond1[(df_cond1["judge_id"] == model_a_id) & (df_cond1["respondent_id"] == peer)]
            .set_index("iteration_id")["score_given"]
        )
        shared_p = a_gives_0.index.intersection(a_gives_1.index)
        if len(shared_p) >= 5:
            observed_diff = float(a_gives_1.loc[shared_p].mean() - a_gives_0.loc[shared_p].mean())
            pooled = np.concatenate([a_gives_0.loc[shared_p].values, a_gives_1.loc[shared_p].values])
            n_perm = 5000
            rng = np.random.default_rng(0)
            count_exceed = 0
            for _ in range(n_perm):
                rng.shuffle(pooled)
                perm_diff = pooled[len(shared_p):].mean() - pooled[:len(shared_p)].mean()
                if abs(perm_diff) >= abs(observed_diff):
                    count_exceed += 1
            reciprocity_perm_p[peer] = round(count_exceed / n_perm, 4)
        else:
            reciprocity_perm_p[peer] = float("nan")

    # ── 4. Variance shift ─────────────────────────────────────
    vd0 = compute_variance_dynamics(df_cond0, rolling_window)
    vd1 = compute_variance_dynamics(df_cond1, rolling_window)
    mean_var_0 = float(np.nanmean(vd0.rolling_variance))
    mean_var_1 = float(np.nanmean(vd1.rolling_variance))

    # ── 5. Score distribution KS test ────────────────────────
    all_a_given_0 = (
        df_cond0[df_cond0["judge_id"] == model_a_id]["score_given"].dropna().values
    )
    all_a_given_1 = (
        df_cond1[df_cond1["judge_id"] == model_a_id]["score_given"].dropna().values
    )
    if len(all_a_given_0) >= 2 and len(all_a_given_1) >= 2:
        ks_stat, ks_p = stats.ks_2samp(all_a_given_0, all_a_given_1)
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    return ComparativeMetrics(
        a_given_t_stat=round(float(t_stat), 4),
        a_given_p_value=round(float(p_val), 6),
        a_given_effect_size_d=round(float(effect_d), 4),
        a_given_mean_cond0=round(float(v0.mean()) if len(v0) > 0 else float("nan"), 4),
        a_given_mean_cond1=round(float(v1.mean()) if len(v1) > 0 else float("nan"), 4),
        a_given_n_pairs=n_pairs,
        drift_received_slope_cond0=drift_rec_0.ols_slope,
        drift_received_slope_cond1=drift_rec_1.ols_slope,
        drift_received_slope_delta=round(drift_rec_1.ols_slope - drift_rec_0.ols_slope, 6),
        drift_given_slope_cond0=drift_giv_0.ols_slope,
        drift_given_slope_cond1=drift_giv_1.ols_slope,
        drift_given_slope_delta=round(drift_giv_1.ols_slope - drift_giv_0.ols_slope, 6),
        reciprocity_delta=reciprocity_delta,
        reciprocity_permutation_p=reciprocity_perm_p,
        mean_variance_cond0=round(mean_var_0, 4),
        mean_variance_cond1=round(mean_var_1, 4),
        variance_shift=round(mean_var_1 - mean_var_0, 4),
        score_dist_ks_stat=round(float(ks_stat), 4),
        score_dist_ks_p_value=round(float(ks_p), 6),
    )


def save_comparative_metrics(
    comp: ComparativeMetrics,
    output_path: str | Path,
) -> None:
    """Serialise ComparativeMetrics to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = {
        "a_given_paired_t_test": {
            "t_stat": comp.a_given_t_stat,
            "p_value": comp.a_given_p_value,
            "effect_size_cohens_d": comp.a_given_effect_size_d,
            "mean_cond0_opaque": comp.a_given_mean_cond0,
            "mean_cond1_visible": comp.a_given_mean_cond1,
            "n_pairs": comp.a_given_n_pairs,
        },
        "drift_slope_comparison": {
            "drift_received_slope_cond0": comp.drift_received_slope_cond0,
            "drift_received_slope_cond1": comp.drift_received_slope_cond1,
            "drift_received_slope_delta": comp.drift_received_slope_delta,
            "drift_given_slope_cond0": comp.drift_given_slope_cond0,
            "drift_given_slope_cond1": comp.drift_given_slope_cond1,
            "drift_given_slope_delta": comp.drift_given_slope_delta,
        },
        "reciprocity_delta_per_peer": comp.reciprocity_delta,
        "reciprocity_permutation_p": comp.reciprocity_permutation_p,
        "variance_shift": {
            "mean_variance_cond0": comp.mean_variance_cond0,
            "mean_variance_cond1": comp.mean_variance_cond1,
            "shift": comp.variance_shift,
        },
        "score_distribution_ks_test": {
            "ks_stat": comp.score_dist_ks_stat,
            "ks_p_value": comp.score_dist_ks_p_value,
        },
    }
    out = _safe_convert(out)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Comparative metrics saved to %s", output_path)
