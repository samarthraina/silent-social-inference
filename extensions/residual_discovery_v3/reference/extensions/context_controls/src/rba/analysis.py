"""Prespecified analysis code for both studies (stdlib only). Run only on synthetic data in
this phase. See artifacts/ANALYSIS_PLAN_v1.md for the plan these functions implement.

Calibration: only VALID outcomes (A5) enter score cells; invalid and missing cells stay
explicit. Estimates are reported (a) complete-case with a task-clustered bootstrap and (b)
as worst/best-case bounds over ALL planned tasks using the 0-10 outcome range.

Audit: the per-justification primary endpoint (>=1 objectively incorrect criticism) is
compared self_low_2 vs no_history within task x scorer pairs; unassessable or missing
items stay in planned denominators and enter [0,1] bounds.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

OUTCOME_RANGE = (0.0, 10.0)
CONTEXTS = ("no_history", "self_low_2", "unrelated_low_2")
RUBRICS = ("original", "explicit_v1")


class AnalysisInputError(ValueError):
    pass


# ---- calibration --------------------------------------------------------------------------

def cells_from_records(planned: Sequence[dict], accepted_payloads: Mapping[str, dict]
                       ) -> Dict[Tuple[str, str, str, str], dict]:
    """(model, task, rubric, context) -> {'score': float|None, 'status': str}.

    `accepted_payloads` maps unit_id -> the final attempt payload. Planned units with no
    final payload are 'missing'. Scores are taken only when analysis_valid is True."""
    out = {}
    for u in planned:
        if u.get("readout") != "legacy_generated_score":
            raise AnalysisInputError("mixed readouts")
        k = (u["model_key"], u["base_task_id"], u["rubric_id"], u["context"])
        if k in out:
            raise AnalysisInputError(f"conflicting duplicate planned cell {k}")
        p = accepted_payloads.get(u["unit_id"])
        if p is None:
            out[k] = {"score": None, "status": "missing"}
        elif p.get("analysis_valid") is True:
            s = p.get("analysis_score")
            if s is None or not (OUTCOME_RANGE[0] <= float(s) <= OUTCOME_RANGE[1]):
                raise AnalysisInputError(f"valid record without in-range score {k}")
            out[k] = {"score": float(s), "status": "valid"}
        else:
            out[k] = {"score": None, "status": p.get("analysis_validity") or "invalid"}
    return out


# A contrast is a dict {(rubric, context): coefficient}.
PRIMARY = {("original", "no_history"): 1, ("original", "self_low_2"): -1,
           ("explicit_v1", "no_history"): -1, ("explicit_v1", "self_low_2"): 1}
SECONDARY = {
    "unrelated_low_interaction": {("original", "no_history"): 1,
                                  ("original", "unrelated_low_2"): -1,
                                  ("explicit_v1", "no_history"): -1,
                                  ("explicit_v1", "unrelated_low_2"): 1},
    "self_vs_unrelated_interaction": {("original", "unrelated_low_2"): 1,
                                      ("original", "self_low_2"): -1,
                                      ("explicit_v1", "unrelated_low_2"): -1,
                                      ("explicit_v1", "self_low_2"): 1},
    "self_low_effect_original": {("original", "no_history"): 1, ("original", "self_low_2"): -1},
    "self_low_effect_explicit": {("explicit_v1", "no_history"): 1,
                                 ("explicit_v1", "self_low_2"): -1},
    "unrelated_low_effect_original": {("original", "no_history"): 1,
                                      ("original", "unrelated_low_2"): -1},
    "unrelated_low_effect_explicit": {("explicit_v1", "no_history"): 1,
                                      ("explicit_v1", "unrelated_low_2"): -1},
    "baseline_shift_no_history": {("explicit_v1", "no_history"): 1,
                                  ("original", "no_history"): -1},
}


def task_contrast(cells, model, task, contrast) -> Optional[float]:
    vals = []
    for (r, c), w in contrast.items():
        s = cells[(model, task, r, c)]["score"]
        if s is None:
            return None
        vals.append(w * s)
    return float(sum(vals))


def task_bounds(cells, model, task, contrast) -> Tuple[float, float]:
    lo = hi = 0.0
    for (r, c), w in contrast.items():
        s = cells[(model, task, r, c)]["score"]
        if s is not None:
            lo += w * s
            hi += w * s
        else:
            lo += w * (OUTCOME_RANGE[0] if w > 0 else OUTCOME_RANGE[1])
            hi += w * (OUTCOME_RANGE[1] if w > 0 else OUTCOME_RANGE[0])
    return lo, hi


def bootstrap_mean(values: Sequence[float], n_boot: int, seed: int,
                   levels=(0.95, 0.975)) -> Dict[str, object]:
    if not values:
        return {"estimate": None, "n": 0, "intervals": {}}
    est = sum(values) / len(values)
    rng = random.Random(seed)
    boots = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                   for _ in range(n_boot))
    iv = {}
    for lv in levels:
        a = (1 - lv) / 2
        iv[f"{lv:.3f}"] = (boots[int(a * n_boot)], boots[min(n_boot - 1, int((1 - a) * n_boot))])
    return {"estimate": est, "n": len(values), "intervals": iv}


def estimate(cells, model, tasks: Sequence[str], contrast, n_boot=10000, seed=20261028
             ) -> Dict[str, object]:
    per_task = {t: task_contrast(cells, model, t, contrast) for t in tasks}
    complete = [v for v in per_task.values() if v is not None]
    bnds = [task_bounds(cells, model, t, contrast) for t in tasks]
    return {"planned_tasks": len(tasks), "complete_tasks": len(complete),
            "complete_case": bootstrap_mean(complete, n_boot, seed),
            "planned_population_bounds": (sum(b[0] for b in bnds) / len(tasks),
                                          sum(b[1] for b in bnds) / len(tasks)),
            "per_task": per_task}


def cell_table(cells, model, tasks):
    out = {}
    for r in RUBRICS:
        for c in CONTEXTS:
            vals = [cells[(model, t, r, c)] for t in tasks]
            v = [x["score"] for x in vals if x["score"] is not None]
            st = {}
            for x in vals:
                st[x["status"]] = st.get(x["status"], 0) + 1
            out[f"{r}|{c}"] = {"planned": len(tasks), "valid": len(v),
                               "mean_valid": (sum(v) / len(v)) if v else None,
                               "status_counts": st}
    return out
