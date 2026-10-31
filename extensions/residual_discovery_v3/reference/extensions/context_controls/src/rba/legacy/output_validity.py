"""Amendment A5: cap-aware output validity.

A response that exhausts its generation ceiling without emitting a terminator did not finish.
Any number the frozen parser extracts from such a response came out of the middle of an
unfinished evaluation, not from a final verdict -- so it is not a score, however
score-shaped it looks. A5 makes that judgement explicit and applies it before analysis.

Measured basis (36-call condition-blind preflight, OLMo-3.1-32B): four responses that hit the
512 ceiling nonetheless parsed, and when the same generations were continued to completion the
scores were 10.0 -> 9.0, 1.0 -> 9.5, 2.5 -> 9.75 and 10.0 -> 9.8. The parser had latched onto
an interim number. Two of those four were off by more than seven points on a ten-point scale.

This module does not touch the frozen parser and does not re-parse anything. It reads what the
parser and the generation call already reported and decides whether the observation may enter
the analysis. Deliberately pure and record-shaped so it is testable without a GPU.

Design rules:
  - capped_incomplete => invalid for analysis, even with a parsed score;
  - the raw response, the parser's numeric match and all cap metadata are RETAINED for audit;
  - capped_incomplete never enters human score recovery, because an explicitly final score
    cannot be inferred from an unfinished generation -- there is nothing to recover;
  - recovery is unchanged for uncapped parser failures;
  - a response that terminated exactly on the ceiling boundary is complete, not capped.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

CAPPED_INCOMPLETE = "capped_incomplete"
VALID = "valid"
PARSE_FAILED = "parse_failed"
GENERATION_ERROR = "generation_error"


def emitted_terminator(record: Dict[str, Any]) -> Optional[bool]:
    """True/False when the generation path reported it; None when unknown.

    Unknown is treated conservatively by is_capped_incomplete: a record that hit the ceiling
    but carries no terminator evidence is assumed unfinished, because the alternative would
    silently admit exactly the observations A5 exists to exclude."""
    v = record.get("emitted_terminator")
    return None if v is None else bool(v)


def is_capped_incomplete(record: Dict[str, Any]) -> bool:
    """The A5 predicate: generation exhausted max_new_tokens without terminating.

    Terminator presence is authoritative, so a response whose final token IS the terminator
    is complete even though it used the whole ceiling -- that is a response that finished on
    the boundary, not one that was cut off."""
    if not record.get("hit_token_cap"):
        return False
    return emitted_terminator(record) is not True


def classify(record: Dict[str, Any]) -> Dict[str, Any]:
    """Decide analysis validity for one scoring record.

    Returns the validity verdict plus the analysis score. Never mutates the input, never
    invents a score, and never discards the parser's own output -- `parsed_score` stays
    exactly as the frozen parser left it, for auditing."""
    capped = is_capped_incomplete(record)
    parse_ok = bool(record.get("parse_ok"))
    parsed = record.get("parsed_score")

    if record.get("generation_error"):
        status, valid = GENERATION_ERROR, False
    elif capped:
        # Invalid even when parse_ok is True and parsed_score is a number in range.
        status, valid = CAPPED_INCOMPLETE, False
    elif parse_ok and parsed is not None:
        status, valid = VALID, True
    else:
        status, valid = PARSE_FAILED, False

    return {
        "analysis_validity": status,
        "analysis_valid": valid,
        # Missing, not imputed and not borrowed from parsed_score.
        "analysis_score": parsed if valid else None,
        "capped_incomplete": capped,
        # Retained for audit: what the parser saw, even when it is not usable as a score.
        "retained_parsed_score": parsed,
        "retained_parse_status": record.get("parse_status"),
        "retained_matched_text": record.get("matched_text") or record.get("parse_matched_text"),
        "output_tokens": record.get("output_tokens"),
        "hit_token_cap": bool(record.get("hit_token_cap")),
        "emitted_terminator": emitted_terminator(record),
        "max_new_tokens": record.get("max_new_tokens"),
        "eligible_for_human_recovery": (not capped) and (not valid)
                                       and not record.get("generation_error"),
    }


def annotate(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `record` carrying the A5 verdict. Additive: no existing field is
    removed or rewritten, so `parse_ok` and `parsed_score` keep their original meaning and
    every prior consumer still reads what it always read."""
    return {**record, **classify(record)}


def analysis_score(record: Dict[str, Any]) -> Optional[float]:
    """The value analysis may use, or None. Records already annotated are honoured as-is so a
    second pass cannot upgrade an invalid observation."""
    if "analysis_valid" in record:
        return record["analysis_score"] if record["analysis_valid"] else None
    return classify(record)["analysis_score"]


def is_valid(record: Dict[str, Any]) -> bool:
    """Whether the observation may enter a condition mean or a paired contrast.

    Reads a stored verdict when present, so resume, adjudication and analysis all reach the
    same answer for the same record rather than each re-deriving it from different fields."""
    if "analysis_valid" in record:
        return bool(record["analysis_valid"])
    return classify(record)["analysis_valid"]


def eligible_for_human_recovery(record: Dict[str, Any]) -> bool:
    """A5 point 4/5: uncapped parser failures still go to the frozen two-reader recovery
    process; capped-incomplete ones never do."""
    if "eligible_for_human_recovery" in record:
        return bool(record["eligible_for_human_recovery"])
    return classify(record)["eligible_for_human_recovery"]


def missingness_report(records) -> Dict[str, Any]:
    """A5 point 7: missingness by scorer x dataset x condition, split by reason, plus the
    retained-pair counts the frozen paired complete-case rule actually depends on."""
    from collections import defaultdict

    cells = defaultdict(lambda: {"n_total": 0, "n_valid": 0, "n_capped_incomplete": 0,
                                 "n_parse_failed": 0, "n_generation_error": 0})
    for r in records:
        v = r if "analysis_validity" in r else classify(r)
        key = (r.get("scorer_key"), r.get("dataset"), r.get("condition"))
        c = cells[key]
        c["n_total"] += 1
        if v["analysis_valid"]:
            c["n_valid"] += 1
        elif v["analysis_validity"] == CAPPED_INCOMPLETE:
            c["n_capped_incomplete"] += 1
        elif v["analysis_validity"] == GENERATION_ERROR:
            c["n_generation_error"] += 1
        else:
            c["n_parse_failed"] += 1

    out = []
    for (scorer, dataset, condition), c in sorted(
            cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        out.append({"scorer_key": scorer, "dataset": dataset, "condition": condition, **c,
                    "missing_rate": (c["n_total"] - c["n_valid"]) / c["n_total"]
                                    if c["n_total"] else None})
    return {"cells": out,
            "n_cells": len(out),
            "n_total": sum(c["n_total"] for c in cells.values()),
            "n_valid": sum(c["n_valid"] for c in cells.values()),
            "n_capped_incomplete": sum(c["n_capped_incomplete"] for c in cells.values()),
            "n_parse_failed": sum(c["n_parse_failed"] for c in cells.values()),
            "n_generation_error": sum(c["n_generation_error"] for c in cells.values())}


def retained_pairs(records, condition_a: str, condition_b: str) -> Dict[str, Any]:
    """The frozen paired complete-case rule, counted per scorer x dataset: a task contributes
    to a contrast only when BOTH of its condition scores are valid. Reported so attrition is
    visible per cell rather than only in the total."""
    from collections import defaultdict

    by_task = defaultdict(dict)
    for r in records:
        key = (r.get("scorer_key"), r.get("dataset"), r.get("source_task_id"))
        by_task[key][r.get("condition")] = is_valid(r)

    cells = defaultdict(lambda: {"n_tasks": 0, "n_retained": 0})
    for (scorer, dataset, _task), conds in by_task.items():
        if condition_a not in conds or condition_b not in conds:
            continue
        c = cells[(scorer, dataset)]
        c["n_tasks"] += 1
        c["n_retained"] += int(conds[condition_a] and conds[condition_b])

    out = []
    for (scorer, dataset), c in sorted(cells.items(),
                                       key=lambda kv: tuple(str(x) for x in kv[0])):
        out.append({"scorer_key": scorer, "dataset": dataset, **c,
                    "retention_rate": c["n_retained"] / c["n_tasks"] if c["n_tasks"] else None})
    return {"contrast": [condition_a, condition_b], "cells": out,
            "n_tasks": sum(c["n_tasks"] for c in cells.values()),
            "n_retained": sum(c["n_retained"] for c in cells.values())}
