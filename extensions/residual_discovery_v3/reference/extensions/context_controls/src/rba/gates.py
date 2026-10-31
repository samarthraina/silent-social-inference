"""Execution gates (fail closed). Every gate starts unconfirmed; only the researcher
may confirm one, by editing the gates file with evidence. Nothing in this package confirms
a gate itself.

  gpu_budget_confirmed        remaining GPU budget confirmed (prior spend known) and a stage
                              cap approved; the durable ledger enforces the numbers
  release_rules_confirmed    release requirements and the rules for including new results
  annotators_confirmed        two annotators and an adjudicator assigned, with exposure
                              recorded (needed before any packet is distributed)
  researcher_code_review  Researcher review of correction pass R1
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from .budget import GateFailure

GATES = ("gpu_budget_confirmed", "release_rules_confirmed", "annotators_confirmed",
         "researcher_code_review", "engineering_preflight_authorized",
         "mhp_spec_reviewed", "research_execution_approved",
         "mhp_qwen_spec_reviewed")
REQUIRED = {
    # Research execution is separate from permission to INCLUDE results in a release
    # ('results_inclusion' below, which needs the release requirements).
    "calibration_run": ("gpu_budget_confirmed", "research_execution_approved"),
    "results_inclusion": ("release_rules_confirmed",),
    "annotation_distribution": ("annotators_confirmed", "release_rules_confirmed",
                                "researcher_code_review"),
    # Scoped: engineering-only units (readout 'engineering_timing_only'), bounded by a
    # provisional ledger allowance. Confirms nothing about budget, rules or annotation.
    "engineering_preflight": ("engineering_preflight_authorized",),
    # Matched-history patching study (rba-mhp-v1): blocked until Researcher reviews the frozen
    # specification and a budget/stage cap is confirmed.
    "mhp_study_run": ("gpu_budget_confirmed", "mhp_spec_reviewed"),
    "mhp_qwen_study_run": ("gpu_budget_confirmed", "mhp_qwen_spec_reviewed"),
}


def blank() -> Dict[str, object]:
    return {"record_id": "rba_execution_gates", "version": "v1",
            "note": "All gates unconfirmed. Only the researcher may confirm a gate, "
                    "with evidence. This file authorizes nothing by itself.",
            "gates": {g: {"confirmed": False, "evidence": None, "confirmed_by": None}
                      for g in GATES}}


_TOKEN = object()      # module-private: Authorization can only be minted by require()


@dataclass(frozen=True)
class Authorization:
    action: str
    gates_file_sha256: str
    confirmed_by: tuple
    _token: object = None

    def __post_init__(self):
        if self._token is not _TOKEN:
            raise GateFailure("Authorization can only be issued by gates.require()")


def check_authorization(auth, action) -> None:
    """Raise unless `auth` is a genuine Authorization for `action` (or one of a tuple)."""
    allowed = (action,) if isinstance(action, str) else tuple(action)
    if not isinstance(auth, Authorization) or auth._token is not _TOKEN:
        raise GateFailure(f"{allowed[0]} requires an Authorization from gates.require()")
    if auth.action not in allowed:
        raise GateFailure(f"authorization is for {auth.action!r}, not {allowed[0]!r}")


def require(path: Path, action: str) -> "Authorization":
    """Raise GateFailure unless every gate required for `action` is confirmed with evidence;
    otherwise return an Authorization that the real backend requires."""
    if action not in REQUIRED:
        raise GateFailure(f"unknown action {action!r}")
    p = Path(path)
    if not p.exists():
        raise GateFailure("no execution-gates file")
    d = json.loads(p.read_text(encoding="utf-8"))
    missing = []
    for g in REQUIRED[action]:
        rec = d.get("gates", {}).get(g) or {}
        if not (rec.get("confirmed") is True and rec.get("evidence") and rec.get("confirmed_by")):
            missing.append(g)
    if missing:
        raise GateFailure(f"{action} blocked; unconfirmed gates: {missing}")
    return Authorization(action, hashlib.sha256(p.read_bytes()).hexdigest(),
                         tuple(d["gates"][g]["confirmed_by"] for g in REQUIRED[action]), _TOKEN)
