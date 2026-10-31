"""Durable aggregate rented-GPU-hour ledger (stdlib only).

Append-only JSONL of events, fsync'd on every write, so accounting survives restarts:

  prior_spend   hours spent before this ledger existed, with status 'verified' or
                'owner_reported'; an 'unknown' entry blocks every new allocation
  open          rental started: allocation id, stage, positive integer device count, time
  close         release CONFIRMED by the operator/provider, with time
  stop          a gate or budget stop, with reason (informational)

Charged hours = wall time from 'open' (rental start, so setup, download, loading and idle
time are included) to 'close', times device count. An allocation that was never closed --
for example because the process crashed -- keeps accruing up to "now" after a restart, until a
close is recorded. The ceiling is 30 aggregate A100 hours for the whole project, the 4-hour
protected reserve stays locked unless the researcher releases it, and every check includes a
shutdown margin. BudgetExhausted and GateFailure are never caught by the runner.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import LEDGER_SCHEMA

TOTAL_CEILING_HOURS = 30.0
PROTECTED_RESERVE_HOURS = 4.0


class BudgetExhausted(RuntimeError):
    pass


class GateFailure(RuntimeError):
    pass


class LedgerError(RuntimeError):
    pass


@dataclass
class Ledger:
    path: Path
    clock: Callable[[], float] = time.time
    stage_caps: Optional[Dict[str, float]] = None
    shutdown_margin_hours: float = 0.25
    reserve_released: bool = False

    def __post_init__(self):
        self.path = Path(self.path)
        self.events: List[dict] = []
        if self.path.exists():
            raw = self.path.read_bytes()
            lines = raw.split(b"\n")
            if raw and not raw.endswith(b"\n"):
                raise LedgerError("ledger has an unterminated last line; inspect before use")
            for i, line in enumerate(lines[:-1] if raw else []):
                ev = json.loads(line)
                if ev.get("schema") != LEDGER_SCHEMA:
                    raise LedgerError(f"line {i + 1}: unknown ledger schema")
                self.events.append(ev)
        self._check_consistency()

    # ---- persistence -------------------------------------------------------------------
    def _write(self, ev: dict) -> dict:
        ev = dict(ev, schema=LEDGER_SCHEMA, recorded_at=self.clock())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.events.append(ev)
        return ev

    def _check_consistency(self):
        opened, closed = {}, set()
        for ev in self.events:
            if ev["type"] == "open":
                if ev["allocation_id"] in opened:
                    raise LedgerError("duplicate allocation id")
                opened[ev["allocation_id"]] = ev
            elif ev["type"] == "close":
                if ev["allocation_id"] not in opened or ev["allocation_id"] in closed:
                    raise LedgerError("close without a matching open allocation")
                closed.add(ev["allocation_id"])

    # ---- events ------------------------------------------------------------------------
    def record_prior_spend(self, hours: Optional[float], status: str, source: str) -> dict:
        if status not in ("verified", "owner_reported", "unknown"):
            raise LedgerError("prior spend status must be verified, owner_reported or unknown")
        if status != "unknown" and (hours is None or hours < 0):
            raise LedgerError("known prior spend needs non-negative hours")
        return self._write({"type": "prior_spend", "hours": hours, "status": status,
                            "source": source})

    def record_provisional_authorization(self, stage: str, hours: float, source: str,
                                         authorized_by: str) -> dict:
        """A researcher-granted, stage-scoped allowance usable while prior spend is UNKNOWN.
        It does not record prior spend as zero and unlocks no other stage."""
        if not (isinstance(hours, (int, float)) and 0 < hours <= 8.0):
            raise LedgerError("provisional allowance must be in (0, 8] hours")
        if not source or not authorized_by:
            raise LedgerError("provisional allowance needs a source and an authorizer")
        return self._write({"type": "provisional_authorization", "stage": stage,
                            "hours": float(hours), "source": source,
                            "authorized_by": authorized_by})

    def provisional_hours(self, stage: str):
        hs = [e["hours"] for e in self.events
              if e["type"] == "provisional_authorization" and e["stage"] == stage]
        return min(hs) if hs else None

    def open(self, stage: str, devices: int, note: str = "",
             projected_hours: float = 0.0, started_at: float = None) -> str:
        if not isinstance(devices, int) or isinstance(devices, bool) or devices < 1:
            raise LedgerError("device count must be a positive integer")
        if self.open_allocations():
            raise LedgerError("another allocation is still open; confirm its release first")
        start = self.clock() if started_at is None else float(started_at)
        if start > self.clock():
            raise LedgerError("allocation cannot start in the future")
        aid = uuid.uuid4().hex
        # check with the already-elapsed time (backdated to rental start) included
        self.check(stage, projected_hours + max(0.0, self.clock() - start) / 3600.0 * devices)
        self._write({"type": "open", "allocation_id": aid, "stage": stage, "devices": devices,
                     "start": start, "note": note})
        return aid

    def close(self, allocation_id: str, confirmed_by: str, end_at: float = None) -> dict:
        opened = {e["allocation_id"]: e for e in self.open_allocations()}
        if allocation_id not in opened:
            raise LedgerError("allocation is not open")
        end = self.clock() if end_at is None else float(end_at)
        if end < opened[allocation_id]["start"] or end > self.clock():
            raise LedgerError("close time must lie between allocation start and now")
        return self._write({"type": "close", "allocation_id": allocation_id,
                            "end": end, "confirmed_by": confirmed_by})

    def note(self, text: str) -> dict:
        return self._write({"type": "note", "text": text})

    def stop(self, reason: str) -> dict:
        return self._write({"type": "stop", "reason": reason})

    # ---- accounting -------------------------------------------------------------------
    def open_allocations(self) -> List[dict]:
        closed = {e["allocation_id"] for e in self.events if e["type"] == "close"}
        return [e for e in self.events if e["type"] == "open" and e["allocation_id"] not in closed]

    def prior_spend(self) -> float:
        return sum(e["hours"] for e in self.events
                   if e["type"] == "prior_spend" and e["status"] != "unknown")

    def has_unknown_prior_spend(self) -> bool:
        return any(e["type"] == "prior_spend" and e["status"] == "unknown" for e in self.events)

    def spent(self, stage: Optional[str] = None) -> float:
        now = self.clock()
        ends = {e["allocation_id"]: e["end"] for e in self.events if e["type"] == "close"}
        tot = 0.0
        for e in self.events:
            if e["type"] != "open" or (stage is not None and e["stage"] != stage):
                continue
            end = ends.get(e["allocation_id"], now)
            tot += max(0.0, end - e["start"]) / 3600.0 * e["devices"]
        if stage is None:
            tot += self.prior_spend()
        return tot

    def usable_total(self) -> float:
        return TOTAL_CEILING_HOURS - (0.0 if self.reserve_released else PROTECTED_RESERVE_HOURS)

    def check(self, stage: str, projected_hours: float = 0.0) -> None:
        """Raise before work that would exceed a cap, including the shutdown margin."""
        prov = self.provisional_hours(stage)
        margin = self.shutdown_margin_hours
        if prov is not None:
            # Researcher-granted, stage-scoped allowance: bounded by its own hours only.
            if self.spent(stage) + projected_hours + margin > prov + 1e-12:
                raise BudgetExhausted(f"stage {stage!r} would exceed its provisional "
                                      f"{prov} h allowance (including {margin} h margin)")
            return
        if self.has_unknown_prior_spend():
            raise GateFailure("prior GPU spending is unknown; record verified or owner-reported "
                              "hours before any allocation")
        if not any(e["type"] == "prior_spend" for e in self.events):
            raise GateFailure("no prior-spend record; inventory earlier spending first")
        if self.stage_caps is None or stage not in self.stage_caps:
            raise GateFailure(f"no approved cap for stage {stage!r}")
        if self.spent(stage) + projected_hours + margin > self.stage_caps[stage] + 1e-12:
            raise BudgetExhausted(f"stage {stage!r} would exceed its {self.stage_caps[stage]} h cap "
                                  f"(including {margin} h shutdown margin)")
        if self.spent() + projected_hours + margin > self.usable_total() + 1e-12:
            raise BudgetExhausted(f"aggregate would exceed {self.usable_total()} usable hours")

    def summary(self) -> Dict[str, object]:
        return {"ceiling_hours": TOTAL_CEILING_HOURS, "reserve_locked": not self.reserve_released,
                "prior_spend_hours": self.prior_spend(),
                "unknown_prior_spend": self.has_unknown_prior_spend(),
                "spent_hours_including_prior": round(self.spent(), 6),
                "open_allocations": len(self.open_allocations())}


class FakeClock:
    def __init__(self, t: float = 1_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds
