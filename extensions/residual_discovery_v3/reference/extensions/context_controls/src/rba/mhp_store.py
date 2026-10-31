"""Durable, resumable attempt storage for the patching executor.

Files in a run directory, read in this order:
  attempts.jsonl            the original (pre-recovery) file; never modified after recovery
  attempts.seg0001.jsonl    one new segment per writer session; fsync after every record
  ...
Load rules:
  * a malformed or newline-less LAST line of a file is QUARANTINED: excluded, reported with
    its byte offset and sha256, and the file itself is left byte-for-byte unchanged
  * any malformed interior line, payload-hash mismatch, or second final attempt for a unit
    raises StoreCorruption
  * attempt_index must be contiguous per unit across files
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

FINAL = ("accepted", "permanent_failure")


class StoreCorruption(RuntimeError):
    pass


def canon(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def payload_sha(p) -> str:
    return hashlib.sha256(canon(p).encode()).hexdigest()


def files(d: Path) -> List[Path]:
    d = Path(d)
    out = [d / "attempts.jsonl"] if (d / "attempts.jsonl").exists() else []
    return out + sorted(d.glob("attempts.seg*.jsonl"))


def load(d: Path) -> Tuple[List[dict], List[dict]]:
    """Return (records, quarantined_tails)."""
    recs, quarantined = [], []
    nxt: Dict[str, int] = {}
    final = set()
    for f in files(d):
        raw = f.read_bytes()
        if not raw:
            continue
        parts = raw.split(b"\n")
        body = parts[:-1] if raw.endswith(b"\n") else parts
        offset = 0
        for i, line in enumerate(body):
            last = i == len(body) - 1
            try:
                if last and not raw.endswith(b"\n"):
                    raise ValueError("no terminating newline")
                r = json.loads(line.decode("utf-8"))
                if not isinstance(r, dict) or "unit_id" not in r:
                    raise ValueError("not an attempt record")
            except ValueError as e:
                if last:
                    quarantined.append({"file": f.name, "line": i + 1, "byte_offset": offset,
                                        "bytes": len(line), "reason": str(e),
                                        "sha256": hashlib.sha256(line).hexdigest()})
                    break
                raise StoreCorruption(f"{f.name} line {i + 1}: {e}") from None
            offset += len(line) + 1
            if payload_sha(r["payload"]) != r["payload_sha256"]:
                raise StoreCorruption(f"{f.name} line {i + 1}: payload hash mismatch")
            u = r["unit_id"]
            if r["attempt_index"] != nxt.get(u, 0):
                raise StoreCorruption(f"{u}: attempt index {r['attempt_index']} != {nxt.get(u, 0)}")
            if u in final:
                raise StoreCorruption(f"{u}: attempt after a final attempt")
            nxt[u] = r["attempt_index"] + 1
            if r["status"] in FINAL:
                final.add(u)
            recs.append(r)
    return recs, quarantined


class Writer:
    """Appends to a NEW segment; earlier files are never opened for writing."""

    def __init__(self, d: Path):
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.records, self.quarantined = load(self.dir)
        self.next = {}
        self.final = set()
        for r in self.records:
            self.next[r["unit_id"]] = r["attempt_index"] + 1
            if r["status"] in FINAL:
                self.final.add(r["unit_id"])
        n = len(list(self.dir.glob("attempts.seg*.jsonl"))) + 1
        self.path = self.dir / f"attempts.seg{n:04d}.jsonl"
        self._fh = None
        if self.quarantined:
            (self.dir / f"quarantine_report_seg{n:04d}.json").write_text(
                json.dumps(self.quarantined, indent=2))

    def append(self, unit_id: str, status: str, payload: dict, recorded_utc: str) -> dict:
        if unit_id in self.final:
            raise StoreCorruption(f"{unit_id} already final; refusing duplicate")
        idx = self.next.get(unit_id, 0)
        rec = {"unit_id": unit_id, "attempt_index": idx, "status": status, "payload": payload,
               "payload_sha256": payload_sha(payload), "recorded_utc": recorded_utc}
        if self._fh is None:
            self._fh = open(self.path, "x", encoding="utf-8")
        self._fh.write(canon(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.next[unit_id] = idx + 1
        if status in FINAL:
            self.final.add(unit_id)
        return rec

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
