"""Strict, versioned identities for planned units, executable units and results; and an
append-only, validated attempt store. Stdlib only.

Three layers:
  planned unit    (UNIT_SCHEMA)  every scientific/decoding field fixed by the run manifest;
                                 runtime-only fields listed in PENDING_RUNTIME_FIELDS = null
  executable unit                the planned unit plus resolved runtime fields; resolving
                                 rejects any null required field
  result key                     sha256 over the canonical executable identity; a stored
                                 attempt can satisfy only the identical executable unit

Store: <dir>/attempts.<seg:04d>.jsonl, one new segment per writer session. On load every
record is re-validated: schema, required fields, attempt_id == sha256(result_key|index),
contiguous attempt indices per result key, payload_sha256 recomputed from the payload,
no conflicting accepted payloads, no interior corruption. A malformed last line of a
segment is preserved byte-for-byte and reported, never repaired.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from . import RESULT_SCHEMA, UNIT_SCHEMA

PLANNED_REQUIRED = (
    "unit_schema", "unit_id", "run_manifest_version", "design_version", "base_task_id",
    "dataset", "source_task_id", "model_key", "model_id", "model_revision", "tokenizer_id",
    "tokenizer_revision", "rubric_id", "rubric_block_sha256", "context", "prompt_sha256",
    "messages_sha256", "decoding", "decoding_sha256", "readout", "parser_sha256",
    "validity_policy_sha256", "generation_code_sha256", "attn_implementation", "precision",
    "retry_policy",
)
PENDING_RUNTIME_FIELDS = (
    "run_manifest_sha256", "input_token_ids_sha256", "chat_template_sha256",
    "transformers_version", "torch_version", "device_name",
)
EXECUTABLE_REQUIRED = PLANNED_REQUIRED + PENDING_RUNTIME_FIELDS
ATTEMPT_STATUSES = ("accepted", "technical_failure", "permanent_failure")


class SchemaError(ValueError):
    pass


class InteriorCorruption(RuntimeError):
    pass


class ConflictError(RuntimeError):
    pass


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_obj(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def check_planned(u: Dict[str, object]) -> None:
    if u.get("unit_schema") != UNIT_SCHEMA:
        raise SchemaError(f"unknown unit schema {u.get('unit_schema')!r}")
    missing = [k for k in PLANNED_REQUIRED if u.get(k) in (None, "")]
    if missing:
        raise SchemaError(f"planned unit missing {missing}")
    extra = set(u) - set(PLANNED_REQUIRED) - set(PENDING_RUNTIME_FIELDS) - {"provenance"}
    if extra:
        raise SchemaError(f"unknown planned-unit fields {sorted(extra)}")
    if sha256_obj(u["decoding"]) != u["decoding_sha256"]:
        raise SchemaError("decoding_sha256 does not match decoding")


def resolve(planned: Dict[str, object], runtime: Dict[str, object]) -> Dict[str, object]:
    """Planned + runtime -> executable unit. Rejects unknown or unresolved fields and any
    runtime value that would overwrite a planned field."""
    check_planned(planned)
    unknown = set(runtime) - set(PENDING_RUNTIME_FIELDS)
    if unknown:
        raise SchemaError(f"runtime may set only pending fields; got {sorted(unknown)}")
    ex = {k: v for k, v in planned.items() if k != "provenance"}
    ex.update(runtime)
    unresolved = [k for k in EXECUTABLE_REQUIRED if ex.get(k) in (None, "")]
    if unresolved:
        raise SchemaError(f"unresolved executable fields {unresolved}")
    return ex


def result_key(executable: Dict[str, object]) -> str:
    missing = [k for k in EXECUTABLE_REQUIRED if executable.get(k) in (None, "")]
    if missing:
        raise SchemaError(f"cannot key an unresolved unit: {missing}")
    return sha256_obj({"schema": RESULT_SCHEMA,
                       "identity": {k: executable[k] for k in EXECUTABLE_REQUIRED}})


def attempt_id(rkey: str, index: int) -> str:
    return hashlib.sha256(f"{rkey}\x1f{index}".encode()).hexdigest()


@dataclass
class LoadReport:
    attempts: List[dict] = field(default_factory=list)
    partial_tails: List[dict] = field(default_factory=list)
    segments: List[str] = field(default_factory=list)


def _validate_record(rec: dict) -> None:
    if rec.get("schema") != RESULT_SCHEMA:
        raise SchemaError(f"unknown result schema {rec.get('schema')!r}")
    for k in ("result_key", "unit_id", "attempt_index", "attempt_id", "status", "payload",
              "payload_sha256", "executable"):
        if k not in rec:
            raise SchemaError(f"attempt record missing {k}")
    if rec["status"] not in ATTEMPT_STATUSES:
        raise SchemaError(f"unknown status {rec['status']!r}")
    if result_key(rec["executable"]) != rec["result_key"]:
        raise SchemaError("result_key does not match the stored executable identity")
    if rec["executable"].get("unit_id") != rec["unit_id"]:
        raise SchemaError("unit_id does not match executable identity")
    if attempt_id(rec["result_key"], rec["attempt_index"]) != rec["attempt_id"]:
        raise SchemaError("attempt_id does not match result_key/attempt_index")
    if sha256_obj(rec["payload"]) != rec["payload_sha256"]:
        raise SchemaError("payload_sha256 does not match payload (tampered or corrupt)")


def load(d: Path) -> LoadReport:
    rep = LoadReport()
    next_index: Dict[str, int] = {}
    accepted: Dict[str, str] = {}
    for seg in sorted(Path(d).glob("attempts.*.jsonl")):
        rep.segments.append(seg.name)
        raw = seg.read_bytes()
        ends_nl = raw.endswith(b"\n")
        lines = raw.split(b"\n")
        body = lines[:-1] if ends_nl else lines
        if raw == b"":
            body = []
        for i, line in enumerate(body):
            last = i == len(body) - 1
            try:
                if not line.strip():
                    raise ValueError("blank line")
                rec = json.loads(line.decode("utf-8"))
                if not isinstance(rec, dict):
                    raise ValueError("not an object")
                if last and not ends_nl:
                    raise ValueError("no terminating newline")
            except ValueError as exc:
                if last:
                    rep.partial_tails.append({"segment": seg.name, "line": i + 1,
                                              "bytes": len(line), "reason": str(exc)})
                    continue
                raise InteriorCorruption(f"{seg.name} line {i + 1}: {exc}") from None
            _validate_record(rec)
            rk = rec["result_key"]
            exp = next_index.get(rk, 0)
            if rec["attempt_index"] != exp:
                raise ConflictError(f"attempt index {rec['attempt_index']} for "
                                    f"{rec['unit_id']} (expected {exp})")
            if rk in accepted:
                raise ConflictError(f"attempt recorded after acceptance for {rec['unit_id']}")
            next_index[rk] = exp + 1
            if rec["status"] == "accepted":
                accepted[rk] = rec["payload_sha256"]
            rep.attempts.append(rec)
    return rep


class Store:
    def __init__(self, d: Path):
        self.dir = Path(d)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.report = load(self.dir)
        n = len(sorted(self.dir.glob("attempts.*.jsonl")))
        self.path = self.dir / f"attempts.{n:04d}.jsonl"
        self._fh = None
        self._next: Dict[str, int] = {}
        self._accepted: Dict[str, dict] = {}
        self._final: set = set()
        for r in self.report.attempts:
            self._next[r["result_key"]] = r["attempt_index"] + 1
            if r["status"] == "accepted":
                self._accepted[r["result_key"]] = r
            if r["status"] in ("accepted", "permanent_failure"):
                self._final.add(r["result_key"])

    def is_final(self, rkey: str) -> bool:
        return rkey in self._final

    def accepted(self, rkey: str) -> Optional[dict]:
        return self._accepted.get(rkey)

    def attempts_for(self, rkey: str) -> int:
        return self._next.get(rkey, 0)

    def append(self, executable: Dict[str, object], status: str, payload: dict) -> dict:
        rk = result_key(executable)
        if rk in self._final:
            raise ConflictError(f"{executable['unit_id']} already has a final attempt")
        idx = self._next.get(rk, 0)
        rec = {"schema": RESULT_SCHEMA, "result_key": rk, "unit_id": executable["unit_id"],
               "attempt_index": idx, "attempt_id": attempt_id(rk, idx), "status": status,
               "payload": payload, "payload_sha256": sha256_obj(payload),
               "executable": executable}
        _validate_record(rec)
        if self._fh is None:
            self._fh = open(self.path, "x", encoding="utf-8")
        self._fh.write(canonical(rec) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._next[rk] = idx + 1
        if status == "accepted":
            self._accepted[rk] = rec
        if status in ("accepted", "permanent_failure"):
            self._final.add(rk)
        return rec

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
