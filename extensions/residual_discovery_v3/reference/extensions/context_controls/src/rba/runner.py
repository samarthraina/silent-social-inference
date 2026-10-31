"""Execution loop for the calibration manifest. Never run against a real model in this phase.

Guarantees (tested with the deterministic MockBackend):
  * the frozen run manifest hash is checked and every unit is schema-checked first
  * duplicate planned unit ids are rejected before any call
  * each unit's prompt and messages are re-rendered from verified inputs and must match the
    planned hashes (prompt identity); a mismatch is a GateFailure
  * a unit with a final attempt (accepted or permanent failure) under the identical
    executable identity is never executed again; the done set updates after every unit
  * the budget is checked before every call; BudgetExhausted / GateFailure propagate
    immediately (a 'stop' event is written first) and no further call is made
  * technical failures (exception or empty text) are retried up to the frozen limit, each
    attempt recorded immutably; a parse failure is a valid record of an invalid outcome,
    not a technical failure, and is not retried
  * validity is decided by the vendored A5 policy: capped => invalid even if parseable
  * a unit exceeding the wall-time watchdog is recorded, then the run stops (GateFailure)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import units as U
from .budget import BudgetExhausted, GateFailure, Ledger
from .legacy import output_validity as ov
from .legacy import scaled_parsing as sp

MAX_ATTEMPTS = 3            # frozen MAX_RETRIES_PER_UNIT of the original scoring path


def load_manifest(path: Path, expected_sha256: str) -> List[dict]:
    raw = Path(path).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha256:
        raise GateFailure(f"run manifest hash {got[:12]} != frozen {expected_sha256[:12]}")
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    ids = [r["unit_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise GateFailure("duplicate planned unit ids in the run manifest")
    for r in rows:
        U.check_planned(r)
    return rows


def payload_from_generation(gen: Optional[dict], err: Optional[str], max_new_tokens: int
                            ) -> dict:
    text = None if gen is None else gen.get("text")
    pr = sp.parse_final_score(text) if err is None else sp.ParseResult(
        sp.EMPTY_OUTPUT, flagged_for_review=True, error=err)
    rec = {"generation_error": err, "response": text,
           "hit_token_cap": None if gen is None else gen.get("hit_token_cap"),
           "emitted_terminator": None if gen is None else gen.get("emitted_terminator"),
           "output_tokens": None if gen is None else gen.get("output_tokens"),
           "max_new_tokens": max_new_tokens, "parse_ok": pr.ok, "parsed_score": pr.score,
           "parse_status": pr.status, "parse_tier": pr.tier,
           "parse_matched_text": pr.matched_text}
    rec = ov.annotate(rec)
    if gen is not None:
        for k in ("prompt_tokens", "generation_seconds", "generated_token_ids_sha256",
                  "first_input_token_id", "last_input_token_id"):
            rec[k] = gen.get(k)
    return rec


def _technical_failure(text: Optional[str], err: Optional[str]) -> bool:
    return err is not None or text is None or not text.strip()


def run(units: Sequence[dict], *, manifest_sha256: str, backend, store: U.Store,
        ledger: Ledger, stage: str, verify_identity: Callable[[dict], List[dict]],
        projected_unit_hours: float, unit_timeout_s: float,
        log: Callable[[str], None] = lambda s: None) -> Dict[str, int]:
    stats = {"executed_calls": 0, "accepted": 0, "permanent_failure": 0, "skipped_final": 0}
    seen = set()
    for u in units:
        if u["unit_id"] in seen:
            raise GateFailure(f"duplicate unit {u['unit_id']} in this call")
        seen.add(u["unit_id"])
        messages = verify_identity(u)       # raises GateFailure on any prompt mismatch
        runtime = backend.runtime_fields(u, messages)
        runtime["run_manifest_sha256"] = manifest_sha256
        ex = U.resolve(u, runtime)
        rk = U.result_key(ex)
        if store.is_final(rk):
            stats["skipped_final"] += 1
            continue
        while True:
            try:
                ledger.check(stage, projected_unit_hours)
            except (BudgetExhausted, GateFailure) as exc:
                ledger.stop(f"{type(exc).__name__}: {exc}")
                raise
            err, gen = None, None
            try:
                gen = backend.generate(u, messages)
            except (BudgetExhausted, GateFailure):
                raise
            except Exception as exc:            # technical failure: recorded, maybe retried
                err = f"{type(exc).__name__}: {exc}"
            stats["executed_calls"] += 1
            payload = payload_from_generation(gen, err, u["decoding"]["max_new_tokens"])
            n_prev = store.attempts_for(rk)
            if _technical_failure(None if gen is None else gen.get("text"), err):
                status = "permanent_failure" if n_prev + 1 >= MAX_ATTEMPTS else "technical_failure"
            else:
                status = "accepted"
            store.append(ex, status, payload)
            if status == "accepted":
                stats["accepted"] += 1
            elif status == "permanent_failure":
                stats["permanent_failure"] += 1
            secs = (gen or {}).get("generation_seconds") or 0.0
            if secs > unit_timeout_s:
                msg = f"unit {u['unit_id']} took {secs:.1f}s > watchdog {unit_timeout_s}s"
                ledger.stop(msg)
                raise GateFailure(msg)
            if status != "technical_failure":
                break
    return stats


# ---- backends --------------------------------------------------------------------------------

class MockBackend:
    """Deterministic CPU stand-in. Its identity fields can never equal a real backend's, so
    mock results cannot satisfy real units."""

    def __init__(self, clock=None, seconds_per_call: float = 10.0, fail_ids=(), empty_ids=(),
                 capped_ids=(), raise_budget_on=None):
        self.clock = clock
        self.seconds = seconds_per_call
        self.fail_ids, self.empty_ids, self.capped_ids = set(fail_ids), set(empty_ids), set(capped_ids)
        self.raise_budget_on = raise_budget_on
        self.calls: List[str] = []
        self.loaded = False

    def load(self):
        self.loaded = True

    def runtime_fields(self, u, messages):
        ids_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        return {"input_token_ids_sha256": "mock:" + ids_hash, "chat_template_sha256": "mock",
                "transformers_version": "mock", "torch_version": "mock",
                "device_name": "mock-cpu"}

    def generate(self, u, messages):
        self.calls.append(u["unit_id"])
        if self.clock is not None:
            self.clock.advance(self.seconds)
        if self.raise_budget_on == u["unit_id"]:
            raise BudgetExhausted("provider reported budget exhaustion")
        if u["unit_id"] in self.fail_ids:
            raise RuntimeError("simulated CUDA error")
        if u["unit_id"] in self.empty_ids:
            return {"text": "", "output_tokens": 0, "hit_token_cap": False,
                    "emitted_terminator": True, "generation_seconds": self.seconds}
        x = int(hashlib.sha256(u["unit_id"].encode()).hexdigest(), 16) % 11
        capped = u["unit_id"] in self.capped_ids
        text = f"Evaluation for mock unit. Final Score: {x}/10"
        n = u["decoding"]["max_new_tokens"] if capped else 40
        return {"text": text, "output_tokens": n, "hit_token_cap": n >= u["decoding"]["max_new_tokens"],
                "emitted_terminator": not capped, "generation_seconds": self.seconds,
                "prompt_tokens": 100}


# Actions whose Authorization may construct/load the real backend. Annotation and results
# inclusion can never load a model.
BACKEND_ACTIONS = ("calibration_run", "engineering_preflight", "mhp_study_run",
                   "mhp_qwen_study_run")


class HFBackend:
    """Real path, reusing the vendored original generation code byte-identically
    (load_exact_model + generate_exact: bf16, sdpa, cuda:0, greedy, seed 42, batch 1).
    NOT exercised in this phase: no model weights are available or permitted."""

    def __init__(self, model_key: str, hf_id: str, revision: str, disable_thinking: bool,
                 max_new_tokens: int = 512, *, authorization=None):
        # Checked BEFORE anything is imported or loaded: no approval, no backend.
        from .gates import check_authorization
        check_authorization(authorization, BACKEND_ACTIONS)
        self.authorization = authorization
        from .legacy import exact_generation as eg
        self.eg = eg
        self.spec = eg.ExactModelSpec(key=model_key, hf_id=hf_id, revision=revision,
                                      disable_thinking=disable_thinking,
                                      max_new_tokens=max_new_tokens)
        self.loaded = None

    def load(self):
        from .gates import check_authorization
        check_authorization(self.authorization, BACKEND_ACTIONS)
        self.loaded = self.eg.load_exact_model(self.spec)

    def runtime_fields(self, u, messages):
        import torch
        import transformers
        if u["model_revision"] != self.spec.revision or u["model_id"] != self.spec.hf_id:
            raise GateFailure("unit model identity does not match the loaded model")
        enc = self.eg.encode_chat(self.loaded.tokenizer, messages,
                                  disable_thinking=self.spec.disable_thinking)
        ids = enc["input_ids"][0].tolist()
        return {"input_token_ids_sha256": hashlib.sha256(str(ids).encode()).hexdigest(),
                "chat_template_sha256": self.loaded.chat_template_sha256,
                "transformers_version": transformers.__version__,
                "torch_version": torch.__version__,
                "device_name": torch.cuda.get_device_name(0)}

    def generate(self, u, messages):
        from .gates import check_authorization
        check_authorization(self.authorization, BACKEND_ACTIONS)
        if self.authorization.action == "engineering_preflight" and \
                u.get("readout") != "engineering_timing_only":
            raise GateFailure(f"engineering-preflight authorization cannot generate study "
                              f"unit {u.get('unit_id')}")
        return self.eg.generate_exact(self.loaded, messages,
                                      max_new_tokens=u["decoding"]["max_new_tokens"],
                                      seed=u["decoding"]["seed"])
