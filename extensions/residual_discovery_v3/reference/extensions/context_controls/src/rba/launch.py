"""The calibration execution entry point. Order is fixed and fail-closed:

  1. execution gates (gates.require -> Authorization)       BEFORE anything else
  2. frozen run manifest (hash + schema + duplicate checks)
  3. durable ledger: prior spend recorded, approved stage cap, projected cost fits
  4. backend construction (requires the Authorization) and model loading
  5. runner.run for each model in turn

Any failure in steps 1-3 raises before a backend object exists, so no model can be loaded
and nothing can be generated. Never invoked in this preparation phase.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

from . import calib as C, gates as G, runner as RN, units as U
from .budget import GateFailure, Ledger

STAGE = "context_calibration_run"


def default_backend_factory(model_key: str, authorization):
    spec = C.MODELS[model_key]
    return RN.HFBackend(model_key, spec["hf_id"], spec["revision"], spec["disable_thinking"],
                        authorization=authorization)


def launch_calibration(*, gates_path: Path, manifest_path: Path, manifest_sha256: str,
                       store_dir: Path, ledger: Ledger, verify_identity: Callable,
                       projected_unit_hours: float, unit_timeout_s: float,
                       backend_factory: Callable = default_backend_factory,
                       models=("llama", "qwen"), log=print, stage: str = STAGE) -> Dict[str, object]:
    auth = G.require(gates_path, "calibration_run")                  # 1
    units = RN.load_manifest(manifest_path, manifest_sha256)         # 2
    by_model = {m: [u for u in units if u["model_key"] == m] for m in models}
    ledger.check(stage, projected_unit_hours * len(units))           # 3 (raises)
    out = {}
    for m in models:
        backend = backend_factory(m, auth)                           # 4
        backend.load()
        store = U.Store(store_dir / m)
        try:
            out[m] = RN.run(by_model[m], manifest_sha256=manifest_sha256, backend=backend,
                            store=store, ledger=ledger, stage=stage,
                            verify_identity=verify_identity,
                            projected_unit_hours=projected_unit_hours,
                            unit_timeout_s=unit_timeout_s, log=log)  # 5
        finally:
            store.close()
    return out
