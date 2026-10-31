"""Read-only, hash-checked public input registry; override root with RBA_INPUT_ROOT."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict

EXT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = EXT.parents[1]
PKG = "."
INPUTS = {'answers': 'code/data/candidates/final_olmo_answers.json', 'a9_tasks': 'code/data/prompts/a9_scoring_tasks_seed20260818.json', 'merged_raw': 'results/behavioral/merged_raw_results.jsonl', 'ext72_manifest': 'mechanistic/manifest/mech_ext72_manifest.json'}

EXPECTED_FILE = EXT / "config" / "inputs_v1.json"


class InputIntegrityError(RuntimeError):
    pass


def root() -> Path:
    return Path(os.environ.get("RBA_INPUT_ROOT", str(DEFAULT_ROOT))).resolve()


def path(name: str) -> Path:
    return root() / INPUTS[name]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def expected() -> Dict[str, str]:
    return json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))["expected_sha256"]


def verified(name: str) -> Path:
    """Path to an input after checking its hash; raises InputIntegrityError otherwise."""
    p = path(name)
    exp = expected().get(name)
    if exp is None:
        raise InputIntegrityError(f"no expected hash recorded for input {name!r}")
    got = sha256_file(p)
    if got != exp:
        raise InputIntegrityError(f"input {name!r} hash {got[:12]} != expected {exp[:12]}")
    return p


def load_json(name: str):
    return json.loads(verified(name).read_text(encoding="utf-8"))


def iter_jsonl(name: str):
    with open(verified(name), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
