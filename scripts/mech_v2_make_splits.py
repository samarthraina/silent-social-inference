"""Freeze the mech-v2 discovery/validation splits (metadata only, no outcomes consulted).

Pool = the 234 A9 study tasks minus every task already used by any mechanistic stage
(Llama discovery 12, validation 24, ext72, cross-model discovery 12) and the engineering
timing-gate tasks. Order within each benchmark is sha256(SEED + task_id); D takes the first
12 per benchmark, V the next 24. The first 4 per benchmark of D form the S1-S3 triage subset.

Usage: python scripts/mech_v2_make_splits.py <path-to-silent-influence-checkout>
"""
import ast
import hashlib
import json
import sys
from pathlib import Path

SEED = "mech_v2_splits_20261030"
N_D, N_V, N_TRIAGE = 12, 24, 4


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def main(src):
    src = Path(src)
    a9 = load(src / "code/data/prompts/a9_scoring_tasks_seed20260818.json")
    study = {f"{d}/{t}" for d, v in lit(a9["datasets"]).items() for t in lit(v["selected"])}
    mt = load(src / "mechanistic/manifest/mech_task_manifest.json")
    ext = load(src / "mechanistic/manifest/mech_ext72_manifest.json")
    cm = load(src / "mechanistic/manifest/crossmodel_discovery_12.json")
    used = {f"{x['dataset']}/{x['task_id']}" for x in mt["discovery"] + mt["validation"] + ext["tasks"] + cm["tasks"]}
    timing = {json.loads(l)["base_task_id"] for l in open(
        src / "extensions/context_controls/artifacts/engineering/timing_gate_manifest_v1.jsonl")}
    pool = study - used - timing

    answers = {f"{x['dataset']}/{x['source_task_id']}": x for x in
               load(src / "code/data/candidates/final_olmo_answers.json")["accepted"]}
    key = lambda t: hashlib.sha256((SEED + t).encode()).hexdigest()
    splits = {"triage": [], "D": [], "V": []}
    for ds in ("mtbench", "arena_hard", "wildbench"):
        ordered = sorted((t for t in pool if t.startswith(ds + "/")), key=key)
        assert len(ordered) >= N_D + N_V, (ds, len(ordered))
        splits["D"] += ordered[:N_D]
        splits["triage"] += ordered[:N_TRIAGE]
        splits["V"] += ordered[N_D:N_D + N_V]
    assert not set(splits["D"]) & set(splits["V"])

    tasks = {t: {k: answers[t][k] for k in ("dataset", "source_task_id", "category", "question",
                                             "question_sha256", "answer", "answer_sha256")}
             for t in splits["D"] + splits["V"]}
    for t, x in tasks.items():
        assert hashlib.sha256(x["answer"].encode()).hexdigest() == x["answer_sha256"], t
    out = {"seed": SEED, "pool_size": len(pool), "excluded_used": len(used), "excluded_timing": len(timing),
           "selection_inputs": ["dataset", "source_task_id"], "outcomes_consulted": False,
           "splits": splits, "tasks": tasks}
    dst = Path(__file__).resolve().parents[1] / "data/mech_v2/splits_v1.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True).encode()
    dst.write_bytes(raw)
    dst.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest() + "  splits_v1.json\n")
    print(dst, {k: len(v) for k, v in splits.items()}, "pool", len(pool))


if __name__ == "__main__":
    main(sys.argv[1])
