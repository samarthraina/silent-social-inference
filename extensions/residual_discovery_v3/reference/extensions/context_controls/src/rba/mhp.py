"""Matched-history patching (feedback-valence follow-up), rba-mhp-v1. Stdlib-only planning
and analysis; the GPU executor is scripts/run_mhp.py and is gated separately.

Question: does the previously selected Llama site (legacy layer 24 = hidden_states[24] =
output of decoder block 23, final input position) contribute to the behavioral difference
between POSITIVE and NEGATIVE attributed feedback when a history sentence is present in
both prompts (self_high_10 vs self_low_2)?

Per task (10 generations, plus 2 prefill-only capture passes):
  baseline_high, baseline_low                          unpatched
  patch_L24_high_to_low, patch_L24_low_to_high         source final-input state -> target
  patch_L2_high_to_low,  patch_L2_low_to_high          early-layer control
  noop_high, noop_low                                  hook installed at L24, writes nothing
  same_high, same_low                                  own L24 state written back
"""
from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

MHP_VERSION = "rba-mhp-v1"
CONDS = ("self_high_10", "self_low_2")
SITE_LAYER = 24            # legacy hidden_states index; hook on model.model.layers[23]
CONTROL_LAYER = 2
POSITION = "final_input"   # last token of each NATIVE prompt, located per prompt
MODEL = {"key": "llama", "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
         "revision": "0e9e39f249a16976918f6564b8830bc894c89659"}
# Qwen variant (rba-mhp-qwen-v1): layer 27 selected in the original cross-model discovery
# (logit-lens JSD, low vs no-history); its original patching did NOT meet its criteria.
MODELS = {"llama": {**MODEL, "site_layer": 24, "version": MHP_VERSION, "disable_thinking": False},
          "qwen": {"key": "qwen", "hf_id": "Qwen/Qwen3-8B",
                   "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                   "site_layer": 27, "version": "rba-mhp-qwen-v1", "disable_thinking": True}}
DECODING = {"do_sample": False, "max_new_tokens": 512, "seed": 42, "batch_size": 1,
            "attn_implementation": "sdpa", "precision": "bfloat16"}
DIRECTIONS = (("self_high_10", "self_low_2"), ("self_low_2", "self_high_10"))
N_BOOT, SEED = 10000, 20261029


def unit_kinds(site_layer: int = SITE_LAYER) -> List[Dict[str, object]]:
    k = [{"kind": "baseline", "target": c, "source": None, "layer": None} for c in CONDS]
    for L in (site_layer, CONTROL_LAYER):
        for s, t in DIRECTIONS:
            k.append({"kind": "patch", "target": t, "source": s, "layer": L})
    k += [{"kind": "noop", "target": c, "source": None, "layer": site_layer} for c in CONDS]
    k += [{"kind": "same_state", "target": c, "source": c, "layer": site_layer} for c in CONDS]
    return k


def unit_id(base_task_id: str, k: Dict[str, object], version: str = MHP_VERSION) -> str:
    return "|".join([version, base_task_id, k["kind"], str(k["source"]), k["target"],
                     f"L{k['layer']}"])


def unit_key(u: Dict[str, object]) -> str:
    """Analysis key: 'baseline:<c>', 'patch:L<l>:<s>-><t>', 'noop:<c>', 'same_state:<c>'."""
    if u["kind"] == "baseline":
        return f"baseline:{u['target']}"
    if u["kind"] == "patch":
        return f"patch:L{u['layer']}:{u['source']}->{u['target']}"
    return f"{u['kind']}:{u['target']}"


def build_units(task_ids: Sequence[str], model_key: str = "llama") -> List[Dict[str, object]]:
    m = MODELS[model_key]
    model = {"key": m["key"], "hf_id": m["hf_id"], "revision": m["revision"]}
    if model_key == "llama":
        model = MODEL
    out = []
    for b in task_ids:
        for k in unit_kinds(m["site_layer"]):
            out.append({"mhp_version": m["version"], "unit_id": unit_id(b, k, m["version"]),
                        "base_task_id": b, **k, "position": POSITION, "model": model,
                        "decoding": DECODING, "readout": "legacy_generated_score"})
    return out


# ---- analysis --------------------------------------------------------------------------------

def _boot(vals: Sequence[float], level: float) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    if not vals:
        return None, None
    est = sum(vals) / len(vals)
    rng = random.Random(SEED)
    bs = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(N_BOOT))
    a = (1 - level) / 2
    return est, (bs[int(a * N_BOOT)], bs[min(N_BOOT - 1, int((1 - a) * N_BOOT))])


def analyze(scores: Mapping[Tuple[str, str], Optional[float]], task_ids: Sequence[str],
            min_gap: float = 0.0, site_layer: int = SITE_LAYER,
            exclude_tasks: Sequence[str] = ()) -> Dict[str, object]:
    """scores[(task, unit_kind_key)] -> VALID score or None (invalid/missing stays None).

    unit_kind_key: 'baseline:<cond>' or 'patch:L<layer>:<src>-><tgt>'.
    For each direction and layer, per task with valid target, source and patched scores:
      raw movement      = S_patch - S_target          (all such tasks, zero gaps included)
      gap               = S_source - S_target
      normalized        = raw / gap                   (only gap != 0 and |gap| >= min_gap)
      unchanged         = S_patch == S_target
    Paired site-minus-control normalized difference uses tasks valid at both layers.
    Normalized movement is NOT a percentage of decisions changed and NOT a mediation fraction.
    """
    excluded = sorted(set(exclude_tasks) & set(task_ids))
    task_ids = [t for t in task_ids if t not in set(excluded)]
    out: Dict[str, object] = {"planned_tasks": len(task_ids) + len(excluded), "min_gap": min_gap,
                              "site_layer": site_layer,
                              "excluded_tasks_pending_identity_investigation": excluded}
    gaps = []
    for t in task_ids:
        h, l = scores.get((t, "baseline:self_high_10")), scores.get((t, "baseline:self_low_2"))
        gaps.append(None if h is None or l is None else h - l)
    known = [g for g in gaps if g is not None]
    out["baseline_high_minus_low"] = {"n_valid": len(known),
                                      "mean": (sum(known) / len(known)) if known else None,
                                      "n_zero": sum(g == 0 for g in known),
                                      "n_negative": sum(g < 0 for g in known)}
    for s, t in DIRECTIONS:
        d = f"{s}->{t}"
        res = {}
        norm_by_layer = {}
        directed_by_layer = {}
        for L in (site_layer, CONTROL_LAYER):
            raw, norm, unchanged, zero_excl, small_excl, missing = [], {}, 0, 0, 0, 0
            directed = {}
            for task in task_ids:
                st = scores.get((task, f"baseline:{t}"))
                ss = scores.get((task, f"baseline:{s}"))
                sp = scores.get((task, f"patch:L{L}:{d}"))
                if st is None or ss is None or sp is None:
                    missing += 1
                    continue
                raw.append(sp - st)
                unchanged += (sp == st)
                gap = ss - st
                if gap == 0:
                    zero_excl += 1
                    continue
                # absolute source-directed movement, in score points (sign of the gap)
                directed[task] = (sp - st) * (1 if gap > 0 else -1)
                if abs(gap) < min_gap:          # inclusive threshold: |gap| >= min_gap kept
                    small_excl += 1
                    continue
                norm[task] = (sp - st) / gap
            norm_by_layer[L] = norm
            directed_by_layer[L] = directed
            e95 = _boot(list(norm.values()), 0.95)
            e975 = _boot(list(norm.values()), 0.975)
            d95 = _boot(list(directed.values()), 0.95)
            res[f"L{L}"] = {"n_complete": len(raw), "missing_or_invalid": missing,
                            "raw_movement_mean": (sum(raw) / len(raw)) if raw else None,
                            "unchanged_fraction": (unchanged / len(raw)) if raw else None,
                            "zero_gap_excluded": zero_excl, "small_gap_excluded": small_excl,
                            "normalized_n": len(norm), "normalized_mean": e95[0],
                            "normalized_ci95": e95[1], "normalized_ci975": e975[1],
                            "source_directed_points_n": len(directed),
                            "source_directed_points_mean": d95[0],
                            "source_directed_points_ci95": d95[1]}
        common = sorted(set(norm_by_layer[site_layer]) & set(norm_by_layer[CONTROL_LAYER]))
        diff = [norm_by_layer[site_layer][x] - norm_by_layer[CONTROL_LAYER][x] for x in common]
        e95, e975 = _boot(diff, 0.95), _boot(diff, 0.975)
        cd = sorted(set(directed_by_layer[site_layer]) & set(directed_by_layer[CONTROL_LAYER]))
        dd = [directed_by_layer[site_layer][x] - directed_by_layer[CONTROL_LAYER][x] for x in cd]
        f95 = _boot(dd, 0.95)
        res["paired_site_minus_control"] = {"n": len(diff), "mean": e95[0], "ci95": e95[1],
                                            "ci975_bonferroni_primary": e975[1],
                                            "source_directed_points_n": len(dd),
                                            "source_directed_points_mean": f95[0],
                                            "source_directed_points_ci95": f95[1]}
        out[d] = res
    return out


def identity_report(tokens: Mapping[Tuple[str, str], Optional[str]], task_ids: Sequence[str]
                    ) -> Dict[str, object]:
    """tokens[(task, key)] -> generated-token sha. No-op and same-state must equal baseline."""
    fails = []
    n = 0
    for t in task_ids:
        for c in CONDS:
            b = tokens.get((t, f"baseline:{c}"))
            for k in (f"noop:{c}", f"same_state:{c}"):
                v = tokens.get((t, k))
                if b is None or v is None:
                    continue
                n += 1
                if v != b:
                    fails.append((t, k))
    return {"checked": n, "failures": fails, "all_identical": not fails and n > 0}
