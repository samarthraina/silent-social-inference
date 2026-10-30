"""Offline (CPU) summaries and pre-registered gate rules for mech-v2 stages.

Usage: python -m mech_v2.analysis <stage> <rows.jsonl> [--model llama] [--lo L --hi L]
"""
from __future__ import annotations

import argparse
import json
from statistics import mean

HIGH, LOW = "self_high_10", "self_low_2"
READOUT_AGREE_MIN = 0.8      # S1: readout gap sign must match greedy gap sign on >= 80% of tasks
POS_CONTROL_MIN = 0.8        # S1: knock-out-everything removes >= 80% of the gap
TRIAGE_MIN = 0.3             # S2: winner must remove >= 30% of the gap, else "distributed"


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _m(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def ko_effect(base, ko, tids, hi=HIGH, lo=LOW):
    """1 - mean knockout gap / mean baseline gap over tasks with both conditions."""
    ok = [t for t in tids if all((t, c) in base and (t, c) in ko for c in (hi, lo))]
    gb = mean(base[(t, hi)] - base[(t, lo)] for t in ok)
    gk = mean(ko[(t, hi)] - ko[(t, lo)] for t in ok)
    return None if not ok or abs(gb) < 1e-6 else 1 - gk / gb


def s1(rows):
    out = {}
    greedy = {(r["task"], r["cond"]): r["score"] for r in rows if r.get("unit") == "greedy"}
    for R in ("direct", "forced"):
        rr = [r for r in rows if r.get("R") == R]
        base = {(r["task"], r["cond"]): r["E"] for r in rr if r["unit"] == "base"}
        tids = sorted({t for t, _ in base})
        agree, gaps = [], []
        for t in tids:
            if (t, HIGH) in base and (t, LOW) in base:
                g = base[(t, HIGH)] - base[(t, LOW)]
                gaps.append(g)
                gg = None if None in (greedy.get((t, HIGH)), greedy.get((t, LOW))) else greedy[(t, HIGH)] - greedy[(t, LOW)]
                if gg:
                    agree.append((g > 0) == (gg > 0))
        ko = {(r["task"], r["cond"]): r["E"] for r in rr if r["unit"] == "ko_all"}
        moves = {}
        pairs = {(r["task"], r["src"], r["tgt"]): r for r in rr if r["unit"] == "pair"}
        for r in rr:
            if r["unit"] in ("final_input_patch", "pos_patch_L1") and "E" in r:
                p = pairs[(r["task"], r["src"], r["tgt"])]
                d = p["E_src"] - p["E_tgt"]
                if abs(d) > 1e-6:
                    moves.setdefault(f'{r["unit"]}:L{r["layer"]}:{r["src"][5:]}->{r["tgt"][5:]}', []).append((r["E"] - p["E_tgt"]) / d)
        out[R] = {
            "n_tasks": len(gaps), "mean_gap": _m(gaps), "sign_agree": _m(agree), "n_agree_tasks": len(agree),
            "mean_mass": _m([r["mass"] for r in rr if r["unit"] == "base"]),
            "mask_identity_max_abs_dE": max((abs(r["dE"]) for r in rr if r["unit"] == "mask_identity"), default=None),
            "same_state_max_abs_dE": max((abs(r["E"] - base[(r["task"], r["tgt"])]) for r in rr
                                          if r["unit"] == "same_state" and (r["task"], r["tgt"]) in base), default=None),
            "ko_all_reduction": ko_effect(base, ko, tids),
            "moves": {k: {"mean": _m(v), "n": len(v)} for k, v in sorted(moves.items())},
        }
        out[R]["passes"] = bool(out[R]["sign_agree"] is not None and out[R]["sign_agree"] >= READOUT_AGREE_MIN
                                and (out[R]["mean_gap"] or 0) > 0)
    ok = [R for R in ("forced", "direct") if out[R]["passes"]]           # tie -> forced (keeps protocol)
    out["chosen_readout"] = max(ok, key=lambda R: out[R]["sign_agree"]) if ok else None
    if out["chosen_readout"]:
        c = out[out["chosen_readout"]]
        out["positive_control_pass"] = (c["ko_all_reduction"] or 0) >= POS_CONTROL_MIN
    out["GO"] = bool(out["chosen_readout"] and out.get("positive_control_pass"))
    return out


def s2(rows):
    base = {(r["task"], r["cond"]): r["E"] for r in rows if r.get("unit") == "base"}
    tids = sorted({t for t, _ in base})
    cells = {}
    for r in rows:
        if r.get("unit") == "ko":
            cells.setdefault((r["readers"], r["lo"], r["hi"]), {})[(r["task"], r["cond"])] = r["E"]
    eff = {f"{w}:{lo}-{hi}": ko_effect(base, v, tids) for (w, lo, hi), v in sorted(cells.items())}
    best = max(eff, key=lambda k: eff[k] if eff[k] is not None else -9)
    w, rng = best.split(":")
    lo, hi = map(int, rng.split("-"))
    return {"effects": eff, "winner": {"readers": w, "lo": lo, "hi": hi, "effect": eff[best]},
            "GO_bisect": (eff[best] or 0) >= TRIAGE_MIN}


def s3(rows):
    win = [r for r in rows if r.get("unit") == "frozen_window"]
    moves = {}
    for r in rows:
        if r.get("unit") == "span_patch":
            moves.setdefault(f'L{r["layer"]}:{r["src"][5:]}->{r["tgt"][5:]}', []).append(r["move"])
    return {"windows": [(r["readers"], r["lo"], r["hi"], r["effect"]) for r in rows if r.get("unit") == "ko_window"],
            "frozen": win[-1] if win else None,
            "span_patch_moves": {k: {"mean": _m(v), "n": len([x for x in v if x is not None])} for k, v in sorted(moves.items())}}


def s4(rows, lo, hi, top=10):
    """Two rankings. Readers: heads in blocks lo..hi by mean attention from post-history
    positions to NUM+HIST (the S2/S3 read window). Writers: heads in any block by mean
    (HIGH - LOW) linearized direct effect on E[score] at the read position."""
    by = {}
    for r in rows:
        if r.get("unit") == "heads":
            by.setdefault(r["task"], {})[r["cond"]] = r
    tasks = [t for t, v in by.items() if HIGH in v and LOW in v]
    any_r = by[tasks[0]][HIGH]
    L, H = len(any_r["dla"]), len(any_r["dla"][0])
    avg = lambda f, l, h: mean((by[t][HIGH][f][l][h] + by[t][LOW][f][l][h]) / 2 for t in tasks)
    readers, writers = [], []
    for l in range(L):
        for h in range(H):
            dd = [by[t][HIGH]["dla"][l][h] - by[t][LOW]["dla"][l][h] for t in tasks]
            m = mean(dd)
            writers.append({"layer": l, "head": h, "mean_d_dla": m,
                            "sign_consistency": mean((x > 0) == (m > 0) for x in dd)})
            if lo <= l <= hi:
                readers.append({"layer": l, "head": h, "att_ALL": avg("att_ALL", l, h),
                                "att_TAIL": avg("att_TAIL", l, h), "att_read_num": avg("att_read_num", l, h)})
    readers.sort(key=lambda s: -s["att_ALL"])
    writers.sort(key=lambda s: -abs(s["mean_d_dla"]))
    return {"n_tasks": len(tasks), "n_layers": L, "readers": readers[:top], "writers": writers[:top]}


def s5(rows):
    base = {(r["task"], r["cond"]): r["E"] for r in rows if r.get("unit") == "base"}
    tids = sorted({t for t, _ in base})

    def eff(unit, **match):
        ko = {(r["task"], r["cond"]): r["E"] for r in rows if r.get("unit") == unit
              and all(r.get(k) == v for k, v in match.items())}
        return ko_effect(base, ko, tids)
    design = next(r for r in rows if r.get("unit") == "design")
    rand = sorted(e for e in (eff("ko_random", draw=j) for j in range(len(design["random_sets"]))) if e is not None)
    moves = {}
    for r in rows:
        if r.get("unit") in ("patch_resid", "patch_heads"):
            m = normalized_move_(r["E"], r["E_tgt"], r["E_src"])
            moves.setdefault(f'{r["unit"]}:{r["src"][5:]}->{r["tgt"][5:]}', []).append(m)
    return {"n_tasks": len(tids), "window": eff("ko_window"), "topk": eff("ko_topk"),
            "random": {"mean": _m(rand), "max": rand[-1] if rand else None, "n": len(rand)},
            "singles": {f"{h[0]}.{h[1]}": eff("ko_single", head=list(h)) for h in design["top"]},
            "patch": {k: {"mean": _m(v), "n": len([x for x in v if x is not None])} for k, v in sorted(moves.items())}}


def boot(tasks, stat, n=10000, seed=20261030):
    """Point estimate and 95% percentile CI; tasks resampled within benchmark strata."""
    import random
    rng = random.Random(seed)
    strata = {}
    for t in tasks:
        strata.setdefault(t.split("/")[0], []).append(t)
    est, draws = stat(tasks), []
    for _ in range(n):
        smp = [rng.choice(v) for v in strata.values() for _ in v]
        x = stat(smp)
        if x is not None:
            draws.append(x)
    draws.sort()
    if est is None or not draws:
        return {"est": est, "lo": None, "hi": None}
    return {"est": est, "lo": draws[int(0.025 * len(draws))], "hi": draws[int(0.975 * len(draws)) - 1]}


def s6(rows, spec, n_boot=10000):
    """Pre-registered estimates and success flags for the frozen V run."""
    get = lambda unit, **m: {r["task"]: r for r in rows if r.get("unit") == unit and all(r.get(k) == v for k, v in m.items())}
    base = {(r["task"], r["cond"]): r for r in rows if r.get("unit") == "base"}
    tasks = sorted({t for t, _ in base})
    out = {"n_tasks": len(tasks)}

    def nec(w):
        ko = {c: get("ko", cond=c, window=list(w)) for c in (HIGH, LOW)}
        def f(ts):
            ts = [t for t in ts if t in ko[HIGH] and t in ko[LOW]]
            gb = mean(base[(t, HIGH)]["E_mask"] - base[(t, LOW)]["E_mask"] for t in ts)
            gk = mean(ko[HIGH][t]["E"] - ko[LOW][t]["E"] for t in ts)
            return None if abs(gb) < 1e-6 else 1 - gk / gb
        return boot(tasks, f, n_boot)
    out["necessity_window"] = nec(spec["window"])
    out["necessity_controls"] = {f"{w[0]}-{w[1]}": nec(w) for w in spec["control_windows"]}

    def suff(unit, s, t, L):
        rr = get(unit, src=s, tgt=t, layer=L)
        def f(ts):
            v = [normalized_move_(rr[x]["E"], rr[x]["E_tgt"], rr[x]["E_src"]) for x in ts if x in rr]
            return _m(v)
        return boot(tasks, f, n_boot)

    def diff(unit, s, t):
        a, c = get(unit, src=s, tgt=t, layer=spec["patch_layer"]), get(unit, src=s, tgt=t, layer=spec["control_layer"])
        def f(ts):
            v = [normalized_move_(a[x]["E"], a[x]["E_tgt"], a[x]["E_src"]) - normalized_move_(c[x]["E"], c[x]["E_tgt"], c[x]["E_src"])
                 for x in ts if x in a and x in c and abs(a[x]["E_src"] - a[x]["E_tgt"]) > 1e-6]
            return _m(v)
        return boot(tasks, f, n_boot)
    for unit, pairs in (("patch", ((HIGH, LOW), (LOW, HIGH))), ("patch_cont", (("self_low_2", "no_history"), ("no_history", "self_low_2")))):
        for s, t in pairs:
            k = f"{unit}:{s}->{t}"
            out[k] = {"site": suff(unit, s, t, spec["patch_layer"]), "control": suff(unit, s, t, spec["control_layer"]),
                      "site_minus_control": diff(unit, s, t)}

    # greedy generations: score shift toward source (patch) and greedy-gap reduction (knockout)
    gb = {(r["task"], r["cond"]): r["score"] for r in rows if r.get("unit") == "gen_base"}
    gk = {(r["task"], r["cond"]): r["score"] for r in rows if r.get("unit") == "gen_ko"}
    gt = sorted({t for t, _ in gb})
    for s, t in ((HIGH, LOW), (LOW, HIGH)):
        gp = {r["task"]: r["score"] for r in rows if r.get("unit") == "gen_patch" and r["src"] == s and r["tgt"] == t}
        sign = 1 if s == HIGH else -1             # high source should raise the low target's score
        def f(ts, gp=gp, sign=sign, t_=t):
            v = [sign * (gp[x] - gb[(x, t_)]) for x in ts if gp.get(x) is not None and gb.get((x, t_)) is not None]
            return _m(v)
        out[f"gen_patch_shift:{s}->{t}"] = boot(gt, f, n_boot)

    def gko(ts):
        ok = [x for x in ts if None not in (gb.get((x, HIGH)), gb.get((x, LOW)), gk.get((x, HIGH)), gk.get((x, LOW)))]
        g0 = mean(gb[(x, HIGH)] - gb[(x, LOW)] for x in ok) if ok else 0
        g1 = mean(gk[(x, HIGH)] - gk[(x, LOW)] for x in ok) if ok else 0
        return None if abs(g0) < 1e-6 else 1 - g1 / g0
    out["gen_ko_gap_reduction"] = boot(gt, gko, n_boot)

    th = spec["thresholds"]
    p = [out[f"patch:{s}->{t}"] for s, t in ((HIGH, LOW), (LOW, HIGH))]
    out["success"] = {
        "sufficiency_both_dirs": all(x["site"]["est"] >= th["sufficiency_min"] and x["site_minus_control"]["lo"] > 0 for x in p),
        "necessity": out["necessity_window"]["est"] >= th["necessity_min"] and out["necessity_window"]["lo"] > max(
            c["hi"] for c in out["necessity_controls"].values()),
        "behavioral_patch": all(out[f"gen_patch_shift:{s}->{t}"]["lo"] > 0 for s, t in ((HIGH, LOW), (LOW, HIGH))),
        "behavioral_knockout": out["gen_ko_gap_reduction"]["lo"] > 0,
    }
    return out


def normalized_move_(p, t, s):
    return None if abs(s - t) < 1e-6 else (p - t) / (s - t)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("s1", "s2", "s3", "s4", "s5", "s6"))
    ap.add_argument("rows")
    ap.add_argument("--lo", type=int)
    ap.add_argument("--hi", type=int)
    ap.add_argument("--spec")
    a = ap.parse_args()
    rows = load(a.rows)
    if a.stage == "s4":
        res = s4(rows, a.lo, a.hi)
    elif a.stage == "s6":
        res = s6(rows, json.load(open(a.spec, encoding="utf-8")))
    else:
        res = globals()[a.stage](rows)
    print(json.dumps(res, indent=1))
