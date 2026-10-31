"""Post-hoc check: does the explicit-mask path alone change the greedy gap? (docs/mech_plan_v2.md)

Compares gen_base and gen_ko (s6_<model>.jsonl) with gen_mask (s6m_<model>.jsonl).
CIs: benchmark-stratified task bootstrap, 10,000 draws.
Usage: python scripts/mech_v2_mask_check.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HIGH, LOW = "self_high_10", "self_low_2"


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def unit(rs, name):
    return {(r["task"], r["cond"]): r for r in rs if r.get("unit") == name}


def gap(d, ts):
    return sum(d[t, HIGH]["score"] - d[t, LOW]["score"] for t in ts) / len(ts)


def ci(v):
    v = sorted(v)
    return round(v[int(.025 * len(v))], 1), round(v[int(.975 * len(v))], 1)


def main():
    for m in ("llama", "qwen"):
        rs = rows(ROOT / f"results/mech_v2/results/s6_{m}.jsonl")
        base, ko = unit(rs, "gen_base"), unit(rs, "gen_ko")
        mask = unit(rows(ROOT / f"results/mech_v2/results/s6m_{m}.jsonl"), "gen_mask")
        assert set(mask) == set(base), "gen_mask must cover the same tasks as gen_base"
        print(m, "outputs", len(mask), "identical text", sum(mask[k]["text"] == base[k]["text"] for k in mask),
              "same score", sum(mask[k]["score"] == base[k]["score"] for k in mask))
        ok = sorted({t for t, _ in mask if all(d[t, c]["score"] is not None for d in (base, mask, ko) for c in (HIGH, LOW))})
        print("  complete", len(ok), "gaps base/mask/ko", *(round(gap(d, ok), 3) for d in (base, mask, ko)))
        print("  reduction from base %", round(100 * (1 - gap(ko, ok) / gap(base, ok)), 1),
              "from mask %", round(100 * (1 - gap(ko, ok) / gap(mask, ok)), 1),
              "gap change base->mask %", round(100 * (gap(mask, ok) / gap(base, ok) - 1), 1))
        groups = {}
        for t in ok:
            groups.setdefault(t.split("/")[0], []).append(t)
        random.seed(0)
        red, change = [], []
        for _ in range(10000):
            s = [t for g in groups.values() for t in random.choices(g, k=len(g))]
            g0, g1 = gap(base, s), gap(mask, s)
            if abs(g1) > 1e-6:
                red.append(100 * (1 - gap(ko, s) / g1))
            if abs(g0) > 1e-6:
                change.append(100 * (g1 / g0 - 1))
        print("  CI reduction from mask", ci(red), "CI gap change", ci(change))


if __name__ == "__main__":
    main()
