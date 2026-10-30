"""mech-v2 GPU stages S1-S4 (see docs/mech_plan_v2.md, section 1b).

Each stage appends one JSON row per measurement to <out>/<stage>_<model>.jsonl and a
{"task_done": ...} row after each task, so a crashed or timed-out run resumes where it
stopped. Gates are evaluated offline by mech_v2.analysis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import core as C

HIGH, LOW, NONE = "self_high_10", "self_low_2", "no_history"
DIRS = ((HIGH, LOW), (LOW, HIGH))


class Store:
    def __init__(self, path: Path, commit=None):
        self.path, self.commit = path, commit or (lambda: None)
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
        self.done = {r["task_done"] for r in self.rows if "task_done" in r}

    def add(self, **row):
        row["t"] = time.strftime("%H:%M:%S")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def finish(self, task):
        self.add(task_done=task)
        self.done.add(task)
        self.commit()


class Ctx:
    """Shared per-run state: model, splits, cached greedy generations."""

    def __init__(self, model_key, splits_path, out, commit, attn="sdpa"):
        sp = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        self.tasks, self.splits = sp["tasks"], sp["splits"]
        self.out, self.commit, self.key = Path(out), commit, model_key
        self.out.mkdir(parents=True, exist_ok=True)
        self.S = C.Scorer(model_key, attn=attn)
        self.gens = Store(self.out / f"gens_{model_key}.jsonl", commit)
        self.gen = {(r["task"], r["cond"]): r for r in self.gens.rows if "cond" in r}

    def greedy(self, tid, cond):
        if (tid, cond) not in self.gen:
            g = self.S.generate(self.tasks[tid], cond)
            self.gens.add(task=tid, cond=cond, **g)
            self.gen[(tid, cond)] = dict(task=tid, cond=cond, **g)
            self.commit()
        return self.gen[(tid, cond)]

    def suffix(self, readout, tid, text_cond):
        """Assistant text before the score token. 'forced' uses the greedy justification of
        `text_cond`; in a source->target patch both runs use the TARGET's text so positions align."""
        if readout == "direct":
            return C.DIRECT_PREFIX
        fp = self.greedy(tid, text_cond)["forced_prefix"]
        return None if fp is None else fp

    def build(self, readout, tid, cond, text_cond=None):
        sfx = self.suffix(readout, tid, text_cond or cond)
        return None if sfx is None else self.S.build(self.tasks[tid], cond, sfx)


def _post_hist(b):
    return b["groups"]["HIST"][-1] + 1


def _keys(b):
    return sorted(b["groups"]["HIST"] + b["groups"]["NUM"])


def _readers(b, which):
    return {"ANS": b["groups"]["ANS"], "TAIL": b["groups"]["TAIL"],
            "ALL": list(range(_post_hist(b), len(b["ids"])))}[which]


def _patch_post(cx, bs, bt, start_t, L):
    """Patch hidden_states[L] at all aligned target positions >= start_t from the source."""
    ps, pt = C.align_after(len(bs["ids"]), len(bt["ids"]), start_t)
    if [bs["ids"][p] for p in ps] != [bt["ids"][p] for p in pt]:
        raise ValueError("aligned source/target positions hold different tokens")
    st = cx.S.states(bs, ps, [L])[L]
    return cx.S.readout(bt, cx.S.patch([(L, pt, st)]))


# ---- S1: readout gate, harness checks, released-site re-run --------------------------------
def s1(cx: Ctx, args):
    st = Store(cx.out / f"s1_{cx.key}.jsonl", cx.commit)
    S, site = cx.S, S_site(cx)
    for tid in cx.splits["triage"]:
        if tid in st.done:
            continue
        for c in (HIGH, LOW, NONE):
            g = cx.greedy(tid, c)
            st.add(task=tid, unit="greedy", cond=c, score=g["score"], capped=g["capped"])
        for R in ("direct", "forced"):
            base = {}
            for c in (HIGH, LOW, NONE):
                b = cx.build(R, tid, c)
                if b is None:
                    continue
                base[c] = S.readout(b)
                st.add(task=tid, R=R, unit="base", cond=c, **base[c])
            if HIGH not in base or LOW not in base:
                continue
            # explicit-mask path must reproduce the default path
            b = cx.build(R, tid, HIGH)
            r = S.readout(b, attn_mask=S.base_mask(len(b["ids"])))
            st.add(task=tid, R=R, unit="mask_identity", cond=HIGH, dE=r["E"] - base[HIGH]["E"], **r)
            # alignment-free positive control: history unreadable by every later position
            for c in (HIGH, LOW):
                b = cx.build(R, tid, c)
                n = len(b["ids"])
                with S.knockout(n, _readers(b, "ALL"), _keys(b), range(S.n_layers)) as _:
                    r = S.readout(b, attn_mask=S.base_mask(n))
                st.add(task=tid, R=R, unit="ko_all", cond=c, **r)
            for s, t in DIRS:
                bt, bs = cx.build(R, tid, t), cx.build(R, tid, s, text_cond=t)
                src = S.readout(bs)
                tgt = base[t] if R == "direct" else S.readout(bt)
                st.add(task=tid, R=R, unit="pair", src=s, tgt=t, E_src=src["E"], E_tgt=tgt["E"])
                fi = bt["final_input"]
                own = S.states(bt, [fi], [site])[site]
                r = S.readout(bt, S.patch([(site, [fi], own)]))
                st.add(task=tid, R=R, unit="same_state", src=t, tgt=t, layer=site, **r)
                for L in (site, C.CONTROL_LAYER):
                    sv = S.states(bs, [bs["final_input"]], [L])[L]
                    r = S.readout(bt, S.patch([(L, [fi], sv)]))
                    st.add(task=tid, R=R, unit="final_input_patch", src=s, tgt=t, layer=L, **r)
                try:
                    ps, pt = C.align(bs["groups"], bt["groups"], ("HIST", "NUM"))
                except ValueError as e:
                    st.add(task=tid, R=R, unit="pos_patch_L1", src=s, tgt=t, skipped=str(e))
                    continue
                sv = S.states(bs, ps, [1])[1]
                r = S.readout(bt, S.patch([(1, pt, sv)]))
                st.add(task=tid, R=R, unit="pos_patch_L1", src=s, tgt=t, layer=1, **r)
        st.finish(tid)


def S_site(cx):
    return C.MODELS[cx.key]["site_layer"]


# ---- S2: path triage by attention knockout ------------------------------------------------
def blocks(n_layers, k=4):
    step = n_layers // k
    return [(i * step, n_layers - 1 if i == k - 1 else (i + 1) * step - 1) for i in range(k)]


def _ko(cx, b, which, lo, hi):
    n = len(b["ids"])
    with cx.S.knockout(n, _readers(b, which), _keys(b), range(lo, hi + 1)) as _:
        return cx.S.readout(b, attn_mask=cx.S.base_mask(n))


def s2(cx: Ctx, args):
    R = args["readout"]
    st = Store(cx.out / f"s2_{cx.key}.jsonl", cx.commit)
    for tid in cx.splits["triage"]:
        if tid in st.done:
            continue
        for c in (HIGH, LOW):
            b = cx.build(R, tid, c)
            if b is None:
                continue
            # baseline through the same explicit-mask path as the knockouts (S1: mask path differs by up to ~0.1)
            st.add(task=tid, cond=c, unit="base", **cx.S.readout(b, attn_mask=cx.S.base_mask(len(b["ids"]))))
            for which in ("ANS", "TAIL", "ALL"):
                for lo, hi in blocks(cx.S.n_layers):
                    st.add(task=tid, cond=c, unit="ko", readers=which, lo=lo, hi=hi, **_ko(cx, b, which, lo, hi))
        st.finish(tid)


# ---- S3: bisection inside the winning block, then span patching ---------------------------
def s3(cx: Ctx, args):
    """args: readout, readers, lo, hi (the S2 winner). Bisection is pre-registered:
    descend into the stronger half while it keeps >= 60% of the parent's effect."""
    from .analysis import ko_effect
    R, which, lo, hi = args["readout"], args["readers"], int(args["lo"]), int(args["hi"])
    st = Store(cx.out / f"s3_{cx.key}.jsonl", cx.commit)
    tids = cx.splits["triage"]
    bs = {(t, c): cx.build(R, t, c) for t in tids for c in (HIGH, LOW)}
    bs = {k: v for k, v in bs.items() if v is not None}
    base = {k: cx.S.readout(b, attn_mask=cx.S.base_mask(len(b["ids"])))["E"] for k, b in bs.items()}

    def effect(a, z):
        rows = {k: _ko(cx, b, which, a, z)["E"] for k, b in bs.items()}
        e = ko_effect(base, rows, tids, HIGH, LOW)
        st.add(unit="ko_window", readers=which, lo=a, hi=z, effect=e, rows={f"{k[0]}|{k[1]}": v for k, v in rows.items()})
        cx.commit()
        return e

    parent, win = effect(lo, hi), (lo, hi)
    while win[1] > win[0]:
        mid = (win[0] + win[1]) // 2
        halves = [((win[0], mid), effect(win[0], mid)), ((mid + 1, win[1]), effect(mid + 1, win[1]))]
        nxt = C.bisect_choice(parent, halves)
        if nxt is None:
            break
        win, parent = nxt, dict(halves)[nxt]
    st.add(unit="frozen_window", readers=which, lo=win[0], hi=win[1], effect=parent)

    # residual patch of the reader span after / before the window, plus L2 control
    group_start = {"ANS": lambda b: b["groups"]["ANS"][0], "TAIL": lambda b: b["groups"]["TAIL"][0],
                   "ALL": _post_hist}[which]
    for tid in tids:
        for s, t in DIRS:
            bt, bsrc = cx.build(R, tid, t), cx.build(R, tid, s, text_cond=t)
            if bt is None or bsrc is None:
                continue
            E_src, E_tgt = cx.S.readout(bsrc)["E"], cx.S.readout(bt)["E"]
            for L in sorted({win[0], win[1] + 1, C.CONTROL_LAYER}):
                r = _patch_post(cx, bsrc, bt, group_start(bt), L)
                st.add(task=tid, unit="span_patch", readers=which, src=s, tgt=t, layer=L,
                       E_src=E_src, E_tgt=E_tgt, move=C.normalized_move(r["E"], E_tgt, E_src), **r)
        cx.commit()
    st.add(task_done="ALL")
    cx.commit()


# ---- S4: observational head scan (eager attention) ----------------------------------------
def s4(cx: Ctx, args):
    """Per head at the score-read position: linearized direct effect on E[score] and
    attention mass to NUM / HIST. Two forward passes per task (HIGH, LOW)."""
    import torch
    R = args["readout"]
    S = cx.S
    st = Store(cx.out / f"s4_{cx.key}.jsonl", cx.commit)
    cfg = S.model.config
    H = cfg.num_attention_heads
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    WU = S.model.lm_head.weight
    norm = S.model.model.norm
    toks = list(S.score_ids)
    WU_s = WU[[S.score_ids[k] for k in toks]].float()            # [ntok, hidden]
    for tid in cx.splits["D"]:
        if tid in st.done:
            continue
        for c in (HIGH, LOW):
            b = cx.build(R, tid, c)
            if b is None:
                continue
            rp = b["read"]
            kh = _keys(b)
            # S2: history is read by ALL post-history positions (TAIL mostly) in blocks 8-15,
            # so record each head's mean attention to NUM+HIST from those reader sets too
            rsets = {"read": [rp], "TAIL": b["groups"]["TAIL"], "ALL": _readers(b, "ALL")}
            xo, att, fin = {}, {}, {}
            hooks = []

            def grab(i, A):
                A = A[0].float()                                          # [H, T, T]
                att[i] = {k: A[:, r][:, :, kh].sum(-1).mean(-1).tolist() for k, r in rsets.items()}
                att[i]["read_num"] = A[:, rp, b["groups"]["NUM"]].sum(-1).tolist()
            for i, layer in enumerate(S.layers):
                hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
                    lambda m, a, i=i: xo.__setitem__(i, a[0][0, rp].float().clone())))
                hooks.append(layer.self_attn.register_forward_hook(lambda m, a, o, i=i: grab(i, o[1])))
            hooks.append(norm.register_forward_pre_hook(lambda m, a: fin.__setitem__(0, a[0][0, rp].float().clone())))
            try:
                r = S.readout(b)
            finally:
                for h in hooks:
                    h.remove()
            # linearize E around this run: dE/dlogit_k = p_k (v_k - E)
            E, mass, vals, p = C.score_distribution(r["pf"], r["q"], S.single_ten)
            w = torch.tensor([p[k] * (vals[k] - E) for k in toks], device="cuda")
            x = fin[0]
            scale = norm.weight.float() / torch.sqrt(x.pow(2).mean() + norm.variance_epsilon)
            dla = []
            for i, layer in enumerate(S.layers):
                Wo = layer.self_attn.o_proj.weight.float().view(-1, H, d)           # [hidden, H, d]
                contrib = torch.einsum("khd,hd->hk", Wo, xo[i].view(H, d))         # [H, hidden]
                dla.append(((contrib * scale) @ WU_s.T @ w).tolist())
            st.add(task=tid, cond=c, unit="heads", E=r["E"], E_lin=E, mass=mass, dla=dla,
                   **{f"att_{k}": [att[i][k] for i in range(S.n_layers)] for k in ("read", "TAIL", "ALL", "read_num")})
        st.finish(tid)


# ---- S5: exact head-level tests on D ------------------------------------------------------
def s5(cx: Ctx, args):
    """args: readout, heads [[block, head], ...] (S4 reader ranking, top-k), lo, hi (read window),
    n_random. Per task and condition: window knockout (reference), joint top-k head knockout,
    each head alone, and n_random matched random head sets (same per-block counts, excluding
    the top-k). Per direction: residual patch of all post-history positions at hidden_states[hi+1]
    (reference) and patch of the top-k heads' outputs at all post-history positions."""
    import random
    R, lo, hi = args["readout"], int(args["lo"]), int(args["hi"])
    top = [tuple(x) for x in args["heads"]]
    H = cx.S.model.config.num_attention_heads
    rng = random.Random(20261030)
    per_block = {}
    for i, h in top:
        per_block.setdefault(i, []).append(h)
    pool = {i: [h for h in range(H) if h not in per_block[i]] for i in per_block}
    rand_sets = [{i: rng.sample(pool[i], len(hs)) for i, hs in per_block.items()}
                 for _ in range(int(args.get("n_random", 10)))]
    st = Store(cx.out / f"s5_{cx.key}{args.get('tag', '')}.jsonl", cx.commit)
    if not st.rows:
        st.add(unit="design", top=top, random_sets=[{str(k): v for k, v in s.items()} for s in rand_sets])
    for tid in cx.splits["D"]:
        if tid in st.done:
            continue
        for c in (HIGH, LOW):
            b = cx.build(R, tid, c)
            n, rd, ks = len(b["ids"]), _readers(b, "ALL"), _keys(b)
            mk = cx.S.base_mask(n)

            def ko(blocks, heads=None):
                with cx.S.knockout(n, rd, ks, blocks, heads) as _:
                    return cx.S.readout(b, attn_mask=mk)["E"]
            st.add(task=tid, cond=c, unit="base", E=cx.S.readout(b, attn_mask=mk)["E"])
            st.add(task=tid, cond=c, unit="ko_window", E=ko(range(lo, hi + 1)))
            st.add(task=tid, cond=c, unit="ko_topk", E=ko(per_block, per_block))
            for i, h in top:
                st.add(task=tid, cond=c, unit="ko_single", head=[i, h], E=ko([i], {i: [h]}))
            for j, rs in enumerate(rand_sets):
                st.add(task=tid, cond=c, unit="ko_random", draw=j, E=ko(rs, rs))
        for s, t in DIRS:
            bt, bs = cx.build(R, tid, t), cx.build(R, tid, s, text_cond=t)
            E_src, E_tgt = cx.S.readout(bs)["E"], cx.S.readout(bt)["E"]
            r = _patch_post(cx, bs, bt, _post_hist(bt), hi + 1)
            st.add(task=tid, unit="patch_resid", src=s, tgt=t, layer=hi + 1, E=r["E"], E_src=E_src, E_tgt=E_tgt)
            ps, pt = C.align_after(len(bs["ids"]), len(bt["ids"]), _post_hist(bt))
            src_in = cx.S.head_inputs(bs, ps, list(per_block))
            d = cx.S.head_dim()
            edits = [(i, h, pt, src_in[i][:, h * d:(h + 1) * d]) for i, h in top]
            r = cx.S.readout(bt, cx.S.patch_heads(edits))
            st.add(task=tid, unit="patch_heads", src=s, tgt=t, E=r["E"], E_src=E_src, E_tgt=E_tgt)
        st.finish(tid)


# ---- S6: frozen validation on V -----------------------------------------------------------
def s6(cx: Ctx, args):
    """Frozen spec: data/mech_v2/freeze_s6_<model>.json (passed as args). Readout arms on all V
    tasks; greedy-generation arms on the first gen_per_dataset V tasks of each benchmark."""
    R, (lo, hi), PL = args["readout"], args["window"], int(args["patch_layer"])
    ctrl_windows, ctrl_layer = [tuple(w) for w in args["control_windows"]], int(args["control_layer"])
    gen_tasks = {t for ds in ("mtbench", "arena_hard", "wildbench")
                 for t in [x for x in cx.splits["V"] if x.startswith(ds + "/")][:int(args["gen_per_dataset"])]}
    S = cx.S
    st = Store(cx.out / f"s6_{cx.key}.jsonl", cx.commit)
    for tid in cx.splits["V"]:
        if tid in st.done:
            continue
        b = {c: cx.build(R, tid, c) for c in (HIGH, LOW, NONE)}
        for c in (HIGH, LOW, NONE):
            st.add(task=tid, cond=c, unit="base", E=S.readout(b[c])["E"],
                   E_mask=S.readout(b[c], attn_mask=S.base_mask(len(b[c]["ids"])))["E"])
        for c in (HIGH, LOW):
            for w in [(lo, hi)] + ctrl_windows:
                st.add(task=tid, cond=c, unit="ko", window=list(w), E=_ko(cx, b[c], "ALL", w[0], w[1])["E"])
        for s, t in DIRS:
            bs = cx.build(R, tid, s, text_cond=t)
            for L in (PL, ctrl_layer):
                r = _patch_post(cx, bs, b[t], _post_hist(b[t]), L)
                st.add(task=tid, unit="patch", src=s, tgt=t, layer=L, E=r["E"],
                       E_src=S.readout(bs)["E"], E_tgt=S.readout(b[t])["E"])
        for s, t in ((LOW, NONE), (NONE, LOW)):            # continuity with the paper's contrast
            bs = cx.build(R, tid, s, text_cond=t)
            for L in (PL, ctrl_layer):
                r = _patch_post(cx, bs, b[t], b[t]["score_line_start"], L)
                st.add(task=tid, unit="patch_cont", src=s, tgt=t, layer=L, E=r["E"],
                       E_src=S.readout(bs)["E"], E_tgt=S.readout(b[t])["E"])
        if tid in gen_tasks:
            p = {c: S.build(cx.tasks[tid], c, "") for c in (HIGH, LOW)}
            ids = {c: p[c]["ids"][:-1] for c in p}                     # drop the readout '1'
            for c in (HIGH, LOW):
                st.add(task=tid, unit="gen_base", cond=c, **S.greedy_loop(ids[c]))
                ks, start = _keys(p[c]), _post_hist(p[c])
                g = S.greedy_loop(ids[c], masked=True, ctx_factory=lambda T, ks=ks, start=start:
                                  S.knockout(T, range(start, T), ks, range(lo, hi + 1)))
                st.add(task=tid, unit="gen_ko", cond=c, **g)
            for s, t in DIRS:
                ps, pt = C.align_after(len(ids[s]), len(ids[t]), _post_hist(p[t]))
                sv = S.states({"ids": ids[s]}, ps, [PL])[PL]
                g = S.greedy_loop(ids[t], ctx_factory=lambda T, pt=pt, sv=sv: S.patch([(PL, pt, sv)]))
                st.add(task=tid, unit="gen_patch", src=s, tgt=t, layer=PL, **g)
        st.finish(tid)


STAGES = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6}
