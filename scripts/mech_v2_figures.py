"""Static PNG figures for docs/mech_v2_TLDR.md (the interactive version is results/mech_v2/analysis.html).

S6 values come from the summary JSONs; S1/S2 discovery values are copied from the run log in docs/mech_plan_v2.md.
Usage: python scripts/mech_v2_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mech_v2/figures"
COL = {"llama": "#2a78d6", "qwen": "#eb6834"}
NAME = {"llama": "Llama-3.1-8B", "qwen": "Qwen3-8B"}
CTRL = "#9a988f"
S6 = {m: json.loads((ROOT / f"results/mech_v2/s6_{m}_summary.json").read_text()) for m in COL}
RELEASED = {"llama": (-0.001, -0.004), "qwen": (-0.001, 0.005)}       # S1, single final-input token
S2 = {"llama": (["0–7", "8–15", "16–23", "24–31"], [[23, 96, 6, -1], [1, 57, 6, 0], [0, 9, -1, 0]]),
      "qwen": (["0–8", "9–17", "18–26", "27–35"], [[0, 70, 75, 7], [4, 16, 57, 8], [1, 0, 4, 1]])}
READERS = ["All post-history", "Template tail", "Answer tokens"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#b9b7b0", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#0b0b0b", "grid.color": "#e9e7e2", "savefig.dpi": 160, "savefig.bbox": "tight"})


def est(d):
    return d["est"], d["lo"], d["hi"]


def fig_sites():
    rows = ["Released site:\nsingle final-input token", "All post-history positions,\nhigh → low", "All post-history positions,\nlow → high"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for j, m in enumerate(COL):
        s = S6[m]
        vals = [sum(RELEASED[m]) / 2, s["patch:self_high_10->self_low_2"]["site"]["est"], s["patch:self_low_2->self_high_10"]["site"]["est"]]
        ys = [i + (-0.18 if j == 0 else 0.18) for i in range(3)]
        ax.barh(ys, vals, height=0.34, color=COL[m], label=NAME[m])
        for y, v in zip(ys, vals):
            ax.text(max(v, 0) + 0.01, y, "≈ 0" if abs(v) < 0.005 else f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(3), rows)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Normalized move toward source (1 = full effect)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Patch site: released single token vs all post-history positions", loc="left", fontsize=11)
    fig.savefig(OUT / "fig1_patch_site.png")
    plt.close(fig)


def fig_forest():
    tests = [("Necessity: block read window", "necessity_window", None),
             ("Sufficiency high → low", "patch:self_high_10->self_low_2", "site"),
             ("Sufficiency low → high", "patch:self_low_2->self_high_10", "site"),
             ("Continuity low → no history", "patch_cont:self_low_2->no_history", "site"),
             ("Continuity no history → low", "patch_cont:no_history->self_low_2", "site")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (_, key, sub) in enumerate(tests):
        for j, m in enumerate(COL):
            y = i + (-0.15 if j == 0 else 0.15)
            s = S6[m]
            if sub is None:
                main, ctrls = s[key], list(s["necessity_controls"].values())
            else:
                main, ctrls = s[key]["site"], [s[key]["control"]]
            for c in ctrls:
                e, lo, hi = est(c)
                ax.plot([lo, hi], [y, y], color=CTRL, lw=1.5)
                ax.plot(e, y, "o", mfc="white", mec=CTRL, mew=1.5, ms=6)
            e, lo, hi = est(main)
            ax.plot([lo, hi], [y, y], color=COL[m], lw=2)
            ax.plot(e, y, "o", color=COL[m], ms=6, label=NAME[m] if i == 0 else None)
    ax.plot([], [], "o", mfc="white", mec=CTRL, label="Control (wrong site)")
    ax.axvline(0.5, color="#b9b7b0", ls="--", lw=1)
    ax.axvline(0, color="#b9b7b0", lw=1)
    ax.set_yticks(range(len(tests)), [t[0] for t in tests])
    ax.invert_yaxis()
    ax.set_xlim(-0.15, 1.05)
    ax.set_xlabel("Gap removed (necessity) or normalized move (patches); 95% CI")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.55, 0.5))
    ax.set_title("Frozen validation, 72 held-out tasks per model", loc="left", fontsize=11)
    fig.savefig(OUT / "fig2_validation.png")
    plt.close(fig)


def fig_heat():
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.6))
    for ax, m in zip(axes, COL):
        cols, data = S2[m]
        ax.imshow([[max(v, 0) for v in r] for r in data], cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i, r in enumerate(data):
            for j, v in enumerate(r):
                ax.text(j, i, f"{v}%", ha="center", va="center", color="white" if v > 50 else "#0b0b0b", fontsize=9)
        ax.set_xticks(range(4), [f"blocks {c}" for c in cols], fontsize=8)
        ax.set_yticks(range(3), READERS if m == "llama" else [""] * 3)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(NAME[m], loc="left", fontsize=10)
    fig.suptitle("Gap removed when readers cannot attend to the history (12 triage tasks)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_read_window.png")
    plt.close(fig)


def fig_behavior():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 2.6), gridspec_kw={"width_ratios": [1, 1.3]})
    for i, m in enumerate(COL):
        e, lo, hi = est(S6[m]["gen_ko_gap_reduction"])
        a.plot([lo * 100, hi * 100], [i, i], color=COL[m], lw=2)
        a.plot(e * 100, i, "o", color=COL[m], ms=6)
    a.set_yticks(range(2), [NAME[m] for m in COL])
    a.invert_yaxis()
    a.set_xlim(0, 110)
    a.set_xlabel("Greedy score gap removed (%)")
    a.set_title("Knockout: works", loc="left", fontsize=10)
    labels = []
    for m in COL:
        for d, k in (("hi → lo", "gen_patch_shift:self_high_10->self_low_2"), ("lo → hi", "gen_patch_shift:self_low_2->self_high_10")):
            i = len(labels)
            e, lo, hi = est(S6[m][k])
            b.plot([lo, hi], [i, i], color=COL[m], lw=2)
            b.plot(e, i, "o", color=COL[m], ms=6)
            labels.append(f"{NAME[m].split('-')[0]} {d}")
    b.axvline(0, color="#b9b7b0", lw=1)
    b.set_yticks(range(4), labels)
    b.invert_yaxis()
    b.set_xlabel("Greedy score shift toward source (points; full gap ≈ 2)")
    b.set_title("Prompt-only patch: fails", loc="left", fontsize=10)
    for ax in (a, b):
        ax.grid(axis="x")
        ax.set_axisbelow(True)
    fig.suptitle("Free generation, 36 held-out tasks (95% CI)", x=0.02, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_generation.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_sites()
    fig_forest()
    fig_heat()
    fig_behavior()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
