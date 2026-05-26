# JudgeBench

> Earlier exploratory work is archived in [`_archive/`](./_archive/).

**Controlled cross-judging experiments on open LLMs: do competition framing, identity reveals, and injected prior scores shift what LLM judges output?**

This repository contains the code, experimental prompts, and aggregated results for a cross-judging study of five open-weights 7-9B language models acting as judges of each other's work across nine conditions at two temperatures.

## What's in this repository

- `judgebench.py` — single-file pipeline (generation + judging + stats + plots)
- `requirements.txt` — Python dependencies
- `data/prompts/` — the 30 Arena-Hard-v2.0 prompts used (pinned, seed=42)
- `results/figures/` — 18 publication-ready figures (PNG + PDF)
- `results/stats/` — paired bootstrap CIs, manipulation-check results
- `results/summaries/` — aggregated per-cell and per-prompt CSVs
- `scripts/` — small helper scripts for reproducing analysis

**Large raw outputs** (full judge-draw logs, per-cell rows with justifications) are hosted separately on Zenodo (see [Reproducing from scratch](#reproducing-from-scratch)) because they exceed GitHub's file-size limits.

## Headline findings

All reported effects are paired-bootstrap 95% CIs over (judge, evaluated, prompt) cells; all replicated across T=0.3 and T=0.9.

1. **Competition framing inflates scores by ~0.22 points** (n>580 per comparison, p<0.0001), across 3 prompt domains and 4 of 5 judges.
2. **Prior-score information adds nothing when competition framing is present, and decreases scores when absent.** The competition wording — not the scores — drives the effect (wording-only effect: +0.206 at T=0.3, +0.234 at T=0.9).
3. **Score faking constitutes a prompt-injection vulnerability.** Llama-3.1 shifts output scores by 1.66 points when shown fabricated prior scores (low=2/10 → high=8/10); Gemma-2 shifts 0.76; Mistral and DeepSeek-R1-Distill are near-robust.
4. **Qwen3 scores its own work ~1 point lower when told it was produced by another model.** Text-identical control; only label changes.
5. **Judges rarely acknowledge the manipulation in their justifications** (<3% mention model name, injected score, or prior-score keywords), yet behave as if affected by it.

See `results/figures/` for visualizations and `results/stats/stat_tests.json` for full effect-size tables.

## Experimental design

Nine conditions across two arcs:

| Case | Description |
|------|-------------|
| 1a | Blind baseline (no identity, no framing) |
| 2a | Identity of evaluated model revealed |
| 2b | Identity + competition framing |
| 2c | Competition framing, anonymous |
| 3a | Real prior scores injected + competition framing ("revenge" condition) |
| 3b | Real prior scores injected, no competition framing |
| 4a | Judge's own response shown, mislabeled as another model's |
| 4b | Another model's response shown, mislabeled as the judge's own |
| 4c | Fabricated prior scores (2 / 5 / 8 / actual) injected |

Five open-weights judges:
- Qwen3-8B
- Llama-3.1-8B-Instruct
- Gemma-2-9B-IT
- Mistral-7B-Instruct-v0.3
- DeepSeek-R1-Distill-Qwen-7B (reasoning-trained)

Two generation temperatures (0.3, 0.9), K=5 judge samples per call, N=5 generation runs per prompt, seed=42 throughout.

## Quick start

Running the full pipeline requires a GPU with >=24GB VRAM (tested on NVIDIA RTX 6000 Ada, 48GB).

```bash
pip install -r requirements.txt

# Accept Hugging Face licenses for gated models:
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#   https://huggingface.co/google/gemma-2-9b-it
# Then:
export HF_TOKEN=hf_...

# Stage 1: validate environment + all 5 models (one-time, ~15 min)
python judgebench.py preflight

# Stage 2: download + pin the prompt set (already in data/prompts/, this is for fresh runs)
python judgebench.py prompts

# Stage 3: run the full pipeline (~7-8 hours on A6000, resumable)
python judgebench.py run

# Stage 4 (only if plotting was skipped): regenerate plots from saved results
python judgebench.py plots
```

Full CLI reference in the docstring at the top of `judgebench.py`.

## Reproducing from scratch

The results in this repository are reproducible end-to-end from `judgebench.py` given the pinned prompt set in `data/prompts/arena_hard_v2_30_seed42.json`. Seeds are fixed at 42 throughout for both generation and bootstrap resampling.

Full per-cell output and per-draw justifications (`cot_log_T*.jsonl`, `full_results_T*.json`) are archived on Zenodo: **[link TBD]**. These are not tracked in git because they total ~500 MB.

To exactly reproduce the aggregated results:
```bash
python judgebench.py run           # regenerate all outputs
python judgebench.py plots         # regenerate figures
```

## Statistical methods

- **Paired bootstrap (10,000 resamples)** at the (judge, evaluated, prompt) cell level for all pairwise comparisons between conditions.
- **Per-judge breakdowns** reported alongside pooled estimates.
- Reporting both **means and medians** for aggregated scores (severe judge-severity differences make medians meaningful).
- **Manipulation-check regex analysis** over justifications: counts explicit mention of model names (Case 2a), score-keyword mentions (Cases 3a/3b), alleged-score-number mentions (Case 4c).

## Known limitations

1. **DeepSeek-R1-Distill's blind-condition success rate was 9.9%** (74/750 cells at T=0.3). The model consistently exceeded a 2048-token budget producing reasoning traces before the JSON response on blind prompts. Reasoning-vs-non-reasoning comparisons involving DeepSeek are based on subsets where it successfully produced parseable output and should be interpreted as lower-bound estimates.
2. **Mistral-7B-v0.3 hits ceiling 71.6% of the time** in the baseline condition (scores ≥ 9), which compresses its upward effect sizes.
3. Qwen3's Case 4c (score faking) has n=17 cells out of 480 expected — insufficient for Qwen-specific Case 4c claims.
4. No closed-source judges (GPT-4, Claude, Gemini) are included; scope is restricted to open-weights models for reproducibility.

## License

Code: MIT (see LICENSE).

Data: the Arena-Hard-v2.0 prompts are redistributed under the original license from [lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto); aggregated results released under CC-BY-4.0.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{judgebench2026,
  author = {[AUTHOR]},
  title  = {JudgeBench: Controlled Cross-Judging Experiments on Open LLMs},
  year   = {2026},
  url    = {[GITHUB URL]}
}
```

## Contact

Open an issue on this repository for questions or replication requests.
