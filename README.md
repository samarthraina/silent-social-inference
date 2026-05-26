# Silent Social Inference in Large Language Models

> Code, data, and experimental outputs for the EMNLP 2026 submission *"Silent
> Social Inference in Large Language Models."* Earlier exploratory work is
> archived in [`_archive/`](./_archive/).

LLMs silently integrate social context into their judgments. When prompts
surrounding a fixed task carry implied identity, competitive stakes, or a
fabricated reputational history, model outputs shift in response — yet the
models almost never mention this context in their stated justifications.
This repository contains the code, experimental prompts, and aggregated
results for a controlled study of this phenomenon across five open-weight
model families, plus a mechanistic analysis localizing the integration to
specific layers of the residual stream.

📄 **Paper**: will be added to [`paper/`](./paper/) at camera-ready.

---

## Headline findings

All reported effects are paired-bootstrap 95% CIs (10,000 resamples) at the
(judge, evaluated model, prompt) cell level. Effects replicate across
*T=0.3* and *T=0.9*, two paraphrase variants, and an independent earlier
experimental run.

1. **Three social-context manipulations shift LLM judges' scores by
   0.17–0.74 points on a 10-point scale**, all *p* < 0.001:
   - Implied identity of the response author: +0.174
   - Anonymous competitive framing: +0.238
   - Fabricated low prior score (2/10): −0.739

2. **Models almost never acknowledge the manipulation in their stated
   justifications.** Across 11 conditions and 5,500 rated justifications,
   the rate of mentioning the injected context never exceeds 5.4% under
   either of two independent measurement methods (keyword search and a
   cross-family LLM-as-judge protocol), while behavioral effects span up
   to 0.74 points. **The gap between behavior and justification is at
   least an order of magnitude.**

3. **Fabricated priors produce a monotonic dose–response.** Low priors
   (2/10) deflate scores by −0.74; mid (5/10) by −0.43; matched high
   (8/10, at or above the model's natural baseline) produce no measurable
   inflation (+0.03, n.s.).

4. **The integration is causally localized to the early-to-mid residual
   stream, with a model-specific cliff.** Logit-level activation patching
   across three focal models (Gemma-2-9B, Llama-3.1-8B, Qwen3-8B) shows:
   - Single-token patching recovers only 3–10% of the behavioral shift.
   - Multi-token patching across the manipulation span recovers ≥1.0×.
   - The recoverable signal cliffs sharply at layer ∼15 in Llama,
     ∼25 in Gemma, and ∼25–28 in Qwen — reproducible per model and
     consistent across all three manipulations.
   - A minimal-pair (9.0/10 vs 2.0/10) ablation isolates the valence of
     an injected score to the same early-layer range in 2 of 3 models.

See [`results/`](./results/) for the underlying data.

---

## Repository structure

```
silent-social-inference/
├── src/
│   ├── behavioral/                       # Behavioral pipeline (paper §4)
│   │   ├── behavioral_original.py        # Original-arm runs (5 judges, 9 cases)
│   │   └── behavioral_paraphrase.py      # Paraphrase variants (para1, para2)
│   ├── mechanistic/                      # Activation patching (paper §5)
│   │   └── mech_interp_v8_multipos.py    # Multi-position patching + valence ablation
│   └── analysis/
│       └── manipulation_check_llm_judge.py  # GPT-OSS-20B judge over justifications
├── results/
│   ├── behavioral/
│   │   ├── data_original/                # Original-arm CSVs, JSONs, checkpoints
│   │   └── data_paraphrase/              # Paraphrase variants
│   ├── mechanistic/
│   │   └── mech_interp_v8/               # Per-model patching JSONs (gemma, llama, qwen)
│   ├── manipulation_check/
│   │   ├── T0.3/                         # GPT-OSS-20B judge outputs at T=0.3
│   │   └── T0.9/                         # GPT-OSS-20B judge outputs at T=0.9
│   ├── figures/                          # Behavioral exploratory figures
│   ├── stats/                            # Bootstrap CIs, effect-size tables
│   └── summaries/                        # Aggregated per-cell and per-prompt CSVs
├── data/
│   └── prompts/                          # 30 pinned Arena-Hard-v2 prompts (seed=42)
├── figures/                              # Final figures referenced by the paper
├── paper/                                # LaTeX source + compiled PDF (added at camera-ready)
└── _archive/                             # Earlier exploratory codebase, preserved
```

---

## Code overview

The pipeline is split into three independently-runnable scripts that mirror
the paper structure.

### `src/behavioral/behavioral_original.py`

Behavioral cross-judging pipeline for the 9 conditions (1a, 2a, 2b, 2c, 3a,
3b, 4a, 4b, 4c). Subcommands:

```bash
python src/behavioral/behavioral_original.py preflight   # validate environment + all 5 models
python src/behavioral/behavioral_original.py prompts     # download + pin the Arena-Hard prompts
python src/behavioral/behavioral_original.py diagnose    # stand-alone model diagnostic
python src/behavioral/behavioral_original.py run         # full generation + judging + stats + plots
python src/behavioral/behavioral_original.py plots       # regenerate plots from saved results (no GPU)
```

The `run` subcommand is checkpointed via `results/checkpoints/` and resumable
after interruption.

### `src/behavioral/behavioral_paraphrase.py`

V2 extension of the behavioral pipeline with paraphrased system prompts
(para1, para2) and supplemental judging with two fine-tuned models (Tulu-3-SFT,
Tulu-3-DPO). Same subcommand interface as `behavioral_original.py`.

### `src/mechanistic/mech_interp_v8_multipos.py`

Multi-position activation patching across Gemma-2-9B, Llama-3.1-8B, and
Qwen3-8B for the three social-context manipulations. Also runs the
minimal-pair (9.0/10 vs 2.0/10) score-faking ablation that produces
Figures 3 and 4.

```bash
python src/mechanistic/mech_interp_v8_multipos.py \
    --prompts-file data/prompts/arena_hard_v2_30_seed42.json \
    --generation-checkpoint results/behavioral/data_original/checkpoints/generation_complete.json \
    --models gemma llama qwen
```

Outputs per-model patching JSONs to `results/mechanistic/`.

### `src/analysis/manipulation_check_llm_judge.py`

GPT-OSS-20B applied to the justification corpus to classify whether each
justification references the injected context. Produces the upper-bound
manipulation-check rate reported in paper §4.5 / §B.2.

See the script's docstring for usage. Outputs to
`results/manipulation_check/T{0.3,0.9}/`.

---

## Experimental setup

**Models.** Five open-weight families in the 7–9B parameter range:

| Model | HuggingFace ID |
|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Gemma-2-9B-it | `google/gemma-2-9b-it` |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |

All five act both as **judges** and as **evaluated models**, yielding a
fully crossed judge-evaluated design. The mechanistic analysis focuses on
three (Gemma, Llama, Qwen) with the cleanest behavioral signal.

**Task.** LLM-as-judge over 30 prompts from
[Arena-Hard-v2.0-Preview](https://github.com/lmarena/arena-hard-auto)
(pinned, seed=42), stratified equally across coding, creative writing,
and math. The judge receives a question, a candidate response, and a
rubric, and emits a JSON object scoring Relevance and Reasoning Quality
on a 1–10 scale.

**Conditions.** Nine conditions across two arcs:

| Case | Description |
|---|---|
| 1a | Blind baseline (no identity, no framing) |
| 2a | Identity of evaluated model revealed |
| 2b | Competition framing + identity |
| 2c | Competition framing, anonymous |
| 3a | Real prior scores + explicit competition ("revenge") |
| 3b | Real prior scores, no competition framing |
| 4a | Judge's own response, mislabeled as another model's |
| 4b | Another model's response, mislabeled as the judge's own |
| 4c | Fabricated prior scores (2 / 5 / 8 / actual) injected |

Two paraphrase variants (para1, para2) of each system prompt are run via
`behavioral_paraphrase.py` for the robustness check in paper §4.7.

**Sampling.** *K* = 5 judge samples per (judge, evaluated, condition,
prompt) cell at each of two temperatures (*T* = 0.3, *T* = 0.9). Seed = 42
throughout.

**Statistics.** Paired bootstrap with *B* = 10,000 resamples, paired at the
(prompt, evaluated_model) cell. All effect sizes reported with 95% CIs.

---

## Hardware and runtime

Running the full pipeline requires a GPU with ≥24 GB VRAM (tested on NVIDIA
A6000 48GB and A100 80GB).

| Stage | Hardware | Wall-clock |
|---|---|---|
| Behavioral run (5 judges × 5 evaluated × 9 conditions × 30 prompts × 2 temps) | 1× A6000 48GB | ~7–8 hours |
| Mechanistic patching (3 focal models × multi-position + ablation) | 1× A100 80GB | ~6–8 hours |
| Manipulation-check LLM judge (5,500 justifications) | 1× 48GB GPU | ~2 hours |
| Regenerating plots from saved results | CPU | <5 minutes |

---

## Reproducibility notes

- All random seeds fixed at 42 (generation, sampling, bootstrap).
- All comparisons paired at the (prompt, evaluated model) cell level.
- The pinned prompt set is in
  [`data/prompts/arena_hard_v2_30_seed42.json`](./data/prompts/).
- Behavioral pipelines are checkpointed via `results/checkpoints/` and
  resumable after interruption.
- The mechanistic v8 protocol verifies batch-size-invariant attention per
  focal model (required for deterministic patching).
- Large per-call justification logs (~900 MB total) are not tracked in git
  due to file-size limits. They are hosted as a separate dataset that will
  be made available at camera-ready.

To reproduce the full pipeline:

```bash
# 1. Set HuggingFace token (required for gated models: Llama, Gemma)
export HF_TOKEN=hf_...

# 2. Validate environment
python src/behavioral/behavioral_original.py preflight

# 3. (Optional) Re-pin the prompts
python src/behavioral/behavioral_original.py prompts

# 4. Run the behavioral pipeline (resumable)
python src/behavioral/behavioral_original.py run

# 5. Run the paraphrase variants
python src/behavioral/behavioral_paraphrase.py run

# 6. Run the mechanistic patching
python src/mechanistic/mech_interp_v8_multipos.py \
    --prompts-file data/prompts/arena_hard_v2_30_seed42.json \
    --generation-checkpoint results/behavioral/data_original/checkpoints/generation_complete.json \
    --models gemma llama qwen

# 7. Run the manipulation-check LLM-judge analysis
python src/analysis/manipulation_check_llm_judge.py
```

---

## Known caveats

1. **Mistral-7B-v0.3** exhibits score saturation at the rubric ceiling
   and a sign reversal on named-competition. We report results but treat
   them as sensitivity checks.
2. **DeepSeek-R1-Distill** has elevated parse-failure rates (28–42%) due
   to long reasoning traces exceeding the 2048-token budget on some
   conditions. Conclusions involving DeepSeek are based on subsets that
   successfully produced parseable JSON.
3. The mechanistic analysis covers 3 of 5 models (Gemma, Llama, Qwen);
   for Qwen the cell count after filtering is modest. Cross-model cliff
   claims are anchored in Gemma and Llama.
4. All experiments use open-weight models in the 7–9B range. We do not
   test frontier proprietary models or substantially smaller/larger
   open-weight models.

---

## License

Code: MIT (see [`LICENSE`](./LICENSE)).

Data: the Arena-Hard-v2.0 prompts are redistributed under the original
license from
[lmarena/arena-hard-auto](https://github.com/lmarena/arena-hard-auto);
aggregated results released under CC-BY-4.0.

---

## Citation

If you use this code or data, please cite (camera-ready details TBA):

```bibtex
@inproceedings{silent_social_inference_2026,
  title  = {Silent Social Inference in Large Language Models},
  author = {Anonymous},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year   = {2026}
}
```

A `CITATION.cff` file is provided for GitHub's "Cite this repository"
feature; it will be updated with author details upon publication.

---

## Contact

Questions and replication requests welcome via this repository's issue
tracker. For double-blind review, please direct correspondence through
the anonymous submission system.
