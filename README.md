# Silent Social Inference in Large Language Models

> Code, data, and experimental outputs for the EMNLP 2026 submission *"Silent
> Social Inference in Large Language Models."* Earlier exploratory work is
> archived in [`_archive/`](./_archive/).

LLMs silently integrate social context into their judgments. When prompts
surrounding a fixed task carry implied identity, competitive stakes, or a
fabricated reputational history, model outputs shift in response — yet the
models almost never mention this context in their stated justifications.
This repository contains the code, experimental prompts, and results for a
controlled study of this phenomenon across five open-weight model families,
plus a mechanistic analysis localizing the integration to specific layers
of the residual stream.

📄 **Paper**: [`paper/silent_social_inference.pdf`](./paper/) *(added at camera-ready)*

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

See [`paper/`](./paper/) for the full report and [`results/`](./results/)
for the underlying data.

---

## Repository structure

```
silent-social-inference/
├── paper/                 # LaTeX source, bibliography, and compiled PDF
├── data/
│   ├── prompts/           # 30 pinned Arena-Hard-v2 prompts (seed=42)
│   └── conditions/        # Prompt templates per experimental condition
├── src/
│   ├── behavioral/        # Behavioral experiment code (paper §4)
│   ├── mechanistic/       # Activation patching code (paper §5)
│   ├── analysis/          # Bootstrap CIs, regression, manipulation checks
│   └── plotting/          # Figure generation scripts
├── configs/               # YAML specs, one per experiment
├── results/
│   ├── behavioral/        # Raw outputs per condition × temperature
│   ├── mechanistic/       # Per-model patching JSON, valence ablation
│   ├── manipulation_check/# GPT-OSS-20B judge outputs over justifications
│   ├── figures/           # Per-experiment exploratory figures
│   ├── stats/             # Bootstrap CIs, effect-size tables
│   └── summaries/         # Aggregated per-cell and per-prompt CSVs
├── figures/               # Final figures referenced by the paper
├── scripts/               # One-shot reproduce scripts
├── notebooks/             # Curated walkthroughs
├── docs/                  # REPRODUCTION.md, DATA_FORMAT.md, HARDWARE.md
└── _archive/              # Earlier exploratory codebase, preserved
```

---

## Quickstart

```bash
# clone this repository
cd silent-social-inference
pip install -r requirements.txt
```

**Regenerate all figures from saved results** (no GPU needed):

```bash
bash scripts/reproduce_figures.sh
```

**Run a single behavioral cell** (one judge, one condition, one prompt) to
verify the pipeline works end-to-end:

```bash
python -m src.behavioral.run_experiment \
    --config configs/behavioral_T0.3.yaml \
    --judge gemma-2-9b-it \
    --condition blind \
    --prompt-id 1
```

See [`docs/REPRODUCTION.md`](./docs/REPRODUCTION.md) for the full pipeline.

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
fully crossed judge-evaluated design. The mechanistic analysis in §5
focuses on three (Gemma, Llama, Qwen) with the cleanest behavioral
signal.

**Task.** LLM-as-judge over 30 prompts from
[Arena-Hard-v2.0-Preview](https://github.com/lmarena/arena-hard-auto)
(pinned, seed=42), stratified equally across coding, creative writing,
and math. The judge receives a question, a candidate response, and a
rubric, and emits a JSON object scoring Relevance and Reasoning Quality
on a 1–10 scale.

**Conditions.** One baseline plus four manipulations:

| Case | Condition | Description |
|---|---|---|
| 1a | Blind | Question, response, rubric only |
| 2a | Identity | Response label = actual model name; judge's identity in system prompt |
| 2b | Competition, named | Competitive framing with competitor names |
| 2c | Competition, anonymous | Competitive framing without identity information |
| 4c | Score-faking | Fabricated prior score injected (low/mid/original/high) |

Plus two paraphrase variants of each manipulation prompt for robustness.

**Sampling.** *K* = 5 samples per (judge, evaluated, condition, prompt)
cell at each of two temperatures (*T* = 0.3, *T* = 0.9). Bootstrap
resampling with *B* = 10,000 paired at the (prompt, evaluated model)
cell.

**Manipulation-check.** Two independent methods over the full
justification corpus: keyword search (lower bound) and GPT-OSS-20B as a
cross-family LLM-judge (upper bound) applied to a stratified sample of
5,500 justifications.

**Mechanistic protocol.** Logit-level expected score at the rubric-token
positions:

$$\mathbb{E}[\text{score}] = \sum_{k=0}^{10} k \cdot \frac{\exp(\ell_k)}{\sum_j \exp(\ell_j)}$$

Single-position vs. multi-position activation patching across the
manipulation span. Minimal-pair (9.0/10 vs 2.0/10) ablation for valence
isolation.

See [`docs/EXPERIMENTAL_SETUP.md`](./docs/EXPERIMENTAL_SETUP.md) for the
full protocol and [paper §3 / §5.1] for narrative justification.

---

## Hardware and runtime

| Stage | Hardware | Wall-clock |
|---|---|---|
| Full behavioral run (5 judges × 5 evaluated × 11 conditions × 30 prompts × 2 temps) | 1× A100 80GB or A6000 48GB | ~7–8 hours |
| Full mechanistic patching (3 focal models × 3 manipulations × 30 cells × all layers) | 1× A100 80GB | ~4–8 hours per model |
| Manipulation-check LLM-judge (5,500 justifications) | 1× 48GB GPU | ~2 hours |
| Figure regeneration from saved results | CPU | <5 minutes |

See [`docs/HARDWARE.md`](./docs/HARDWARE.md) for memory profiles and
batch-size configuration.

---

## Reproducibility notes

- All random seeds fixed at 42 (generation, sampling, bootstrap).
- All comparisons paired at the (prompt, evaluated model) cell level.
- Manipulation prompts and templates released in
  [`data/conditions/`](./data/conditions/) verbatim.
- The attention kernel is verified batch-size-invariant for each focal
  model (required for activation patching to be deterministic).
- Large raw outputs (full per-draw justifications, ~500 MB) are hosted
  separately on Zenodo: **[link TBD]** — they are not tracked in git due
  to file-size limits.

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
