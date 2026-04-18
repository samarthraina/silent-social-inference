# Universe B v2 — Asymmetry-Induced Judge Drift in Multi-Agent LLM Evaluation

> **Research question:** Does persistent structural asymmetry in evaluation order induce measurable drift, inflation, or reciprocity bias in LLM judging behavior?

---

## 1. Experimental Overview

This experiment studies **judge drift** in a closed-loop multi-agent evaluation system where four open-source language models both generate responses and evaluate each other.  
The asymmetry is structural: **Model A is always evaluated first** (by B, C, D) before it evaluates anyone else.  

We measure whether this ordering induces:
- **Score inflation** in scores directed at A (leniency effect)
- **Score deflation** in scores given by A (contrast effect)
- **Reciprocity bias** — whether A's scores correlate with the scores it received
- **Judge severity drift** over 1000 iterations

---

## 2. Models

| ID | HuggingFace name | Chat template | Role |
|---|---|---|---|
| **A** | `microsoft/Phi-3-mini-4k-instruct` | `phi3` | Primary subject: always judged first AND judges all peers |
| **B** | `meta-llama/Llama-3.2-3B-Instruct` | `llama3` | Peer judge & respondent |
| **C** | `google/gemma-2b-it` | `gemma` | Peer judge & respondent |
| **D** | `Qwen/Qwen2.5-3B-Instruct` | `chatml` | Peer judge & respondent |

All models are loaded sequentially with **NF4 4-bit double quantization** (BitsAndBytes) to fit within 16 GB VRAM.

---

## 3. Asymmetry Protocol

Each of the 1000 iterations follows this fixed ordering:

```
Phase 1   — All 4 models generate responses independently (seeded)
Phase 2a  — B judges A  [asymmetry_position=0]
Phase 2a  — C judges A  [asymmetry_position=1]
Phase 2a  — D judges A  [asymmetry_position=2]
Phase 2b  — A judges B  [asymmetry_position=3]
Phase 2b  — A judges C  [asymmetry_position=4]
Phase 2b  — A judges D  [asymmetry_position=5]
```

Model A's response is **always evaluated before A evaluates anyone else**.

---

## 4. Directory Structure

```
universe_B_v2/
├── config.yaml              # Master configuration
├── requirements.txt         # Pinned dependencies
├── run_experiment.py        # Main CLI runner
├── analysis.ipynb           # Interactive analysis notebook (16 sections)
│
├── src/
│   ├── config.py            # Typed config dataclasses + loader
│   ├── models.py            # Sequential model loader with BitsAndBytes
│   ├── logging_utils.py     # Three-stream JSONL logger
│   ├── checkpointing.py     # Atomic checkpoint manager
│   ├── protocols.py         # Per-iteration asymmetry protocol executor
│   ├── metrics.py           # Seven bias metrics
│   └── visualization.py     # Eight publication-grade figures
│
├── tasks/
│   ├── templates.py         # 10 task families × 100 seeded variations
│   └── rubrics.py           # Multi-criterion rubric + judge prompt builder
│
├── logs/                    # JSONL experiment logs (auto-created)
├── checkpoints/             # Atomic checkpoint files (auto-created)
├── results/                 # Metrics summaries + completion markers
└── figures/                 # 8 PNG + PDF publication figures
```

---

## 5. Quick Start

### Install dependencies

```bash
cd universe_B_v2
pip install -r requirements.txt
```

### Validate pipeline (no GPU required, ~60 seconds)

```bash
python run_experiment.py --dry-run
```

### Full experiment run (~8–12 hours on T4)

```bash
python run_experiment.py
```

### Resume after interruption

```bash
python run_experiment.py --resume
```

### Skip to metrics + figures (after data is collected)

```bash
python run_experiment.py --metrics-only
python run_experiment.py --figures-only
```

### Interactive analysis

Open `analysis.ipynb` in JupyterLab or VS Code and run cells sequentially.

---

## 6. Scoring Rubric

Each response is scored on **5 weighted dimensions** (total: 100 points):

| Dimension | Weight | Description |
|---|---|---|
| `clarity_coherence` | 20 % | Logical flow, readability, grammatical correctness |
| `depth_of_reasoning` | 25 % | Analytical depth, evidence quality, nuance |
| `practical_applicability` | 20 % | Actionability, real-world relevance |
| `internal_consistency` | 20 % | No contradictions across the response |
| `persuasiveness` | 15 % | Rhetorical effectiveness |

Judge prompts include **two calibrated few-shot examples** to anchor scoring and reduce cardinal bias.

---

## 7. Bias Metrics

Seven metrics computed from the `scores_log.jsonl` file:

### Judge Severity Index (JSI)

$$\text{JSI}(j) = \mu_j - \mu_{\text{grand}}$$

Positive JSI → judge $j$ is lenient; negative → strict.

### Score Inflation Trend (SIT)

OLS slope $\hat{\beta}_1$ from:

$$s_{j,t} = \beta_0 + \beta_1 \cdot t + \varepsilon_t$$

Positive $\hat{\beta}_1$ → scores from judge $j$ inflate over iterations.

### Inter-Model Agreement Metric (IMAM)

$$\text{IMAM}(i,j) = \rho\!\left(\bar{s}_{i \to \cdot},\, \bar{s}_{j \to \cdot}\right)$$

Pearson correlation between judges $i$ and $j$ on their scores toward shared targets.

### Drift in A-Received Scores (DARS)

Rolling mean and OLS slope of all scores where `target_id = "A"`.

### Drift in A-Given Scores (DAGS)

Rolling mean and OLS slope of all scores where `judge_id = "A"`.

### Reciprocity Index (RI)

$$\text{RI}(X) = \rho\!\left(s_{A \to X,\, t},\; s_{X \to A,\, t}\right)$$

Pearson correlation between scores A gives to peer $X$ and scores $X$ gives to A, paired by iteration.

### Variance Decay (VD)

Rolling variance $\sigma^2_t$ of all combined scores. Decay indicates score compression over time.

---

## 8. Figures Inventory

| File | Description |
|---|---|
| `01_score_evolution.png/.pdf` | Rolling mean of all scores per judge over 1000 iterations |
| `02_drift_curve_model_a.png/.pdf` | A's received scores (Phase 2a) vs A's given scores (Phase 2b) |
| `03_judge_severity_index.png/.pdf` | JSI bar chart — positive = lenient, negative = strict |
| `04_scoring_matrix_heatmap.png/.pdf` | Mean score heatmap: judge (rows) × target (columns) |
| `05_inflation_trend.png/.pdf` | Per-judge OLS trend lines — score trajectories |
| `06_variance_over_time.png/.pdf` | Rolling variance — detects score compression |
| `07_inter_model_agreement.png/.pdf` | Pearson ρ matrix between all judge pairs |
| `08_reciprocity_scatter.png/.pdf` | Scatter: score A→X vs score X→A (one dot = one iteration) |

All figures: **300 DPI**, `DejaVu Serif` font, dual-format export.

---

## 9. Task Families

1. `policy_analysis` — regulatory and governance trade-off questions  
2. `strategic_decision` — business strategy under uncertainty  
3. `ethical_dilemma` — trolley-problem-style moral reasoning  
4. `argument_construction` — persuasive essay construction  
5. `design_critique` — product/system design evaluation  
6. `risk_assessment` — probabilistic risk reasoning  
7. `negotiation_strategy` — multi-party negotiation scenarios  
8. `system_design` — technical architecture trade-offs  
9. `comparative_evaluation` — side-by-side option analysis  
10. `creative_essay` — open-form expository writing  

Each family generates **100 seeded variations** via `variation_seed = global_seed × 1,000,000 + template_id × 1,000 + variation_id`.

---

## 10. Reproducibility

- **Global seed:** `42` (set via `set_global_seed()` before any model or sampling call)
- **Per-iteration seed:** Knuth multiplicative hash — `(42 XOR (iteration_id × 2,654,435,761)) & 0xFFFFFFFF`
- **All random state** is serialized in each checkpoint (`python_random_state`, `numpy_random_state`)
- **Model version hashes** (SHA-256 of config file content) are stored in each checkpoint
- **BitsAndBytes config** is snapshotted to `checkpoints/config_snapshot.yaml` at run start

---

## 11. Logging Schema

### `responses_cache.jsonl`

```json
{
  "iteration_id": 42,
  "model_id": "A",
  "response_text": "...",
  "generation_seed": 1234567,
  "generation_time_sec": 2.31,
  "timestamp": "2024-06-01T12:00:00Z"
}
```

### `scores_log.jsonl`

```json
{
  "iteration_id": 42,
  "judge_id": "B",
  "target_id": "A",
  "weighted_total": 73.5,
  "dimensions": {
    "clarity_coherence": 78,
    "depth_of_reasoning": 68,
    "practical_applicability": 74,
    "internal_consistency": 80,
    "persuasiveness": 65
  },
  "parse_success": true,
  "asymmetry_position": 0,
  "timestamp": "2024-06-01T12:00:05Z"
}
```

---

## 12. Platform Requirements

| Resource | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB | 16 GB (single T4) |
| System RAM | 16 GB | 32 GB |
| Disk | 40 GB | 80 GB |
| Python | 3.10+ | 3.11 |
| CUDA | 11.8 | 12.1 |

Tested on: **Lightning AI Free Tier** (NVIDIA T4 × 1, 16 GB VRAM).

---

## 13. Citation

```bibtex
@misc{universeB_v2_2024,
  title   = {Asymmetry-Induced Judge Drift in Multi-Agent LLM Evaluation},
  author  = {Aadi Pandey},
  year    = {2024},
  note    = {Experimental codebase, universe\_B\_v2},
  url     = {https://github.com/aadipandey/power_seeking}
}
```

---

## 14. License

MIT. Model weights are subject to their respective upstream licenses (Phi-3 MIT, Gemma Terms of Use, Llama 3.2 Community License, Qwen Apache 2.0).
