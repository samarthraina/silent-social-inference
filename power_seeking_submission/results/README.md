# Results

This directory contains aggregated outputs from the full pipeline run. See the root `README.md` for experimental design details.

## Contents

### `figures/`
18 publication-ready plots (9 chart types × 2 temperatures, each as PNG + PDF):

- `01_case_cascade_T{0.3,0.9}` — Mean total score across 6 main conditions
- `02_self_bias_T{0.3,0.9}` — Self-preference (self − others) per judge in Case 2a
- `03_heatmap_case{1a,2a,3a}_T{0.3,0.9}` — Judge × evaluated score matrices
- `04_score_faking_dose_response_T{0.3,0.9}` — Case 4c: scores vs. alleged prior score
- `05_reasoning_vs_nonreasoning_T{0.3,0.9}` — DeepSeek-R1-Distill vs. non-reasoning judges
- `06_per_prompt_violin_T{0.3,0.9}` — Distribution of cell-level scores across conditions
- `07_stat_tests_T{0.3,0.9}` — Paired-bootstrap effect sizes with 95% CIs

### `stats/`
- `stat_tests.json` — paired-bootstrap effect sizes for all 6 pairwise comparisons, pooled and per-judge, at both temperatures
- `manipulation_check.json` — fraction of judge justifications that explicitly mention model names (Case 2a), score keywords (Cases 3a/3b), or alleged score numbers (Case 4c)

### `summaries/`
- `summary_T{0.3,0.9}.csv` — one row per (judge, case, evaluated_model) with n_cells, avg_relevance, avg_rq, avg_total, std_total, median_total
- `per_prompt_T{0.3,0.9}.csv` — one row per (judge, case, evaluated_model, prompt_id) — used for paired bootstrap

## Files NOT tracked in this repo

These are available on Zenodo (see root README):

- `full_results_T{0.3,0.9}.json` (~34 MB each) — complete per-cell data including justifications
- `cot_log_T{0.3,0.9}.jsonl` (~210 MB each) — every individual judge draw (K=5 per cell) with raw output, parsed fields, and metadata
- `event_log.jsonl` (~14 MB) — timing and progress events
- `checkpoints/` (~82 MB) — intermediate checkpoint files; regenerated automatically when rerunning

Regenerate any of the above by running `python judgebench.py run` from the repo root.
