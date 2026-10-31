# Residual discovery v3: free-generation history-state interventions

This independent extension asks whether replacing a localized internal **prompt** state transfers the positive-versus-negative history effect during ordinary free generation. It complements the attention-window interventions elsewhere in this branch. It does not replicate their score-prefix readout or claim a validated circuit.

**Start with [the two-page TLDR](residual_discovery_TLDR.pdf).**

## Completed results

Same 18 discovery task identities per model, balanced across MT-Bench, Arena-Hard and WildBench; fixed candidate answers. Native self_high_10 / self_low_2 prompts, greedy free-text evaluation plus a numeric score, 512-token cap. Llama-3.1-8B-Instruct and Qwen3-8B use pinned revisions in the reference loader. Runs used BF16, SDPA, cached decoding, seed 42, one A100 PCIe 40 GB, sequentially. Runtime libraries are recorded in each run's metadata and pip-freeze file.

| Result | Llama | Qwen |
|---|---:|---:|
| Mean score: high history | 9.33 | 9.69 |
| Mean score: low history | 7.22 | 7.22 |
| Mean paired gap (points) | 2.11 | 2.47 |
| Valid cross-patches with unchanged score | 685 / 862 (79.5%) | 770 / 861 (89.4%) |
| Discovery elapsed GPU-run time | 167.31 min | 191.19 min |

The grid contains 24 residual sites: post-block layers 4,8,12,16,20,24,28,32 x history_end / answer / final. `history_end` means the last native token overlapping the history sentence, not the full history or rating numeral. `answer` covers the frozen answer; `final` is the final prompt token. Source and target retain their native lengths; semantic positions must align bijectively. Each direction replaces states only on prefill and leaves decoding untouched. Other information and caches may still carry the history effect.

Each model has 1,764 planned records: 36 baselines, 864 same-state controls and 864 cross-patches. All 1,728 discovery same-state checks across models reproduced exact generated token hashes. All 24 embedding-replay positive controls passed. Five discovery scores are unusable (four capped, one parse failure); they remain missing, never zero. All raw records are retained.

Modest, post-search bidirectional leads occur at Llama L12/history_end (+0.306 / +0.235 points; n=18/17) and Qwen L16/history_end (+0.389 / +0.139; n=18/18). They maximize the smaller directional mean. Median effects are zero, and exploratory 98.75% intervals reach/cross zero; these intervals do not adjust for searching 24 sites. No unseen-task mechanism claim is supported yet. Qwen L27 was not sampled.

## Contents and CPU reproduction

- Exact executed `run_targeted.py`, `interventions.py`, configurations and frozen split.
- `reference/`: minimal original-study input/helper snapshot; upstream revision and third-party terms in [PROVENANCE.md](PROVENANCE.md).
- `results/`: successful discovery raw records, metadata, summaries, smoke controls, token preflights and environments.
- `verify_results.py`: checks all payload hashes, planned unit coverage, frozen executor/hooks/configuration hashes, same-state outputs and positive controls; reconstructs score effects and baseline intervals.
- `analysis/`: numerical audit, all 96 directional grid effects and both figures.
- `make_tldr.py`: rebuilds both figures and the short PDF.
- `MANIFEST.sha256`: SHA-256 inventory. No weights, access tokens or server-administration scripts are included.

From this directory, in a Python 3.12+ environment:

```bash
python -m pip install numpy matplotlib reportlab
python verify_results.py
python make_tldr.py
```

The audit uses a 20,000-draw task-stratified bootstrap (seed 20261031) for 95% baseline gap intervals. Site intervals come from the recorded original analyzer (20,000 draws, seed 20261030). To recompute the original site summaries independently, use a fresh output path:

```bash
python analyze_run.py --run results/discovery_v3_llama_20261030T231649Z_m73tYe/run --split split.json --candidates discovery_v3_llama.json --out /tmp/llama-residual-summary.json
python analyze_run.py --run results/discovery_v3_qwen_20261030T195652Z_STN5KL/run --split split.json --candidates discovery_v3_qwen.json --out /tmp/qwen-residual-summary.json
```

For the 20 unit/architecture tests, install PyTorch and Transformers, then:

```bash
REPO="$PWD/reference" HF_HUB_OFFLINE=1 python -m unittest discover -s . -p test_followup.py -v
```

These architecture tests use tiny randomly initialized models; they download no weights. Recorded research inference used torch 2.14.1+cu126 and transformers 5.15.0. The local portability tests also passed using torch 2.15.0 and transformers 5.22.0; this is not a claim of identical inference across library versions.

## GPU rerun (only if needed)

Use a separate, empty output directory. The executor never resumes into existing directories and never silently skips partial tasks. Authenticate locally with `hf auth login` if required for gated Llama weights. Do not paste tokens into source or commands committed to Git.

```bash
python -m pip install -r reference/extensions/context_controls/requirements-gpu.txt
python preflight_tokens.py --repo reference --package . --model qwen --out /tmp/qwen-residual-preflight.json
python run_targeted.py --repo reference --split split.json --candidates smoke_v3_qwen.json --phase engineering --model qwen --out /tmp/qwen-residual-smoke --max-calls 180 --max-seconds 3600
python run_targeted.py --repo reference --split split.json --candidates discovery_v3_qwen.json --phase discovery --model qwen --out /tmp/qwen-residual-discovery --max-calls 3492 --max-seconds 21600
```

Use `llama` and matching llama configuration filenames for Llama. Review engineering identity and embedding-recovery controls before discovery. Running these commands incurs provider compute charges; this scientific package does not stop or destroy an instance.

## Not completed / next experiments

The reserved 72-task validation inference has **not** been run. The split was frozen against the original recorded manifests, but other collaborators' experiments overlap parts of it. Audit joint exposure before choosing a fresh confirmation set. Do not describe all 72 tasks as unseen.

Proposed complementary experiments, not implemented or run in this release:

1. Block attention to history only for generated-token readers, leaving prompt processing intact.
2. A four-arm factorial test: baseline, prompt-state replacement, generated-token blocking, both together. Use the identical mask implementation even in the unblocked baseline. Experiments 1 and 2 share arms.
3. Compare ordinary generation with controlled, history-independent rationale prefixes, with matched length/readout controls. This can test rationale dependence, but changes to a rationale also change the scoring context and require careful interpretation.

Use collaborators' window findings as credited, externally informed hypotheses. Start with a small discovery pilot, fix the interventions and analysis, then validate on jointly audited held-out tasks. The tested grid alone does not establish that generated-token re-reading explains the effect.

Planning estimate for both models sequentially on one A100 40 GB: **3-6 GPU hours for a 12-task/model pilot of all three experiments**, plus **3-6+ additional GPU hours for confirmation**, depending on task count, output length, masking overhead and controls. Implementation/testing/analysis adds roughly 1-2 working days. These are planning ranges, not measurements of the unimplemented interventions; time a small pilot before committing the full budget.
