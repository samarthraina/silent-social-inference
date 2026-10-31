# Focused confirmation: does generated-token access to history carry the score effect?

**Status: prospective code prepared; no held-out GPU outcomes have been collected by this package.** The task-exposure author review and a reachable GPU instance are still needed to start a real run. A test creates only temporary, explicitly synthetic locks; those are never research authorization or results.

The completed pilot found history-gap reductions of about 45% (Llama) and 71% (Qwen), with positive history-versus-content-control contrasts. It did not establish a reliable patch interaction, and its rationale experiment was unusable because of generation limits. This confirmation therefore tests the history-reading result, while preserving those other pilot outcomes as reported. It is a prospectively selected follow-up, not an independent preregistration of the exploratory pilot.

## Exactly what runs

Each model evaluates each frozen task under high and low history in four conditions:

1. `baseline`: matched explicit causal mask, no blocked links.
2. `native_baseline`: actual original Hugging Face cached generator.
3. `generated_block`: only generated input-token readers are prevented from attending to history, in all decoder blocks (32 Llama, 36 Qwen).
4. `control_block`: same number of question/answer keys blocked for the same readers and layers.

These are exactly the already tested v3 backend arms, with the same 512-token limit, BF16/SDPA, seed 42, native prompts, frozen answers and cap-aware score parser. **No changes are made to the v3 backend.** The native baseline measures computation-path sensitivity; it is not the baseline for the causal mask contrast. The first output prediction remains a prompt-reader prediction and must be unchanged by generated-reader-only blocking.

The recorded exposure manifests leave **44 candidate tasks: 1 MT-Bench, 21 Arena-Hard, 22 WildBench**. See `candidate_task_ids.txt`. Additional author-reported task IDs and exact-question hashes are excluded before freezing. Final task count can therefore be smaller. This is not a balanced three-benchmark replication, and tasks are not claimed behaviorally unseen, semantically novel or absent from pretraining.

At 44 tasks, there are 352 scored generations per model (704 total). Each model first repeats its six engineering condition checks on the three previously exposed engineering tasks. Those engineering records never enter the confirmation's task-level statistics.

## Prospective outcomes and decision rule

For every task, define gap = high-history score minus low-history score.

- Primary 1: gap(matched baseline) minus gap(history blocking).
- Primary 2: gap(content control) minus gap(history blocking).

Report all four tests (two outcomes x two models) with two-sided 98.75% stratified paired-task bootstrap intervals (100,000 draws, seed 20261031), preserving complete-case counts by dataset. Preserve raw invalid/capped outputs as missing. Also report score-range [0,10] worst-case bounds over all planned tasks; no zero imputation or after-the-fact score recovery.

For a model, the program flags both predeclared checks positive only if both interval lower limits and both planned-task missingness-bound lower limits exceed zero. This flag is a numerical check, not a full mechanistic conclusion. Inspect absolute scores, raw generation quality and benchmark counts: question/answer-key blocking is a competing-content control, not an inert intervention. Small-sample percentile bootstrap coverage is approximate.

The native-versus-explicit gap difference gets a descriptive 95% interval and missingness bounds. It is separate from the four primary tests. An interval including zero does not establish native-path equivalence. Report null and negative outcomes as well as positive ones.

A confirmed result supports a causal contribution from direct history access during generation within the matched computation path. It does not identify a unique layer/head, a complete circuit, natural rationale mediation, a social grudge, real interaction effects or frontier-model generalization.

## Proof files and exposure review

`evidence/` in the complete local upload archive contains byte-preserved v3 engineering and pilot records. It is omitted from GitHub; it contains unpublished research outputs. The runner verifies source/configuration hashes, raw record coverage, payloads and the full original pilot controls. GitHub-only users must supply those evidence directories separately.

Copy `exposure_review.template.json` to a new review file. Fill the reviewer name, actual time zone-aware review date, actual exposure statement, and any additional exposed task IDs/question hashes. Set `author_confirms_remaining_tasks_not_used_mechanistically` true only after checking with the team. This is missing scientific provenance, not an extra permission to spend GPU credits. Do not invent the attestation. The original and teammate branches were checked against their frozen recorded revisions during preparation, but unrecorded/local team experiments cannot be inferred from GitHub.

Freeze the confirmation before running any held-out task:

```bash
python freeze_confirmation.py --audit /workspace/exposure-review.json \
  --out /workspace/history-confirmation-locks
```

This verifies evidence for both models, excludes additional exposures, and writes immutable source/plan/protocol/audit snapshots. It refuses to overwrite an existing lock directory. The frozen endpoints are selected from the reported pilot before confirmation outcomes, and the old unusable experiments are not erased or relabeled as successes.

## GPU launch after freezing

Keep this folder next to `generation_path_v1` and `residual_discovery_v3`, exactly as in the complete upload archive. Use the same isolated environment that successfully ran v3; the code checks the original software versions, model revisions and chat template. The current GPU type is recorded and the original engineering checks are rerun on it.

From this folder, with the environment active and Hugging Face login cache configured:

```bash
bash run_confirmation_both.sh /workspace/history-confirmation-locks /workspace/history-confirmation-results
```

An external launcher can activate the already tested isolated environment, freeze a reviewed plan if needed, and start the sequential batch:

```bash
bash launch_confirmation.sh /workspace/exposure-review.json \
  /workspace/history-confirmation-locks /workspace/history-confirmation-results
```

It uses `/workspace/generation-path-venv` and `/workspace/hf-cache` by default, or explicit `CONFIRM_VENV` / `HF_HOME` values. It refuses a changed exposure review for an already frozen lock. It does not create an author attestation or start a rented instance. If setting up a fresh server, create an ordinary isolated venv (without system-package inheritance), install `requirements-gpu.txt`, and log in with `hf auth login` using the same `HF_HOME`.

Qwen runs before Llama. The script exits on failed checks or an execution error. It does not run the rationale or residual-patch arms. It does not rent/start/stop/destroy a cloud instance. After both models finish, it generates JSON and a brief Markdown table.

For a single model:

```bash
python confirm.py --locks /workspace/history-confirmation-locks --model qwen \
  --out /workspace/history-confirmation-results/qwen --max-seconds 10800
```

The three-hour wall-clock limit is per model/invocation. Interruption or timeout preserves synced completed units. Rerun the batch script to resume. A lock records the first output path to prevent accidentally starting the same confirmation in a second directory; restore interrupted results to that same path. Source/protocol/runtime changes are refused. Do not retune or selectively rerun confirmation outcomes.

For unattended SSH use, make an external launcher that activates the existing Python environment, exports the correct `HF_HOME`, changes to this directory and executes the batch. Run that launcher in `tmux`. The package deliberately does not embed a host address or credentials. The previous address refused connections during preparation; obtain the current SSH Connect command before launching.

Based on the measured pilot's benchmark-specific timing, the four-arm 44-task subset projects to about 94 minutes of generation for both models, plus engineering rechecks, loading and task-length uncertainty. Budget roughly **2–3 GPU hours**, then revise from actual progress. Longer outputs or a different GPU can change this.

## Verification and analysis

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s . -p test_confirmation.py -v
```

Tests verify exposure filtering, missing-review refusal, source/lock changes, interruption/resume, score and mask invariants, and full summary coverage using explicitly synthetic temporary data. They also verify original real pilot evidence. They generate no held-out model predictions.

To reanalyze completed confirmation runs:

```bash
python summarize_confirmation.py \
  --run /workspace/history-confirmation-results/qwen \
  --run /workspace/history-confirmation-results/llama \
  --out /workspace/history-confirmation-results/recomputed_summary.json
```

Only complete runs with full rechecks, matching locks and all planned units enter analysis. The output includes all primary effects, complete-case intervals, worst-case missingness bounds, absolute mean scores, per-benchmark gaps, invalid-output counts and native-path sensitivity.

## Reproducibility

Source backend: `silent-social-inference` branch `residual-discovery-v3`, commit c03183a39e4e15fab8dceff0da9b0725699fb69c. That backend retains the original `silent-influence` snapshot f8010273445adcf276c094cf5b228fc7bd0acf2a. Teammate membership source: commit 630b1c6813a9b7578d9a7825f0ef1dd17a49faa8. The original and teammate branch heads still matched those recorded revisions when checked for this package. Included upstream license/notices remain applicable. No weights are bundled.

Optional figures, after installing Matplotlib, show both primary outcomes for both models with the adjusted intervals and separate missing-score bounds:

```bash
python plot_confirmation.py --summary /workspace/history-confirmation-results/confirmation_summary.json --out /workspace/history-confirmation-results/figures
```
