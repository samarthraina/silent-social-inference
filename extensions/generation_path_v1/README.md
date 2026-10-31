# Generation-path follow-up: three complementary causal experiments

**Status: implemented and locally tested; the real 8B GPU experiments have NOT run.**

This follows the completed [residual discovery v3](../residual_discovery_v3/README.md). It asks whether the small prompt-state transfers missed information that is read again during generation, and whether an evaluation prefix can carry history-sensitive scoring information. It complements the collaborator's prompt/window work rather than repeating their layer-window search.

## What the three experiments do

1. **Block history reading during generation only.** The original prompt is processed normally. Starting with generated input tokens, attention to the history sentence is blocked in all 32 blocks. Compare with a baseline using the identical explicit causal mask, and a control blocking an equal number of non-history question/answer tokens. This tests direct generated-reader access to history. Other prompt states, previous generated tokens and cached states can still carry history information.
2. **Combine that blocking with your state replacements.** Compare baseline, blocking alone, state replacement alone and both together, in both high-to-low and low-to-high directions. Use your discovery leads (Llama L12/history_end; Qwen L16/history_end), a fixed L2 control, and same-state identity controls. The interaction tests whether blocking changes the efficacy of the state replacement. A positive interaction would support that specific causal interaction, not establish a complete circuit.
3. **Hold the written evaluation prefix fixed.** Generate short qualitative evaluations under high, low and no history. For each task, take the same token length from all three (minimum available length, capped at 96; at least 8). Reuse each prefix under both target histories, then generate a final score. Include an equal-length generic prefix and an empty-prefix reference. This tests controlled rationale-prefix dependence while reducing length confounding. It is a different output protocol from ordinary free generation; it does not by itself prove natural reasoning mediation or truthful explanations.

The first output token is predicted from the final **prompt input** token, so generated-reader-only blocking must not alter that first prediction. The code and analysis explicitly check this boundary.

## Task split and scope

The frozen task plan is selected using task IDs, question hashes and recorded exposure only; no scores are consulted.

| Stage | Tasks per model | Purpose |
|---|---:|---|
| Engineering | 3 | One per benchmark; plumbing checks, 64 generated tokens per equivalence check |
| Pilot | 12 | Four per benchmark, from the already exposed discovery split |
| Candidate confirmation | 44 | Excludes both collaborators' recorded mechanistic discovery/validation tasks |

Confirmation composition: **1 MT-Bench, 21 Arena-Hard, 22 WildBench**. These tasks are mechanistically unexposed according to the included manifests and exact-question hashes; they are not necessarily behaviorally unseen, semantically novel or free of unrecorded exposure. An author must audit additional exposure before freezing validation. New MT-Bench inputs would be needed for a balanced benchmark claim.

The pilot has **348 planned units/model** (29/task): 16 ordinary evaluation generations, 3 rationale donors and 10 score continuations. An invalid rationale set produces explicit missing placeholders; it is never silently replaced or scored as zero. Both models together have 696 pilot units. Confirmation would have 1,276 units/model (2,552 total).

## Safeguards and analysis

- Same native prompts, pinned model revisions, BF16/SDPA, greedy decoding, cache-enabled generation and seed 42.
- Identical explicit causal masks in every scientific arm. Controls check native generation, automatic masking, no-cache recomputation, no-op hooks and same-state patching before the pilot.
- Active mask plumbing control plus real Llama/Qwen tiny-model attention tests. Source capture and residual editing use one-based post-block layers; attention blocks are zero-based.
- Each completed unit is flushed and synced immediately. Resume skips only checksum-verified completed units. Output directories, model/runtime versions, protocol, inputs and source hashes are bound; changed definitions cannot be mixed into an existing run.
- A wall-clock bound pauses the run safely. `Ctrl-C` or termination preserves completed units. A dropped SSH connection does not stop a run inside `tmux`.
- Six predeclared primary contrasts/model, with a 12-test family across both models. Task-level paired/stratified bootstrap, two-sided Bonferroni-adjusted intervals (99.5833%, 100,000 draws). Pilot results remain exploratory.
- Incomplete/capped outputs remain missing under the original frozen score parser. Report complete-case intervals, every missing unit, and worst-case bounds over all planned tasks. No observation-level pseudoreplication, hidden score recovery, denominator-based effect inflation or candidate search on validation.

See [METHODS.md](METHODS.md) for the exact contrasts and interpretation limits.

## Running on the GPU server

Keep this folder next to `residual_discovery_v3`; the packaged archive preserves that structure. Activate your existing Python environment. If starting a new environment, install:

```bash
python -m pip install -r requirements-gpu.txt
```

If required, authenticate interactively with `hf auth login` in that environment and the same `HF_HOME` used by the run. Never paste a token into source code. Existing cached authenticated weights can be reused.

From this folder, start **engineering then pilot, Qwen then Llama, sequentially**:

```bash
tmux new-session -d -s generation-path -c "$PWD" \
  -e "HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}" \
  'bash run_pilot_both.sh /workspace/generation-path-results > /workspace/generation-path.log 2>&1'
tail -f /workspace/generation-path.log
```

The four-hour limit applies to each model/stage invocation, not the whole batch. If it pauses or is interrupted, rerun the same batch command; existing verified units are resumed. Do not start a second copy while one is active. `Ctrl-C` during `tail -f` stops watching the log; it does not stop the experiment. To interrupt the process in its tmux session, attach using `tmux attach -t generation-path`, then press `Ctrl-C`.

The batch exits on a failed engineering check or any other error. It **does not run validation, stop an instance, or destroy data**. Back up the result directory and stop the GPU instance in your provider dashboard when finished.

For one stage/model explicitly:

```bash
python run.py --plan frozen/plan.json --stage engineering --model qwen \
  --out /workspace/generation-path-results/qwen_engineering --max-seconds 3600
python run.py --plan frozen/plan.json --stage pilot --model qwen \
  --engineering /workspace/generation-path-results/qwen_engineering \
  --out /workspace/generation-path-results/qwen_pilot --max-seconds 14400
```

Use `--resume` with the same paths to continue an interrupted run. Do not re-upload/redefine code between interruptions; resume intentionally rejects changed source. For an unfinished final JSONL line after a hard shutdown, `python repair_tail.py /path/to/raw.jsonl` first creates a backup and removes only an invalid incomplete final line. It refuses to discard a complete JSON record.

## Validation after reviewing the pilot

Do not silently adjust the protocol after pilot results and call the original pilot confirmatory. Changes require new versioned protocol/code, a new engineering gate and pilot, and a fresh lock before validation. The included candidate confirmation set must not be used for tuning.

1. Review the two pilot summaries and engineering checks.
2. Make an exposure audit from `exposure_audit.template.json`: bind the plan SHA-256, list any additional task exposure, and enter an author name/date. Do not attest until the team has checked all intervening work.
3. Freeze each model's lock, then run validation once. Existing partial runs can resume under the same lock.

```bash
python lock_validation.py --model qwen --plan frozen/plan.json \
  --peer-split frozen/peer_splits.json \
  --engineering /workspace/generation-path-results/qwen_engineering \
  --pilot /workspace/generation-path-results/qwen_pilot \
  --exposure-audit /workspace/exposure-audit.json --out /workspace/qwen-validation-lock.json
python run.py --plan frozen/plan.json --stage validation --model qwen \
  --engineering /workspace/generation-path-results/qwen_engineering \
  --lock /workspace/qwen-validation-lock.json \
  --out /workspace/generation-path-results/qwen_validation --max-seconds 28800
```

Repeat with `llama`; never run the two models concurrently on this 40 GB GPU. Analyze both complete validation runs together:

```bash
python analyze.py --run /workspace/generation-path-results/qwen_validation \
  --run /workspace/generation-path-results/llama_validation \
  --out /workspace/generation-path-results/validation_summary.json
```

The same analysis command with `*_pilot` paths reconstructs the exploratory pilot summary. The batch does this automatically after both pilots finish. Optional figures can be made on your Mac or server after installing `requirements-analysis.txt`:

```bash
python plot_results.py --summary /path/to/pilot_summary.json --out /path/to/new-figure-directory
```

This plots every primary contrast, null results included, with complete-case confidence intervals and separate all-task missing-score bounds.

## Local checks and practical timing

```bash
HF_HUB_OFFLINE=1 python -m unittest discover -s . -p test_generation_path.py -v
```

Tests use tiny randomly initialized real Llama/Qwen architectures and clearly synthetic temporary adapter data. They do not download 8B weights or count as research results. See `TEST_REPORT.json` for verified local environments.

Previous actual A100 discovery times were 167 minutes for Llama and 191 for Qwen. Those involved a different mixture of generations/captures; the new runner also uses caching to avoid full-prompt recomputation at every generation step. Budget roughly **3-6 GPU hours for the two-model pilot**, provisionally, and time it before estimating the 44-task confirmation. Confirmation has 3.67 times as many tasks as the pilot; longer tasks or invalid-donor rates can change throughput. There is no measured GPU runtime for this new protocol yet.

No claim about frontier models, real agent interactions, or the frequency of factually incorrect criticisms follows from these interventions alone.
