# Mech v2 TL;DR: where the history effect is carried

Full plan and run log: [mech_plan_v2.md](mech_plan_v2.md). Charts: [results/mech_v2/analysis.html](../results/mech_v2/analysis.html).

## The question
A judge model that sees "previously scored your answer 10/10" vs "2/10" gives very different scores (gap ≈ 2 points).
The released matched-history patch (single final-input token, Llama L24 / Qwen L27) moved ~0%.
Reviewers read that as "small / Llama-only" mech evidence. We asked where the effect is actually carried.

## What we did
- **Checked the patching setup first (S1).** Continuous readout (expected score from the score-token distribution) agrees in sign with greedy scores on 12/12 tasks. Positive controls pass. The released single-token null **replicates** (move ≈ 0, same as an L2 control). It was not readout noise; it was the wrong site.
- **Discovery on frozen discovery tasks only (36 tasks, S2–S5).** Attention-edge knockouts (post-history positions ← history tokens) by block window, bisection, residual patching at all post-history positions, head ranking.
- **One frozen validation on 72 unseen tasks per model (S6).** Splits, windows, sites, thresholds and success rules were hashed before the run. 95% CIs from a benchmark-stratified task bootstrap (10k draws).
- **Second model (Qwen3-8B)** with a cross-model prediction written before any Qwen GPU run.

## Result (72 held-out tasks per model)

| | Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| Released single-token site | 0% | 0% |
| Read window (post-history reads history) | blocks 8–15 | blocks 9–26 |
| **Necessity**: block that reading → gap removed | **95%** [93, 97] | **97%** [96, 98] |
| Control windows | 21%, 3% | −7%, 5% |
| **Sufficiency**: patch post-window state, high→low / low→high | **0.99 / 0.91** | **0.95 / 0.91** |
| L2 control patch | ≈ 0 | ≈ 0 |
| low↔no-history continuity (paper had 22%) | 0.98 / 0.97 | 0.89 / 0.99 |
| Greedy generation: knockout removes gap | **92%** [83, 101] | **93%** [79, 105] |
| Greedy generation: prompt-only patch shifts score | fails | fails |

**One sentence:** the history sentence is read by the tokens after it (mostly the template tail, not the answer) in a mid-depth block window, and the state written there is necessary (~96%) and sufficient (~0.9–1.0) for the score shift, in both models, on unseen tasks.

## How this answers the reviewers
- **"Too small (~20%)":** the 20% came from patching one token. Patching the right positions recovers 0.89–0.99; blocking the read removes 95–97%.
- **"Llama only":** the same form holds on Qwen3-8B with frozen pre-registered tests.

## Honest limits (report them)
- **Prompt-only patching does not move free generation** (both models). Likely cause: generated tokens re-read the history in the same window; the knockout (which blocks that) does remove 92–93% of the greedy gap. The confirming test is not run.
- **No head-level circuit.** Attention-ranked head sets were at random-control level; single heads have opposing signs. The claim is window-level.
- **Qwen prediction partly missed.** Predicted window 9–17; actual reading spans 9–26. Qwen control windows are narrower (9 vs 18 blocks).
- **Readout deviation.** Pre-registered gate picked a "forced" readout whose control failed (58%, because the model's own justification carries part of the gap). Switched to the direct readout after S1, before any S2 data, logged in the run log.

## Deferred (not run)
Causal per-head scan over blocks 8–15; rank-1 valence-direction patch; prompt patch + generated-token knockout.

## Reproduce
```bash
PYTHONPATH=src python -m pytest tests/test_mech_v2.py
PYTHONPATH=src python -m mech_v2.analysis s6 results/mech_v2/results/s6_llama.jsonl --spec data/mech_v2/freeze_s6_llama.json
```
GPU stages: `modal run src/mech_v2/modal_app.py --stage s1..s6 --model llama|qwen` (A100-80GB, roughly $10–12 total for both models).
