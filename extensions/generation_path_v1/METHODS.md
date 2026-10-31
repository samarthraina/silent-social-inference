# Predeclared estimands and interpretation

## Experiment 1: direct history access by generated input tokens

For each task let gap(arm) = score(high history, arm) - score(low history, arm).

1. `history_gap_reduction` = gap(baseline) - gap(generated_block).
2. `history_vs_control_specificity` = gap(control_block) - gap(generated_block).

Both are in score points. A positive reduction alone is insufficient for a history-specific conclusion: broad content disruption, generic score collapse and missingness must be assessed using the competing-content control, raw responses, descriptive scores/gaps and missing-output bounds. Question/answer tokens are not inert. The control blocks the same number of native prompt keys in the same layers, deterministically choosing question/answer positions in prompt order. It can itself impair judging.

Every generation uses a newly allocated cache. Intervention/control arms use the same explicit-mask path; an additional native-baseline arm calls the original Hugging Face generator and is reported separately. Blocked keys are all native tokens overlapping the history sentence. Readers are only input positions at or beyond the native prompt length. The first sample comes from a prompt reader and is unchanged by this intervention. Prompt states are never recomputed under a blocked prompt mask. Later generated states can still encode history indirectly through unblocked prompt keys or previous generated tokens.

## Experiment 2: state replacement x generated-reader blocking

Let movement(patch, block) average the signed movements toward the opposite-history donor across the two target histories:

- For a low-history target: score(patched target) - score(unpatched target).
- For a high-history target: score(unpatched target) - score(patched target).

Use the unpatched target under the same blocking setting. Then:

3. `patch_block_interaction` = movement(candidate patch, blocked) - movement(candidate patch, unblocked).
4. `patch_interaction_vs_l2` = candidate interaction - L2 interaction.

The candidate sites are fixed from our previous exploratory search, not selected on new validation results. The candidate span is exactly the last native token overlapping the history sentence. It is not a whole-history or numeral replacement. The L2 control uses that same span. The source states come from an ordinary unmodified explicit-mask prefill. Cross-history semantic-anchor token sequences must match despite native high/low length differences. Source/target caches are never shared; downstream caches reflect only the target run and the single prefill residual edit. The same-state patch must reproduce every output token.

A positive interaction can show that these two interventions interact. A null interaction cannot rule out distributed carriers, redundant routes, other token locations or other mechanisms. An improvement at the candidate without separation from L2 does not support layer specificity.

## Experiment 3: controlled evaluation-prefix carryover

Three qualitative donor evaluations are generated under high, low and no history with the same appended request. The request changes the output protocol; it is not part of the natural free-generation endpoint. Donors must terminate, contain text, and contain no explicit numeric-rating marker. All donors are retained even when invalid.

Tokenize the cleaned donor texts without special tokens. For a task define n = min(96, all three donor token counts). Require n >= 8 and all three usable donors; otherwise every nonempty-prefix score arm stays explicitly missing. Reuse each donor's first n tokens under both target histories. This is a token-prefix intervention, not a claim about a complete natural rationale. A generic task-independent text is repeated/truncated to n tokens as a length control. The score cue is tokenized separately and appended identically: `\n\nFinal score: `. An empty-prefix reference has that same cue but zero rationale tokens. Only the newly generated score completion enters the frozen final-score parser.

5. `neutral_rationale_gap_reduction` = gap(equal-length generic prefix) - gap(no-history donor prefix).
6. `rationale_donor_effect` = mean over both target histories of score(high-history donor prefix) - score(low-history donor prefix).

Nonempty prefixes have exactly the same token count per task, and a given prefix is byte-for-token identical across target histories. Generic and no-history prefixes differ in content, so this controls length but does not isolate every semantic property. Prefix truncation may remove later criticism or praise. Residual history influence, instruction effects and calibration can coexist. These experiments can establish controlled context dependence; they cannot identify a unique natural indirect effect or certify faithful chain-of-thought.

## Inference, missingness and confirmation

Each primary outcome is first calculated as one paired linear contrast per task, preserving within-task dependence across arms and directions. Bootstrap tasks within datasets, preserving the complete-case counts per dataset. Use 100,000 draws and seed 20261031. Report two-sided percentile intervals with alpha .05 / 12 for six endpoints x two models. These are finite-sample bootstrap approximations, not exact coverage guarantees. Report benchmark counts and avoid a balanced MT-Bench generalization claim (only one candidate MT-Bench confirmation task remains).

For each planned task, construct a worst-case lower/upper bound for each missing score using the legal score range [0,10]. Combine duplicate coefficients before bounding. Average these task bounds over all planned tasks, separately from the complete-case mean/interval. Thus a capped answer or invalid donor never becomes a numeric zero. Inspect how conclusions depend on missingness; a positive complete-case interval with bounds spanning zero is not robust to arbitrary missing outcomes.

All six outcomes and both model families are reported regardless of sign. The pilot is exploratory. Before validation, an author must review additional task exposure and freeze protocol, plan, sources and the completed engineering/pilot proofs. All existing completed raw units survive interruptions; no retuning or selective reruns of validation outputs are allowed under the same experiment label.

The exposure filter excludes recorded experiments by task ID and exact question hash, not semantic similarity or unknown collaborator runs. Candidate confirmation therefore requires the external exposure audit included in the lock. The frozen pool has 44 tasks (1 MT-Bench, 21 Arena-Hard, 22 WildBench). These are not newly collected tasks or necessarily behaviorally unseen examples.

## Provenance

Prompt rendering, model pins, cap-aware validity and score parsing come from the original `silent-influence` repository at f8010273445adcf276c094cf5b228fc7bd0acf2a, through the adjacent `residual_discovery_v3/reference` snapshot. Previous residual leads come from the completed runs in that extension. The collaborator's task exposure file is copied from `silent-social-inference` branch `mech-v2-window-validation`, commit 630b1c6813a9b7578d9a7825f0ef1dd17a49faa8. The explicit-mask knockout concept follows the collaborator's `src/mech_v2/core.py`; this implementation uses cached generation, generated-only readers, matched baselines and separate local tests. The original reference licenses/notices continue to apply. No model weights are included.


## Protocol v2 correction

The initial runner incorrectly assumed 32 decoder blocks for both models. The pinned Qwen3-8B configuration has 36; Llama has 32. Protocol v2 blocks all layers separately for each model and uses model-specific counts throughout the engineering checks and analysis. Architecture is checked before loading weights and again on the loaded model. The layer-count failure occurred before any engineering units ran. Start a fresh result directory under this corrected protocol; do not mix v1/v2 run metadata.


## Protocol v3 numerical-path sensitivity

Exact cross-path text equality is not a premise of the intervention estimand: explicit causal masks, automatic masks and recomputation can produce different floating-point predictions. Mandatory controls are exact repeated logits/tokens within the explicit path, same-state and no-op output identity, unchanged prompt predictions under generated-only blocking, correct blocked edge counts and an active mask control. The engineering gate verifies saved hashes and cannot pass if any of those controls fail. Cross-path token hashes and their agreement flags are retained regardless of sign.

To quantify score-level consequences, add actual native Hugging Face cached generation under both histories for every pilot and validation task (two extra units per task). All original primary intervention contrasts remain within the explicit-mask path. Report the native gap, explicit gap, and paired explicit-minus-native gap difference, with a descriptive 95% task-stratified bootstrap interval and planned-task missing bounds. That diagnostic is not an extra confirmatory primary test and is not evidence of equivalence merely because its interval includes zero. It limits the scope of any result that fails to transfer across computational paths.

The version change follows engineering-only evidence before the new pilot/validation; it is not a retroactive declaration that failed v2 checks passed. Preserve v2 records and freeze v3 before the new runs. No interventions are applied to the native baseline.
