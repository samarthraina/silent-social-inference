# Where is the history effect carried? Mech plan v2 (draft for team review)

Target contrast: `self_high_10` vs `self_low_2` (matched history; valence only).
Secondary: `self_low_2` vs `no_history` (continuity with the paper's 22% result).
Models: Llama-3.1-8B-Instruct, Qwen3-8B (thinking off). Same pinned revisions as `extensions/context_controls`.

## 0. Audit of the completed matched-history patching (MHP)

Implementation is correct: hook indexing matches `hidden_states[L]`, the prefill-only guard
is right under KV caching, 288/288 identity checks pass. The null is a property of the
design, not a bug:

1. **The site cannot block the signal.** The history sentence is line 3 of the user turn,
   ahead of the question and the frozen answer (prompt median ~470 tokens, max ~2.7k).
   Its tokens keep their own keys/values at every layer. We overwrite one template token
   (`\n\n` after the assistant header) at one layer. Every later prompt token already
   read the history, and each of the up to 512 generated tokens re-reads it directly.
   In matched history the target also contains a *competing* sentence, so a single-site
   patch is outvoted. (In the paper's low→none patch, the none target has no competing
   sentence, which is a plausible reason that design moved ~20% and this one moves 0.)
2. **The readout is chaotic.** Offline audit of the released raw records
   (`scratchpad/audit.py`):
   - Llama L24 patches change the generated text in 52/72 and 61/72 tasks. The median first
     differing character is 0, so the first token flips, but the score moves toward the
     source in only 15–22 tasks and the mean signed move is slightly *negative*.
   - The **L2 control also changes the text in 46–53/72 tasks**. Text changes are
     butterfly effects of a 512-token greedy rollout, not valence transfer.
   - One integer score, parsed at the end of a long rollout, is too coarse to detect partial mediation.
3. **Qwen positions are misaligned for any multi-token design.** Qwen splits digits, so
   `10/10` is one token longer than `2/10` (all 72 tasks differ by exactly +1 token).
   Llama lengths are identical. Position-wise patching must align by span, with
   everything after the numeral shifted by the offset.
4. **Inherited sites.** L24 and L27 were selected on low vs none (presence plus valence) with a
   lens-JSD criterion. Nothing says the high-vs-low signal lives there.
5. Minor: rationales often contain two scores (e.g. "0/10 … However … 5/10"). The tier-1 parser
   takes one of them, which adds more readout noise.

## 0b. Harness checks on GPU (D only, before any search)

The audit above was offline. Before searching we confirm on GPU:

- **Positive control:** patch all of `NUM ∪ HIST` from source to target at layer 1. This should
  move essentially the whole gap (R ≈ 1). If it doesn't, the harness or readout is broken and
  nothing downstream is interpretable.
- **Null diagnosis:** re-run the released sites (Llama L24, Qwen L27, final-input token) under the
  continuous readout. If R is still ≈ 0, the site itself carries little valence. If R > 0, the
  released null was mostly readout noise. Either outcome goes in the paper.
- **Qwen alignment:** check that span-aligned patching with the +1 token offset reproduces the
  same-state identity exactly.

## 0c. Hypotheses and the targeted intervention for each

These are fixed before discovery. They are derived from the audit (the history tokens keep their
own keys/values and are re-read downstream) and the prompt order (history → question → answer → tail).

| Hypothesis | Where valence is carried | Targeted intervention that tests it |
|---|---|---|
| H-relay | Mid-layer answer-token representations, colored by attention to the history | Block ANS←(NUM∪HIST) attention in mid windows. Patch the `ANS` span. |
| H-direct | Late layers at the tail/score positions, which read the history directly | Block TAIL←(NUM∪HIST) attention in late windows. Patch `TAIL` across a layer window, not one layer. |
| H-token | The numeral token's own residual stream, read by a few late heads | Patch `NUM` only. Find and patch the heads that attend to `NUM` from the score position. |
| H-distributed | No compact carrier | All of the above stay below 50% necessity. Report the curves as the result. |

Discovery decides **which** hypothesis survives and **at which layers/heads**. It doesn't invent new
intervention types after seeing the data.

## 1. Readout: replace the rollout with a continuous, deterministic metric

All search runs use one forward pass per intervention:

- **R-direct.** Append a fixed assistant prefix ending in `Final score:` and read p(0..10) at
  the next token. Metric: expected score `E[s] = Σ s·p(s)`, plus the logit difference between
  the high and low halves of the scale.
- **R-forced.** Teacher-force the target condition's own baseline justification up to the
  score token (the parsed span) and read p(0..10) there. This keeps the original
  justify-then-score protocol.
- **Validity gate (on discovery tasks, before any search):** the unpatched high−low gap under
  each readout must be positive on ≥80% of tasks, and its task-level correlation with the
  greedy gap is reported. Only gated readouts are used.
- Normalized recovery `R = (m_patched − m_target)/(m_source − m_target)` on the continuous metric,
  which has no zero-gap exclusions. Raw differences are reported as well.

Greedy generation is used only for final confirmation (section 4).

## 1b. Budget mode: staged funnel (supersedes the grid sweeps in 3a–3c and the budget in 5)

Budget: ₹500–1000. No blind grid. Each stage is cheap, ends in a go/no-go gate, and narrows what
the next stage looks at. Llama goes first. Qwen gets only the frozen hypothesis that survived on
Llama, plus its own layer bisection. Pass counts are per model.

| Stage | Tasks | What it does | Passes | Gate / what it narrows |
|---|---|---|---|---|
| S0 (CPU, free) | released records | Already done: audit, null mechanism, Qwen offset | 0 | — |
| S1 sanity | 12 D | Readout gate, positive control (layer-1 history patch), released-site re-run on the new readout | ~150 | Stop if the readout gap disagrees with greedy or the positive control fails |
| S2 path triage | 12 D | Attention-knockout, 3 reader sets (ANS, TAIL, all) × 4 blocks of 8 layers × 2 conditions | ~300 | Picks the hypothesis (relay / direct / token) and an 8-layer block |
| S3 bisection | 12 D | Split the winning block into halves, recursively (8 → 4 → 2 → 1); patch the winning prompt part only | ~250 | Narrows to a 1–2 layer site and one span |
| S4 head scan (observational) | 36 D | Two forward passes per task: (a) direct logit attribution of every head's output at the score position onto (high − low score tokens); (b) each head's attention from the score position to `NUM`/`HIST` | ~75 | Keep the top ~10 heads that sit inside the S3 layers. No gradients, so it fits 16–24 GB |
| S5 exact confirm | 36 D | Patch and ablate the top heads, singly and jointly; random-head control | ~800 | Freeze site S*, intervention and thresholds (hash) |
| S6 validation | 72 V | Frozen S* only: sufficiency both directions, necessity, L2 + 10 random controls on the readout; greedy generation on 36 V for 6 arms | ~1.8k passes + ~216 generations | Pre-registered success criteria (section 4) |

Totals: about 3.4k forward passes plus ~216 generations per model, versus ~41k passes plus ~575
generations in the full plan.

Estimated cost, both models (to be checked against a real run):
- RTX 4090 bf16: ~2.5–3.5 h → roughly ₹100–250.
- Kaggle 2×T4 fp16: ~8–10 h of the free weekly quota → ₹0.

The ₹500–1000 cap covers one full rerun if a stage fails for engineering reasons.

**Stopping rules:**
- If S2 shows no knockout block reaching ≥30% of the gap on any reader set, skip S3–S5. Run the
  distributed-hypothesis report (S2 curves plus a coarse 8-layer patch map) and stop.
- If S3 bisection gets flat (both halves < 60% of the parent block's effect), freeze at the parent
  block instead of a single layer.

## 2. Splits (frozen by hash before any GPU run)

From the 234 study tasks, exclude the 12 discovery, 24 validation and 72 extension tasks already used.
That leaves about 126 fresh tasks. Stratify by benchmark:

| Split | n | Use |
|---|---|---|
| D (discovery) | 36 (12/benchmark) | all searches, site/head selection, threshold setting |
| V (validation) | 72 (24/benchmark) | one frozen confirmatory run, no retuning |
| E (existing 72) | 72 | secondary replication; prior MHP nulls stay in the record |

Selection uses only dataset, id and category, never scores.

## 3. Discovery (D only), per model

**3a. Layer × position-group patching (both directions, both readouts).**
Position groups are aligned spans:
- `NUM`: the numeral `2` vs `10`
- `HIST`: the rest of the history sentence
- `INSTR`: the instruction lines
- `Q`: the question
- `ANS`: the frozen answer
- `TAIL`: the template tail plus the final input token

Patch the residual stream for one group at a time at each layer. This gives an
information-flow map: where valence sits at each depth, and when (or whether) it leaves `NUM/HIST`.

**3b. Attention-edge knockout.** Block attention from a set of reader positions to
`NUM ∪ HIST` within a sliding window of 5 layers. Reader sets:
- (i) `ANS` tokens
- (ii) `TAIL` plus the score position
- (iii) all later positions

This separates two competing hypotheses:
- **H-relay:** the history changes how the *answer content* is encoded in mid layers. Readers then
  see an already-colored answer, so knocking out the ANS←HIST edges kills the effect.
- **H-direct:** late layers at the score/TAIL positions read the history directly. Knocking out the
  TAIL←HIST edges in late layers kills the effect.
- **H-distributed:** no window or edge set removes ≥50%. If that is what we find, we report it
  honestly, with the curves.

**3c. Component attribution.** Run attribution patching (AtP*) over all attention heads and MLPs
for high↔low on the readout, then do exact activation patching on the top 50 components to
remove AtP false positives. Rank by exact effect.

**3d. Directions (only if 3a/3b localize a site).** Compute the mean-difference valence direction
at the selected group and layer, then test:
- directional ablation (necessity)
- additive steering with β ∈ {−2,−1,−0.5,0,0.5,1,2} (dose-response)

The released `ablate_positionwise`/`add_positionwise` hooks already exist.

**Freeze file** (written and hashed on D, before touching V), per model:
- the chosen site S*: a group/layer window, an edge set, or a head set of size k
- the intervention type
- the readout
- the primary metric and success thresholds below

## 4. Validation (V, one run, no retuning)

For S* per model, both directions:

1. **Sufficiency:** patch S* from the source into the target. Primary metric is mean R.
2. **Necessity:** knock out or ablate S* and measure the reduction of the high−low gap.
3. **Specificity controls:**
   - the same intervention at L2
   - k random heads (or random positions) matched in count and norm (20 draws)
   - the no-op and same-state identity checks
4. **Behavioral confirmation:** greedy 512-token generation with the original parser for the
   sufficiency and necessity arms on V. Report score movement and the text-change rate
   versus the L2 control.
5. **Continuity:** the same S* applied to low vs none, to connect with the paper's 22%.

**Pre-registered success:**
- sufficiency R ≥ 0.5, with the 95% task-bootstrap CI excluding the random-control upper bound
- necessity removes ≥ 50% of the gap
- both directions hold
- the behavioral arm moves in the predicted direction with a CI excluding 0

Partial outcomes are reported against these thresholds and are never relabeled after the fact.
The bootstrap uses 10,000 task draws, stratified by benchmark.

## 5. Budget (A100 80GB, bf16)

Forward-only runs on D come to about 0.1–0.4 s per pass:

| Run | Passes | Time |
|---|---|---|
| 3a: 32 layers × 6 groups × 2 dirs × 2 readouts × 36 tasks | ≈ 28k | ~2 h per model |
| 3b: windows × 3 reader sets | ≈ 5k | < 1 h |
| 3c: AtP is 2 backward passes per task; exact top-50 | ≈ 7k | < 1 h |
| V, readout arms | — | < 1 h |
| V, greedy confirmation (~8 arms × 72 tasks × ~5 s) | — | ~1 h |

Total is roughly 10–12 GPU-hours for both models.

## Run log

### S1, Llama (12 triage tasks, 2026-10-30), `results/mech_v2/results/s1_llama.jsonl`

| Check | Direct readout | Forced readout |
|---|---|---|
| Readout gap sign = greedy gap sign | 12/12 | 12/12 |
| Mean high−low gap (expected score) | 1.68 | 2.30 |
| Score-token mass | 0.995 | 0.999 |
| Same-state identity, max abs ΔE | 0 | 0 |
| Layer-1 patch of NUM+HIST, normalized move | 0.99 / 1.00 | 0.99 / 1.00 |
| Knock out all attention to history, gap removed | 100% | 58% |
| Released site, final-input L24, move (both directions) | −0.001 / −0.004 | +0.003 / −0.005 |
| L2 control, final-input, move | +0.001 / −0.003 | −0.018 / −0.001 |
| Explicit-mask path vs default, max abs ΔE | 0.117 | 0.037 |

**Pre-registered gate: NO-GO.** On a tie the rule selects the forced readout, whose
knockout positive control removed 58% < 80%.

**Deviation (decided 2026-10-30, after seeing S1, before any S2 data).** S2–S4 use the
**direct** readout. Reason: in the forced readout each condition keeps its own greedy
justification, which was written under the history. Blocking attention to the history
sentence cannot remove the part of the gap carried by that self-generated text. The
control therefore mixes two paths, and is not a harness failure: the layer-1 patch control
passes at 0.99 under both readouts. The direct readout passes every S1 check.

Findings retained for the paper:
- (i) The released L24 final-input null replicates under a continuous readout (move ≈ 0,
  equal to L2), so that null was not readout noise.
- (ii) Under the forced readout, about 42% of the gap survives complete removal of
  attention to history. That share is carried by the self-generated justification.

Fix applied before S2: S2/S3 knockout baselines now use the same explicit-mask path as
the knockouts.

### S2, Llama (12 triage tasks, direct readout), `results/mech_v2/results/s2_llama.jsonl`

Share of the high−low gap (baseline 1.66) removed by blocking attention to NUM+HIST:

| Readers \ blocks | 0–7 | 8–15 | 16–23 | 24–31 |
|---|---|---|---|---|
| ALL post-history positions | 23% | **96%** (12/12 tasks > 50%) | 6% | −1% |
| TAIL (template tail + score prefix) | 1% | 57% (7/12) | 6% | 0% |
| ANS (frozen answer) | 0% | 9% | −1% | 0% |

**Gate:** GO (winner ALL 8–15, 96% ≥ 30%).

Reading:
- The history is read out of its own tokens almost entirely in decoder blocks 8–15. From block 16
  onward, blocking it changes nothing, so by then the valence is already stored at
  post-history positions.
- The tail positions are the main direct reader (57%). The answer tokens carry little (9%).
  This favors H-direct over H-relay, and leaves ~30% that goes through other post-history
  positions or redundant routes.
- This is consistent with the released null: the signal sits across several tail
  positions after block 15, so overwriting one final-input token at L24 cannot move it.

### S3, Llama (12 triage tasks, direct readout), `results/mech_v2/results/s3_llama.jsonl`

Bisection by knockout (ALL post-history readers ← NUM+HIST), share of gap removed:

| Window | 8–15 | 8–11 | 12–15 | 12–13 | 14–15 | 12 | 13 |
|---|---|---|---|---|---|---|---|
| Removed | 96% | 58% | 67% | 53% | 16% | 18% | 40% |

- The pre-registered rule froze block 13 (40%).
- The halves overlap: 58% + 67% > 96%. Reading is redundant across blocks 8–15, and no single
  block is necessary on its own.

Sufficiency: patch the residual stream at all post-history positions from source into target.
Values are normalized movement toward the source.

| Site | high→low | low→high |
|---|---|---|
| `hidden_states[13]` (input to block 13) | 0.54 | 0.32 |
| `hidden_states[14]` (output of block 13) | **0.81** | **0.73** |
| `hidden_states[2]` control | 0.01 | 0.02 |

Reading: by the output of block 13, about 77% of the valence effect sits in the post-history
residual stream. Compare the 0% for the single final-input token at L24, and the paper's 22%
for low→none.

**S4 adjustment (discovery-side, before S4 data):** S4 now also records each head's mean
attention to NUM+HIST from TAIL and from all post-history positions. Reader heads are ranked
inside blocks 8–15. Direct-logit "writer" heads are ranked over all blocks.

### S4, Llama (36 D tasks, direct readout, eager attention), `results/mech_v2/results/s4_llama.jsonl`

Reader heads in blocks 8–15, by mean attention to NUM+HIST:
- **Primary** (pre-specified: from ALL post-history positions), top 10:
  8.31, 9.3, 10.25, 9.19, 9.23, 13.14, 11.15, 11.13, 13.3, 9.2.
  The ranking is flat: top 0.043 against a median of 0.010 and a p95 of 0.055.
- **Secondary** (exploratory: from TAIL positions), top 10:
  8.31, 11.5, 13.3, 13.19, 13.14, 13.8, 14.20, 11.12, 13.16, 13.17.
  Masses run 0.15–0.23, and 6 of the 10 are in block 13, which agrees with the S3 bisection.

Writer heads (direct effect on E[score], high − low):
- 22.14 (−0.33), 28.17 (+0.33), 28.18 (−0.21), 22.12 (+0.12), 28.6 (+0.09).
- Sign consistency is 0.86–1.00 across the 36 tasks.
- These are not tested yet.

S5 runs both reader sets on D. Only one set will be frozen for V, and the choice uses D results only.

### S5, Llama (36 D tasks, direct readout), `results/mech_v2/results/s5_llama_{primary,tail}.jsonl`

| Intervention | Primary head set (pre-specified) | Tail head set (exploratory) |
|---|---|---|
| Knock out ALL←HIST, blocks 8–15 (window) | **96%** of gap removed | 96% (same runs) |
| Knock out top-10 heads only | 4% | −6% |
| 10 random matched head sets: mean / max | 3% / 28% | 15% / 40% |
| Patch top-10 head outputs (high→low / low→high) | 0.02 / 0.11 | −0.18 / 0.04 |
| **Residual patch, all post-history positions, `hidden_states[16]`** | **0.99 / 0.90** | same |

Single tail-set heads have mixed signs:
- 13.19 (+19%) and 14.20 (+10%) carry the valence.
- 13.16 (−34%) and 13.17 (−17%) counteract it: blocking them *widens* the gap.

Conclusions on D:
1. Window-level necessity (96%) and post-window sufficiency (0.90–0.99) replicate on 36 tasks.
2. Attention-mass ranking does not identify causal heads. The top-10 sets are at
   random-control level, and reading is spread over many heads with opposing signs. **No
   head-level claim is supported.** The causal site is the window, not a small head set.
3. Decision (made on D only): freeze the window-level site for V. Head-level circuit work
   would need a causal per-head scan (all 256 heads in blocks 8–15). That is optional and not
   part of the validation.

### Deferred follow-ups for the failed head-level test (noted 2026-10-30, not run)

The S5 failure is narrow. Ranking heads by attention mass did not find causal heads. That does
not show a head-level mechanism is absent. The paper reports the window-level claim plus
redundancy (single heads have opposing signs). Two options, discovery tasks only, ~$1–2 total:

1. **Causal per-head scan.**
   - Knock out each of the 256 heads in blocks 8–15 singly on the 12 triage tasks, and rank by
     gap removed (carriers and counteracting heads kept separate).
   - Plot a cumulative top-k joint-knockout curve against random sets.
   - Confirm on the other 24 D tasks.
   - If a small set passes, freeze it and validate on V as a separate, labelled run.
2. **Rank-1 valence direction.**
   - Take the mean-difference direction (high − low) of `hidden_states[16]` at post-history
     positions, fitted on D.
   - Patch only that 1-D component, with random-direction controls.
   - Tests whether a single linear direction carries the effect, even though the reading is distributed.

## Qwen3-8B: prepared, not run (2026-10-30)

**CPU pre-check (Qwen tokenizer, all 108 D+V tasks):** all pass.
- Spans build.
- NUM is 2 tokens for 10/10 and 1 token for 2/10.
- Post-history positions align token-for-token between high and low.
- Score-line alignment holds for low↔no-history.
- Generation-prompt alignment holds.
- Prompt length: median 620, max 2,982 tokens.

**Pre-registered cross-model prediction (written before any Qwen GPU run):** the Llama result
transfers in form.
- History is read by post-history positions within a contiguous mid-depth window.
- Llama blocks 8–15 of 32 = 25–47% depth. The prediction is that Qwen's S2 winner is the
  **ALL readers, blocks 9–17** cell (of 36).
- Post-window residual patching at post-history positions should be sufficient (≥ 0.5 both directions).
- Late blocks should be ≈ 0.
- Any other outcome is reported as non-transfer.

**Sequence:** identical rules to Llama. Qwen gets its own S2/S3 window and its own freeze.

| Step | Command (`modal run src/mech_v2/modal_app.py ...`) | Notes |
|---|---|---|
| S1 | `--stage s1 --model qwen` | The NUM-span layer-1 patch is skipped (unaligned). The positive control is knockout-all. |
| S2 | `--stage s2 --model qwen --args '{"readout":"<S1 choice>"}'` | 4 blocks of 9 |
| S3 | `--stage s3 --model qwen --args '{"readout":...,"readers":...,"lo":..,"hi":..}'` | |
| S5 | `--stage s5 --model qwen --args '{"readout":...,"lo":..,"hi":..,"heads":[],"n_random":0,"tag":"_window"}'` | Window-level only. Heads are deferred, as for Llama. |
| S6 | new `data/mech_v2/freeze_s6_qwen.json` (hashed), then `--stage s6 --model qwen` | |

Estimated cost: about $5–7 in total, mostly S6 generation.

### S6, Llama: frozen validation on V (72 held-out tasks), `results/mech_v2/results/s6_llama.jsonl`, summary `results/mech_v2/s6_llama_summary.json`

- Spec `freeze_s6_llama.json` (sha d9c577a2…), hash checked before launch.
- Unpatched V gap: 1.93, positive on 72/72 tasks.
- 95% CIs come from a benchmark-stratified task bootstrap with 10,000 draws.

| Pre-registered test | Estimate [95% CI] | Rule | Result |
|---|---|---|---|
| Necessity: knock out post-history ← NUM+HIST, blocks 8–15 | **95.4%** [92.9, 97.1] | ≥ 50%, and CI above the control CIs | **PASS** |
| … control blocks 0–7 / 16–23 | 20.8% [17.6, 24.1] / 3.4% [1.9, 4.8] | — | — |
| Sufficiency high→low: residual patch at `hidden_states[16]`, all post-history positions | **0.99** [0.98, 1.00] (L2 control 0.02) | ≥ 0.5, and site − control CI > 0 | **PASS** |
| Sufficiency low→high | **0.91** [0.89, 0.94] (L2 control 0.01) | same | **PASS** |
| Behavioral knockout: reduction in the greedy high−low score gap (36 tasks) | **92%** [83, 101] | CI lower bound > 0 | **PASS** |
| Behavioral patch: greedy score shift toward source, high→low | +0.32 [−0.39, +0.94] | CI lower bound > 0, both directions | **FAIL** |
| … low→high | +0.31 [+0.02, +0.63] | | (pass alone) |

Secondary (continuity with the paper's low-vs-none contrast and its 22%): the same site moves
**0.98** [0.96, 1.00] for low→none and **0.97** [0.96, 0.98] for none→low. The L2 control is about 0.

Integrity:
- One task was interrupted by a local network drop (DNS failure) and rerun on resume.
  Its duplicate rows are bit-identical.
- Generation: 7 of 219 outputs have no parseable score, and 12 hit the 512-token cap.
- The explicit-mask path differs from the default path by at most 0.18 in E. Knockouts are
  compared against mask-path baselines.

Reading of the failed behavioral-patch criterion. This is an interpretation, not a rescue: the
patch writes prompt positions only. Every generated token is itself a post-history position, and
it can still read the history sentence in blocks 8–15. Necessity shows that reading is what
carries the effect. So prompt-only patching is expected to move the single-pass readout fully but
free generation only partially. That would be tested by patching the prompt states while also
blocking generated tokens from reading NUM+HIST. Not run yet; exploratory if run.

### S1, Qwen3-8B (12 triage tasks), `results/mech_v2/results/s1_qwen.jsonl`

| Check | Direct | Forced |
|---|---|---|
| Sign agreement with greedy | 12/12 | 12/12 |
| Mean gap | 2.03 | 2.73 |
| Score-token mass | 1.00 | 1.00 |
| Same-state identity, max abs ΔE | 0 | 0 |
| Knockout-all positive control | **100%** | 65% |
| Released site, final-input L27, move | −0.001 / +0.005 | +0.003 / −0.000 |
| L2 control | +0.006 / −0.004 | −0.003 / +0.005 |
| Mask path vs default, max abs ΔE | 0.075 | 0.064 |

The NUM-span layer-1 patch is skipped (unaligned, as pre-declared). The original rule gives
NO-GO for the same reason as Llama: the forced readout's own justification carries about 35%
of the gap.

**The Llama deviation is applied unchanged (direct readout).** It was decided before any Qwen
data, so no new choice was made.

The released Qwen L27 null replicates under the continuous readout, so it was not readout noise.

### S2, Qwen (12 triage tasks, direct), `results/mech_v2/results/s2_qwen.jsonl`

Baseline gap is 2.05. Share of the gap removed by knocking out → NUM+HIST:

| Readers \ blocks | 0–8 | 9–17 | 18–26 | 27–35 |
|---|---|---|---|---|
| ALL post-history | 0% | 70% (11/12 tasks > 50%) | **75%** (10/12) | 7% |
| TAIL | 4% | 16% | 57% (7/12) | 8% |
| ANS | 1% | 0% | 4% | 1% |

**Pre-registered prediction (winner = ALL 9–17): NOT MET.** The winner is ALL 18–26, with 9–17
close behind. What does transfer from Llama:
- History is read by post-history positions in a contiguous mid-depth range.
- The answer tokens play no role.
- Late blocks are ≈ 0.
- The tail is the main direct reader, now in blocks 18–26.

What differs: Qwen's reading is spread over a wider and later range, blocks 9–26
(25–75% depth), where Llama's is blocks 8–15 (25–47%). No single 9-block window reaches Llama's 96%.

Gate: GO. S3 bisects 18–26 per rule.

### S3, Qwen (12 triage, direct), `results/mech_v2/results/s3_qwen.jsonl`

| Window | 18–26 | 18–22 | 23–26 | 18–20 | 21–22 |
|---|---|---|---|---|---|
| Removed | 75% | **64%** (frozen by rule) | 12% | 36% | 20% |

Sufficiency: residual patch at all post-history positions.

| Site | high→low | low→high |
|---|---|---|
| `hidden_states[18]` | 0.34 | 0.34 |
| `hidden_states[23]` (after block 22) | **0.81** | **0.83** |
| L2 control | 0.00 | 0.00 |

**S5 window rule for Qwen (written before any Qwen S5 data).** Llama froze its S2 winner block.
Qwen reading is split across blocks 9–17 and 18–26, so S5 tests two candidate windows on all 36 D tasks:

| Candidate | Knockout window | Patch site |
|---|---|---|
| (a) | ALL ← HIST, 18–26 | `hidden_states[27]` |
| (b) | ALL ← HIST, 9–26 | `hidden_states[27]` |

**Freeze (a) if its D necessity is ≥ 0.80, else (b).** The controls in S6 will be the other
equal-width blocks.

### S5, Qwen (36 D tasks), `results/mech_v2/results/s5_qwen_{w18_26,w9_26}.jsonl`

| Candidate | Knockout necessity | Residual patch at `hidden_states[27]`, high→low / low→high |
|---|---|---|
| (a) 18–26 | 72% | 0.92 / 0.96 |
| (b) 9–26 | **98.5%** | 0.92 / 0.96 |

Rule: (a) is below 0.80, so **freeze (b), blocks 9–26**.

S6 controls are the complementary blocks 0–8 and 27–35. These are 9 blocks wide against 18
for the target window, since equal width is impossible without overlap. This is a stated limitation.

### S6, Qwen: frozen validation on V (72 held-out tasks), `results/mech_v2/results/s6_qwen.jsonl`, summary `results/mech_v2/s6_qwen_summary.json`

- Spec `freeze_s6_qwen.json` (sha f519abbf…).
- Unpatched V gap: 2.10, positive on 72/72 tasks.

| Pre-registered test | Estimate [95% CI] | Result |
|---|---|---|
| Necessity: knock out post-history ← NUM+HIST, blocks 9–26 | **96.9%** [95.6, 98.0] (controls 0–8: −7.1%, 27–35: 4.7%) | **PASS** |
| Sufficiency high→low, `hidden_states[27]` | **0.95** [0.93, 0.96] (L2: 0.00) | **PASS** |
| Sufficiency low→high | **0.91** [0.88, 0.95] (L2: −0.00) | **PASS** |
| Behavioral knockout: greedy gap reduction (36 tasks) | **93%** [79, 105] | **PASS** |
| Behavioral patch: greedy shift toward source, high→low / low→high | −0.06 [−0.36, 0.31] / −0.03 [−0.11, 0.06] | **FAIL** |

Continuity, low↔no-history at the same site: **0.89** [0.84, 0.92] and **0.99** [0.98, 1.00].

Integrity:
- Generation: 2 of 216 outputs unparsed, 2 capped.
- The explicit-mask vs default path difference in E is larger for Qwen than for Llama. The
  maximum is 0.48 (distribution printed with the log). Knockouts are compared only against
  mask-path baselines, so the estimates are internally consistent.

## Cross-model summary (frozen validations, 72 held-out tasks each)

| | Llama-3.1-8B | Qwen3-8B |
|---|---|---|
| Released single-token site (final input) | L24: 0% | L27: 0% |
| Read window (post-history ← history) | blocks 8–15 | blocks 9–26 |
| Necessity | 95% | 97% |
| Sufficiency, post-window state (both directions) | 0.99 / 0.91 | 0.95 / 0.91 |
| low↔none continuity (paper: 22%) | 0.98 / 0.97 | 0.89 / 0.99 |
| Greedy knockout | 92% | 93% |
| Greedy prompt-only patch | fails | fails (≈ 0) |
| Answer tokens as readers (S2) | 9% | ≤ 4% |

The behavioral-patch failure replicates in both models, and is stronger in Qwen. This fits
generated tokens re-reading the history sentence directly. The test that would confirm it is
an untested exploratory patch combined with a generated-token knockout.
