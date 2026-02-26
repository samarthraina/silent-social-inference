"""
src/protocols.py
─────────────────────────────────────────────────────────────
Iteration protocol executor for the asymmetry experiment.

Protocol per iteration
──────────────────────
Phase 1 — Generation (all models, no order asymmetry):
  For each model M ∈ {A, B, C, D}:
    Load M → generate response to task → unload M → cache response

Phase 2a — Judging of Model A (asymmetry: A is ALWAYS judged first):
  Load B → judge A's response → unload B → log score
  Load C → judge A's response → unload C → log score
  Load D → judge A's response → unload D → log score
  Compute mean score received by A.

Phase 2b — A judges others:
  Load A → judge B's response → log score
           judge C's response → log score
           judge D's response → log score
  Unload A.

Asymmetry properties
────────────────────
• Model A's response is always evaluated BEFORE A evaluates others.
• Within Phase 2a, B, C, D judge A sequentially (no parallel access).
• All identities are masked in judge prompts (no model names revealed).
• No feedback is communicated between any models.
• The asymmetry flag is set to True for all records.

Memory management
─────────────────
• Only one model occupies GPU memory at any time.
• Responses are cached in a Python dict (not GPU) before judging begins.
• Aggressive GPU clear (gc + cuda.empty_cache) after each model unload.

Retry logic
───────────
• Generation: up to 2 attempts on OOM or RuntimeError.
• Judging/parsing: up to parse_retries attempts per score (configurable).

Iteration seed derivation
─────────────────────────
    iteration_seed = global_seed XOR (iteration_id * 2654435761)  [Knuth hash]
    model_seed     = iteration_seed XOR model_index
This gives uncorrelated seeds per-model per-iteration while remaining
deterministic from global_seed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Seed derivation
# ──────────────────────────────────────────────────────────────────────────────

_KNUTH_MULTIPLIER = 2654435761  # Knuth multiplicative hash


def _derive_iteration_seed(global_seed: int, iteration_id: int) -> int:
    return (global_seed ^ (iteration_id * _KNUTH_MULTIPLIER)) & 0xFFFFFFFF


def _derive_model_seed(iteration_seed: int, model_index: int) -> int:
    return (iteration_seed ^ model_index) & 0xFFFFFFFF


# ──────────────────────────────────────────────────────────────────────────────
# Per-iteration result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IterationResult:
    iteration_id: int
    task_template_id: int
    task_variation_id: int
    # {model_key: GenerationResult}
    responses: Dict[str, "GenerationResult"]  # noqa: F821
    # {(judge_key, respondent_key): JudgeScore}
    scores: Dict[Tuple[str, str], "JudgeScore"]  # noqa: F821
    # Mean score received by model_a across its judges
    model_a_received_mean: float
    wall_clock_seconds: float

    def score_given(self, judge_key: str, respondent_key: str) -> Optional[float]:
        s = self.scores.get((judge_key, respondent_key))
        return s.weighted_total if s else None


# ──────────────────────────────────────────────────────────────────────────────
# Protocol executor
# ──────────────────────────────────────────────────────────────────────────────

class IterationProtocol:
    """
    Executes one full iteration of the experiment.

    Parameters
    ----------
    cfg      : ExperimentConfig
    handles  : {model_key: ModelHandle}  — unloaded handles
    task_reg : TaskRegistry
    exp_log  : ExperimentLogger
    """

    MAX_GENERATION_RETRIES = 2

    def __init__(
        self,
        cfg,
        handles: Dict[str, "ModelHandle"],   # noqa: F821
        task_reg: "TaskRegistry",            # noqa: F821
        exp_log: "ExperimentLogger",         # noqa: F821
    ):
        self.cfg = cfg
        self.handles = handles
        self.task_reg = task_reg
        self.exp_log = exp_log

        # Ordered model keys
        self.model_keys: List[str] = list(cfg.models.keys())
        self.model_key_to_index: Dict[str, int] = {
            k: i for i, k in enumerate(self.model_keys)
        }

        # Asymmetry config shortcuts
        self.primary_key: str = cfg.asymmetry.model_judged_first
        self.judges_of_a: List[str] = cfg.asymmetry.judges_of_a
        self.models_a_judges: List[str] = cfg.asymmetry.models_a_judges

    # ── Public entry point ────────────────────────────────────

    def run_iteration(self, iteration_id: int) -> IterationResult:
        """
        Execute one full iteration. Returns IterationResult.
        """
        t_start = time.perf_counter()

        task = self.task_reg.get_by_iteration_id(
            iteration_id,
            variations_per_template=self.cfg.tasks.variations_per_template,
        )
        iter_seed = _derive_iteration_seed(self.cfg.seed, iteration_id)

        logger.info(
            "─── Iteration %4d / %d  [%s var=%d  seed=%d] ───",
            iteration_id,
            self.cfg.total_iterations - 1,
            task.family_name,
            task.variation_id,
            iter_seed,
        )

        # ── Phase 1: Generation ───────────────────────────────
        response_cache: Dict[str, "GenerationResult"] = {}
        for model_idx, model_key in enumerate(self.model_keys):
            model_seed = _derive_model_seed(iter_seed, model_idx)
            result = self._generate_response(
                model_key=model_key,
                task_prompt=task.prompt,
                seed=model_seed,
                iteration_id=iteration_id,
                task=task,
            )
            response_cache[model_key] = result

        # ── Phase 2a: Judge Model A ───────────────────────────
        scores: Dict[Tuple[str, str], "JudgeScore"] = {}
        a_received_scores: List[float] = []

        a_response_text = response_cache[self.primary_key].text
        for judge_key in self.judges_of_a:
            judge_score = self._judge_response(
                judge_key=judge_key,
                respondent_response=a_response_text,
                task_prompt=task.prompt,
                iteration_id=iteration_id,
                task=task,
                respondent_key=self.primary_key,
            )
            scores[(judge_key, self.primary_key)] = judge_score
            a_received_scores.append(judge_score.weighted_total)

        model_a_received_mean = (
            sum(a_received_scores) / len(a_received_scores)
            if a_received_scores else 0.0
        )

        # ── Phase 2b: Model A judges others ───────────────────
        handle_a = self.handles[self.primary_key]
        with handle_a:
            for respondent_key in self.models_a_judges:
                judge_score = self._judge_response_with_loaded_model(
                    model_handle=handle_a,
                    respondent_response=response_cache[respondent_key].text,
                    task_prompt=task.prompt,
                    iteration_id=iteration_id,
                    task=task,
                    judge_key=self.primary_key,
                    respondent_key=respondent_key,
                )
                scores[(self.primary_key, respondent_key)] = judge_score

        wall_clock = time.perf_counter() - t_start
        logger.info("Iteration %d complete in %.1fs", iteration_id, wall_clock)

        return IterationResult(
            iteration_id=iteration_id,
            task_template_id=task.template_id,
            task_variation_id=task.variation_id,
            responses=response_cache,
            scores=scores,
            model_a_received_mean=model_a_received_mean,
            wall_clock_seconds=wall_clock,
        )

    # ── Phase 1 helper ────────────────────────────────────────

    def _generate_response(
        self,
        model_key: str,
        task_prompt: str,
        seed: int,
        iteration_id: int,
        task: "TaskVariation",  # noqa: F821
    ) -> "GenerationResult":
        """Load model, generate, unload, log, return."""
        from src.models import format_prompt

        model_cfg = self.cfg.models[model_key]
        handle = self.handles[model_key]

        system_prompt = (
            "You are a knowledgeable expert providing a thorough, well-reasoned response. "
            "Write clearly and directly. Do not add meta-commentary about the task."
        )
        prompt = format_prompt(system_prompt, task_prompt, model_cfg.chat_template)

        gen_cfg = self.cfg.generation
        last_exc = None

        for attempt in range(1, self.MAX_GENERATION_RETRIES + 1):
            try:
                with handle as h:
                    result = h.generate(
                        prompt=prompt,
                        max_new_tokens=gen_cfg.max_new_tokens,
                        temperature=gen_cfg.temperature,
                        do_sample=gen_cfg.do_sample,
                        top_p=gen_cfg.top_p,
                        top_k=gen_cfg.top_k,
                        repetition_penalty=gen_cfg.repetition_penalty,
                        seed=seed,
                    )
                    version_hash = h.version_hash

                self.exp_log.log_response(
                    iteration_id=iteration_id,
                    task_template_id=task.template_id,
                    task_variation_id=task.variation_id,
                    model_id=model_cfg.display_name,
                    response_text=result.text,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    elapsed_seconds=result.elapsed_seconds,
                    model_version_hash=version_hash,
                    asymmetry_flag=True,
                )
                return result

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Generation attempt %d/%d failed for %s at iter %d: %s",
                    attempt, self.MAX_GENERATION_RETRIES, model_key, iteration_id, exc,
                )
                from src.models import aggressive_gpu_clear
                aggressive_gpu_clear()

        # All retries failed — log and return placeholder
        logger.error(
            "All generation retries exhausted for %s at iter %d.", model_key, iteration_id
        )
        self.exp_log.log_error(
            iteration_id=iteration_id,
            error_type="GenerationFailure",
            error_message=str(last_exc),
            context={"model_key": model_key},
        )
        from src.models import GenerationResult
        return GenerationResult(
            text="[GENERATION_FAILED]",
            input_tokens=0,
            output_tokens=0,
            elapsed_seconds=0.0,
            model_id=model_cfg.display_name,
        )

    # ── Phase 2 helpers ───────────────────────────────────────

    def _judge_response(
        self,
        judge_key: str,
        respondent_response: str,
        task_prompt: str,
        iteration_id: int,
        task: "TaskVariation",  # noqa: F821
        respondent_key: str,
    ) -> "JudgeScore":
        """Load judge model, score, unload."""
        handle = self.handles[judge_key]
        with handle as h:
            return self._judge_response_with_loaded_model(
                model_handle=h,
                respondent_response=respondent_response,
                task_prompt=task_prompt,
                iteration_id=iteration_id,
                task=task,
                judge_key=judge_key,
                respondent_key=respondent_key,
            )

    def _judge_response_with_loaded_model(
        self,
        model_handle: "ModelHandle",  # noqa: F821
        respondent_response: str,
        task_prompt: str,
        iteration_id: int,
        task: "TaskVariation",   # noqa: F821
        judge_key: str,
        respondent_key: str,
    ) -> "JudgeScore":
        """Score a response using an already-loaded ModelHandle."""
        from tasks.rubrics import build_judge_prompt, parse_judge_output
        from src.models import format_prompt

        judge_cfg = self.cfg.models[judge_key]
        judge_params = self.cfg.judging

        system_prompt = (
            "You are a calibrated expert evaluator. "
            "Your task is to score a written response strictly according to the rubric provided. "
            "Output ONLY valid JSON as specified. Do not include any other text."
        )
        rubric_prompt = build_judge_prompt(
            task_prompt=task_prompt,
            response_text=respondent_response,
            rubric_dimensions=self.cfg.rubric.dimensions,
        )
        full_prompt = format_prompt(system_prompt, rubric_prompt, judge_cfg.chat_template)

        best_score = None
        for attempt in range(1, judge_params.parse_retries + 1):
            try:
                result = model_handle.generate(
                    prompt=full_prompt,
                    max_new_tokens=judge_params.max_new_tokens,
                    temperature=judge_params.temperature,
                    do_sample=judge_params.do_sample,
                )
                score = parse_judge_output(
                    raw_output=result.text,
                    rubric_dimensions=self.cfg.rubric.dimensions,
                    attempt=attempt,
                )
                if score.parse_success:
                    best_score = score
                    break
                elif best_score is None:
                    best_score = score  # keep fallback
            except Exception as exc:
                logger.warning(
                    "Judge attempt %d/%d failed (%s → %s): %s",
                    attempt, judge_params.parse_retries, judge_key, respondent_key, exc,
                )

        if best_score is None:
            from tasks.rubrics import JudgeScore, DimensionScores
            best_score = JudgeScore(
                dimension_scores=DimensionScores(50, 50, 50, 50, 50),
                weighted_total=50.0,
                brief_rationale="[ALL_JUDGE_ATTEMPTS_FAILED]",
                raw_output="",
                parse_success=False,
                parse_attempts=judge_params.parse_retries,
            )

        self.exp_log.log_score(
            iteration_id=iteration_id,
            task_template_id=task.template_id,
            task_variation_id=task.variation_id,
            judge_id=self.cfg.models[judge_key].display_name,
            respondent_id=self.cfg.models[respondent_key].display_name,
            score_given=best_score.weighted_total,
            dimension_scores=best_score.dimension_scores.as_dict(),
            brief_rationale=best_score.brief_rationale,
            parse_success=best_score.parse_success,
            parse_attempts=best_score.parse_attempts,
        )

        return best_score
