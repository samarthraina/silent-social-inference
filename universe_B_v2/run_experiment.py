#!/usr/bin/env python3
"""
run_experiment.py
─────────────────────────────────────────────────────────────
Main experiment runner for: Asymmetry-Induced Judge Drift

Entry points
────────────
    python run_experiment.py                          # Full run from config.yaml
    python run_experiment.py --config path/to/config  # Custom config
    python run_experiment.py --dry-run                # 20-iteration dry run
    python run_experiment.py --resume                 # Resume from checkpoint
    python run_experiment.py --metrics-only           # Skip experiment; recompute metrics
    python run_experiment.py --figures-only           # Skip experiment; re-render figures

Runtime pipeline
────────────────
1. Load & validate config (config.yaml)
2. Setup logging
3. Seed all random sources deterministically
4. Load or create checkpoint
5. Save config snapshot for reproducibility
6. For each iteration [resume_point → total]:
   a. Run IterationProtocol (Phase 1 + 2a + 2b)
   b. Update and save checkpoint every N iterations
7. Compute all bias metrics
8. Save metrics summary JSON
9. Render all publication figures
10. Log experiment completion

Failure handling
────────────────
• uncaught exceptions → log_error + graceful checkpoint write
• OOM errors → model unload + GPU clear + retry (up to 2x per iteration)
• If an iteration fails entirely, insert a SKIPPED record and continue.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

# Add parent to sys.path for clean imports regardless of CWD
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Asymmetry-Induced Judge Drift Experiment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(_THIS_DIR / "config.yaml"),
        help="Path to experiment config YAML (default: config.yaml in script dir)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Override config: run only dry_run_iterations iterations",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing checkpoint (default behaviour if checkpoint exists)",
    )
    p.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip the LLM experiment. Load existing logs and recompute metrics + figures.",
    )
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip the LLM experiment and metrics. Only re-render figures.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level for console output",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Seed management
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Set all random sources to a deterministic state."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info("Global seed set: %d", seed)


# ──────────────────────────────────────────────────────────────────────────────
# Consistency checks
# ──────────────────────────────────────────────────────────────────────────────

def run_consistency_assertions(cfg) -> None:
    """
    Pre-flight assertions to verify environment and config sanity.
    Raises AssertionError if any check fails.
    """
    # All model keys in asymmetry section resolve
    for ref in cfg.asymmetry.judges_of_a + cfg.asymmetry.models_a_judges:
        assert ref in cfg.models, f"Asymmetry references undefined model: '{ref}'"

    # Rubric weights
    total_w = sum(d.weight for d in cfg.rubric.dimensions.values())
    assert abs(total_w - 1.0) < 1e-4, f"Rubric weights must sum to 1.0, got {total_w}"

    # Task family count
    assert cfg.tasks.num_templates == len(cfg.tasks.template_families), (
        f"num_templates ({cfg.tasks.num_templates}) != len(template_families) "
        f"({len(cfg.tasks.template_families)})"
    )

    logger.info("Consistency assertions passed.")


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment_loop(cfg, handles, task_reg, exp_log, ckpt_mgr, ckpt) -> None:
    """
    Core iteration loop.

    On resume, skips iterations ≤ ckpt.last_completed_iteration.
    """
    from src.protocols import IterationProtocol
    from src.models import aggressive_gpu_clear

    protocol = IterationProtocol(cfg, handles, task_reg, exp_log)
    total = cfg.total_iterations
    resume_from = ckpt.next_iteration
    model_a_id = cfg.models[cfg.asymmetry.model_judged_first].display_name

    logger.info(
        "Starting experiment loop: %d → %d  (total=%d)",
        resume_from, total - 1, total,
    )

    for iteration_id in range(resume_from, total):
        iter_start = time.perf_counter()
        try:
            result = protocol.run_iteration(iteration_id)

            iter_duration = time.perf_counter() - iter_start
            ckpt_mgr.update(
                ckpt=ckpt,
                iteration_id=iteration_id,
                template_id=result.task_template_id,
                variation_id=result.task_variation_id,
                iteration_duration_sec=iter_duration,
            )

            # Periodic checkpoint save
            if (iteration_id + 1) % cfg.checkpoint_interval == 0:
                ckpt_mgr.save(ckpt)
                exp_log.log_checkpoint(
                    iteration_id=iteration_id,
                    checkpoint_path=str(ckpt_mgr.checkpoint_path),
                )

            # Progress report
            pct = (iteration_id + 1) / total * 100
            eta_str = ""
            if ckpt.estimated_remaining_seconds is not None:
                eta_s = ckpt.estimated_remaining_seconds
                eta_str = f"  ETA: {eta_s/60:.1f}min"
            logger.info(
                "  [%5.1f%%] iter=%d  A_recv=%.1f%s",
                pct, iteration_id, result.model_a_received_mean, eta_str,
            )

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt at iteration %d. Saving checkpoint.", iteration_id)
            ckpt_mgr.save(ckpt)
            raise

        except Exception as exc:
            logger.error(
                "Iteration %d failed: %s\n%s",
                iteration_id, exc, traceback.format_exc(),
            )
            exp_log.log_error(
                iteration_id=iteration_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            # Attempt GPU recovery and continue
            aggressive_gpu_clear()
            continue

    # Final checkpoint
    ckpt_mgr.save(ckpt)
    logger.info("Experiment loop complete.")


# ──────────────────────────────────────────────────────────────────────────────
# Post-processing
# ──────────────────────────────────────────────────────────────────────────────

def run_post_processing(cfg, exp_log) -> None:
    """Compute metrics and render all figures after the experiment loop."""
    from src.metrics import compute_all_metrics, save_metrics_summary
    from src.visualization import render_all_figures

    base = Path(cfg.base_dir)
    scores_path = base / cfg.paths.scores_log_file
    figures_dir = base / cfg.paths.figures_dir
    metrics_path = base / cfg.paths.metrics_output_file

    model_a_id = cfg.models[cfg.asymmetry.model_judged_first].display_name

    logger.info("Computing bias metrics...")
    try:
        summary = compute_all_metrics(
            scores_log_path=scores_path,
            model_a_id=model_a_id,
            rolling_window=20,
            config_hash=cfg.config_hash,
        )
        save_metrics_summary(summary, metrics_path)

        logger.info("Rendering figures...")
        render_all_figures(
            scores_log_path=scores_path,
            metrics_summary=summary,
            figures_dir=figures_dir,
            model_a_id=model_a_id,
            rolling_window=20,
        )
    except Exception as exc:
        logger.error("Post-processing failed: %s\n%s", exc, traceback.format_exc())
        exp_log.log_error(
            iteration_id=None,
            error_type="PostProcessingFailure",
            error_message=str(exc),
        )
        raise


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────
    from src.config import load_config, set_global_config, save_config_snapshot
    cfg = load_config(args.config)
    if args.dry_run:
        cfg.dry_run = True
    set_global_config(cfg)

    # ── Setup logging ─────────────────────────────────────────
    from src.logging_utils import setup_python_logging, ExperimentLogger
    base = Path(cfg.base_dir)
    log_level = getattr(logging, args.log_level, logging.INFO)
    setup_python_logging(
        log_dir=base / cfg.paths.log_dir,
        experiment_name=cfg.name,
        level=log_level,
    )
    logger.info("=" * 60)
    logger.info("Experiment: %s  (config_hash=%s)", cfg.name, cfg.config_hash)
    logger.info("Dry run: %s  |  Total iterations: %d", cfg.dry_run, cfg.total_iterations)
    logger.info("=" * 60)

    # ── Initialise experiment logger ──────────────────────────
    exp_log = ExperimentLogger(
        responses_path=base / cfg.paths.responses_cache_file,
        scores_path=base / cfg.paths.scores_log_file,
        events_path=base / cfg.paths.events_log_file,
        global_seed=cfg.seed,
    )

    # ── Metrics / figures-only shortcuts ──────────────────────
    if args.figures_only:
        logger.info("--figures-only: skipping experiment, re-rendering figures only.")
        run_post_processing(cfg, exp_log)
        return

    if args.metrics_only:
        logger.info("--metrics-only: skipping experiment, recomputing metrics + figures.")
        run_post_processing(cfg, exp_log)
        return

    # ── Consistency checks ────────────────────────────────────
    run_consistency_assertions(cfg)

    # ── Save config snapshot ──────────────────────────────────
    snapshot_path = save_config_snapshot(cfg, output_dir=base / cfg.paths.results_dir)
    logger.info("Config snapshot saved: %s", snapshot_path)

    # ── Seed RNGs ─────────────────────────────────────────────
    set_global_seed(cfg.seed)

    # ── Checkpoint ────────────────────────────────────────────
    from src.checkpointing import CheckpointManager
    ckpt_mgr = CheckpointManager(cfg)
    ckpt = ckpt_mgr.load_or_create()
    if ckpt.last_completed_iteration >= 0:
        ckpt_mgr.restore_random_state(ckpt)

    if ckpt.is_complete:
        logger.info("Experiment already complete per checkpoint. Running post-processing only.")
        run_post_processing(cfg, exp_log)
        return

    # ── Build model handles (not yet loaded) ──────────────────
    from src.models import create_handles
    handles = create_handles(cfg)

    # ── Build task registry ───────────────────────────────────
    from tasks.templates import TaskRegistry
    task_reg = TaskRegistry(global_seed=cfg.seed)

    # ── Log experiment start ──────────────────────────────────
    exp_log.log_experiment_start(
        config_hash=cfg.config_hash,
        total_iterations=cfg.total_iterations,
    )

    # ── Main loop ─────────────────────────────────────────────
    experiment_start_time = time.time()
    try:
        run_experiment_loop(cfg, handles, task_reg, exp_log, ckpt_mgr, ckpt)
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user.")
        exp_log.log_event(
            event_type="experiment_interrupted",
            message="KeyboardInterrupt received.",
            iteration_id=ckpt.last_completed_iteration,
        )
        sys.exit(0)
    except Exception as exc:
        logger.critical("Unhandled exception in experiment loop: %s", exc)
        exp_log.log_error(
            iteration_id=ckpt.last_completed_iteration,
            error_type="UnhandledError",
            error_message=str(exc),
            context={"traceback": traceback.format_exc()},
        )
        raise

    total_time = time.time() - experiment_start_time
    logger.info("Experiment loop finished in %.1f seconds (%.1f min).", total_time, total_time / 60)

    # ── Post-processing ───────────────────────────────────────
    run_post_processing(cfg, exp_log)

    # ── Complete ──────────────────────────────────────────────
    exp_log.log_experiment_end(total_iterations_completed=ckpt.last_completed_iteration + 1)
    logger.info("Experiment complete. Results in: %s", base / cfg.paths.results_dir)
    logger.info("Figures in:  %s", base / cfg.paths.figures_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
