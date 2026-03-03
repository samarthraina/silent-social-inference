"""
src/logging_utils.py
─────────────────────────────────────────────────────────────
Structured JSONL logging for all experiment events.

Three log streams
─────────────────
1. responses_cache.jsonl   — One record per model response (Phase 1 output).
2. scores_log.jsonl        — One record per judging act (Phase 2 output).
3. events_log.jsonl        — Experiment lifecycle events (start, checkpoint, error).

Each stream appends one JSON object per line (JSONL / newline-delimited JSON).
This format is:
  • Appendable without loading the whole file.
  • Readable by pandas with pd.read_json(..., lines=True).
  • Resumable — no corruption from partial writes.

All writers are thread-safe via file-level locking (though the experiment
runs single-threaded).

Schema specification
────────────────────
ResponseRecord
  iteration_id        int      : Global iteration counter (0-indexed)
  task_template_id    int      : Template family index (0-9)
  task_variation_id   int      : Variation index within template (0-99)
  model_id            str      : Responding model display name
  response_text       str      : Full generated response
  input_tokens        int      : Prompt token count
  output_tokens       int      : Response token count
  elapsed_seconds     float    : Wall-clock generation time
  global_seed         int      : Experiment global seed
  model_version_hash  str      : SHA-256[:16] of model config files
  asymmetry_flag      bool     : Always True (structural asymmetry applied)
  timestamp           str      : ISO-8601 UTC timestamp

ScoreRecord
  iteration_id        int
  task_template_id    int
  task_variation_id   int
  judge_id            str      : Judging model display name
  respondent_id       str      : Responding model display name
  score_given         float    : Weighted total score awarded by judge
  score_received      float    : Same score from respondent's perspective
  dimension_scores    dict     : Per-dimension raw scores
  brief_rationale     str      : Judge's 1-3 sentence rationale
  parse_success       bool     : Whether JSON parse succeeded
  parse_attempts      int      : Number of parse attempts used
  score_visibility_mode  int   : 0 = Opaque, 1 = Visible (config-level constant)
  global_seed         int
  timestamp           str

VisibilityEventRecord  (written to events_log.jsonl)
  event_type             str   : "visibility_notification"
  iteration_id           int
  score_visibility_mode  int   : 0 or 1
  average_score_for_a    float : Mean score received by Model A in this iteration
  visibility_timestamp   str   : ISO-8601 UTC timestamp of notification
  message                str
  metadata               dict

EventRecord
  event_type          str      : "experiment_start" | "checkpoint" | "iteration_complete"
                                 | "error" | "experiment_end"
  iteration_id        int | None
  message             str
  metadata            dict     : Arbitrary key-value data
  timestamp           str
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_jsonl(filepath: str | Path, record: Dict[str, Any]) -> None:
    """
    Atomically append one JSON record to a JSONL file.
    Uses OS-level advisory lock to prevent corruption.
    """
    filepath = _ensure_dir(filepath)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with open(filepath, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment logger
# ──────────────────────────────────────────────────────────────────────────────

class ExperimentLogger:
    """
    Manages all three JSONL log streams for an experiment run.

    Parameters
    ----------
    responses_path : str | Path
    scores_path    : str | Path
    events_path    : str | Path
    global_seed    : int
    """

    def __init__(
        self,
        responses_path: str | Path,
        scores_path: str | Path,
        events_path: str | Path,
        global_seed: int = 42,
        score_visibility_mode: int = 0,
    ):
        self.responses_path = Path(responses_path)
        self.scores_path = Path(scores_path)
        self.events_path = Path(events_path)
        self.global_seed = global_seed
        self.score_visibility_mode = score_visibility_mode
        self._iteration_count: int = 0
        self._score_count: int = 0

    # ── Response logging ──────────────────────────────────────

    def log_response(
        self,
        iteration_id: int,
        task_template_id: int,
        task_variation_id: int,
        model_id: str,
        response_text: str,
        input_tokens: int,
        output_tokens: int,
        elapsed_seconds: float,
        model_version_hash: str,
        asymmetry_flag: bool = True,
    ) -> None:
        record = {
            "record_type": "response",
            "iteration_id": iteration_id,
            "task_template_id": task_template_id,
            "task_variation_id": task_variation_id,
            "model_id": model_id,
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "global_seed": self.global_seed,
            "model_version_hash": model_version_hash,
            "asymmetry_flag": asymmetry_flag,
            "timestamp": _utc_now(),
        }
        _append_jsonl(self.responses_path, record)
        logger.debug(
            "Response logged  iter=%d  model=%s  tokens=%d",
            iteration_id, model_id, output_tokens,
        )

    # ── Score logging ─────────────────────────────────────────

    def log_score(
        self,
        iteration_id: int,
        task_template_id: int,
        task_variation_id: int,
        judge_id: str,
        respondent_id: str,
        score_given: float,
        dimension_scores: Dict[str, int],
        brief_rationale: str,
        parse_success: bool,
        parse_attempts: int,
    ) -> None:
        record = {
            "record_type": "score",
            "iteration_id": iteration_id,
            "task_template_id": task_template_id,
            "task_variation_id": task_variation_id,
            "judge_id": judge_id,
            "respondent_id": respondent_id,
            "score_given": round(score_given, 2),
            "score_received": round(score_given, 2),  # Same value, dual perspective field
            "dimension_scores": dimension_scores,
            "brief_rationale": brief_rationale,
            "parse_success": parse_success,
            "parse_attempts": parse_attempts,
            "score_visibility_mode": self.score_visibility_mode,
            "global_seed": self.global_seed,
            "timestamp": _utc_now(),
        }
        _append_jsonl(self.scores_path, record)
        self._score_count += 1
        logger.debug(
            "Score logged  iter=%d  judge=%s → respondent=%s  score=%.1f",
            iteration_id, judge_id, respondent_id, score_given,
        )

    # ── Visibility event logging ────────────────────────────────────────────

    def log_visibility_event(
        self,
        iteration_id: int,
        average_score_for_a: float,
    ) -> None:
        """
        Log the per-iteration score visibility notification event.

        This event fires after Phase 2a (Model A has been judged) and before
        Phase 2b (Model A judges others), recording:
          - which condition was active (score_visibility_mode)
          - the average score Model A received
          - the precise UTC timestamp of the visibility notification

        Written to events_log.jsonl with event_type="visibility_notification".
        """
        visibility_ts = _utc_now()
        condition_label = (
            "opaque_awareness" if self.score_visibility_mode == 0
            else "visible_score_awareness"
        )
        self.log_event(
            event_type="visibility_notification",
            message=(
                f"Model A visibility notification issued (mode={self.score_visibility_mode}: "
                f"{condition_label}). "
                f"average_score_for_a={average_score_for_a:.2f}"
            ),
            iteration_id=iteration_id,
            metadata={
                "score_visibility_mode": self.score_visibility_mode,
                "average_score_for_a": round(average_score_for_a, 4),
                "visibility_timestamp": visibility_ts,
                "condition_label": condition_label,
            },
        )
        logger.debug(
            "Visibility event logged  iter=%d  mode=%d  avg_score=%.2f",
            iteration_id, self.score_visibility_mode, average_score_for_a,
        )

    # ── Event logging ─────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        message: str,
        iteration_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "record_type": "event",
            "event_type": event_type,
            "iteration_id": iteration_id,
            "message": message,
            "metadata": metadata or {},
            "timestamp": _utc_now(),
        }
        _append_jsonl(self.events_path, record)
        logger.info("[EVENT] %s | iter=%s | %s", event_type, iteration_id, message)

    # ── Convenience helpers ───────────────────────────────────

    def log_experiment_start(self, config_hash: str, total_iterations: int) -> None:
        self.log_event(
            event_type="experiment_start",
            message="Experiment starting.",
            metadata={
                "config_hash": config_hash,
                "total_iterations": total_iterations,
                "global_seed": self.global_seed,
            },
        )

    def log_experiment_end(self, total_iterations_completed: int) -> None:
        self.log_event(
            event_type="experiment_end",
            message="Experiment completed successfully.",
            metadata={
                "total_iterations_completed": total_iterations_completed,
                "total_scores_logged": self._score_count,
            },
        )

    def log_checkpoint(self, iteration_id: int, checkpoint_path: str) -> None:
        self.log_event(
            event_type="checkpoint",
            message=f"Checkpoint saved at iteration {iteration_id}.",
            iteration_id=iteration_id,
            metadata={"checkpoint_path": checkpoint_path},
        )

    def log_error(
        self,
        iteration_id: int,
        error_type: str,
        error_message: str,
        context: Optional[Dict] = None,
    ) -> None:
        self.log_event(
            event_type="error",
            message=f"{error_type}: {error_message}",
            iteration_id=iteration_id,
            metadata={"error_type": error_type, "context": context or {}},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Python logging setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_python_logging(
    log_dir: str | Path,
    experiment_name: str,
    level: int = logging.INFO,
) -> None:
    """
    Configure root Python logger with:
    • Colour-coded console handler (INFO+).
    • Rotating file handler writing to logs/<experiment_name>.log.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{experiment_name}.log"

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Let handlers filter

    # Remove existing handlers (idempotent re-init)
    for h in root.handlers[:]:
        root.removeHandler(h)

    # ── File handler ─────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # ── Console handler ───────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    try:
        import colorlog  # optional dependency
        console_fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        console_handler.setFormatter(console_fmt)
    except ImportError:
        console_fmt = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_fmt)

    root.addHandler(file_handler)
    root.addHandler(console_handler)


# ──────────────────────────────────────────────────────────────────────────────
# JSONL reader (for analysis & resume)
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> list:
    """Load all records from a JSONL file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Malformed JSONL at %s line %d: %s", path.name, lineno, e)
    return records
