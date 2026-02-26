"""
src/checkpointing.py
─────────────────────────────────────────────────────────────
Experiment checkpoint manager for resume-from-failure support.

Design
──────
• A single JSON checkpoint file stores the minimal state needed
  to resume an interrupted experiment.
• Atomic write (write-to-temp + rename) prevents corrupt checkpoints.
• Random state is serialised as a list so it survives pickling.
• On resume, the runner must fast-forward through already-logged
  iterations rather than re-running them.

Checkpoint schema
─────────────────
{
  "schema_version":            "1.0",
  "experiment_name":           str,
  "config_hash":               str,
  "global_seed":               int,
  "last_completed_iteration":  int,   # -1 means not started
  "total_iterations":          int,
  "task_pointer": {
    "template_id":             int,
    "variation_id":            int,
  },
  "python_random_state":       list,  # random.getstate()
  "numpy_random_state": {            # np.random.get_state()
    "name":                    str,
    "keys":                    list,
    "pos":                     int,
    "has_gauss":               int,
    "cached_gaussian":         float,
  },
  "timestamps": {
    "created_at":              str,   # ISO-8601 UTC
    "last_updated_at":         str,
  },
  "iteration_durations_sec":   list[float],  # wall-clock per iteration
  "model_version_hashes":      dict,  # {model_key: hash}
}
"""

from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint data structure
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    schema_version: str = SCHEMA_VERSION
    experiment_name: str = ""
    config_hash: str = ""
    global_seed: int = 42
    last_completed_iteration: int = -1   # -1 = not started
    total_iterations: int = 1000
    template_id: int = 0
    variation_id: int = 0
    python_random_state: Optional[list] = None
    numpy_random_state: Optional[dict] = None
    created_at: str = ""
    last_updated_at: str = ""
    iteration_durations_sec: List[float] = field(default_factory=list)
    model_version_hashes: Dict[str, str] = field(default_factory=dict)

    @property
    def next_iteration(self) -> int:
        return self.last_completed_iteration + 1

    @property
    def is_complete(self) -> bool:
        return self.last_completed_iteration >= self.total_iterations - 1

    @property
    def progress_fraction(self) -> float:
        if self.total_iterations == 0:
            return 0.0
        return (self.last_completed_iteration + 1) / self.total_iterations

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        if not self.iteration_durations_sec:
            return None
        recent = self.iteration_durations_sec[-min(20, len(self.iteration_durations_sec)):]
        mean_dur = sum(recent) / len(recent)
        remaining = self.total_iterations - (self.last_completed_iteration + 1)
        return mean_dur * remaining

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "config_hash": self.config_hash,
            "global_seed": self.global_seed,
            "last_completed_iteration": self.last_completed_iteration,
            "total_iterations": self.total_iterations,
            "task_pointer": {
                "template_id": self.template_id,
                "variation_id": self.variation_id,
            },
            "python_random_state": self.python_random_state,
            "numpy_random_state": self.numpy_random_state,
            "timestamps": {
                "created_at": self.created_at,
                "last_updated_at": self.last_updated_at,
            },
            "iteration_durations_sec": self.iteration_durations_sec,
            "model_version_hashes": self.model_version_hashes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        tp = d.get("task_pointer", {})
        ts = d.get("timestamps", {})
        return cls(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            experiment_name=d.get("experiment_name", ""),
            config_hash=d.get("config_hash", ""),
            global_seed=d.get("global_seed", 42),
            last_completed_iteration=d.get("last_completed_iteration", -1),
            total_iterations=d.get("total_iterations", 1000),
            template_id=tp.get("template_id", 0),
            variation_id=tp.get("variation_id", 0),
            python_random_state=d.get("python_random_state"),
            numpy_random_state=d.get("numpy_random_state"),
            created_at=ts.get("created_at", ""),
            last_updated_at=ts.get("last_updated_at", ""),
            iteration_durations_sec=d.get("iteration_durations_sec", []),
            model_version_hashes=d.get("model_version_hashes", {}),
        )


# ──────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ──────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    """
    Manages reading and writing experiment checkpoints.

    Usage
    -----
        cm = CheckpointManager(cfg)
        ckpt = cm.load_or_create()              # Resume if file exists
        ...
        cm.update(ckpt, iteration_id=5, ...)    # Update fields
        cm.save(ckpt)                            # Atomic write
    """

    def __init__(self, cfg):
        """cfg: ExperimentConfig"""
        self.cfg = cfg
        self.checkpoint_path = Path(cfg.base_dir) / cfg.paths.checkpoint_file
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load / Create ─────────────────────────────────────────

    def load_or_create(self) -> Checkpoint:
        """
        Load existing checkpoint if present and hash-compatible.
        Otherwise create a fresh checkpoint.
        """
        if self.checkpoint_path.exists():
            try:
                ckpt = self._load()
                if ckpt.config_hash != self.cfg.config_hash:
                    logger.warning(
                        "Checkpoint config_hash mismatch "
                        "(stored=%s, current=%s). Starting fresh.",
                        ckpt.config_hash,
                        self.cfg.config_hash,
                    )
                    return self._create_fresh()
                if ckpt.is_complete:
                    logger.info(
                        "Checkpoint shows experiment already complete (%d/%d iterations).",
                        ckpt.last_completed_iteration + 1, ckpt.total_iterations,
                    )
                else:
                    logger.info(
                        "Resuming from checkpoint: iteration %d/%d "
                        "(%.1f%% complete).",
                        ckpt.next_iteration,
                        ckpt.total_iterations,
                        ckpt.progress_fraction * 100,
                    )
                return ckpt
            except Exception as exc:
                logger.error("Failed to load checkpoint: %s — starting fresh.", exc)
                return self._create_fresh()
        else:
            logger.info("No checkpoint found. Starting fresh experiment.")
            return self._create_fresh()

    def _create_fresh(self) -> Checkpoint:
        now = datetime.now(timezone.utc).isoformat()
        return Checkpoint(
            experiment_name=self.cfg.name,
            config_hash=self.cfg.config_hash,
            global_seed=self.cfg.seed,
            last_completed_iteration=-1,
            total_iterations=self.cfg.total_iterations,
            template_id=0,
            variation_id=0,
            created_at=now,
            last_updated_at=now,
        )

    def _load(self) -> Checkpoint:
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._check_schema(data)
        return Checkpoint.from_dict(data)

    @staticmethod
    def _check_schema(data: dict) -> None:
        stored = data.get("schema_version", "unknown")
        if stored != SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema version mismatch: stored={stored}, expected={SCHEMA_VERSION}"
            )

    # ── Update ────────────────────────────────────────────────

    def update(
        self,
        ckpt: Checkpoint,
        iteration_id: int,
        template_id: int,
        variation_id: int,
        iteration_duration_sec: float,
        model_version_hashes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Update checkpoint fields after completing an iteration.
        Captures Python and NumPy random states.
        """
        ckpt.last_completed_iteration = iteration_id
        ckpt.template_id = template_id
        ckpt.variation_id = variation_id
        ckpt.last_updated_at = datetime.now(timezone.utc).isoformat()
        ckpt.iteration_durations_sec.append(round(iteration_duration_sec, 3))
        # Keep only the last 100 durations (prevent unbounded growth)
        if len(ckpt.iteration_durations_sec) > 100:
            ckpt.iteration_durations_sec = ckpt.iteration_durations_sec[-100:]

        if model_version_hashes:
            ckpt.model_version_hashes.update(model_version_hashes)

        # Capture random states for perfect resume
        ckpt.python_random_state = list(random.getstate())
        try:
            import numpy as np
            np_state = np.random.get_state()
            ckpt.numpy_random_state = {
                "name": np_state[0],
                "keys": np_state[1].tolist(),
                "pos": int(np_state[2]),
                "has_gauss": int(np_state[3]),
                "cached_gaussian": float(np_state[4]),
            }
        except ImportError:
            ckpt.numpy_random_state = None

    def restore_random_state(self, ckpt: Checkpoint) -> None:
        """Restore Python and NumPy random states from checkpoint."""
        if ckpt.python_random_state is not None:
            try:
                random.setstate(tuple(ckpt.python_random_state))
                logger.debug("Python random state restored from checkpoint.")
            except Exception as e:
                logger.warning("Could not restore Python random state: %s", e)

        if ckpt.numpy_random_state is not None:
            try:
                import numpy as np
                s = ckpt.numpy_random_state
                np.random.set_state((
                    s["name"],
                    np.array(s["keys"], dtype=np.uint32),
                    s["pos"],
                    s["has_gauss"],
                    s["cached_gaussian"],
                ))
                logger.debug("NumPy random state restored from checkpoint.")
            except ImportError:
                pass
            except Exception as e:
                logger.warning("Could not restore NumPy random state: %s", e)

    # ── Save (atomic) ─────────────────────────────────────────

    def save(self, ckpt: Checkpoint) -> None:
        """
        Atomically write checkpoint to disk.
        Writes to a temp file then renames — prevents partial writes.
        """
        data = ckpt.to_dict()
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self.checkpoint_path.parent,
            prefix=".tmp_ckpt_",
            suffix=".json",
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.checkpoint_path)
            logger.debug(
                "Checkpoint saved: iteration %d/%d",
                ckpt.last_completed_iteration + 1,
                ckpt.total_iterations,
            )
        except Exception as exc:
            # Clean up temp on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise RuntimeError(f"Checkpoint write failed: {exc}") from exc

    # ── Consistency assertion ─────────────────────────────────

    def assert_config_consistency(self, ckpt: Checkpoint) -> None:
        """
        Assert that the loaded checkpoint is consistent with the current config.
        Raises AssertionError with a diagnostic message on mismatch.
        """
        assert ckpt.global_seed == self.cfg.seed, (
            f"Seed mismatch: checkpoint={ckpt.global_seed}, config={self.cfg.seed}"
        )
        assert ckpt.total_iterations == self.cfg.total_iterations, (
            f"Total iterations mismatch: checkpoint={ckpt.total_iterations}, "
            f"config={self.cfg.total_iterations}"
        )
