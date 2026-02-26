"""
src/config.py
─────────────────────────────────────────────────────────────
Configuration loader, validator, and runtime snapshot writer.

Responsibilities
----------------
• Parse config.yaml into a typed ExperimentConfig dataclass tree.
• Validate all values and surface meaningful errors early.
• Persist a frozen JSON snapshot of the runtime config alongside
  every experiment run for exact reproducibility.
• Expose a singleton accessor so all modules share one config object.
"""

from __future__ import annotations

import json
import os
import hashlib
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Sub-config dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    hf_name: str
    display_name: str
    quantization: str
    role: str
    chat_template: str

    def __post_init__(self):
        assert self.quantization in {"4bit", "8bit", "none"}, \
            f"Unsupported quantization: {self.quantization}"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1


@dataclass
class JudgingConfig:
    max_new_tokens: int = 512
    temperature: float = 0.05
    do_sample: bool = False
    score_range: Tuple[int, int] = (0, 100)
    parse_retries: int = 3


@dataclass
class RubricDimension:
    weight: float
    description: str


@dataclass
class RubricConfig:
    dimensions: Dict[str, RubricDimension] = field(default_factory=dict)

    def __post_init__(self):
        total_weight = sum(d.weight for d in self.dimensions.values())
        assert abs(total_weight - 1.0) < 1e-6, \
            f"Rubric weights must sum to 1.0, got {total_weight:.4f}"


@dataclass
class TaskConfig:
    num_templates: int = 10
    variations_per_template: int = 100
    template_families: List[str] = field(default_factory=list)

    @property
    def total_iterations(self) -> int:
        return self.num_templates * self.variations_per_template


@dataclass
class AsymmetryConfig:
    model_judged_first: str = "model_a"
    judges_of_a: List[str] = field(default_factory=list)
    models_a_judges: List[str] = field(default_factory=list)


@dataclass
class PathConfig:
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    figures_dir: str = "figures"
    responses_cache_file: str = "logs/responses_cache.jsonl"
    scores_log_file: str = "logs/scores_log.jsonl"
    events_log_file: str = "logs/events_log.jsonl"
    metrics_output_file: str = "results/metrics_summary.json"
    checkpoint_file: str = "checkpoints/experiment_checkpoint.json"


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_enable_fp32_cpu_offload: bool = False


@dataclass
class ExperimentConfig:
    # Top-level fields
    name: str = "asymmetry_judge_drift_v1"
    description: str = ""
    seed: int = 42
    dry_run: bool = False
    dry_run_iterations: int = 20
    checkpoint_interval: int = 10

    # Sub-configs
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judging: JudgingConfig = field(default_factory=JudgingConfig)
    rubric: RubricConfig = field(default_factory=RubricConfig)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    asymmetry: AsymmetryConfig = field(default_factory=AsymmetryConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Set at load time
    config_hash: str = ""
    base_dir: str = ""

    @property
    def total_iterations(self) -> int:
        if self.dry_run:
            return self.dry_run_iterations
        return self.tasks.total_iterations

    @property
    def model_ids(self) -> List[str]:
        return list(self.models.keys())

    def effective_iterations(self) -> int:
        return self.total_iterations


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Parse config.yaml → ExperimentConfig, validate, compute content hash.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Compute a deterministic hash of the raw config for reproducibility
    raw_bytes = json.dumps(raw, sort_keys=True).encode()
    config_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]

    exp_section = raw.get("experiment", {})

    # ── Models ────────────────────────────────────────────────
    models: Dict[str, ModelConfig] = {}
    for model_key, mv in raw.get("models", {}).items():
        models[model_key] = ModelConfig(
            hf_name=mv["hf_name"],
            display_name=mv["display_name"],
            quantization=mv["quantization"],
            role=mv["role"],
            chat_template=mv["chat_template"],
        )

    # ── Generation ────────────────────────────────────────────
    gv = raw.get("generation", {})
    generation = GenerationConfig(
        max_new_tokens=gv.get("max_new_tokens", 512),
        temperature=gv.get("temperature", 0.7),
        do_sample=gv.get("do_sample", True),
        top_p=gv.get("top_p", 0.9),
        top_k=gv.get("top_k", 50),
        repetition_penalty=gv.get("repetition_penalty", 1.1),
    )

    # ── Judging ───────────────────────────────────────────────
    jv = raw.get("judging", {})
    score_range_raw = jv.get("score_range", [0, 100])
    judging = JudgingConfig(
        max_new_tokens=jv.get("max_new_tokens", 512),
        temperature=jv.get("temperature", 0.05),
        do_sample=jv.get("do_sample", False),
        score_range=tuple(score_range_raw),
        parse_retries=jv.get("parse_retries", 3),
    )

    # ── Rubric ────────────────────────────────────────────────
    rv = raw.get("rubric", {})
    rubric_dims: Dict[str, RubricDimension] = {}
    for dim_name, dv in rv.get("dimensions", {}).items():
        rubric_dims[dim_name] = RubricDimension(
            weight=dv["weight"],
            description=dv["description"],
        )
    rubric = RubricConfig(dimensions=rubric_dims)

    # ── Tasks ─────────────────────────────────────────────────
    tv = raw.get("tasks", {})
    tasks = TaskConfig(
        num_templates=tv.get("num_templates", 10),
        variations_per_template=tv.get("variations_per_template", 100),
        template_families=tv.get("template_families", []),
    )

    # ── Asymmetry ─────────────────────────────────────────────
    av = raw.get("asymmetry", {})
    asymmetry = AsymmetryConfig(
        model_judged_first=av.get("model_judged_first", "model_a"),
        judges_of_a=av.get("judges_of_a", []),
        models_a_judges=av.get("models_a_judges", []),
    )

    # ── Paths ─────────────────────────────────────────────────
    pv = raw.get("paths", {})
    paths = PathConfig(**{k: v for k, v in pv.items()})

    # ── Quantization ──────────────────────────────────────────
    qv = raw.get("quantization", {})
    quantization = QuantizationConfig(
        load_in_4bit=qv.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=qv.get("bnb_4bit_compute_dtype", "float16"),
        bnb_4bit_quant_type=qv.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=qv.get("bnb_4bit_use_double_quant", True),
        llm_int8_enable_fp32_cpu_offload=qv.get("llm_int8_enable_fp32_cpu_offload", False),
    )

    cfg = ExperimentConfig(
        name=exp_section.get("name", "asymmetry_judge_drift_v1"),
        description=exp_section.get("description", ""),
        seed=exp_section.get("seed", 42),
        dry_run=exp_section.get("dry_run", False),
        dry_run_iterations=exp_section.get("dry_run_iterations", 20),
        checkpoint_interval=exp_section.get("checkpoint_interval", 10),
        models=models,
        generation=generation,
        judging=judging,
        rubric=rubric,
        tasks=tasks,
        asymmetry=asymmetry,
        paths=paths,
        quantization=quantization,
        config_hash=config_hash,
        base_dir=str(config_path.parent),
    )

    _validate_config(cfg)
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate_config(cfg: ExperimentConfig) -> None:
    """Assert all cross-field constraints."""
    assert len(cfg.models) >= 2, "Experiment requires at least 2 models."
    assert cfg.seed >= 0, "Seed must be non-negative."
    assert cfg.tasks.num_templates > 0
    assert cfg.tasks.variations_per_template > 0
    assert cfg.generation.temperature > 0
    assert cfg.judging.temperature >= 0

    # Asymmetry references must resolve
    for ref in cfg.asymmetry.judges_of_a + cfg.asymmetry.models_a_judges:
        assert ref in cfg.models, \
            f"Asymmetry references unknown model key: '{ref}'"
    assert cfg.asymmetry.model_judged_first in cfg.models, \
        f"model_judged_first '{cfg.asymmetry.model_judged_first}' not in models."

    # Rubric weight sum
    total_w = sum(d.weight for d in cfg.rubric.dimensions.values())
    assert abs(total_w - 1.0) < 1e-5, \
        f"Rubric weights must sum to 1.0, got {total_w}"


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot persistence
# ──────────────────────────────────────────────────────────────────────────────

def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses (and nested dicts/lists) to plain dicts."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(i) for i in obj]
    return obj


def save_config_snapshot(cfg: ExperimentConfig, output_dir: str | Path) -> Path:
    """
    Write a frozen JSON snapshot of the full config to output_dir.
    Returns the path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"config_snapshot_{cfg.config_hash}.json"
    snapshot = _dataclass_to_dict(cfg)
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    return snapshot_path


# ──────────────────────────────────────────────────────────────────────────────
# Singleton accessor
# ──────────────────────────────────────────────────────────────────────────────

_GLOBAL_CONFIG: Optional[ExperimentConfig] = None


def set_global_config(cfg: ExperimentConfig) -> None:
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = cfg


def get_config() -> ExperimentConfig:
    if _GLOBAL_CONFIG is None:
        raise RuntimeError("Global config has not been set. Call set_global_config() first.")
    return _GLOBAL_CONFIG
