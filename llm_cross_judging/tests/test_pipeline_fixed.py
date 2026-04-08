import sys
from unittest.mock import MagicMock

# Mock vllm and torch before importing pipeline (not installed locally)
sys.modules['vllm'] = MagicMock()
sys.modules['torch'] = MagicMock()

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import llm_cross_judging.LLM_Cross_judging_fixed as pipe


@pytest.fixture(autouse=True)
def reset_state():
    if hasattr(pipe, '_reset_event_log'):
        pipe._reset_event_log()
    yield
    if hasattr(pipe, '_reset_event_log'):
        pipe._reset_event_log()


def test_model_config_keys_match_models():
    assert set(pipe.MODEL_CONFIG.keys()) == set(pipe.MODELS.keys())

def test_model_config_has_required_fields():
    for key, cfg in pipe.MODEL_CONFIG.items():
        assert "is_sft" in cfg
        assert "label" in cfg
        assert isinstance(cfg["is_sft"], bool)
        assert isinstance(cfg["label"], str) and cfg["label"]

def test_labels_are_unique():
    labels = [cfg["label"] for cfg in pipe.MODEL_CONFIG.values()]
    assert len(labels) == len(set(labels))

def test_nine_models():
    assert len(pipe.MODELS) == 9

def test_config_constants():
    assert pipe.N_RUNS_PER_PROMPT == 10
    assert pipe.K_JUDGE_SAMPLES == 3
    assert pipe.TEMPERATURES[0] == 0.0
    assert pipe.TEMPERATURES[-1] == 1.0
    assert len(pipe.TEMPERATURES) == 11

def test_predict_runtime_returns_float():
    result = pipe.predict_runtime(num_prompts=2, num_models=2, num_temperatures=1, n_runs=2, k_judge_samples=1)
    assert isinstance(result, float)
    assert result > 0

def test_predict_runtime_scales_with_prompts():
    t1 = pipe.predict_runtime(num_prompts=1, num_models=2, num_temperatures=1, n_runs=2, k_judge_samples=1)
    t2 = pipe.predict_runtime(num_prompts=2, num_models=2, num_temperatures=1, n_runs=2, k_judge_samples=1)
    assert t2 > t1

def test_predict_runtime_scales_with_temperatures():
    t1 = pipe.predict_runtime(num_prompts=2, num_models=2, num_temperatures=1, n_runs=2, k_judge_samples=1)
    t2 = pipe.predict_runtime(num_prompts=2, num_models=2, num_temperatures=2, n_runs=2, k_judge_samples=1)
    assert t2 > t1

def test_predict_runtime_defaults_use_globals():
    result_default = pipe.predict_runtime()
    result_explicit = pipe.predict_runtime(
        num_prompts=len(pipe.PROMPTS),
        num_models=len(pipe.MODELS),
        num_temperatures=len(pipe.TEMPERATURES),
        n_runs=pipe.N_RUNS_PER_PROMPT,
        k_judge_samples=pipe.K_JUDGE_SAMPLES,
    )
    assert result_default == result_explicit
