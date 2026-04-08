import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Avoid importing torch/transformers at module level in tests
import importlib, types

def _load_pipeline_config():
    """Import only the config-level names from the pipeline, without loading models."""
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    return pipe

def test_model_config_keys_match_models():
    pipe = _load_pipeline_config()
    assert set(pipe.MODEL_CONFIG.keys()) == set(pipe.MODELS.keys()), \
        "MODEL_CONFIG must have exactly the same keys as MODELS"

def test_model_config_has_required_fields():
    pipe = _load_pipeline_config()
    for key, cfg in pipe.MODEL_CONFIG.items():
        assert "is_sft" in cfg, f"{key} missing 'is_sft'"
        assert "label" in cfg, f"{key} missing 'label'"
        assert isinstance(cfg["is_sft"], bool), f"{key} is_sft must be bool"
        assert isinstance(cfg["label"], str) and cfg["label"], f"{key} label must be non-empty str"

def test_labels_are_unique():
    pipe = _load_pipeline_config()
    labels = [cfg["label"] for cfg in pipe.MODEL_CONFIG.values()]
    assert len(labels) == len(set(labels)), "All MODEL_CONFIG labels must be unique"

def test_nine_models():
    pipe = _load_pipeline_config()
    assert len(pipe.MODELS) == 9, f"Expected 9 models, got {len(pipe.MODELS)}"
