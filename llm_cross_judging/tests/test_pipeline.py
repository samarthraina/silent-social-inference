import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

def _load_pipeline_config():
    """Import only the config-level names from the pipeline, without loading models."""
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    return pipe

@pytest.fixture(autouse=True)
def reset_pipeline_state():
    pipe = _load_pipeline_config()
    pipe._loaded_model = None
    pipe._loaded_tokenizer = None
    pipe._loaded_key = None
    pipe._failed_models = set()
    yield
    pipe._loaded_model = None
    pipe._loaded_tokenizer = None
    pipe._loaded_key = None
    pipe._failed_models = set()

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

from unittest.mock import MagicMock, patch

def test_unload_clears_globals():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    # Manually set state to simulate a loaded model
    pipe._loaded_model = MagicMock()
    pipe._loaded_tokenizer = MagicMock()
    pipe._loaded_key = "llama"

    with patch("gc.collect"), patch("torch.cuda.is_available", return_value=False):
        pipe.unload_current_model()

    assert pipe._loaded_model is None
    assert pipe._loaded_tokenizer is None
    assert pipe._loaded_key is None

def test_load_model_skips_if_already_loaded():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._loaded_model = MagicMock()
    pipe._loaded_tokenizer = MagicMock()
    pipe._loaded_key = "llama"

    # load_model("llama") should return immediately without calling from_pretrained
    with patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_load:
        pipe.load_model("llama")
        mock_load.assert_not_called()
