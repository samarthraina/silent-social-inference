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
    pipe._reset_event_log()
    yield
    pipe._loaded_model = None
    pipe._loaded_tokenizer = None
    pipe._loaded_key = None
    pipe._failed_models = set()
    pipe._reset_event_log()

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

def test_log_event_appends_to_log():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._reset_event_log()
    pipe.log_event("generation", prompt_id=0, model_key="llama", phase="collect",
                   iteration_idx=2, input_text="hi", output_text="hello", duration_s=1.0)
    assert len(pipe._event_log) == 1
    entry = pipe._event_log[0]
    assert entry["event"] == "generation"
    assert entry["prompt_id"] == 0
    assert entry["iteration_idx"] == 2
    assert entry["model_key"] == "llama"
    assert "ts" in entry

def test_get_log_entry_filters_prompt_id():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._reset_event_log()
    pipe.log_event("generation", prompt_id=0, model_key="llama", phase="collect",
                   iteration_idx=0, input_text="a", output_text="b", duration_s=1.0)
    pipe.log_event("generation", prompt_id=1, model_key="llama", phase="collect",
                   iteration_idx=0, input_text="c", output_text="d", duration_s=1.0)
    results = pipe.get_log_entry(prompt_id=1)
    assert len(results) == 1
    assert results[0]["prompt_id"] == 1

def test_get_log_entry_multi_filter():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._reset_event_log()
    # Add 3 entries: same prompt/iteration but different k_sample_idx
    for k in range(3):
        pipe.log_event("judge_draw", prompt_id=4, model_key="qwen", phase="judge",
                       iteration_idx=None, case="case1_blind", judge_key="qwen",
                       k_sample_idx=k, input_text="x", output_text="y",
                       duration_s=0.5, parse_ok=True, parsed_scores=[])
    # Different prompt
    pipe.log_event("judge_draw", prompt_id=5, model_key="qwen", phase="judge",
                   iteration_idx=None, case="case1_blind", judge_key="qwen",
                   k_sample_idx=0, input_text="x", output_text="y",
                   duration_s=0.5, parse_ok=True, parsed_scores=[])

    results = pipe.get_log_entry(prompt_id=4, k_sample_idx=2)
    assert len(results) == 1
    assert results[0]["k_sample_idx"] == 2

def test_get_log_entry_no_filters_returns_all():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._reset_event_log()
    for i in range(5):
        pipe.log_event("generation", prompt_id=i, model_key="llama", phase="collect",
                       iteration_idx=0, input_text="x", output_text="y", duration_s=1.0)
    assert len(pipe.get_log_entry()) == 5

def test_build_judge_messages_instruct_model():
    pipe = _load_pipeline_config()
    msgs = pipe.build_judge_messages("qwen", "Rate this answer.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Rate this answer."
    assert "Example format" not in msgs[0]["content"]

def test_build_judge_messages_sft_model():
    pipe = _load_pipeline_config()
    msgs = pipe.build_judge_messages("olmo_sft", "Rate this answer.")
    assert len(msgs) == 2
    assert "Example format" in msgs[0]["content"]
    assert '"evaluations"' in msgs[0]["content"]

def test_build_judge_messages_tulu_sft():
    pipe = _load_pipeline_config()
    msgs = pipe.build_judge_messages("tulu_sft", "Rate this answer.")
    assert "Example format" in msgs[0]["content"]

def test_build_judge_messages_tulu_dpo_is_not_sft():
    pipe = _load_pipeline_config()
    msgs = pipe.build_judge_messages("tulu_dpo", "Rate this answer.")
    assert "Example format" not in msgs[0]["content"]

def test_temperature_config():
    pipe = _load_pipeline_config()
    assert hasattr(pipe, "TEMPERATURES"), "TEMPERATURES constant missing"
    assert pipe.TEMPERATURES[0] == 0.0
    assert pipe.TEMPERATURES[-1] == 1.0
    assert len(pipe.TEMPERATURES) == 11
    assert hasattr(pipe, "N_TEMPERATURE_RUNS"), "N_TEMPERATURE_RUNS constant missing"
    assert pipe.N_TEMPERATURE_RUNS == 10
    assert pipe.N_RUNS_PER_PROMPT == 10

def test_log_event_includes_temperature_fields():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._reset_event_log()
    pipe.log_event(
        "generation",
        prompt_id=0, model_key="llama", phase="collect",
        iteration_idx=0, input_text="hi", output_text="hello", duration_s=1.0,
        temperature=0.5, temp_run_idx=3,
    )
    entry = pipe._event_log[0]
    assert entry["temperature"] == 0.5
    assert entry["temp_run_idx"] == 3


def test_get_log_entry_filters_temperature():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._reset_event_log()
    pipe.log_event("generation", prompt_id=0, model_key="llama", phase="collect",
                   iteration_idx=0, input_text="a", output_text="b", duration_s=1.0,
                   temperature=0.3, temp_run_idx=0)
    pipe.log_event("generation", prompt_id=1, model_key="llama", phase="collect",
                   iteration_idx=0, input_text="c", output_text="d", duration_s=1.0,
                   temperature=0.7, temp_run_idx=0)
    results = pipe.get_log_entry(temperature=0.3)
    assert len(results) == 1
    assert results[0]["temperature"] == 0.3


def test_get_log_entry_filters_temp_run_idx():
    import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe
    pipe._reset_event_log()
    for run in range(3):
        pipe.log_event("generation", prompt_id=0, model_key="llama", phase="collect",
                       iteration_idx=0, input_text="a", output_text="b", duration_s=1.0,
                       temperature=0.5, temp_run_idx=run)
    results = pipe.get_log_entry(temp_run_idx=1)
    assert len(results) == 1
    assert results[0]["temp_run_idx"] == 1
