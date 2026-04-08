import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import random
import pytest
from unittest.mock import MagicMock, patch
import llm_cross_judging.LLM_Cross_Judging_Pipeline_KSampling as pipe

@pytest.fixture(autouse=True)
def reset_state():
    pipe._loaded_llm = None
    pipe._loaded_key = None
    pipe._failed_models = set()
    pipe._reset_event_log()
    yield
    pipe._loaded_llm = None
    pipe._loaded_key = None
    pipe._failed_models = set()
    pipe._reset_event_log()


# ============================================================
# Task 2: Config tests
# ============================================================

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

def test_unload_clears_globals():
    pipe._loaded_llm = MagicMock()
    pipe._loaded_key = "llama"
    with patch("gc.collect"), patch("torch.cuda.is_available", return_value=False):
        pipe.unload_current_model()
    assert pipe._loaded_llm is None
    assert pipe._loaded_key is None

def test_load_model_skips_if_already_loaded():
    pipe._loaded_llm = MagicMock()
    pipe._loaded_key = "llama"
    with patch("vllm.LLM") as mock_llm:
        pipe.load_model("llama")
        mock_llm.assert_not_called()

def test_load_model_marks_failed_on_error():
    pipe._loaded_llm = None
    pipe._loaded_key = None
    with patch("vllm.LLM", side_effect=RuntimeError("oom")):
        pipe.load_model("qwen")
    assert "qwen" in pipe._failed_models

def test_load_model_skips_known_failed():
    pipe._failed_models = {"qwen"}
    with patch("vllm.LLM") as mock_llm:
        pipe.load_model("qwen")
        mock_llm.assert_not_called()


# ============================================================
# Task 3: Event logging tests
# ============================================================

def test_log_event_appends():
    pipe._reset_event_log()
    pipe.log_event("generation", prompt_id=0, model_key="llama", phase="collect",
                   run_idx=2, temperature=0.5, input_text="hi", output_text="ok", duration_s=1.0)
    assert len(pipe._event_log) == 1
    e = pipe._event_log[0]
    assert e["event"] == "generation"
    assert e["prompt_id"] == 0
    assert e["run_idx"] == 2
    assert e["temperature"] == 0.5
    assert "ts" in e

def test_log_event_judge_draw():
    pipe._reset_event_log()
    pipe.log_event("judge_draw", prompt_id=0, model_key="llama", phase="judge",
                   case="case1_blind", judge_key="qwen", evaluated_model="llama",
                   k_sample_idx=1, temperature=0.5, input_text="x", output_text="y",
                   duration_s=0.3, parse_ok=True, parsed_scores=[])
    e = pipe._event_log[0]
    assert e["case"] == "case1_blind"
    assert e["evaluated_model"] == "llama"
    assert e["k_sample_idx"] == 1
    assert e["parse_ok"] is True

def test_get_log_entry_filters_prompt_id():
    pipe._reset_event_log()
    pipe.log_event("generation", prompt_id=0, model_key="llama", phase="collect",
                   run_idx=0, temperature=0.5, input_text="a", output_text="b", duration_s=1.0)
    pipe.log_event("generation", prompt_id=1, model_key="llama", phase="collect",
                   run_idx=0, temperature=0.5, input_text="c", output_text="d", duration_s=1.0)
    results = pipe.get_log_entry(prompt_id=1)
    assert len(results) == 1
    assert results[0]["prompt_id"] == 1

def test_get_log_entry_multi_filter():
    pipe._reset_event_log()
    for k in range(3):
        pipe.log_event("judge_draw", prompt_id=4, model_key="qwen", phase="judge",
                       case="case1_blind", judge_key="qwen", evaluated_model="llama",
                       k_sample_idx=k, temperature=0.5, input_text="x", output_text="y",
                       duration_s=0.5, parse_ok=True, parsed_scores=[])
    results = pipe.get_log_entry(prompt_id=4, k_sample_idx=2)
    assert len(results) == 1
    assert results[0]["k_sample_idx"] == 2

def test_get_log_entry_no_filters():
    pipe._reset_event_log()
    for i in range(5):
        pipe.log_event("generation", prompt_id=i, model_key="llama", phase="collect",
                       run_idx=0, temperature=0.5, input_text="x", output_text="y", duration_s=1.0)
    assert len(pipe.get_log_entry()) == 5

def test_get_log_entry_filters_evaluated_model():
    pipe._reset_event_log()
    pipe.log_event("judge_draw", prompt_id=0, model_key="qwen", phase="judge",
                   case="case1_blind", judge_key="qwen", evaluated_model="llama",
                   k_sample_idx=0, temperature=0.5, input_text="x", output_text="y",
                   duration_s=0.3, parse_ok=True, parsed_scores=[])
    pipe.log_event("judge_draw", prompt_id=0, model_key="qwen", phase="judge",
                   case="case1_blind", judge_key="qwen", evaluated_model="gemma",
                   k_sample_idx=0, temperature=0.5, input_text="x", output_text="y",
                   duration_s=0.3, parse_ok=True, parsed_scores=[])
    results = pipe.get_log_entry(evaluated_model="llama")
    assert len(results) == 1
    assert results[0]["evaluated_model"] == "llama"


# ============================================================
# Task 4: Parsing and judge helper tests
# ============================================================

def test_parse_judge_response_valid():
    raw = '{"evaluations": [{"answer_id": "Response_1", "relevance": 7, "reasoning_quality": 8, "justification": "good"}]}'
    result = pipe.parse_judge_response(raw)
    assert result is not None
    assert result[0]["answer_id"] == "Response_1"
    assert result[0]["relevance"] == 7

def test_parse_judge_response_markdown_wrapped():
    raw = '```json\n{"evaluations": [{"answer_id": "R1", "relevance": 5, "reasoning_quality": 6, "justification": "ok"}]}\n```'
    result = pipe.parse_judge_response(raw)
    assert result is not None
    assert result[0]["relevance"] == 5

def test_parse_judge_response_invalid():
    result = pipe.parse_judge_response("not json at all")
    assert result is None

def test_build_judge_messages_instruct():
    msgs = pipe.build_judge_messages("qwen", "Rate this.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "Rate this."
    assert "Example format" not in msgs[0]["content"]

def test_build_judge_messages_sft():
    msgs = pipe.build_judge_messages("olmo_sft", "Rate this.")
    assert "Example format" in msgs[0]["content"]
    assert '"evaluations"' in msgs[0]["content"]

def test_build_judge_messages_tulu_sft():
    msgs = pipe.build_judge_messages("tulu_sft", "Rate this.")
    assert "Example format" in msgs[0]["content"]

def test_build_judge_messages_tulu_dpo_not_sft():
    msgs = pipe.build_judge_messages("tulu_dpo", "Rate this.")
    assert "Example format" not in msgs[0]["content"]

def test_resolve_model_key_exact():
    result = pipe._resolve_model_key("Qwen2.5-7B-Instruct", list(pipe.MODELS.keys()))
    assert result == "qwen"

def test_resolve_model_key_fuzzy():
    result = pipe._resolve_model_key("OLMo-7B-SFT", list(pipe.MODELS.keys()))
    assert result == "olmo_sft"

def test_resolve_model_key_fuzzy_instruct_vs_sft():
    result = pipe._resolve_model_key("OLMo-7B-Instruct", list(pipe.MODELS.keys()))
    assert result == "olmo_instruct"

def test_resolve_model_key_no_match():
    result = pipe._resolve_model_key("SomeRandomModel", list(pipe.MODELS.keys()))
    assert result is None


# ============================================================
# Task 5: collect_responses tests
# ============================================================

def test_collect_responses_structure():
    """collect_responses returns list with one entry per prompt, each with N runs per model."""
    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text=f"response_{i}") for i in range(pipe.N_RUNS_PER_PROMPT)]
    fake_llm.generate.return_value = [fake_output] * pipe.N_RUNS_PER_PROMPT

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "unload_current_model"), \
         patch.object(pipe, "_loaded_llm", fake_llm, create=True), \
         patch.object(pipe, "_loaded_key", "qwen", create=True), \
         patch.object(pipe, "_failed_models", set(pipe.MODELS.keys()) - {"qwen"}):
        result = pipe.collect_responses([pipe.PROMPTS[0]], temperature=0.7)

    assert len(result) == 1
    entry = result[0]
    assert entry["prompt_id"] == 0
    assert entry["temperature"] == 0.7
    assert "qwen" in entry["responses"]
    assert len(entry["responses"]["qwen"]) == pipe.N_RUNS_PER_PROMPT

def test_collect_responses_greedy_at_zero():
    """temperature=0.0 must pass do_sample=False via SamplingParams temperature=0."""
    captured_params = {}
    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="r") for _ in range(pipe.N_RUNS_PER_PROMPT)]
    fake_llm.generate.return_value = [fake_output] * pipe.N_RUNS_PER_PROMPT

    original_sp = __import__("vllm").SamplingParams

    def capture_sp(*args, **kwargs):
        captured_params.update(kwargs)
        return original_sp(*args, **kwargs)

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "unload_current_model"), \
         patch.object(pipe, "_loaded_llm", fake_llm, create=True), \
         patch.object(pipe, "_loaded_key", "qwen", create=True), \
         patch.object(pipe, "_failed_models", set(pipe.MODELS.keys()) - {"qwen"}), \
         patch("vllm.SamplingParams", side_effect=capture_sp):
        pipe.collect_responses([pipe.PROMPTS[0]], temperature=0.0)

    assert captured_params.get("temperature") == 0.0

def test_collect_responses_logs_events():
    """Each generation run must produce one event log entry per model per run."""
    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text=f"r{i}") for i in range(pipe.N_RUNS_PER_PROMPT)]
    fake_llm.generate.return_value = [fake_output] * pipe.N_RUNS_PER_PROMPT

    pipe._reset_event_log()
    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "unload_current_model"), \
         patch.object(pipe, "_loaded_llm", fake_llm, create=True), \
         patch.object(pipe, "_loaded_key", "qwen", create=True), \
         patch.object(pipe, "_failed_models", set(pipe.MODELS.keys()) - {"qwen"}):
        pipe.collect_responses([pipe.PROMPTS[0]], temperature=0.5)

    entries = pipe.get_log_entry(phase="collect", model_key="qwen")
    assert len(entries) == pipe.N_RUNS_PER_PROMPT
    assert all(e["temperature"] == 0.5 for e in entries)
    assert all(e["run_idx"] == i for i, e in enumerate(entries))


# ============================================================
# Task 6: K-sampling judge helper tests
# ============================================================

def test_sample_judge_scores_averages_k_draws():
    """sample_judge_scores returns averaged scores across K valid parses."""
    valid_response = json.dumps({"evaluations": [
        {"answer_id": "Response_1", "relevance": 8, "reasoning_quality": 7, "justification": "good"},
        {"answer_id": "Response_2", "relevance": 6, "reasoning_quality": 5, "justification": "ok"},
    ]})

    fake_llm = MagicMock()
    fake_out = MagicMock()
    fake_out.outputs = [MagicMock(text=valid_response)]
    fake_llm.generate.return_value = [fake_out] * pipe.K_JUDGE_SAMPLES

    with patch.object(pipe, "_loaded_llm", fake_llm, create=True), \
         patch.object(pipe, "_loaded_key", "qwen", create=True):
        averaged, raws = pipe.sample_judge_scores(
            judge_key="qwen",
            formatted_prompt="judge prompt",
            log_ctx={"phase": "judge", "prompt_id": 0, "case": "case1_blind",
                     "judge_key": "qwen", "evaluated_model": "llama", "temperature": 0.5}
        )

    assert averaged is not None
    assert len(averaged) == 2
    r1 = next(a for a in averaged if a["answer_id"] == "Response_1")
    assert r1["relevance"] == 8.0
    assert r1["k_valid"] == pipe.K_JUDGE_SAMPLES
    assert len(r1["relevance_samples"]) == pipe.K_JUDGE_SAMPLES

def test_sample_judge_scores_returns_none_on_all_failures():
    fake_llm = MagicMock()
    fake_out = MagicMock()
    fake_out.outputs = [MagicMock(text="not json")]
    fake_llm.generate.return_value = [fake_out] * pipe.K_JUDGE_SAMPLES

    with patch.object(pipe, "_loaded_llm", fake_llm, create=True), \
         patch.object(pipe, "_loaded_key", "qwen", create=True):
        averaged, raws = pipe.sample_judge_scores(
            judge_key="qwen",
            formatted_prompt="bad prompt",
            log_ctx={"phase": "judge", "prompt_id": 0, "case": "case1_blind",
                     "judge_key": "qwen", "evaluated_model": "llama", "temperature": 0.5}
        )

    assert averaged is None
    assert len(raws) == pipe.K_JUDGE_SAMPLES


# ============================================================
# Task 7: _judge_one_model tests
# ============================================================

def _make_fake_sample_judge(scores_per_response=None):
    """Returns a fake sample_judge_scores that returns deterministic averaged evals."""
    if scores_per_response is None:
        scores_per_response = {"relevance": 7.0, "reasoning_quality": 6.0}

    def fake_sample(judge_key, formatted_prompt, log_ctx, k=None):
        n_responses = log_ctx.get("_n_responses", 10)
        averaged = [
            {
                "answer_id": f"Response_{i+1}",
                "relevance":         scores_per_response["relevance"],
                "reasoning_quality": scores_per_response["reasoning_quality"],
                "justification": "test",
                "k_valid": pipe.K_JUDGE_SAMPLES,
                "relevance_samples": [7.0] * pipe.K_JUDGE_SAMPLES,
                "rq_samples":        [6.0] * pipe.K_JUDGE_SAMPLES,
                "relevance_std": 0.0,
                "rq_std":        0.0,
            }
            for i in range(n_responses)
        ]
        return averaged, ["raw"] * pipe.K_JUDGE_SAMPLES
    return fake_sample


def test_judge_one_model_returns_result_entry():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {
            "llama": [f"response_{i}" for i in range(10)],
        }
    }]

    with patch.object(pipe, "sample_judge_scores", side_effect=_make_fake_sample_judge()), \
         patch.object(pipe, "_apply_chat_template", return_value="formatted"), \
         patch.object(pipe, "_loaded_key", "qwen", create=True):
        results = pipe._judge_one_model(
            judge_key="qwen",
            evaluated_model="llama",
            collected=collected,
            case="case1_blind",
            build_prompt_fn=lambda entry, runs: "judge prompt",
            rng=random.Random(42),
        )

    assert len(results) == 1
    r = results[0]
    assert r["evaluated_model"] == "llama"
    assert r["case"] == "case1_blind"
    assert len(r["individual_response_scores"]) == 10
    assert "model_aggregated_scores" in r
    assert r["model_aggregated_scores"]["n_runs"] == 10

def test_judge_one_model_parse_error_flagged():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {"llama": [f"r{i}" for i in range(10)]}
    }]

    with patch.object(pipe, "sample_judge_scores", return_value=(None, ["bad"] * 3)), \
         patch.object(pipe, "_apply_chat_template", return_value="formatted"), \
         patch.object(pipe, "_loaded_key", "qwen", create=True):
        results = pipe._judge_one_model(
            judge_key="qwen",
            evaluated_model="llama",
            collected=collected,
            case="case1_blind",
            build_prompt_fn=lambda entry, runs: "judge prompt",
            rng=random.Random(42),
        )

    assert results[0].get("parse_error") is True


# ============================================================
# Task 8: case1_blind tests
# ============================================================

def test_judge_case1_blind_structure():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        return [{"prompt_id": 0, "evaluated_model": evaluated_model,
                 "case": case, "individual_response_scores": [],
                 "model_aggregated_scores": {"relevance_avg": 7.0, "rq_avg": 6.5, "n_runs": 10}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        result = pipe.judge_case1_blind("qwen", list(pipe.MODELS.keys()), collected)

    assert result["case"] == "case1_blind"
    assert result["judge"] == "qwen"
    assert len(result["results"]) == len(pipe.MODELS)
    assert all(r["case"] == "case1_blind" for r in result["results"])

def test_judge_case1_blind_prompt_is_anonymous():
    """The judge prompt must not contain model names."""
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]
    captured_prompts = []

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        for entry in collected:
            runs = [{"label": f"Response_{i+1}", "run_idx": i, "text": f"r{i}"} for i in range(10)]
            captured_prompts.append(build_prompt_fn(entry, runs))
        return [{"prompt_id": 0, "evaluated_model": evaluated_model, "case": case,
                 "individual_response_scores": [], "model_aggregated_scores": {}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        pipe.judge_case1_blind("qwen", list(pipe.MODELS.keys()), collected)

    for prompt_text in captured_prompts:
        for cfg in pipe.MODEL_CONFIG.values():
            assert cfg["label"] not in prompt_text, \
                f"Model name '{cfg['label']}' found in blind judge prompt"


# ============================================================
# Task 9: case2a and case2b tests
# ============================================================

def test_judge_case2a_identity_prompt_contains_model_names():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]
    captured_prompts = []

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        for entry in collected:
            runs = [{"label": f"Response_{i+1}", "run_idx": i, "text": f"r{i}"} for i in range(10)]
            captured_prompts.append(build_prompt_fn(entry, runs))
        return [{"prompt_id": 0, "evaluated_model": evaluated_model, "case": case,
                 "individual_response_scores": [], "model_aggregated_scores": {}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        pipe.judge_case2a_identity("qwen", list(pipe.MODELS.keys()), collected)

    found_any = any(
        any(cfg["label"] in p for cfg in pipe.MODEL_CONFIG.values())
        for p in captured_prompts
    )
    assert found_any, "Case 2a prompt must contain at least one model name"

def test_judge_case2a_no_competition_mention():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]
    captured_prompts = []

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        for entry in collected:
            runs = [{"label": f"Response_{i+1}", "run_idx": i, "text": f"r{i}"} for i in range(10)]
            captured_prompts.append(build_prompt_fn(entry, runs))
        return [{"prompt_id": 0, "evaluated_model": evaluated_model, "case": case,
                 "individual_response_scores": [], "model_aggregated_scores": {}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        pipe.judge_case2a_identity("qwen", list(pipe.MODELS.keys()), collected)

    for p in captured_prompts:
        assert "competition" not in p.lower(), "Case 2a must not mention competition"

def test_judge_case2b_competition_prompt_mentions_competition():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]
    captured_prompts = []

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        for entry in collected:
            runs = [{"label": f"Response_{i+1}", "run_idx": i, "text": f"r{i}"} for i in range(10)]
            captured_prompts.append(build_prompt_fn(entry, runs))
        return [{"prompt_id": 0, "evaluated_model": evaluated_model, "case": case,
                 "individual_response_scores": [], "model_aggregated_scores": {}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        pipe.judge_case2b_competition("qwen", list(pipe.MODELS.keys()), collected)

    found_competition = any("competition" in p.lower() for p in captured_prompts)
    assert found_competition, "Case 2b prompt must mention competition"

def test_judge_case2b_structure():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        return [{"prompt_id": 0, "evaluated_model": evaluated_model, "case": case,
                 "individual_response_scores": [], "model_aggregated_scores": {}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        result = pipe.judge_case2b_competition("qwen", list(pipe.MODELS.keys()), collected)

    assert result["case"] == "case2b_competition"
    assert result["judge"] == "qwen"
    assert len(result["results"]) == len(pipe.MODELS)


# ============================================================
# Task 10: case3_revenge tests
# ============================================================

def test_extract_scores_from_case1():
    """extract_scores_from_case1 returns {evaluator_key: avg_total} for a given judge model."""
    case1_results_by_judge = {
        "llama": {
            "case": "case1_blind",
            "judge": "llama",
            "results": [
                {
                    "prompt_id": 0,
                    "evaluated_model": "qwen",
                    "case": "case1_blind",
                    "individual_response_scores": [],
                    "model_aggregated_scores": {"relevance_avg": 7.0, "rq_avg": 6.0, "n_runs": 10},
                }
            ]
        },
        "gemma": {
            "case": "case1_blind",
            "judge": "gemma",
            "results": [
                {
                    "prompt_id": 0,
                    "evaluated_model": "qwen",
                    "case": "case1_blind",
                    "individual_response_scores": [],
                    "model_aggregated_scores": {"relevance_avg": 8.0, "rq_avg": 9.0, "n_runs": 10},
                }
            ]
        },
    }
    scores = pipe.extract_scores_from_case1(
        judge_key="qwen",
        case1_results_by_judge=case1_results_by_judge,
    )
    assert "llama" in scores
    assert "gemma" in scores
    assert scores["llama"] == round((7.0 + 6.0) / 2, 3)
    assert scores["gemma"] == round((8.0 + 9.0) / 2, 3)

def test_judge_case3_revenge_structure():
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]
    individual_scores = {mk: 7.0 for mk in pipe.MODELS if mk != "qwen"}

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        return [{"prompt_id": 0, "evaluated_model": evaluated_model, "case": case,
                 "individual_response_scores": [], "model_aggregated_scores": {}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        result = pipe.judge_case3_revenge(
            judge_key="qwen",
            eval_keys=list(pipe.MODELS.keys()),
            collected=collected,
            individual_scores=individual_scores,
        )

    assert result["case"] == "case3_revenge"
    assert result["judge"] == "qwen"
    assert result["individual_scores_received"] == individual_scores
    assert len(result["results"]) == len(pipe.MODELS)

def test_judge_case3_revenge_prompt_contains_scores():
    """The revenge prompt must contain the scores that other models gave the judge."""
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}
    }]
    individual_scores = {"llama": 6.5, "gemma": 8.0}
    captured_prompts = []

    def fake_judge_one(judge_key, evaluated_model, collected, case, build_prompt_fn, rng):
        for entry in collected:
            runs = [{"label": f"Response_{i+1}", "run_idx": i, "text": f"r{i}"} for i in range(10)]
            captured_prompts.append(build_prompt_fn(entry, runs))
        return [{"prompt_id": 0, "evaluated_model": evaluated_model, "case": case,
                 "individual_response_scores": [], "model_aggregated_scores": {}}]

    with patch.object(pipe, "load_model"), \
         patch.object(pipe, "_judge_one_model", side_effect=fake_judge_one):
        pipe.judge_case3_revenge(
            judge_key="qwen",
            eval_keys=list(pipe.MODELS.keys()),
            collected=collected,
            individual_scores=individual_scores,
        )

    for p in captured_prompts:
        assert "6.5" in p or "8.0" in p, "Revenge prompt must contain the scores received"


# ============================================================
# Task 11: Aggregation tests
# ============================================================

def _make_series(judge="qwen", temperature=0.5):
    result_entry = lambda mk: {
        "evaluated_model": mk,
        "case": "case1_blind",
        "model_aggregated_scores": {"relevance_avg": 7.0, "rq_avg": 6.0, "n_runs": 10},
    }
    return {
        "judge": judge,
        "temperature": temperature,
        "cases": {
            "case1_blind": {"case": "case1_blind", "judge": judge,
                            "results": [result_entry(mk) for mk in pipe.MODELS]},
        },
        "summary": {},
    }

def test_aggregate_series_computes_averages():
    series = _make_series()
    summary = pipe.aggregate_series(series)
    assert "case1_blind" in summary
    for mk in pipe.MODELS:
        assert mk in summary["case1_blind"]
        assert summary["case1_blind"][mk]["avg_relevance"] == 7.0
        assert summary["case1_blind"][mk]["avg_rq"] == 6.0
        assert summary["case1_blind"][mk]["avg_total"] == 6.5

def test_save_results_csv_structure(tmp_path):
    import csv
    series = _make_series(temperature=0.5)
    series["summary"] = pipe.aggregate_series(series)
    pipe.save_results([series], str(tmp_path), temperature=0.5)
    csv_files = list(tmp_path.glob("summary_*.csv"))
    assert len(csv_files) == 1
    with open(csv_files[0]) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(pipe.MODELS)
    assert rows[0]["temperature"] == "0.5"
    assert "avg_relevance" in rows[0]
    assert "avg_rq" in rows[0]

def test_save_results_filename_includes_temperature(tmp_path):
    series = _make_series(temperature=0.3)
    series["summary"] = pipe.aggregate_series(series)
    pipe.save_results([series], str(tmp_path), temperature=0.3)
    json_files = list(tmp_path.glob("full_results_*.json"))
    assert len(json_files) == 1
    assert "T0.3" in json_files[0].name


# ============================================================
# Task 12: run_pipeline tests
# ============================================================

def test_run_pipeline_calls_collect_once_per_temperature():
    collect_calls = []

    def fake_collect(prompts, temperature=0.7):
        collect_calls.append(temperature)
        return [{"prompt_id": 0, "prompt": "q", "temperature": temperature,
                 "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}}]

    def fake_case1(judge_key, eval_keys, collected, seed=42):
        return {"case": "case1_blind", "judge": judge_key, "results": []}

    def fake_case2a(judge_key, eval_keys, collected, seed=42):
        return {"case": "case2a_identity", "judge": judge_key, "results": []}

    def fake_case2b(judge_key, eval_keys, collected, seed=42):
        return {"case": "case2b_competition", "judge": judge_key, "results": []}

    def fake_case3(judge_key, eval_keys, collected, individual_scores, seed=42):
        return {"case": "case3_revenge", "judge": judge_key,
                "individual_scores_received": individual_scores, "results": []}

    with patch.object(pipe, "collect_responses", side_effect=fake_collect), \
         patch.object(pipe, "judge_case1_blind", side_effect=fake_case1), \
         patch.object(pipe, "judge_case2a_identity", side_effect=fake_case2a), \
         patch.object(pipe, "judge_case2b_competition", side_effect=fake_case2b), \
         patch.object(pipe, "judge_case3_revenge", side_effect=fake_case3), \
         patch.object(pipe, "extract_scores_from_case1", return_value={"llama": 7.0}), \
         patch.object(pipe, "save_results", return_value=("/a.json", "/a.csv")), \
         patch.object(pipe, "save_event_log"), \
         patch.object(pipe, "print_summary_table"), \
         patch.object(pipe, "cleanup_gpu"), \
         patch("os.makedirs"), patch("builtins.open", MagicMock()):
        pipe.run_pipeline(prompts=[pipe.PROMPTS[0]])

    assert len(collect_calls) == len(pipe.TEMPERATURES)
    assert collect_calls[0] == 0.0
    assert collect_calls[-1] == 1.0

def test_run_pipeline_all_four_cases_called():
    case1_calls = []
    case2a_calls = []
    case2b_calls = []
    case3_calls = []

    def fake_collect(prompts, temperature=0.7):
        return [{"prompt_id": 0, "prompt": "q", "temperature": temperature,
                 "responses": {mk: [f"r{i}" for i in range(10)] for mk in pipe.MODELS}}]

    def fake_case1(judge_key, eval_keys, collected, seed=42):
        case1_calls.append(judge_key)
        return {"case": "case1_blind", "judge": judge_key, "results": []}

    def fake_case2a(judge_key, eval_keys, collected, seed=42):
        case2a_calls.append(judge_key)
        return {"case": "case2a_identity", "judge": judge_key, "results": []}

    def fake_case2b(judge_key, eval_keys, collected, seed=42):
        case2b_calls.append(judge_key)
        return {"case": "case2b_competition", "judge": judge_key, "results": []}

    def fake_case3(judge_key, eval_keys, collected, individual_scores, seed=42):
        case3_calls.append(judge_key)
        return {"case": "case3_revenge", "judge": judge_key,
                "individual_scores_received": individual_scores, "results": []}

    with patch.object(pipe, "collect_responses", side_effect=fake_collect), \
         patch.object(pipe, "judge_case1_blind", side_effect=fake_case1), \
         patch.object(pipe, "judge_case2a_identity", side_effect=fake_case2a), \
         patch.object(pipe, "judge_case2b_competition", side_effect=fake_case2b), \
         patch.object(pipe, "judge_case3_revenge", side_effect=fake_case3), \
         patch.object(pipe, "extract_scores_from_case1", return_value={"llama": 7.0}), \
         patch.object(pipe, "save_results", return_value=("/a.json", "/a.csv")), \
         patch.object(pipe, "save_event_log"), \
         patch.object(pipe, "print_summary_table"), \
         patch.object(pipe, "cleanup_gpu"), \
         patch("os.makedirs"), patch("builtins.open", MagicMock()):
        pipe.run_pipeline(prompts=[pipe.PROMPTS[0]])

    n_models = len(pipe.MODELS)
    n_temps  = len(pipe.TEMPERATURES)
    assert len(case1_calls)  == n_models * n_temps
    assert len(case2a_calls) == n_models * n_temps
    assert len(case2b_calls) == n_models * n_temps
    assert len(case3_calls)  == n_models * n_temps


# ============================================================
# Task 13: label_map mutation fix test
# ============================================================

def test_judge_one_model_label_map_not_mutated_by_case2a():
    """Case 2a build_prompt_fn must not corrupt label_map in result entry."""
    collected = [{
        "prompt_id": 0,
        "prompt": "Test prompt",
        "temperature": 0.5,
        "responses": {"llama": [f"r{i}" for i in range(10)]}
    }]

    valid_response = json.dumps({"evaluations": [
        {"answer_id": f"Llama-3.1-8B-Instruct_run{i}",
         "relevance": 7, "reasoning_quality": 6, "justification": "ok"}
        for i in range(10)
    ]})
    fake_llm = MagicMock()
    fake_out = MagicMock()
    fake_out.outputs = [MagicMock(text=valid_response)]
    fake_llm.generate.return_value = [fake_out] * pipe.K_JUDGE_SAMPLES

    with patch.object(pipe, "_loaded_llm", fake_llm, create=True), \
         patch.object(pipe, "_loaded_key", "qwen", create=True), \
         patch.object(pipe, "load_model"), \
         patch.object(pipe, "_apply_chat_template", return_value="formatted"):
        result = pipe.judge_case2a_identity(
            "qwen", ["llama"], collected
        )

    for entry in result["results"]:
        for label, meta in entry["label_map"].items():
            assert "model" in meta, f"label_map entry missing 'model': {meta}"
            assert "run_idx" in meta, f"label_map entry missing 'run_idx': {meta}"
