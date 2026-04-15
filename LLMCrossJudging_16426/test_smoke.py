# test_smoke.py — smoke test for LLMCrossJudging_16426.py
# Verifies _process_faking_outputs column keys and save_faking_results CSV structure
# without requiring vLLM or a GPU.

import sys, os, csv, tempfile, json
sys.path.insert(0, os.path.dirname(__file__))

# Mock vLLM before importing the module
import unittest.mock as mock
sys.modules['torch'] = mock.MagicMock()
sys.modules['vllm'] = mock.MagicMock()

import LLMCrossJudging_16426 as p16

# --- Test _process_faking_outputs ---
def make_fake_output(text):
    out = mock.MagicMock()
    out.text = text
    return out

def make_fake_req_output(texts):
    req = mock.MagicMock()
    req.outputs = [make_fake_output(t) for t in texts]
    return req

good_json = json.dumps({
    "evaluations": [{
        "answer_id": "TestModel_run0",
        "relevance": 7,
        "reasoning_quality": 6,
        "justification": "test"
    }]
})

outputs = [make_fake_req_output([good_json, good_json, good_json])]
metas = [{
    "prompt_id": 0,
    "prompt": "test prompt",
    "temperature": 0.3,
    "fake_attribution": "llama",
    "actual_generator": "qwen",
    "response_label": "Llama-3.1-8B-Instruct_run0",
}]

results = p16._process_faking_outputs("qwen", outputs, metas, "identity_faking")
assert len(results) == 1, f"Expected 1 result, got {len(results)}"
r = results[0]
assert r["judge"] == "qwen", f"judge mismatch: {r['judge']}"
assert r["experiment"] == "identity_faking", f"experiment mismatch: {r['experiment']}"
assert r["fake_attribution"] == "llama", f"fake_attribution mismatch: {r['fake_attribution']}"
assert r["actual_generator"] == "qwen", f"actual_generator mismatch: {r['actual_generator']}"
assert "relevance_avg" in r, f"missing relevance_avg: {list(r.keys())}"
assert "total_avg" in r, f"missing total_avg: {list(r.keys())}"
assert "parse_error" not in r, f"unexpected parse_error in result"
print("_process_faking_outputs: OK")

# --- Test save_faking_results ---
identity_results = [dict(r, fake_attribution="llama", actual_generator="qwen")]
reverse_results = [{
    "judge": "qwen", "experiment": "reverse_identity_faking",
    "prompt_id": 0, "temperature": 0.3,
    "fake_attribution": "qwen", "actual_generator": "llama",
    "response_label": "Qwen2.5-7B-Instruct_run0",
    "relevance_avg": 6.0, "rq_avg": 5.5, "total_avg": 5.75,
    "relevance_std": 0.0, "rq_std": 0.0, "k_valid": 3
}]
score_results = [{
    "judge": "qwen", "experiment": "score_faking",
    "prompt_id": 0, "temperature": 0.3,
    "evaluated_model": "llama", "score_condition": "low",
    "alleged_score": 2.0,
    "relevance_avg": 6.0, "rq_avg": 5.5, "total_avg": 5.75,
    "relevance_std": 0.0, "rq_std": 0.0, "k_valid": 3
}]

with tempfile.TemporaryDirectory() as tmpdir:
    p16.save_faking_results(identity_results, reverse_results, score_results, tmpdir, 0.3)

    # Check identity faking CSV
    if_path = os.path.join(tmpdir, "identity_faking_T0.3.csv")
    assert os.path.exists(if_path), f"Missing {if_path}"
    rows = list(csv.DictReader(open(if_path)))
    assert len(rows) == 1
    assert "judge" in rows[0], f"missing 'judge' col: {list(rows[0].keys())}"
    assert "fake_attribution" in rows[0], f"missing 'fake_attribution': {list(rows[0].keys())}"
    assert "total_avg" in rows[0], f"missing 'total_avg': {list(rows[0].keys())}"
    print(f"identity_faking CSV cols: {list(rows[0].keys())}")

    # Check reverse identity faking CSV
    rif_path = os.path.join(tmpdir, "reverse_identity_faking_T0.3.csv")
    assert os.path.exists(rif_path), f"Missing {rif_path}"
    rows = list(csv.DictReader(open(rif_path)))
    assert "actual_generator" in rows[0], f"missing 'actual_generator': {list(rows[0].keys())}"
    print(f"reverse_identity_faking CSV cols: {list(rows[0].keys())}")

    # Check score faking CSV
    sf_path = os.path.join(tmpdir, "score_faking_T0.3.csv")
    assert os.path.exists(sf_path), f"Missing {sf_path}"
    rows = list(csv.DictReader(open(sf_path)))
    assert "score_condition" in rows[0], f"missing 'score_condition': {list(rows[0].keys())}"
    assert "alleged_score" in rows[0], f"missing 'alleged_score': {list(rows[0].keys())}"
    print(f"score_faking CSV cols: {list(rows[0].keys())}")

print("save_faking_results: OK")
print("ALL SMOKE TESTS PASSED")
