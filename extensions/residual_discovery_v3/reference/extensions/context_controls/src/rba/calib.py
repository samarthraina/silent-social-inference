"""Rubric-calibration experiment: prompts, invariance checks and the 360-unit draft manifest.

Original arm  = the original renderer's prompt, byte-for-byte (vendored scaled_conditions).
Explicit arm  = the same prompt with ONE block inserted immediately after the unchanged
                scoring instruction / output request line:

    <instruction>\\n\\n[<context>\\n\\n]<score line>\\n\\n<EXPLICIT_RUBRIC>\\n\\nQUESTION:...

Everything before the insertion (instruction, context sentence and its position, the
output request) and everything after it (question, answer) is byte-identical across arms.
This is a rubric intervention that adds length and wording; it is not a length-matched
isolation of calibration.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

from .legacy import final_study_messages as fsm
from .legacy import scaled_conditions as sc

RUBRICS = ("original", "explicit_v1")
CONTEXTS = ("no_history", "self_low_2", "unrelated_low_2")
MODELS = {
    "llama": {"hf_id": "meta-llama/Llama-3.1-8B-Instruct",
              "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
              "disable_thinking": False},
    "qwen": {"hf_id": "Qwen/Qwen3-8B",
             "revision": "b968826d9c46dd6066d109eabc6255188de91218",
             "disable_thinking": True},
}

# Fixed rubric. Defines grading standards only: no mention of prior feedback,
# history, anchoring or the hypothesis; no instruction to ignore anything; no reference
# answers or demonstrations.
EXPLICIT_RUBRIC = (
    "Scoring rubric: score the content for accuracy and completeness with respect to the "
    "question.\n"
    "- Accuracy: whether the statements, reasoning, calculations or code in the content "
    "are correct.\n"
    "- Completeness: whether the content addresses every part of the question.\n"
    "Score levels:\n"
    "10: fully accurate and complete.\n"
    "8-9: accurate and nearly complete; only minor omissions or imprecisions.\n"
    "6-7: mostly accurate; some noticeable omissions or minor errors.\n"
    "4-5: partly accurate or partly complete; significant omissions or at least one "
    "substantive error.\n"
    "2-3: largely inaccurate or incomplete; addresses only a small part of the question.\n"
    "0-1: wrong, irrelevant or no meaningful response."
)
RUBRIC_INSERT = "\n\n" + EXPLICIT_RUBRIC

DECODING = {"do_sample": False, "max_new_tokens": 512, "seed": 42, "batch_size": 1,
            "pad_token_id": "tokenizer.eos_token_id", "use_cache": "transformers default",
            "envelope": "user message only (final_study_messages)"}
ATTN_IMPLEMENTATION = "sdpa"      # as in the original production path (load_exact_model)
PRECISION = "bfloat16"


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _score_line_end(prompt: str, cond) -> int:
    """Character offset just after the score line, computed from the frozen layout."""
    pos = len(sc.INSTRUCTION_LINE) + 2            # instruction + "\n\n"
    if cond.added_context is not None:
        pos += len(cond.added_context) + 2
    end = pos + len(sc.SCORE_LINE)
    if prompt[pos:end] != sc.SCORE_LINE:
        raise AssertionError("frozen layout changed: score line not at its expected offset")
    return end


def render(question: str, answer: str, context: str, rubric: str) -> Dict[str, object]:
    cond = sc.resolve(context)
    original = sc.render_prompt(question, answer, cond)
    if rubric == "original":
        prompt, rubric_span = original, None
    elif rubric == "explicit_v1":
        k = _score_line_end(original, cond)
        prompt = original[:k] + RUBRIC_INSERT + original[k:]
        rubric_span = (k, k + len(RUBRIC_INSERT))
    else:
        raise KeyError(rubric)
    ctx_span = None
    if cond.added_context is not None:
        s = len(sc.INSTRUCTION_LINE) + 2
        ctx_span = (s, s + len(cond.added_context))
        assert prompt[ctx_span[0]:ctx_span[1]] == cond.added_context
    return {"prompt": prompt, "rubric_span": rubric_span, "context_span": ctx_span,
            "added_context": cond.added_context}


def messages(model_key: str, prompt: str) -> List[Dict[str, str]]:
    return fsm.build_scorer_messages(model_key, prompt)


def assert_invariance(question: str, answer: str) -> None:
    """Prove the explicit arm differs from the original arm only by the inserted block, and
    that contexts differ only by the context sentence."""
    for ctx in CONTEXTS:
        o = render(question, answer, ctx, "original")
        e = render(question, answer, ctx, "explicit_v1")
        s, t = e["rubric_span"]
        if e["prompt"][:s] + e["prompt"][t:] != o["prompt"]:
            raise AssertionError(f"{ctx}: explicit arm differs outside the rubric block")
        if e["prompt"][s:t] != RUBRIC_INSERT:
            raise AssertionError("rubric block content mismatch")
        if o["context_span"] != e["context_span"]:
            raise AssertionError("context placement differs across rubric arms")
        if not o["prompt"].endswith(answer) or not e["prompt"].endswith(answer):
            raise AssertionError("answer bytes not terminal")
        if o["prompt"].count(question) < 1:
            raise AssertionError("question missing")
    base = render(question, answer, "no_history", "original")["prompt"]
    for ctx in ("self_low_2", "unrelated_low_2"):
        r = render(question, answer, ctx, "original")
        s, t = r["context_span"]
        if r["prompt"][:s] + r["prompt"][t + 2:] != base:
            raise AssertionError(f"{ctx}: differs from no_history beyond the context sentence")


def unit_id(model: str, base_task_id: str, rubric: str, context: str) -> str:
    ds, tid = base_task_id.split("/", 1)
    return f"rba-calib-v1__{model}__{ds}__{tid}__{rubric}__{context}"


def historical_unit_id(model: str, base_task_id: str, context: str) -> str:
    ds, tid = base_task_id.split("/", 1)
    return f"{model}__{ds}__{tid}__{context}"
