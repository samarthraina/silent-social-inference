"""Scorer message-wrapping for the final 578-task behavioral study only.

Deliberately separate from scaled_conditions.py (the frozen prompt renderer, pinned by
hash in configs/final_analysis_plan.json) and from scaled_parsing.py: this module adds
no prompt content and no parsing logic, it only decides which chat-message envelope a
given scorer's rendered prompt goes into.

Three of the four scorers (Llama, Qwen, Gemma) score user-message-only, exactly as in the
90-task run. The fourth, allenai/Olmo-3.1-32B-Instruct (Amendment A2, replacing the
never-run mistralai/Mistral-Small-24B-Instruct-2501), requires an explicit EMPTY system
message: its official chat template injects a default persona/knowledge-cutoff system
message whenever no system turn is supplied at all, which would leak "You are Olmo...
built by Ai2... date cutoff is December 2024..." into an otherwise frozen, reproducible
prompt (verified by direct Jinja2 rendering of the pinned template -- see
configs/final_analysis_plan_v2_2.json E8_resolution_olmo32b_template). An explicit empty
system message renders to an empty <|im_start|>system\\n<|im_end|>\\n turn and nothing
else, with zero residual alphabetic words -- verified against the SAME preflight gate
function used on-GPU (silent_influence.final_preflight_checks.check_render).

This is the same class of problem, and the same resolution, that Amendment A1 applied to
the superseded Mistral scorer (its pin file, configs/scorer_mistral_small_24b_pin.json,
is preserved unmodified as a historical artifact even though that scorer was never run).

user-only vs. empty-system-message is CONDITIONAL on the preflight template-audit gate
(scripts/verify_final_preflight_gate.py) passing for that scorer -- this module does not
itself verify cleanliness, it only encodes the wrapping decision the gate result licenses.
"""
from __future__ import annotations

from typing import Dict, List

OLMO32B_KEY = "olmo32b"

# Scorers whose messages are user-message-only, conditional on the preflight gate
# confirming their official template injects no hidden system content when no system
# turn is supplied (Llama, Gemma) or never emits a system block at all under this
# protocol (Qwen -- independently verified: only triggered by a `tools` kwarg, never
# passed here).
USER_ONLY_SCORER_KEYS = ("llama", "qwen", "gemma")

EMPTY_SYSTEM_SCORER_KEYS = (OLMO32B_KEY,)


def build_scorer_messages(scorer_key: str, prompt: str) -> List[Dict[str, str]]:
    """The frozen scoring `prompt` (from scaled_conditions.render_prompt, unmodified) is
    always the entire user turn. The only per-scorer decision is whether an explicit
    empty system turn precedes it."""
    if scorer_key in EMPTY_SYSTEM_SCORER_KEYS:
        return [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
    if scorer_key in USER_ONLY_SCORER_KEYS:
        return [{"role": "user", "content": prompt}]
    raise KeyError(f"unknown scorer key {scorer_key!r}; known: "
                   f"{USER_ONLY_SCORER_KEYS + EMPTY_SYSTEM_SCORER_KEYS}")
