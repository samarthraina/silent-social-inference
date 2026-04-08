# llm_cross_judging/LLM_Cross_judging_fixed.py

import json
import random
import time
import os
import csv
import re
import gc
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
from vllm import LLM, SamplingParams

# ============================================================
# CONFIGURATION
# ============================================================

MODELS = {
    "qwen":          "Qwen/Qwen2.5-7B-Instruct",
    "llama":         "meta-llama/Llama-3.1-8B-Instruct",
    "gemma":         "google/gemma-2-9b-it",
    "mistral":       "mistralai/Mistral-7B-Instruct-v0.3",
    "olmo_instruct": "allenai/OLMo-7B-Instruct",
    "olmo_sft":      "allenai/OLMo-7B-SFT",
    "tulu_sft":      "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "tulu_dpo":      "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "gemma4":        "google/gemma-4-E4B-it",
}

MODEL_CONFIG = {
    "qwen":          {"is_sft": False, "label": "Qwen2.5-7B-Instruct"},
    "llama":         {"is_sft": False, "label": "Llama-3.1-8B-Instruct"},
    "gemma":         {"is_sft": False, "label": "Gemma-2-9B-IT"},
    "mistral":       {"is_sft": False, "label": "Mistral-7B-Instruct-v0.3"},
    "olmo_instruct": {"is_sft": False, "label": "OLMo-7B-Instruct"},
    "olmo_sft":      {"is_sft": True,  "label": "OLMo-7B-SFT"},
    "tulu_sft":      {"is_sft": True,  "label": "Llama-3.1-Tulu-3-8B-SFT"},
    "tulu_dpo":      {"is_sft": False, "label": "Llama-3.1-Tulu-3-8B-DPO"},
    "gemma4":        {"is_sft": False, "label": "Gemma-4-E4B-IT"},
}

N_RUNS_PER_PROMPT = 10
K_JUDGE_SAMPLES   = 3
TEMPERATURES      = [round(i * 0.1, 1) for i in range(11)]   # [0.0, 0.1, ..., 1.0]

GENERATION_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS      = 512
JUDGE_TEMPERATURE     = 0.3
JUDGE_TOP_P           = 0.9

OUTPUT_DIR = "/root/results"
SEED       = 42

if os.path.exists("/huggingface-public"):
    os.environ["HF_HOME"] = "/huggingface-public"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ============================================================
# PROMPTS
# ============================================================

PROMPTS = [
    "A farmer has 3 fields. Field A yields 20% more than Field B, and Field C yields 15% less than Field A. If Field B produces 500 kg, what is the total harvest? Walk through your reasoning.",
    "In exactly 3 sentences, explain why a society that values only efficiency would eventually collapse.",
    "Compare the trade-offs between nuclear fission and solar energy for base-load power generation. Be concise but specific.",
    "A self-driving car must choose between hitting one elderly pedestrian or swerving into a wall, injuring its two passengers. What frameworks apply here, and what would you recommend?",
    "What is the time complexity of finding the longest increasing subsequence using dynamic programming, and why can't we do better than O(n log n)?",
    "Explain the concept of entropy to a 12-year-old using a real-world analogy. Keep it under 100 words.",
    "\"Correlation does not imply causation.\" Give a concrete, original example where ignoring this leads to a dangerous policy decision.",
    "You have 8 identical-looking balls. One is heavier. You have a balance scale and can use it only twice. Describe your strategy.",
]

# ============================================================
# RUBRIC
# ============================================================

RUBRIC = """
SCORING RUBRIC (use for every answer):

1. **Relevance** (1-10): Does the answer directly and completely address what was asked?
   - 9-10: Fully addresses every part of the prompt with precision
   - 6-8:  Addresses the core question but misses nuances or sub-parts
   - 3-5:  Partially relevant, drifts off-topic or is superficial
   - 1-2:  Largely irrelevant or misunderstands the question

2. **Reasoning Quality** (1-10): Is the thinking sound, well-structured, and logically coherent?
   - 9-10: Rigorous logic, clear structure, no fallacies, insightful
   - 6-8:  Generally sound but with minor gaps or unsupported claims
   - 3-5:  Noticeable logical flaws or disorganized thinking
   - 1-2:  Incoherent, contradictory, or no reasoning shown

You MUST respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "evaluations": [
    {
      "answer_id": "<the answer label>",
      "relevance": <int 1-10>,
      "reasoning_quality": <int 1-10>,
      "justification": "<1-2 sentences explaining your scores>"
    }
  ]
}
"""

# ============================================================
# RUNTIME PREDICTION
# ============================================================

def predict_runtime(
    num_prompts: int = None,
    num_models: int = None,
    num_temperatures: int = None,
    n_runs: int = None,
    k_judge_samples: int = None,
) -> float:
    """
    Estimate wall-clock time for the full pipeline run.

    Assumptions (H100 80GB, vLLM bfloat16):
      GEN_S_PER_BATCH = 8.0   — seconds per (model, prompt) batch call (all N runs batched)
      SWAP_S          = 45.0  — seconds per model load/unload swap
      JUDGE_S_PER_CALL = 4.0  — seconds per judge call (K samples batched)

    Returns grand_total_s as float.
    """
    GEN_S_PER_BATCH   = 8.0
    SWAP_S            = 45.0
    JUDGE_S_PER_CALL  = 4.0

    # Apply defaults from globals
    if num_prompts is None:
        num_prompts = len(PROMPTS)
    if num_models is None:
        num_models = len(MODELS)
    if num_temperatures is None:
        num_temperatures = len(TEMPERATURES)
    if n_runs is None:
        n_runs = N_RUNS_PER_PROMPT
    if k_judge_samples is None:
        k_judge_samples = K_JUDGE_SAMPLES

    # Per-temperature costs
    generation_s   = num_models * num_prompts * GEN_S_PER_BATCH
    gen_swaps_s    = num_models * SWAP_S
    judge_calls    = num_models * num_models * 4 * num_prompts
    judging_s      = judge_calls * JUDGE_S_PER_CALL
    judge_swaps    = num_models * num_models * 4
    judge_swaps_s  = judge_swaps * SWAP_S
    total_per_temp = generation_s + gen_swaps_s + judging_s + judge_swaps_s

    grand_total_s  = total_per_temp * num_temperatures

    # Print formatted table
    now             = datetime.now()
    finish_time     = datetime.fromtimestamp(now.timestamp() + grand_total_s)
    sep             = "=" * 60

    print(sep)
    print("RUNTIME ESTIMATE (H100 80GB, vLLM bfloat16)")
    print(sep)
    print(f"Config: {num_models} models, {num_prompts} prompts, {num_temperatures} temperatures, "
          f"N={n_runs} runs, K={k_judge_samples} judge samples")
    print()
    print("Per temperature:")
    print(f"  Generation (batched):  {generation_s:>8,.0f} s  "
          f"({num_models} model loads × {num_prompts} prompts × {GEN_S_PER_BATCH:.0f}s/batch)")
    print(f"  Gen model swaps:       {gen_swaps_s:>8,.0f} s  "
          f"({num_models} models × {SWAP_S:.0f}s/swap)")
    print(f"  Judging (4 cases):     {judging_s:>8,.0f} s  "
          f"({judge_calls} judge calls × {JUDGE_S_PER_CALL:.0f}s/call)")
    print(f"  Judge model swaps:     {judge_swaps_s:>8,.0f} s  "
          f"({judge_swaps} swaps × {SWAP_S:.0f}s/swap)")
    print(f"  Subtotal:              {total_per_temp:>8,.0f} s  "
          f"(~{total_per_temp / 3600:.1f} hours)")
    print()
    print(f"Total ({num_temperatures} temperatures):")
    print(f"  Grand total:           {grand_total_s:>8,.0f} s  "
          f"(~{grand_total_s / 3600:.1f} hours)")
    print(f"  Estimated finish:      {finish_time.strftime('%Y-%m-%d %H:%M')} (local time)")
    print(sep)

    return float(grand_total_s)
