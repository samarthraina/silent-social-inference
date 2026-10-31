"""Exact reproduction of the pilot's model-input path, for the scaled experiment only.

Ground truth is the code cell titled "HELD-OUT MT-BENCH SILENT CONTEXT EXPERIMENT" in
SSI_vs_SI.ipynb (sections 10-19) -- read directly, not reconstructed from the visible
prompt text. Deliberately NOT a change to backends.HFBackend: that class renders the chat
template to a string and re-tokenizes it, which is not guaranteed token-identical to
tokenizing the message list directly (special-token handling can differ), and it also
serves the mechanistic-phase protocol whose existing results must not be perturbed. This
module is a narrow, isolated path used only by generate_scaled_candidates.py and
run_scaled_scoring.py.

Every choice below is copied from the notebook, not inferred:
  - Model loading (all four models, including Gemma -- the mechanistic backend's "eager"
    override for Gemma's attention softcap does NOT apply here; the pilot loaded every
    model, Gemma included, with attn_implementation="sdpa"):
        AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16,
            device_map={"": 0}, low_cpu_mem_usage=True, attn_implementation="sdpa")
  - Chat template: tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_tensors="pt", return_dict=True
        [, enable_thinking=False for Qwen only]) -- tokenizes directly, no
    render-to-string-then-retokenize step.
  - transformers.set_seed(42) immediately before every .generate() call (the pilot never
    calls torch.manual_seed directly).
  - torch.inference_mode(), not torch.no_grad().
  - model.generate(**encoded, do_sample=False, max_new_tokens=N,
        pad_token_id=tokenizer.eos_token_id) -- no top_p (irrelevant under greedy decoding,
    and the pilot never passed it).
  - Decode: tokenizer.decode(new_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=False) -- the explicit False matters: HFBackend's
    generic decode call leaves this at the tokenizer's own default, which is not
    guaranteed False across tokenizers/versions.

The OLMo candidate-generation messages are the other verified piece: the system message
is ONLY the "producing the response" sentence; the "no scratch work" instruction is
appended to the USER turn (after the question, separated by a blank line), not folded
into the system message the way candidate_gen.build_answer_messages() does. Do not use
candidate_gen.build_answer_messages() for this experiment.

Requires transformers==5.15.0, matching the pilot's recorded environment.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_TRANSFORMERS_VERSION = "5.15.0"

OLMO_SYSTEM_PROMPT = (
    "You are producing the response that will be shown to a user. Answer the task "
    "directly and completely. Return only the final user-facing answer."
)
OLMO_ANSWER_INSTRUCTION = (
    "Do not include scratch work, self-corrections, hidden reasoning, planning notes, "
    "or commentary about your reasoning process. Explanations that are naturally "
    "required as part of the final answer are allowed."
)


def build_olmo_messages(question: str) -> List[Dict[str, str]]:
    """Verified against SSI_vs_SI.ipynb generate_olmo_answer(): system = the single
    "producing the response" sentence; user = question + blank line + the instruction
    sentence. NOT the same split as candidate_gen.build_answer_messages()."""
    return [
        {"role": "system", "content": OLMO_SYSTEM_PROMPT},
        {"role": "user", "content": f"{question}\n\n{OLMO_ANSWER_INSTRUCTION}"},
    ]


@dataclass
class ExactModelSpec:
    key: str
    hf_id: str
    revision: str
    disable_thinking: bool = False  # Qwen only -> enable_thinking=False in the template call
    # OLMo-3.1-32B-Instruct only (Amendment A2 E9 / Clarification C1_2): the shipped
    # config.json carries use_cache=False, which if it reached generation would make
    # decoding O(n^2). True here means load_exact_model sets model.config.use_cache and
    # model.generation_config.use_cache to True explicitly, and every generate_exact*
    # call also passes use_cache=True explicitly -- both, not either, per C1's requirement
    # not to rely on GenerationConfig defaults or on config.json inheritance.
    force_use_cache: bool = False
    # OLMo-3.1-32B-Instruct only (Amendment A2 batching / Clarification C1_3): a batch
    # size above 1 requires BOTH the token-ID equivalence smoke test AND a measured
    # memory preflight, because this model leaves only ~13.9 GiB of headroom on an 80 GB
    # card versus the ~31.2 GiB the superseded 24B scorer had. Scorers with this False
    # still require the equivalence test (the study-wide default), just not the
    # additional memory measurement.
    requires_memory_preflight: bool = False
    # Amendment A4: the scorer's generation ceiling. 512 for llama, qwen and gemma -- the
    # frozen value from plan v2.0 -- and 2048 for OLMo-3.1-32B-Instruct alone, because at 512
    # it reached the ceiling on 16/36 condition-blind preflight calls and lost the final score
    # line in 11 of them (uncapped responses parsed 20/20; capped ones 5/16). A ceiling is an
    # upper bound, not a target: raising it cannot alter any token a 512-token run already
    # produced under do_sample=False, so it does not change the other scorers and does not
    # lengthen a response that already terminated.
    max_new_tokens: int = 512


@dataclass
class LoadedModel:
    spec: ExactModelSpec
    tokenizer: Any
    model: Any
    chat_template_sha256: Optional[str]
    attn_implementation: str
    transformers_version: str


def check_transformers_version(log=print) -> str:
    import transformers
    v = transformers.__version__
    if v != REQUIRED_TRANSFORMERS_VERSION:
        log(f"[WARNING] transformers=={v} is installed; the pilot used "
           f"{REQUIRED_TRANSFORMERS_VERSION}. Chat-template output and tokenization "
           f"are not guaranteed identical across versions -- install "
           f"transformers=={REQUIRED_TRANSFORMERS_VERSION} before the real run.")
    return v


def reset_generation_config_for_determinism(model, log=print) -> None:
    """Neutralize any shipped sampling defaults (e.g. Mistral's temperature=0.15,
    do_sample=True) on the loaded model's generation_config, in addition to passing the
    explicit do_sample=False kwarg on every call. Required per Amendment A1
    ("generation_determinism.how"): kwarg precedence alone leaves the shipped values
    resident in the merged GenerationConfig object, and later inspection or a partial
    refactor that drops the explicit kwarg would silently revert to sampling. Applied to
    every scorer and the generator, not just Mistral, since the failure mode is generic.
    """
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    if getattr(gc, "do_sample", False) or getattr(gc, "temperature", None) not in (None, 1.0) \
            or getattr(gc, "top_p", None) is not None or getattr(gc, "top_k", None) is not None:
        log(f"[determinism] resetting shipped generation_config "
            f"(do_sample={getattr(gc, 'do_sample', None)}, "
            f"temperature={getattr(gc, 'temperature', None)}, "
            f"top_p={getattr(gc, 'top_p', None)}, top_k={getattr(gc, 'top_k', None)})")
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None


def apply_use_cache_setting(model, value: bool, log=print) -> None:
    """Amendment A2 Clarification C1_2: explicitly set use_cache on BOTH model.config and
    model.generation_config (when present), never relying on either's shipped default.
    OLMo-3.1-32B-Instruct ships config.json with use_cache=False; if that reached
    generation every new token would re-run the full forward pass over the whole
    sequence (O(n^2) instead of O(n)). This cannot change which tokens greedy decoding
    selects -- it is a cost property, not a validity one -- but a roughly order-of-
    magnitude runtime inflation must be prevented rather than discovered mid-run."""
    prior_config = getattr(getattr(model, "config", None), "use_cache", None)
    if hasattr(model, "config"):
        model.config.use_cache = value
    gc = getattr(model, "generation_config", None)
    prior_gc = getattr(gc, "use_cache", None) if gc is not None else None
    if gc is not None:
        gc.use_cache = value
    if prior_config != value or prior_gc != value:
        log(f"[use_cache] set model.config.use_cache={value} (was {prior_config}), "
           f"model.generation_config.use_cache={value} (was {prior_gc})")


def load_exact_model(spec: ExactModelSpec, hf_token: Optional[str] = None) -> LoadedModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision,
                                              token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, revision=spec.revision, token=hf_token,
        dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    reset_generation_config_for_determinism(model)
    if spec.force_use_cache:
        apply_use_cache_setting(model, True)

    tmpl = getattr(tokenizer, "chat_template", None)
    tmpl_sha = hashlib.sha256(tmpl.encode()).hexdigest() if isinstance(tmpl, str) else None
    import transformers as _tf
    return LoadedModel(
        spec=spec, tokenizer=tokenizer, model=model, chat_template_sha256=tmpl_sha,
        attn_implementation=getattr(model.config, "_attn_implementation", "sdpa"),
        transformers_version=_tf.__version__,
    )


# ---- E2 residency/precision gate (Amendment A1 residency_gate_checks) ----------------

def check_gpu_residency(model, require_bf16_params: bool = True) -> List[str]:
    """Six-point gate: full-GPU residency, no offload, no quantization, BF16 parameters.
    Returns a list of violation strings; an empty list means the gate passes. Never raises
    on its own -- callers decide whether a non-empty list is fatal (it should be, before
    any inference, per Amendment A1's E2 "on_failure": abort and return for review)."""
    import torch
    problems: List[str] = []

    for name, p in model.named_parameters():
        if p.device.type != "cuda":
            problems.append(f"parameter {name!r} is on device {p.device.type} (expected cuda)")
        if require_bf16_params and p.dtype != torch.bfloat16:
            problems.append(f"parameter {name!r} has dtype {p.dtype} (expected bfloat16)")

    for name, b in model.named_buffers():
        if b.device.type != "cuda":
            problems.append(f"buffer {name!r} is on device {b.device.type} (expected cuda)")
        # non-persistent buffers (e.g. rotary inv_freq) are exempt from the dtype check only

    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        bad = {k: v for k, v in hf_device_map.items()
              if str(v) in ("cpu", "disk", "meta")}
        if bad:
            problems.append(f"hf_device_map has offloaded entries: {bad}")

    quant_cfg = getattr(getattr(model, "config", None), "quantization_config", None)
    if quant_cfg is not None:
        problems.append(f"quantization_config present on model.config: {quant_cfg!r}")

    return problems


def check_vram_allocated(expected_bytes: int, tolerance: float = 0.15,
                         upper_slack: float = 4.0) -> List[str]:
    """Amendment A1 residency check 6: post-load torch.cuda.memory_allocated() within
    tolerance of the expected BF16 weight size.

    Below `expected * (1 - tolerance)` means part of the model is not actually resident --
    silent CPU/disk offload that slipped past the device checks. Above
    `expected * (1 + tolerance) * upper_slack` means substantially more is resident than
    one BF16 copy (e.g. a duplicated model, or a leaked KV cache from a previous call);
    the slack is deliberately generous because this is called after a short warm-up
    generation whose freed cache may not have been fully released back yet."""
    import torch
    if not torch.cuda.is_available():
        return ["cuda not available; cannot check VRAM allocation"]
    actual = torch.cuda.memory_allocated()
    lo = expected_bytes * (1 - tolerance)
    hi = expected_bytes * (1 + tolerance) * upper_slack
    problems = []
    if actual < lo:
        problems.append(f"allocated {actual/2**30:.2f} GiB is below the expected "
                        f"{expected_bytes/2**30:.2f} GiB weight size (tolerance {tolerance}); "
                        f"possible CPU/disk offload")
    if actual > hi:
        problems.append(f"allocated {actual/2**30:.2f} GiB exceeds "
                        f"{hi/2**30:.2f} GiB, far above one BF16 copy of the expected "
                        f"{expected_bytes/2**30:.2f} GiB; possible duplicate residency "
                        f"or retained cache")
    return problems


def encode_chat(tokenizer, messages: List[Dict[str, str]], disable_thinking: bool = False):
    kwargs = {"tokenize": True, "add_generation_prompt": True,
             "return_tensors": "pt", "return_dict": True}
    if disable_thinking:
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def build_generate_kwargs(spec: ExactModelSpec, *, max_new_tokens: int, pad_token_id,
                          use_cache: Optional[bool] = None) -> Dict[str, Any]:
    """Pure kwargs construction for model.generate(), deliberately factored out so it is
    unit-testable without torch or a loaded model. The deterministic decoding kwargs
    (do_sample, max_new_tokens, pad_token_id) are always present and are never touched by
    `use_cache`. `use_cache` is included in the returned dict only when the caller passes
    an explicit override (True or False -- used by the C1_2 cache-on/cache-off
    equivalence check to force each arm regardless of the spec), or, if the caller passes
    none, when `spec.force_use_cache` is set (the OLMo-3.1-32B production path -- see
    Amendment A2 Clarification C1_2). For every other scorer this omits the key entirely,
    leaving transformers' own default untouched."""
    kwargs: Dict[str, Any] = {"do_sample": False, "max_new_tokens": max_new_tokens,
                              "pad_token_id": pad_token_id}
    effective_use_cache = use_cache if use_cache is not None else (
        True if spec.force_use_cache else None)
    if effective_use_cache is not None:
        kwargs["use_cache"] = effective_use_cache
    return kwargs


def generate_exact(loaded: LoadedModel, messages: List[Dict[str, str]], *,
                   max_new_tokens: int, seed: int = 42,
                   use_cache: Optional[bool] = None) -> Dict[str, Any]:
    import torch
    from transformers import set_seed

    encoded = encode_chat(loaded.tokenizer, messages,
                          disable_thinking=loaded.spec.disable_thinking)
    device = next(loaded.model.parameters()).device
    encoded = encoded.to(device)

    generate_kwargs = build_generate_kwargs(
        loaded.spec, max_new_tokens=max_new_tokens,
        pad_token_id=loaded.tokenizer.eos_token_id, use_cache=use_cache)

    set_seed(seed)
    t0 = time.time()
    with torch.inference_mode():
        generated = loaded.model.generate(**encoded, **generate_kwargs)
    elapsed = time.time() - t0

    prompt_len = encoded["input_ids"].shape[1]
    new_tokens = generated[0, prompt_len:]
    text = loaded.tokenizer.decode(new_tokens, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False).strip()

    input_ids = encoded["input_ids"][0].tolist()
    raw_new = new_tokens.tolist()
    terms = terminator_ids(loaded)
    trimmed_new = trim_to_first_eos(raw_new, terms)
    # Amendment A5: did generation actually finish, or was it cut off at the ceiling?
    # Detected by looking for a terminator id, NOT by comparing trimmed/raw lengths --
    # trim_to_first_eos is inclusive, so a response ending exactly on its terminator has
    # equal lengths and would be misread as unterminated.
    emitted_terminator = bool(set(raw_new) & set(terms))
    return {
        "emitted_terminator": emitted_terminator,
        "text": text,
        "output_tokens": int(new_tokens.shape[0]),
        "hit_token_cap": int(new_tokens.shape[0]) >= max_new_tokens,
        "generation_seconds": round(elapsed, 3),
        "prompt_tokens": prompt_len,
        "first_input_token_id": input_ids[0] if input_ids else None,
        "last_input_token_id": input_ids[-1] if input_ids else None,
        "input_token_ids": input_ids,
        "generated_token_ids_raw": raw_new,
        "generated_token_ids_trimmed": trimmed_new,
        "generated_token_ids_sha256": hashlib.sha256(str(trimmed_new).encode()).hexdigest(),
    }


def trim_to_first_eos(token_ids: List[int], eos_token_id) -> List[int]:
    """Trim a generated token-ID sequence to its first terminator (inclusive), or return it
    unchanged if none appears (i.e. the sequence hit the token cap). Used only to make the
    batching-equivalence comparison fair: a batched call pads every row's raw output to the
    length of the longest still-generating row in that batch by repeating pad_token_id
    after a finished row's real EOS, which is not a divergence in what the model computed
    -- decode(skip_special_tokens=True) already collapses this difference for text; this
    makes the token-ID comparison agree with it for exactly the same reason, without
    hiding a genuine pre-terminator divergence.

    `eos_token_id` may be a single id OR a collection of ids. Several models declare more
    than one terminator (Qwen3 ships eos_token_id as a list; Llama-3.1 stops on either
    <|eot_id|> or <|end_of_text|>). Trimming on only ONE of them would leave a batched row
    that stopped on a different terminator carrying its trailing pad run while the
    batch-size-1 row does not, producing a hash mismatch that reflects padding rather than
    a real divergence -- which under the frozen policy would needlessly force that model
    to batch size 1 for the whole study."""
    stops = ({int(eos_token_id)} if isinstance(eos_token_id, (int, bool))
            else {int(t) for t in eos_token_id if t is not None})
    for i, tid in enumerate(token_ids):
        if tid in stops:
            return list(token_ids[:i + 1])
    return list(token_ids)


def terminator_ids(loaded: "LoadedModel") -> List[int]:
    """Every token id that can end generation for this model: the tokenizer's eos plus any
    eos_token_id declared on the model's generation_config (often a list). Used for
    equivalence trimming only -- never to change what generate() itself does."""
    ids = []
    tok_eos = getattr(loaded.tokenizer, "eos_token_id", None)
    if tok_eos is not None:
        ids.append(int(tok_eos))
    gc = getattr(loaded.model, "generation_config", None)
    gc_eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(gc_eos, (list, tuple, set)):
        ids += [int(t) for t in gc_eos if t is not None]
    elif gc_eos is not None:
        ids.append(int(gc_eos))
    return sorted(set(ids))


def encode_chat_batch(tokenizer, list_of_messages: List[List[Dict[str, str]]],
                      disable_thinking: bool = False):
    """Tokenize each conversation individually (so per-row special-token handling exactly
    matches the single-call path), then left-pad into one batch. Left padding is required
    for batched causal-LM generation: it keeps every row's real prompt right-aligned so
    that new tokens begin at the same column index for the whole batch."""
    import torch

    per_row_ids: List[List[int]] = []
    for messages in list_of_messages:
        kwargs = {"tokenize": True, "add_generation_prompt": True,
                  "return_tensors": None, "return_dict": True}
        if disable_thinking:
            kwargs["enable_thinking"] = False
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
        # transformers >= 5 returns a BatchEncoding here, and iterating one yields its KEYS.
        # Consuming it directly therefore produced ["input_ids", "attention_mask"] -- two
        # strings -- instead of token ids, which surfaced as a TypeError deep inside
        # torch.tensor(). Take the field by name, as the single-call encode_chat does, and
        # still accept a bare list in case the return shape changes back.
        ids = list(encoded["input_ids"]) if hasattr(encoded, "keys") else list(encoded)
        if not ids or not all(isinstance(t, int) for t in ids):
            raise TypeError(
                f"apply_chat_template returned {type(encoded).__name__} whose input_ids are "
                f"not a list of ints (first element: {ids[0]!r} if any). Refusing to build a "
                f"batch from it rather than failing later inside torch.")
        per_row_ids.append(ids)

    max_len = max(len(ids) for ids in per_row_ids)
    # Llama-3.1 ships no pad token, so pad_token_id is None and torch.full would fail. Fall
    # back to eos, matching the pad_token_id_source the single-call path already records in
    # the frozen decode settings. Which id is used cannot affect the output: every padded
    # column is masked out by attention_mask below, and the batching-equivalence gate that
    # calls this proves token-identical output against the batch-size-1 path.
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer exposes neither pad_token_id nor eos_token_id; cannot "
                         "left-pad a batch deterministically")
    input_ids = torch.full((len(per_row_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(per_row_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(per_row_ids):
        n = len(ids)
        input_ids[i, max_len - n:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - n:] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}, per_row_ids


def generate_exact_batch(loaded: LoadedModel, list_of_messages: List[List[Dict[str, str]]], *,
                         max_new_tokens: int, seed: int = 42,
                         use_cache: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Batched counterpart to generate_exact: one model.generate() call over the whole
    micro-batch. Only used after the frozen batching-equivalence smoke test has proven,
    per-model, that this produces token-identical output to the batch-size-1 path (see
    scripts/run_batching_equivalence_smoke.py); never used unconditionally."""
    import torch
    from transformers import set_seed

    # No tokenizer.padding_side juggling: encode_chat_batch builds the left-padded tensor
    # itself, so the tokenizer's own padding setting is never consulted and setting it
    # would be a no-op that falsely implies otherwise.
    encoded, per_row_prompt_ids = encode_chat_batch(
        loaded.tokenizer, list_of_messages, disable_thinking=loaded.spec.disable_thinking)

    device = next(loaded.model.parameters()).device
    encoded = {k: v.to(device) for k, v in encoded.items()}
    padded_prompt_len = encoded["input_ids"].shape[1]

    generate_kwargs = build_generate_kwargs(
        loaded.spec, max_new_tokens=max_new_tokens,
        pad_token_id=loaded.tokenizer.eos_token_id, use_cache=use_cache)

    set_seed(seed)
    t0 = time.time()
    with torch.inference_mode():
        generated = loaded.model.generate(**encoded, **generate_kwargs)
    elapsed = time.time() - t0

    stops = terminator_ids(loaded)
    stop_set = set(stops)
    results = []
    for i, prompt_ids in enumerate(per_row_prompt_ids):
        raw_new = generated[i, padded_prompt_len:].tolist()
        trimmed_new = trim_to_first_eos(raw_new, stops)
        text = loaded.tokenizer.decode(generated[i, padded_prompt_len:],
                                       skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False).strip()
        results.append({
            "text": text,
            "output_tokens": len(trimmed_new),
            "hit_token_cap": not stop_set.intersection(raw_new),
            "generation_seconds": round(elapsed / len(per_row_prompt_ids), 3),
            "generation_seconds_batch_total": round(elapsed, 3),
            "prompt_tokens": len(prompt_ids),
            "first_input_token_id": prompt_ids[0] if prompt_ids else None,
            "last_input_token_id": prompt_ids[-1] if prompt_ids else None,
            "generated_token_ids_raw": raw_new,
            "generated_token_ids_trimmed": trimmed_new,
            "generated_token_ids_sha256": hashlib.sha256(
                str(trimmed_new).encode()).hexdigest(),
            # A5 completeness signal. Uses this scope's stop_set -- an earlier edit referenced
            # `terms`, which exists only in the single-call path, so every batched call raised
            # NameError. No CPU test caught it because the stub batch path never reaches here.
            "emitted_terminator": bool(stop_set & set(raw_new)),
            "batch_size": len(per_row_prompt_ids),
        })
    return results


def memory_preflight_ok(safety_fraction: float = 0.90) -> Tuple[bool, Dict[str, Any]]:
    """Amendment A2 batching / Clarification C1_3: measures peak GPU memory used since the
    caller's last torch.cuda.reset_peak_memory_stats() and checks it against a safety
    fraction of total device capacity. Required, IN ADDITION TO token-ID equivalence,
    before any scorer with ExactModelSpec.requires_memory_preflight=True may run at a
    batch size above 1 -- OLMo-3.1-32B-Instruct leaves only ~13.9 GiB of headroom on an
    80 GB card, against ~31.2 GiB for the superseded 24B scorer, so "it didn't OOM on the
    smoke set" is not itself sufficient evidence of safety at scale.

    Fails closed: with no CUDA device available this cannot confirm the intended batch
    size is safe, so it reports failure rather than silently assuming success."""
    import torch
    if not torch.cuda.is_available():
        return False, {"reason": "cuda not available; cannot measure GPU memory",
                       "safe": False}
    peak = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    threshold = total * safety_fraction
    safe = peak <= threshold
    return safe, {
        "peak_bytes": peak, "peak_gib": peak / 2**30,
        "total_bytes": total, "total_gib": total / 2**30,
        "threshold_gib": threshold / 2**30, "safety_fraction": safety_fraction,
        "safe": safe,
    }


@dataclass
class CacheEquivalenceResult:
    n_units: int
    n_mismatches: int
    mismatched_indices: List[int]

    @property
    def passed(self) -> bool:
        return self.n_mismatches == 0


def check_cache_equivalence(cache_on_results: List[Dict[str, Any]],
                            cache_off_results: List[Dict[str, Any]]) -> CacheEquivalenceResult:
    """Amendment A2 Clarification C1_2's mandatory cache-on/cache-off equivalence check:
    exact equality of the GENERATED TOKEN-ID sequence (compared via the sha256 that
    generate_exact already computes) between the two arms, for every unit. Under greedy
    decoding the output must be invariant to KV caching; a mismatch indicates a numerical
    or implementation fault that would also threaten run-to-run reproducibility, and per
    C1's requirement it must be investigated -- never silently accepted, never resolved by
    preferring the faster arm. Pure comparison logic: takes already-generated results, so
    it is unit-testable without a GPU."""
    if len(cache_on_results) != len(cache_off_results):
        raise ValueError(f"cache-on/cache-off result counts differ: "
                         f"{len(cache_on_results)} vs {len(cache_off_results)}")
    mismatches = [i for i, (a, b) in enumerate(zip(cache_on_results, cache_off_results))
                 if a["generated_token_ids_sha256"] != b["generated_token_ids_sha256"]]
    return CacheEquivalenceResult(n_units=len(cache_on_results),
                                  n_mismatches=len(mismatches),
                                  mismatched_indices=mismatches)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weight_shard_hashes(shard_dir, expected_sha256: Dict[str, str]) -> List[str]:
    """Verifies already-downloaded weight-shard files against a frozen sha256 map (e.g.
    configs/scorer_olmo3_1_32b_pin.json's weight_shard_lfs_sha256_for_later_verification).
    Returns a list of problem strings; an empty list means every expected shard is present
    and byte-exact. Streams each file in chunks rather than reading it whole, so this is
    safe to run against real multi-GB shards without loading them entirely into memory.
    Never downloads anything -- operates only on files already on disk, and is the required
    check before trusting weights that were fetched on the GPU host."""
    shard_dir = Path(shard_dir)
    problems: List[str] = []
    for filename, expected in expected_sha256.items():
        # Reject anything that is not a bare filename BEFORE touching the filesystem: a
        # key containing a separator or ".." would resolve outside shard_dir and could be
        # satisfied by an unrelated file elsewhere on the host, which is exactly the
        # "wrongly resolved file" case this check exists to prevent. The frozen pin file
        # contains only bare "model-*.safetensors" names, so a violation here means the
        # expected-hash map itself is not the frozen one.
        if Path(filename).name != filename or filename in ("", ".", ".."):
            problems.append(f"invalid shard name (must be a bare filename): {filename!r}")
            continue
        p = shard_dir / filename
        try:
            if not p.is_file():
                problems.append(
                    f"missing shard: {filename}" if not p.exists()
                    else f"not a regular file: {filename}")
                continue
            actual = _sha256_file(p)
        except OSError as e:
            # Never let an unreadable path abort the whole verification with a traceback:
            # report it as a problem so the caller still fails closed with a full list.
            problems.append(f"unreadable shard {filename}: {type(e).__name__}: {e}")
            continue
        if actual != expected:
            problems.append(f"hash mismatch for {filename}: "
                            f"expected {expected}, got {actual}")
    return problems


def unload(loaded: LoadedModel) -> None:
    import gc
    import torch
    del loaded.model
    del loaded.tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ---- No-GPU stub, for plumbing tests only (resume/retry/substitution/idempotency). ----
# Never a source of experimental data; deliberately produces JSON-shaped nonsense text
# so it can never be mistaken for a real model response if it leaked into real output.

class _StubTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    chat_template = "stub"
    padding_side = "right"


class _StubModel:
    class _Cfg:
        _attn_implementation = "stub"
        quantization_config = None
    config = _Cfg()
    hf_device_map = None

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def named_parameters(self):
        import torch
        yield "stub.weight", torch.zeros(1)

    def named_buffers(self):
        import torch
        yield "stub.buffer", torch.zeros(1)


def load_stub_model(spec: ExactModelSpec) -> LoadedModel:
    return LoadedModel(spec=spec, tokenizer=_StubTokenizer(), model=_StubModel(),
                       chat_template_sha256="stub", attn_implementation="stub",
                       transformers_version="stub")


def _stub_token_ids(messages: List[Dict[str, str]], n: int = 10) -> List[int]:
    """Deterministic pseudo token-IDs derived only from message content, so that stub
    batch-1 and stub batched calls over the same messages always agree -- this exercises
    the batching-equivalence harness's comparison logic without a real model."""
    h = hashlib.sha256(str(messages).encode()).digest()
    return [int.from_bytes((h * 4)[i:i + 2], "big") % 32000 for i in range(0, n * 2, 2)]


def generate_exact_stub(loaded: LoadedModel, messages: List[Dict[str, str]], *,
                        max_new_tokens: int, seed: int = 42,
                        use_cache: Optional[bool] = None) -> Dict[str, Any]:
    # use_cache is accepted only for call-signature compatibility with generate_exact --
    # the stub is deterministic and instant regardless of its value, by design (this is
    # what makes the cache-on/cache-off equivalence harness exercisable without a GPU).
    h = hashlib.sha256(str(messages).encode()).hexdigest()
    ids = _stub_token_ids(messages)
    return {
        "text": f"stub response {h[:8]}", "output_tokens": 10, "hit_token_cap": False,
        # A stub response is a well-formed complete one: it terminated normally, so under
        # Amendment A5 it is valid for analysis rather than capped_incomplete.
        "emitted_terminator": True,
        "generation_seconds": 0.0, "prompt_tokens": 5,
        "first_input_token_id": 1, "last_input_token_id": 2,
        "input_token_ids": [1, 2, 3],
        "generated_token_ids_raw": ids, "generated_token_ids_trimmed": ids,
        "generated_token_ids_sha256": hashlib.sha256(str(ids).encode()).hexdigest(),
    }


def generate_exact_batch_stub(loaded: LoadedModel,
                              list_of_messages: List[List[Dict[str, str]]], *,
                              max_new_tokens: int, seed: int = 42,
                              use_cache: Optional[bool] = None) -> List[Dict[str, Any]]:
    return [{**generate_exact_stub(loaded, m, max_new_tokens=max_new_tokens, seed=seed,
                                   use_cache=use_cache),
            "generation_seconds_batch_total": 0.0, "batch_size": len(list_of_messages)}
           for m in list_of_messages]
