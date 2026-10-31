"""Phase 2B TRACE — residual-stream activation patching.

Layer indexing matches Phase 2A exactly. `hidden_states[L]` is the tensor patched at
"layer L":

    L == 0  -> output of embed_tokens          (the embedding itself)
    L >= 1  -> output of transformer block L-1 (a POST-block hook)

So a hook for layer L is registered on `model.model.layers[L-1]`, and patching layer L
replaces the state that Phase 2A read as `hidden_states[L]`. Only the listed token
positions are written; every other position is left untouched.

The hook guards on sequence length: during incremental decoding the block sees a
single-token input and must not be modified, otherwise the patch would be re-applied to
generated tokens rather than to the prompt.

One readback caveat, verified on the model rather than assumed: transformers captures
`hidden_states[L]` from the raw module output, before a user forward hook on that module
runs. A hook installed here therefore does not show up at index L of the same pass's
`hidden_states` — its consequences appear from L+1 onward. Read and write still name the
same point in the computation, so capture/patch round trips are correct; only same-pass
verification at index L is misleading. To observe what a hook actually wrote, register a
second hook after it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Sequence


def _module_for_layer(model, layer: int):
    if layer == 0:
        return model.model.embed_tokens
    return model.model.layers[layer - 1]


@contextmanager
def patch_positions(model, layer: int, positions: Sequence[int], values):
    """Replace the residual state at `positions` with `values` for one forward pass.

    `values` is [len(positions), hidden]. Applied only on a pass whose sequence length
    exceeds 1, i.e. the prefill over the prompt.
    """
    import torch

    pos = torch.as_tensor(list(positions), device=values.device, dtype=torch.long)

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] <= 1:          # incremental decoding step: leave untouched
            return output
        hs = hs.clone()
        hs[0, pos, :] = values.to(hs.dtype)
        return (hs,) + output[1:] if is_tuple else hs

    handle = _module_for_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_direction(model, layer: int, positions: Sequence[int], direction,
                     alpha: float = 1.0):
    """Remove `alpha` times the component along `direction` at `positions`, one layer.

    h' = h - alpha (h . d) d, with d unit norm. The projection is taken in float32 and
    cast back, so alpha=0 is a bitwise identity: a bf16 value survives the round trip
    exactly, which is what makes the alpha=0 arm a real null-implementation control.
    """
    import torch

    pos = torch.as_tensor(list(positions), device=direction.device, dtype=torch.long)
    d = direction.float()
    d = d / d.norm()

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] <= 1:          # incremental decoding step: leave untouched
            return output
        hs = hs.clone()
        sub = hs[0, pos, :].float()
        hs[0, pos, :] = (sub - alpha * (sub @ d).unsqueeze(-1) * d).to(hs.dtype)
        return (hs,) + output[1:] if is_tuple else hs

    handle = _module_for_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_positionwise(model, layer: int, positions: Sequence[int], directions,
                        alpha: float = 1.0):
    """One direction per position, all removals applied in the same forward pass.

    `directions` is [len(positions), hidden], row i belonging to positions[i]. Rows are
    normalized here; a zero row leaves its position untouched, which is how a position
    whose fitted direction vanished is handled without a special case.
    """
    import torch

    pos = torch.as_tensor(list(positions), device=directions.device, dtype=torch.long)
    D = directions.float()
    if D.shape[0] != pos.shape[0]:
        raise ValueError(f"{D.shape[0]} directions for {pos.shape[0]} positions")
    n = D.norm(dim=-1, keepdim=True)
    D = torch.where(n > 0, D / n.clamp(min=1e-12), torch.zeros_like(D))

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] <= 1:          # incremental decoding step: leave untouched
            return output
        hs = hs.clone()
        sub = hs[0, pos, :].float()
        hs[0, pos, :] = (sub - alpha * (sub * D).sum(-1, keepdim=True) * D).to(hs.dtype)
        return (hs,) + output[1:] if is_tuple else hs

    handle = _module_for_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def add_positionwise(model, layer: int, positions: Sequence[int], deltas,
                     beta: float = 1.0):
    """Steering: h'_t = h_t + beta * delta_t at every listed position, one layer.

    `deltas` is [len(positions), hidden] and is used **raw** — not normalized — so the
    per-position magnitude observed between the endpoint conditions is what gets injected.
    beta=0 is a bitwise identity: bf16 -> float32 -> bf16 round-trips exactly and adding
    0*delta changes nothing, which is what makes the beta=0 arm a real no-op control.
    """
    import torch

    pos = torch.as_tensor(list(positions), device=deltas.device, dtype=torch.long)
    D = deltas.float()
    if D.shape[0] != pos.shape[0]:
        raise ValueError(f"{D.shape[0]} deltas for {pos.shape[0]} positions")

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        if hs.shape[1] <= 1:          # incremental decoding step: leave untouched
            return output
        hs = hs.clone()
        hs[0, pos, :] = (hs[0, pos, :].float() + beta * D).to(hs.dtype)
        return (hs,) + output[1:] if is_tuple else hs

    handle = _module_for_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def no_patch(model, layer: int = 1):
    """Hooks installed and firing, but writing nothing.

    The null implementation control: if a hook-enabled pass does not reproduce the frozen
    behavioural score exactly, the harness itself is perturbing the run.
    """
    def hook(_module, _inputs, output):
        return output

    handle = _module_for_layer(model, layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def capture_states(model, input_ids, positions_by_name: Dict[str, List[int]],
                   n_layers: int):
    """One forward pass; returns {position_name: {layer: tensor[len(pos), hidden]}}."""
    import torch

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    assert len(hs) == n_layers + 1, "hidden_states length does not match n_layers+1"
    cap = {}
    for name, idx in positions_by_name.items():
        if not idx:
            continue
        cap[name] = {L: hs[L][0, idx, :].detach().clone() for L in range(n_layers + 1)}
    del out
    return cap
