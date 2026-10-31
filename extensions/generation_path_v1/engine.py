"""Batch-one cached interventions; all research arms receive the same explicit causal mask.

Reader indices denote INPUT tokens, not the tokens predicted by their logits. Generated
readers start at the native prompt length: the first sampled token is therefore unaffected
by generated-reader-only blocking. History can still act through other prompt states.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib


@dataclass
class Patch:
    layer: int                 # one-based decoder-block output
    positions: list
    values: object             # [len(positions), hidden_size]


def digest_ids(ids):
    return hashlib.sha256(str(list(ids)).encode()).hexdigest()


def causal_mask(torch, q_start, q_len, key_len, device, dtype):
    if q_start < 0 or q_len <= 0 or key_len != q_start + q_len:
        raise ValueError('Only contiguous, unpadded, batch-one causal input is supported')
    q = torch.arange(q_start, key_len, device=device)[:, None]
    k = torch.arange(key_len, device=device)[None, :]
    return torch.where(k <= q, torch.tensor(0., device=device, dtype=dtype),
                       torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype))[None, None]


def blocked_mask(mask, *, q_start, prompt_len, keys, reader_policy='generated'):
    """Clone only when blocking; never permit edits to future keys or an all-masked row."""
    import torch
    if reader_policy not in ('generated', 'last_prompt'):
        raise ValueError('Unknown reader policy')
    if len(set(keys)) != len(keys) or any(k < 0 or k >= prompt_len for k in keys):
        raise ValueError('Keys must be unique native prompt positions')
    absolute = torch.arange(q_start, q_start + mask.shape[-2], device=mask.device)
    readers = (absolute >= prompt_len if reader_policy == 'generated'
               else absolute == prompt_len - 1)
    rows = readers.nonzero(as_tuple=True)[0]
    if not keys or rows.numel() == 0:
        return mask, 0
    out = mask.clone()
    cols = torch.tensor(keys, device=mask.device)
    if any(k > int(absolute[rows[0]]) for k in keys):
        raise ValueError('Cannot block noncausal keys')
    out[0, 0, rows[:, None], cols[None, :]] = torch.finfo(out.dtype).min
    if not (out[0, 0, rows].eq(0).any(dim=-1)).all():
        raise ValueError('Blocking would remove every available key')
    return out, int(rows.numel()) * len(keys)


@contextmanager
def forward_edits(model, mask, *, q_start, prompt_len, keys=(), blocks=(), patch=None,
                  reader_policy='generated'):
    """One forward only; strict hook counts and mask shape/content checks, even on no-op."""
    import torch
    layers = model.model.layers
    if keys and not blocks:
        raise ValueError('A key knockout requires selected attention blocks')
    if len(set(blocks)) != len(blocks) or any(i < 0 or i >= len(layers) for i in blocks):
        raise ValueError('Invalid zero-based decoder blocks')
    edited, edges = blocked_mask(mask, q_start=q_start, prompt_len=prompt_len,
                                  keys=list(keys), reader_policy=reader_policy)
    events = {'mask_hooks': 0, 'blocked_edges': 0, 'patch_writes': 0}
    with ExitStack() as stack:
        for block in blocks:
            def before(_m, args, kwargs):
                incoming = kwargs.get('attention_mask')
                if incoming is None or incoming.shape != mask.shape or not torch.equal(incoming, mask):
                    raise RuntimeError('Attention backend did not preserve the explicit causal mask')
                events['mask_hooks'] += 1
                events['blocked_edges'] += edges
                return args, dict(kwargs, attention_mask=edited)
            handle = layers[block].self_attn.register_forward_pre_hook(before, with_kwargs=True)
            stack.callback(handle.remove)
        if patch is not None:
            if not 1 <= patch.layer <= len(layers) or not patch.positions:
                raise ValueError('Invalid residual patch')
            if len(set(patch.positions)) != len(patch.positions) or any(
                    p < 0 or p >= prompt_len for p in patch.positions):
                raise ValueError('Patch positions must be unique native prompt positions')
            def replace(_m, _args, output):
                if q_start != 0:
                    return output
                hs = output[0] if isinstance(output, tuple) else output
                if hs.shape[0] != 1 or tuple(patch.values.shape) != (len(patch.positions), hs.shape[-1]):
                    raise ValueError('Residual donor shape mismatch')
                hs = hs.clone()
                hs[0, patch.positions] = patch.values.to(device=hs.device, dtype=hs.dtype)
                events['patch_writes'] += 1
                return (hs,) + output[1:] if isinstance(output, tuple) else hs
            handle = layers[patch.layer-1].register_forward_hook(replace)
            stack.callback(handle.remove)
        yield events
        if events['mask_hooks'] != len(blocks):
            raise RuntimeError('Every selected attention module must fire exactly once per forward')
        if patch is not None and events['patch_writes'] != int(q_start == 0):
            raise RuntimeError('Residual patch did not fire at the expected prefill')


def forward(model, ids, *, prompt_len, cache=None, q_start=0, keys=(), blocks=(),
            patch=None, explicit=True, reader_policy='generated'):
    import torch
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError('Batch size one required')
    dtype = next(model.parameters()).dtype
    mask = causal_mask(torch, q_start, ids.shape[1], q_start+ids.shape[1], ids.device, dtype)
    if not explicit and (keys or blocks or patch):
        raise ValueError('Unmasked native reference cannot carry interventions')
    kwargs = {'attention_mask': mask} if explicit else {}
    with torch.inference_mode(), forward_edits(model, mask, q_start=q_start,
            prompt_len=prompt_len, keys=keys, blocks=blocks, patch=patch,
            reader_policy=reader_policy) as events:
        output = model(input_ids=ids, past_key_values=cache, use_cache=True, **kwargs)
    return output, events


def capture(model, ids, layer, positions):
    """Capture native unmodified donor states under the same explicit-mask prefill."""
    got = []
    if not positions or not 1 <= layer <= len(model.model.layers):
        raise ValueError('Invalid capture site')
    h = model.model.layers[layer-1].register_forward_hook(
        lambda _m, _a, out: got.append((out[0] if isinstance(out, tuple) else out)
                                     [0, positions].detach().clone()))
    try:
        forward(model, ids, prompt_len=ids.shape[1])
    finally:
        h.remove()
    if len(got) != 1:
        raise RuntimeError('Donor capture must fire once')
    return got[0]


def greedy(model, input_ids, *, prompt_len, max_new, eos_ids, keys=(), blocks=(), patch=None,
           explicit=True, use_cache=True, budget=lambda: None):
    """Cache retained within a single arm only. No state/cache is shared across arms."""
    import torch
    if max_new < 1 or prompt_len < 1 or prompt_len > input_ids.shape[1]:
        raise ValueError('Invalid sequence length/budget')
    if input_ids.shape[1]+max_new > model.config.max_position_embeddings:
        raise ValueError('Requested prompt plus generation exceeds model context capacity')
    seq = input_ids.clone()
    initial = seq.shape[1]
    cache = None
    generated = []
    totals = {'mask_hooks': 0, 'blocked_edges': 0, 'patch_writes': 0, 'forwards': 0}
    first_logits = None
    for step in range(max_new):
        budget()
        if use_cache or step == 0:
            q_start = 0 if step == 0 else seq.shape[1]-1
            ids = seq if step == 0 else seq[:, -1:]
            out, ev = forward(model, ids, prompt_len=prompt_len, cache=cache,
                q_start=q_start, keys=keys, blocks=blocks, patch=patch, explicit=explicit)
        else:
            # Reference recomputation: no history from another arm; repeat the same prompt patch.
            out, ev = forward(model, seq, prompt_len=prompt_len, keys=keys, blocks=blocks,
                              patch=patch, explicit=explicit)
        logits = out.logits[0, -1].float()
        if step == 0:
            first_logits = logits.detach().cpu()
        token = int(logits.argmax())
        generated.append(token)
        totals['forwards'] += 1
        for k in ev:
            totals[k] += ev[k]
        seq = torch.cat((seq, seq.new_tensor([[token]])), dim=1)
        cache = out.past_key_values if use_cache else None
        del out
        if token in eos_ids:
            break
    ended = generated[-1] in eos_ids
    return {'ids': generated, 'input_tokens': initial, 'native_prompt_tokens': prompt_len,
            'emitted_terminator': ended, 'hit_token_cap': len(generated) == max_new,
            'token_sha256': digest_ids(generated), 'events': totals,
            '_first_logits': first_logits}
