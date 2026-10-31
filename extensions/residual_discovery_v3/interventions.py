"""Targeted causal interventions. Importable without torch; requires torch to run.

Hooks capture their own module output, avoiding HF hidden-state/final-norm ambiguity.
Only batch size one is supported. Prefill-only patches require cache-enabled generation.
"""
from contextlib import contextmanager
from difflib import SequenceMatcher

def token_alignment(source, target):
    """Map target indices to source indices; do not guess unequal replacement spans."""
    pairs = {}
    for tag, a, b, c, d in SequenceMatcher(None, source, target, autojunk=False).get_opcodes():
        if tag == 'equal' or (tag == 'replace' and b-a == d-c):
            pairs.update({j: i for i,j in zip(range(a,b), range(c,d))})
    return pairs

def aligned_positions(source_ids, target_ids, target_positions):
    mapping = token_alignment(source_ids, target_ids)
    if any(p not in mapping for p in target_positions):
        raise ValueError('Span has non-bijective token alignment; do not truncate or broadcast')
    return [mapping[p] for p in target_positions], list(target_positions)

def semantic_positions(span, source_ids, target_ids, source_positions, target_positions):
    """Align unchanged semantic spans in each native encoding, not absolute indices.

    For history_end we patch ONLY the final token overlapping the history sentence;
    it can summarize prior tokens, but this is not a whole-history transplantation.
    Unchanged spans must have exactly the same token sequence in both conditions.
    """
    if span in ('final', 'history_end', 'answer', 'question', 'instruction', 'suffix'):
        a, b = list(source_positions), list(target_positions)
        if not a or not b:
            raise ValueError('Empty semantic intervention span')
        if span in ('final','history_end') and (len(a)!=1 or len(b)!=1):
            raise ValueError('Anchor intervention must contain exactly one token')
        if [source_ids[i] for i in a] != [target_ids[i] for i in b]:
            raise ValueError(f'{span}: native semantic-span token sequences differ')
        return a,b
    a,b = aligned_positions(source_ids,target_ids,target_positions)
    if a != list(source_positions):
        raise ValueError(f'{span}: mapping does not cover exactly the donor semantic span')
    return a,b

def module_for(model, component, layer):
    if component == 'embed':
        if layer != 0:
            raise ValueError('Embedding layer must be zero')
        return model.model.embed_tokens
    if not 1 <= layer <= len(model.model.layers):
        raise ValueError('Post-block layer index out of range')
    block = model.model.layers[layer-1]
    return {'resid': block, 'attn': block.self_attn,
            'mlp': block.mlp, 'head': block.self_attn.o_proj}[component]

def _tensor(output):
    return output[0] if isinstance(output, tuple) else output

def _replace(output, value):
    return (value,) + output[1:] if isinstance(output, tuple) else value

def capture(model, input_ids, component, layer, positions):
    import torch
    values = []
    module = module_for(model, component, layer)
    def save(_m, _i, out):
        values.append(_tensor(out)[0, positions, :].detach().clone())
    def save_head(_m, inputs):
        values.append(inputs[0][0, positions, :].detach().clone())
    h = (module.register_forward_pre_hook(save_head) if component == 'head'
         else module.register_forward_hook(save))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        h.remove()
    if len(values) != 1:
        raise RuntimeError(f'Expected exactly one donor capture, got {len(values)}')
    return values[0]

def embedding_replay_control(model, input_ids, baseline_tokens, generate_kwargs, *, seed=42):
    """Plumbing control: recover native generation from a same-length synthetic scaffold.

    This is NOT a high/low history transfer: all scaffold embeddings are replaced
    with the native prompt embeddings. It works even when high/low lengths differ.
    """
    import torch
    from transformers import set_seed
    positions = list(range(input_ids.shape[1]))
    values = capture(model, input_ids, 'embed', 0, positions)
    scaffold = torch.full_like(input_ids, generate_kwargs['pad_token_id'])
    set_seed(seed)
    with transplant(model, 'embed', 0, positions, values, input_ids.shape[1]) as events:
        with torch.inference_mode():
            result = model.generate(input_ids=scaffold, attention_mask=torch.ones_like(scaffold),
                                    **generate_kwargs)
    tokens = result[0, input_ids.shape[1]:].tolist()
    if tokens != list(baseline_tokens):
        raise RuntimeError('Synthetic-scaffold embedding recovery failed')
    return {'control': 'synthetic_scaffold_embedding_recovery_v2',
            'prompt_tokens': input_ids.shape[1], 'generated_tokens': len(tokens),
            'all_tokens_identical': True, 'events': events}

@contextmanager
def transplant(model, component, layer, positions, values, prompt_len, *, head=None):
    """Replace prompt positions once; retain donor influence in downstream cache only.

    `head` patches one attention head BEFORE o_proj, not one grouped-query KV head.
    KV states produced at/before a post-block patch are deliberately NOT overwritten.
    """
    module = module_for(model, component, layer)
    events = {'prefill_writes': 0, 'decode_skips': 0}
    def edit(hs):
        if hs.shape[0] != 1:
            raise ValueError('Batch size one required')
        if hs.shape[1] == 1 and events['prefill_writes']:
            events['decode_skips'] += 1
            return hs
        if hs.shape[1] != prompt_len or events['prefill_writes']:
            raise RuntimeError('Unexpected forward layout/cache mode')
        if tuple(values.shape) != (len(positions), hs.shape[-1]):
            raise ValueError('Donor shape mismatch')
        hs = hs.clone()
        if head is None:
            hs[0, positions, :] = values.to(device=hs.device, dtype=hs.dtype)
        else:
            attn = model.model.layers[layer-1].self_attn
            width = attn.head_dim
            n_heads = hs.shape[-1] // width
            if not 0 <= head < n_heads or component != 'head':
                raise ValueError('Invalid attention output head')
            cols = slice(head*width, (head+1)*width)
            hs[0, positions, cols] = values[:, cols].to(device=hs.device, dtype=hs.dtype)
        events['prefill_writes'] += 1
        return hs
    def post(_m, _i, out):
        return _replace(out, edit(_tensor(out)))
    def pre(_m, inputs):
        return (edit(inputs[0]),) + inputs[1:]
    h = (module.register_forward_pre_hook(pre) if component == 'head'
         else module.register_forward_hook(post))
    try:
        yield events
        if events['prefill_writes'] != 1:
            raise RuntimeError('Patch did not fire exactly once')
    finally:
        h.remove()
