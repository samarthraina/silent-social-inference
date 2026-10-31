"""Bounded GPU executor for explicitly supplied candidate interventions.

Example candidates.json: [{"component":"resid","layer":24,"span":"history"}]
No search/selection on validation is allowed. Run validation once after locking candidates.
GPU loading/inference has NOT been exercised in the authoring environment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from interventions import semantic_positions, capture, transplant, embedding_replay_control

def sha(data):
    return hashlib.sha256(data).hexdigest()

def encode(eg, tok, messages, thinking):
    native = eg.encode_chat(tok, messages, disable_thinking=thinking)['input_ids']
    kw = {'tokenize': False, 'add_generation_prompt': True}
    if thinking:
        kw['enable_thinking'] = False
    rendered = tok.apply_chat_template(messages, **kw)
    offsets = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    if offsets['input_ids'] != native[0].tolist():
        raise RuntimeError('Offset tokenizer does not reproduce native chat token IDs')
    # Only one user message; locate the entire prompt rather than searching task text.
    prompt = messages[-1]['content']
    start = rendered.index(prompt)
    if rendered.count(prompt) != 1:
        raise RuntimeError('Ambiguous prompt location')
    return native, rendered, offsets['offset_mapping'], start

def span_positions(prompt, condition, offsets, start, span, question=None, answer=None):
    if span == 'all':
        return list(range(len(offsets)))
    if span == 'final':
        return [len(offsets)-1]
    history_start = prompt.index(condition.added_context)
    history_end = history_start + len(condition.added_context)
    q = history_end + 2
    # Offsets derive from the frozen layout, never a marker inside task content.
    from rba.legacy.scaled_conditions import SCORE_LINE
    q += len(SCORE_LINE) + len('\n\nQUESTION:\n')
    a = len(prompt)-len(answer) if answer is not None else prompt.index('\nCONTENT TO JUDGE:\n', q)+len('\nCONTENT TO JUDGE:\n')
    if question is not None and prompt[q:q+len(question)] != question:
        raise ValueError('Question span does not match renderer')
    if span in ('history','history_end'):
        lo, hi = history_start, history_end
    elif span == 'rating':
        lo, hi = history_end-len('10/10.' if condition.name == 'self_high_10' else '2/10.'), history_end-1
    elif span == 'instruction':
        lo, hi = history_end, q-len('\nQUESTION:\n')
    elif span == 'question':
        lo, hi = q, a-len('\nCONTENT TO JUDGE:\n')
    elif span == 'answer':
        lo, hi = a, len(prompt)
    elif span == 'suffix':
        lo, hi = len(prompt), max(e for _, e in offsets)-start
    else:
        raise ValueError(f'Unknown span {span}')
    pos = [i for i,(s,e) in enumerate(offsets) if e > start+lo and s < start+hi]
    if not pos:
        raise ValueError(f'Empty span: {span}')
    return pos[-1:] if span == 'history_end' else pos

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--split', required=True, type=Path)
    p.add_argument('--candidates', required=True, type=Path)
    p.add_argument('--phase', choices=['engineering','discovery','validation'], required=True)
    p.add_argument('--model', choices=['llama','qwen'], required=True)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--max-calls', required=True, type=int)
    p.add_argument('--max-seconds', required=True, type=float)
    p.add_argument('--lock', type=Path, help='Required validation lock binding split and candidates hashes')
    args = p.parse_args()
    split_bytes, candidate_bytes = args.split.read_bytes(), args.candidates.read_bytes()
    split, specs = json.loads(split_bytes), json.loads(candidate_bytes)
    if not specs or len({json.dumps(s,sort_keys=True) for s in specs}) != len(specs):
        raise ValueError('Candidates must be nonempty and unique')
    if args.phase == 'validation':
        if args.lock is None or len(specs) != 2:
            raise ValueError('Validation requires one candidate followed by a fixed L2 residual control')
        if specs[1] != {'component':'resid','layer':2,'span':specs[0]['span']} or specs[0] == specs[1]:
            raise ValueError('Validation control must be L2 at the selected token span')
        lock = json.loads(args.lock.read_text())
        expected = {'model': args.model, 'split_sha256': sha(split_bytes),
                    'candidates_sha256': sha(candidate_bytes)}
        if any(lock.get(k) != v for k,v in expected.items()):
            raise ValueError('Validation lock hash/model mismatch')
    sys.path.insert(0, str(args.repo/'extensions/context_controls/src'))
    from rba import calib as C, inputs as I, runner as RN
    from rba.legacy import exact_generation as eg, scaled_conditions as sc
    for name, expected_hash in split['input_sha256'].items():
        if sha((args.repo/name).read_bytes()) != expected_hash:
            raise ValueError(f'Split input changed: {name}')
    tasks = {f"{t['dataset']}/{t['source_task_id']}": t for t in I.load_json('answers')['accepted']}
    # 2 baselines and 4 generations + 4 capture forwards per candidate/task.
    replay_calls = 4 if args.phase == 'engineering' else 0
    required_calls = len(split['splits'][args.phase])*(2+8*len(specs)+replay_calls)
    if args.max_calls < required_calls or args.max_seconds <= 0:
        raise ValueError(f'Need >= {required_calls} bounded calls; no partial task subset allowed')
    args.out.mkdir(parents=True, exist_ok=False)
    began, calls = time.monotonic(), 0
    def budget():
        nonlocal calls
        if calls >= args.max_calls or time.monotonic()-began >= args.max_seconds:
            raise RuntimeError('Execution bound reached; do not treat partial output as confirmatory')
        calls += 1
    meta = {'phase': args.phase, 'model': args.model, 'split_sha256': sha(split_bytes),
            'candidates_sha256': sha(candidate_bytes), 'required_calls': required_calls,
            'status': 'started', 'executor_sha256': sha(Path(__file__).read_bytes()),
            'hooks_sha256': sha(Path(__file__).with_name('interventions.py').read_bytes())}
    if args.phase == 'engineering':
        meta['engineering_protocol'] = 'v2_unequal_length_embedding_replay'
    meta['alignment_protocol'] = 'v3_native_semantic_spans'
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    m = C.MODELS[args.model]
    # Uses the published pinned loader, bf16/SDPA, without renting any resource.
    # Account/provider authorization and shutdown are the operator's responsibility.
    import torch
    import transformers
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    loaded = eg.load_exact_model(eg.ExactModelSpec(key=args.model,hf_id=m['hf_id'],
                                  revision=m['revision'],disable_thinking=m['disable_thinking']))
    model, tok = loaded.model, loaded.tokenizer
    residency_errors = eg.check_gpu_residency(model)
    if residency_errors:
        raise RuntimeError('GPU residency check failed: '+ '; '.join(residency_errors))
    meta.update(torch=torch.__version__,transformers=transformers.__version__,
                device=torch.cuda.get_device_name(0), model_revision=m['revision'],
                chat_template_sha256=loaded.chat_template_sha256)
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))
    conditions = ('self_high_10','self_low_2')
    def generate(messages):
        budget()
        return eg.generate_exact(loaded, messages, max_new_tokens=512,seed=42,use_cache=True)
    with (args.out/'raw.jsonl').open('x') as f:
        def write(row):
            row['payload_sha256'] = sha(json.dumps(row,sort_keys=True).encode())
            f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush()
        for task_record in split['splits'][args.phase]:
            b = task_record['base_task_id']; t = tasks[b]
            print(f"START {args.phase} {args.model} {b}",flush=True)
            for field in ['question','answer']:
                if sha(t[field].encode()) != task_record[field+'_sha256']:
                    raise ValueError('Task content hash mismatch')
            messages, prompts, enc, offsets, starts, base = {}, {}, {}, {}, {}, {}
            for cond in conditions:
                prompts[cond] = sc.render_prompt(t['question'],t['answer'],sc.resolve(cond))
                messages[cond] = C.messages(args.model,prompts[cond])
                ids, _, offsets[cond], starts[cond] = encode(eg,tok,messages[cond],m['disable_thinking'])
                enc[cond] = ids.to(model.device)
                gen = generate(messages[cond]); base[cond] = gen
                write(dict(task=b, kind='baseline', target=cond,
                           input_sha256=sha(str(ids[0].tolist()).encode()),
                           **RN.payload_from_generation(gen,None,512)))
            if args.phase == 'engineering':
                # Native high/low encodings may have different lengths. Test full
                # embedding recovery against a synthetic equal-length scaffold,
                # recorded separately so it cannot be mistaken for valence transfer.
                for cond in conditions:
                    budget(); budget()  # one donor capture plus one generation
                    control = embedding_replay_control(model,enc[cond],
                        base[cond]['generated_token_ids_raw'],
                        eg.build_generate_kwargs(loaded.spec,max_new_tokens=512,
                            pad_token_id=tok.eos_token_id,use_cache=True))
                    control.update(task=b,condition=cond)
                    control['payload_sha256'] = sha(json.dumps(control,sort_keys=True).encode())
                    with (args.out/'positive_controls.jsonl').open('a') as cf:
                        cf.write(json.dumps(control,sort_keys=True)+'\n')
            for spec in specs:
                comp, layer, span = spec['component'], spec['layer'], spec['span']
                for source,target in [conditions,conditions[::-1]]:
                    src = span_positions(prompts[source],sc.resolve(source),offsets[source],starts[source],span,
                                         question=t['question'],answer=t['answer'])
                    dst = span_positions(prompts[target],sc.resolve(target),offsets[target],starts[target],span,
                                         question=t['question'],answer=t['answer'])
                    src,dst = semantic_positions(span,enc[source][0].tolist(),enc[target][0].tolist(),src,dst)
                    if span == 'all' and (len(src)!=enc[source].shape[1] or src!=list(range(len(src)))):
                        raise ValueError('All-token positive control needs equal-length bijective prompts')
                    budget(); donor = capture(model,enc[source],comp,layer,src)
                    budget(); identity = capture(model,enc[target],comp,layer,dst)
                    for kind,values in [('same_state',identity),('cross_patch',donor)]:
                        with transplant(model,comp,layer,dst,values,enc[target].shape[1],head=spec.get('head')) as events:
                            gen = generate(messages[target])
                        if kind == 'same_state' and gen['generated_token_ids_sha256'] != base[target]['generated_token_ids_sha256']:
                            raise RuntimeError('Identity control failed; stop before interpreting effects')
                        if comp == 'embed' and span == 'all' and kind == 'cross_patch':
                            if gen['generated_token_ids_sha256'] != base[source]['generated_token_ids_sha256']:
                                raise RuntimeError('Embedding full-prompt recovery control failed')
                        write(dict(task=b,kind=kind,source=source,target=target,spec=spec,
                                   source_positions=src,target_positions=dst,events=events,
                                   **RN.payload_from_generation(gen,None,512)))
            print(f"DONE {args.phase} {args.model} {b}; elapsed "
                  f"{(time.monotonic()-began)/60:.1f} min",flush=True)
    meta.update(status='complete',calls=calls,elapsed_seconds=time.monotonic()-began)
    (args.out/'metadata.json').write_text(json.dumps(meta,indent=2))

if __name__ == '__main__':
    main()
