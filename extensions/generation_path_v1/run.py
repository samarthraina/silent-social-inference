"""Resumable, bounded GPU runner. Engineering -> pilot -> frozen, exposure-audited validation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from protocol import (HERE, DEPENDENCY, canonical, load, planned_units, ordered_units, read_journal,
                      sha, source_hashes, unit_key, validate_protocol, write_json, model_blocks, validate_model_config, validate_architecture)


class BudgetStop(RuntimeError):
    pass


def gate_proof(path, *, model, stage, binding):
    p=Path(path);m=load(p/'metadata.json')
    if m['status']!='complete' or m['model']!=model or m['stage']!=stage:
        raise ValueError('Required '+stage+' run is not complete for this model')
    for k in ['plan_sha256','protocol_sha256','source_sha256']:
        if m[k]!=binding[k]:raise ValueError('Gate used different '+k)
    tasks=load(p/'plan.json')[stage]
    expected={unit_key(u) for u in planned_units(tasks,stage)}
    records=read_journal(p/'raw.jsonl',expected)
    if set(records)!=expected:raise ValueError('Gate raw coverage is incomplete')
    if stage=='engineering' and not all(r['all_checks_pass'] for r in records.values()):
        raise ValueError('Engineering checks failed')
    if stage=='engineering':
        for r in records.values():
            if len(set(r['equivalence_token_sha256'].values()))!=1:
                raise ValueError('Engineering token-hash identity differs')
            if r['baseline_prefill_logits_sha256']!=r['generated_block_prefill_logits_sha256']:
                raise ValueError('Generated-reader block changed prompt logits')
            if r['active_decode_events']['blocked_edges']!=len(r['history_keys'])*len(model_blocks(load(p/'protocol.json'),model)) or r['prefill_block_events']['blocked_edges']!=0:
                raise ValueError('Engineering reader-boundary checks differ')
            if not r['positive_control_max_logit_difference']>0:
                raise ValueError('Engineering active mask control had no measurable effect')
    return sha((p/'metadata.json').read_bytes()),sha((p/'raw.jsonl').read_bytes())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    ap.add_argument('--stage',choices=['engineering','pilot','validation'],required=True)
    ap.add_argument('--model',choices=['llama','qwen'],required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--engineering',type=Path)
    ap.add_argument('--lock',type=Path)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--max-seconds',type=float,required=True)
    a=ap.parse_args()
    if a.max_seconds<=0:raise ValueError('Positive wall-clock limit required')
    protocol=load(a.protocol);validate_protocol(protocol)
    plan=load(a.plan)
    if plan['outcomes_consulted'] is not False:raise ValueError('Outcome-dependent task selection is forbidden')
    tasks=plan[a.stage]
    if not tasks:raise ValueError('No planned tasks')
    if len({t['base_task_id'] for t in tasks})!=len(tasks):raise ValueError('Duplicate task ID')
    # Verify the bundled inputs against their original frozen split before using text.
    own=load(DEPENDENCY/'split.json')
    if sha((DEPENDENCY/'split.json').read_bytes())!=plan['own_split_sha256']:
        raise ValueError('Original split changed')
    for rel,h in own['input_sha256'].items():
        if sha((DEPENDENCY/'reference'/rel).read_bytes())!=h:raise ValueError('Frozen input changed: '+rel)
    binding=dict(model=a.model,stage=a.stage,plan_sha256=sha(a.plan.read_bytes()),
                 protocol_sha256=sha(a.protocol.read_bytes()),source_sha256=source_hashes())
    if a.stage!='engineering':
        if a.engineering is None:raise ValueError('An engineering gate is required')
        gate=gate_proof(a.engineering,model=a.model,stage='engineering',binding=binding)
        binding['engineering_proof']=list(gate)
    if a.stage=='validation':
        if a.lock is None:raise ValueError('Validation requires a frozen lock and exposure audit')
        lock=load(a.lock)
        for k in ['model','plan_sha256','protocol_sha256','source_sha256','engineering_proof']:
            if lock[k]!=binding[k]:raise ValueError('Validation lock differs: '+k)
        if lock.get('validation_authorized') is not True:raise ValueError('Validation exposure audit was not attested')
        binding['validation_lock_sha256']=sha(a.lock.read_bytes())
    expected={unit_key(u) for u in planned_units(tasks,a.stage)}
    a.out.mkdir(parents=True,exist_ok=a.resume)
    # Serialize every cooperating GPU runner, in addition to the per-run lock.
    gpu_lock=open('/tmp/silent-influence-generation-path-gpu0.lock','a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run_lock=open(a.out/'.lock','a');fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata=a.out/'metadata.json'
    if a.resume and metadata.exists():
        meta=load(metadata)
        for k,v in binding.items():
            if meta[k]!=v:raise ValueError('Resume binding differs: '+k)
    elif a.resume:
        raise ValueError('No metadata to resume; choose a new output directory without --resume')
    else:
        meta=dict(binding,status='started',expected_units=len(expected),accumulated_seconds=0)
        (a.out/'plan.json').write_bytes(a.plan.read_bytes());(a.out/'protocol.json').write_bytes(a.protocol.read_bytes())
        write_json(metadata,meta)
    journal=read_journal(a.out/'raw.jsonl',expected)
    if set(journal)==expected and meta['status']=='complete':
        print('Already complete:',a.out);return
    from study import Study, C, eg
    import torch
    import transformers
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required for research inference')
    from transformers import set_seed
    set_seed(protocol['seed'])
    cfg=C.MODELS[a.model]
    began=time.monotonic()
    def budget():
        if time.monotonic()-began>=a.max_seconds:raise BudgetStop('Wall-clock budget reached; resume later')
    previous_seconds=meta.get('accumulated_seconds',0)
    def save_status(status,**extra):
        meta.update(status=status,completed_units=len(journal),
                    accumulated_seconds=previous_seconds+time.monotonic()-began,**extra)
        write_json(metadata,meta)
    def interrupted(_signum,_frame):raise KeyboardInterrupt('Termination requested; saved units remain resumable')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        from transformers import AutoConfig
        pinned_config=AutoConfig.from_pretrained(cfg['hf_id'],revision=cfg['revision'])
        validate_model_config(pinned_config,protocol,a.model)
        loaded=eg.load_exact_model(eg.ExactModelSpec(key=a.model,hf_id=cfg['hf_id'],
                      revision=cfg['revision'],disable_thinking=cfg['disable_thinking']))
        if eg.check_gpu_residency(loaded.model):raise RuntimeError('Model is not fully resident in BF16 on GPU')
        validate_architecture(loaded.model,protocol,a.model)
        if getattr(loaded.model.config,'sliding_window',None) is not None and getattr(
                loaded.model.config,'use_sliding_window',False):raise ValueError('Sliding-window attention is not supported')
        runtime=dict(device=torch.cuda.get_device_name(0),torch=torch.__version__,transformers=transformers.__version__,
                    model_revision=cfg['revision'],chat_template_sha256=loaded.chat_template_sha256)
        if a.resume and any(k in meta and meta[k]!=v for k,v in runtime.items()):
            raise ValueError('Resume requires the same model/runtime/device type/chat template')
        meta.update(runtime)
        save_status('running')
        answers=load(DEPENDENCY/'reference/code/data/candidates/final_olmo_answers.json')['accepted']
        lookup={f"{t['dataset']}/{t['source_task_id']}":t for t in answers}
        study=Study(loaded,a.model,protocol,budget=budget)
        with (a.out/'raw.jsonl').open('a') as f:
            for unit in ordered_units(tasks,a.stage):
                key=unit_key(unit)
                if key in journal:continue
                task=lookup[unit['task']]
                frozen=next(t for t in tasks if t['base_task_id']==unit['task'])
                for field in ['question','answer']:
                    if sha(task[field].encode())!=frozen[field+'_sha256']:raise ValueError('Task text changed')
                budget();print('START',key,flush=True)
                payload=study.run_unit(unit,task,journal)
                record=dict(unit,model=a.model,**payload)
                record['payload_sha256']=sha(canonical(record))
                f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
                journal[key]=record;save_status('running')
                print('DONE',key,f'[{len(journal)}/{len(expected)}]',flush=True)
        if set(journal)!=expected:raise ValueError('Final journal coverage differs')
        if a.stage=='engineering':
            if not all(r['all_checks_pass'] for r in journal.values()):raise RuntimeError('Engineering gate failed; no pilot permitted')
        else:
            for t in tasks:
                for c in ('self_high_10','self_low_2'):
                    def get(arm):return journal[unit_key(dict(task=t['base_task_id'],kind='factorial',condition=c,arm=arm))]
                    if get('baseline')['ids']!=get('same_state_patch')['ids']:
                        raise RuntimeError('Same-state patch identity failed')
        save_status('complete')
        print('COMPLETE:',a.out,flush=True)
    except (BudgetStop,KeyboardInterrupt) as e:
        save_status('paused',last_error=str(e));print(str(e),file=sys.stderr);raise SystemExit(75)
    except Exception as e:
        save_status('failed',last_error=type(e).__name__+': '+str(e));raise
    finally:
        fcntl.flock(run_lock,fcntl.LOCK_UN);run_lock.close()
        fcntl.flock(gpu_lock,fcntl.LOCK_UN);gpu_lock.close()


if __name__=='__main__':main()
