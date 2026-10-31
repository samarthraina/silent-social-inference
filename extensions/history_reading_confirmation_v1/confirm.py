"""Bounded, resumable confirmation; fresh engineering recheck before held-out inference."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import time
from confirmation_core import (HERE,BACKEND,MODELS,backend,load,sha,verify_lock,verify_evidence,
                               units,ordered_units,verify_row,verify_pairs)


class BudgetStop(RuntimeError):pass


def append_row(file,row,journal):
    import json
    row=dict(row);row['payload_sha256']=sha(backend.canonical(row))
    file.write(json.dumps(row,sort_keys=True)+'\n');file.flush();os.fsync(file.fileno())
    journal[backend.unit_key(row)]=row


def execute_validation(study,tasks,lookup,journal,file,budget,checkpoint):
    """Shared with local recovery tests; requires no synthetic mode in the GPU CLI."""
    for unit in ordered_units(tasks):
        key=backend.unit_key(unit)
        if key in journal:continue
        task=lookup[unit['task']];frozen=next(t for t in tasks if t['base_task_id']==unit['task'])
        for name in ['question','answer']:
            if sha(task[name].encode())!=frozen[name+'_sha256']:raise ValueError('Task text changed')
        budget();print('START confirmation',key,flush=True)
        row=dict(unit,model=study.key,**study.run_unit(unit,task,journal))
        verify_row(row,study.key)
        prospective=dict(journal);prospective[key]=row;verify_pairs(prospective,[frozen])
        append_row(file,row,journal);checkpoint()
        print('DONE confirmation',key,f'[{len(journal)}/{len(tasks)*8}]',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--locks',type=Path,required=True);ap.add_argument('--model',choices=MODELS,required=True)
    ap.add_argument('--evidence',type=Path,default=HERE/'evidence');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-seconds',type=float,default=10800);ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if a.max_seconds<=0:raise ValueError('Positive wall-clock bound required')
    lock,plan,protocol=verify_lock(a.locks)
    if lock['backend_protocol_sha256']!=sha((BACKEND/'protocol.json').read_bytes()):raise ValueError('Backend protocol changed')
    for m in MODELS:
        if lock['pilot_and_engineering_proofs'][m]!=verify_evidence(a.evidence,m):raise ValueError('Original evidence changed')
    expected={backend.unit_key(u) for u in units(plan['tasks'])}
    eexpected={backend.unit_key(u) for u in units(plan['engineering_tasks'],engineering=True)}
    binding={'model':a.model,'phase':'focused_confirmation','lock_sha256':sha((a.locks/'lock.json').read_bytes()),
             'plan_sha256':lock['plan_sha256'],'confirmation_protocol_sha256':lock['confirmation_protocol_sha256'],
             'source_sha256':lock['source_sha256'],'backend_source_sha256':lock['backend_source_sha256']}
    a.out.mkdir(parents=True,exist_ok=a.resume)
    gpu=open('/tmp/silent-influence-generation-path-gpu0.lock','a');fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB)
    runlock=open(a.out/'.lock','a');fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=a.locks/('started_'+a.model+'.json')
    if marker.exists():
        if load(marker)['output_directory']!=str(a.out.resolve()):raise ValueError('This confirmation already started elsewhere; restore/resume its original path')
    else:
        with marker.open('x') as f:
            import json
            json.dump({'output_directory':str(a.out.resolve()),'model':a.model},f)
    meta_path=a.out/'metadata.json'
    if a.resume:
        meta=load(meta_path)
        for k,v in binding.items():
            if meta[k]!=v:raise ValueError('Resume binding changed: '+k)
    else:
        meta=dict(binding,status='started',expected_units=len(expected),accumulated_seconds=0)
        for name in ['lock.json','plan.json','confirmation_protocol.json','exposure_audit.json']:
            shutil.copy2(a.locks/name,a.out/name)
        backend.write_json(meta_path,meta)
    records=backend.read_journal(a.out/'raw.jsonl',expected)
    recheck=backend.read_journal(a.out/'engineering_recheck.jsonl',eexpected)
    for r in records.values():verify_row(r,a.model)
    verify_pairs(records,plan['tasks'])
    bprotocol=load(BACKEND/'protocol.json')
    for r in recheck.values():backend.verify_engineering_record(r,bprotocol,a.model)
    if records and set(recheck)!=eexpected:raise ValueError('Held-out records exist without a full engineering recheck')
    if set(records)==expected and meta['status']=='complete':print('Already complete:',a.out);return
    began=time.monotonic();previous=meta.get('accumulated_seconds',0)
    def budget():
        if time.monotonic()-began>=a.max_seconds:raise BudgetStop('Wall-clock bound reached; resume with the same frozen inputs')
    def checkpoint(status='running',**extra):
        meta.update(status=status,completed_units=len(records),engineering_recheck_units=len(recheck),
                    accumulated_seconds=previous+time.monotonic()-began,**extra)
        backend.write_json(meta_path,meta)
    def stop(_s,_f):raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,stop)
    try:
        from study import Study,C,eg
        import torch,transformers
        if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
        from transformers import AutoConfig,set_seed
        set_seed(42);cfg=C.MODELS[a.model]
        backend.validate_model_config(AutoConfig.from_pretrained(cfg['hf_id'],revision=cfg['revision']),bprotocol,a.model)
        loaded=eg.load_exact_model(eg.ExactModelSpec(key=a.model,hf_id=cfg['hf_id'],revision=cfg['revision'],disable_thinking=cfg['disable_thinking']))
        backend.validate_architecture(loaded.model,bprotocol,a.model)
        if eg.check_gpu_residency(loaded.model):raise ValueError('Model not fully resident in BF16 on GPU')
        runtime={'torch':torch.__version__,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(0),
                 'model_revision':cfg['revision'],'chat_template_sha256':loaded.chat_template_sha256}
        original=load(a.evidence/f'{a.model}_pilot/metadata.json')
        for k in ['torch','transformers','model_revision','chat_template_sha256']:
            if runtime[k]!=original[k]:raise ValueError('Runtime differs from the tested pilot: '+k)
        if a.resume and any(k in meta and meta[k]!=v for k,v in runtime.items()):raise ValueError('Resume runtime/device type differs')
        meta.update(runtime);checkpoint()
        lookup={f"{t['dataset']}/{t['source_task_id']}":t for t in load(backend.DEPENDENCY/'reference/code/data/candidates/final_olmo_answers.json')['accepted']}
        study=Study(loaded,a.model,bprotocol,budget)
        with (a.out/'engineering_recheck.jsonl').open('a') as file:
            for unit in units(plan['engineering_tasks'],engineering=True):
                key=backend.unit_key(unit)
                if key in recheck:continue
                budget();print('START engineering recheck',key,flush=True)
                row=dict(unit,model=a.model,**study.run_unit(unit,lookup[unit['task']],recheck))
                # Save the actual failed check too, then stop; never rewrite it as passing.
                append_row(file,row,recheck);checkpoint();backend.verify_engineering_record(row,bprotocol,a.model)
        if set(recheck)!=eexpected:raise ValueError('Engineering recheck coverage incomplete')
        with (a.out/'raw.jsonl').open('a') as file:
            execute_validation(study,plan['tasks'],lookup,records,file,budget,checkpoint)
        if set(records)!=expected:raise ValueError('Confirmation coverage incomplete')
        checkpoint('complete');print('CONFIRMATION COMPLETE:',a.out,flush=True)
    except (BudgetStop,KeyboardInterrupt) as error:
        checkpoint('paused',last_error=str(error));raise SystemExit(75)
    except BaseException as error:
        checkpoint('failed',last_error=type(error).__name__+': '+str(error));raise
    finally:
        fcntl.flock(runlock,fcntl.LOCK_UN);runlock.close();fcntl.flock(gpu,fcntl.LOCK_UN);gpu.close()


if __name__=='__main__':main()
