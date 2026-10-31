"""Freeze a completed pilot's protocol with explicit joint-exposure attestation. No GPU calls."""
import argparse
from pathlib import Path
from protocol import load, sha, source_hashes, write_json, validate_protocol, HERE, build_plan, DEPENDENCY
from run import gate_proof

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',choices=['llama','qwen'],required=True)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,default=HERE/'protocol.json')
    ap.add_argument('--engineering',type=Path,required=True)
    ap.add_argument('--pilot',type=Path,required=True)
    ap.add_argument('--peer-split',type=Path,required=True)
    ap.add_argument('--exposure-audit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();p=load(a.protocol);validate_protocol(p)
    plan=load(a.plan);peer=load(a.peer_split)
    if sha(a.peer_split.read_bytes())!=plan['peer_split_sha256']:raise ValueError('Peer split changed; rebuild joint-exposure plan')
    reconstructed=build_plan(load(DEPENDENCY/'split.json'),peer)
    if any(plan[k]!=reconstructed[k] for k in reconstructed):raise ValueError('Plan differs from outcome-blind exposure selection')
    audit=load(a.exposure_audit)
    if audit.get('author_attests_no_additional_mechanistic_exposure') is not True or not audit.get('reviewer_name') or not audit.get('reviewed_utc'):
        raise ValueError('An author must attest to the additional-exposure audit before validation')
    if audit['plan_sha256']!=sha(a.plan.read_bytes()):raise ValueError('Exposure audit binds a different plan')
    if set(audit.get('additional_exposed_task_ids',[])) & {t['base_task_id'] for t in plan['validation']}:
        raise ValueError('Validation has additional exposed tasks; do not proceed with this plan')
    lock=dict(model=a.model,plan_sha256=sha(a.plan.read_bytes()),protocol_sha256=sha(a.protocol.read_bytes()),source_sha256=source_hashes())
    lock['engineering_proof']=list(gate_proof(a.engineering,model=a.model,stage='engineering',binding=lock))
    lock['pilot_proof']=list(gate_proof(a.pilot,model=a.model,stage='pilot',binding=lock))
    lock.update(validation_authorized=True,exposure_audit=audit,exposure_audit_sha256=sha(a.exposure_audit.read_bytes()),
                inference_family='12 primary tests across two models; no site/window search on validation')
    # Never overwrite a previously frozen confirmation lock.
    with a.out.open('x') as f:
        import json
        json.dump(lock,f,indent=2,sort_keys=True);f.write('\n')
    print('Frozen validation lock:',a.out)
