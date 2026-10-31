"""Freeze prospective four-arm confirmation after author review of task exposure."""
import argparse
from pathlib import Path
import shutil
from confirmation_core import (HERE,BACKEND,MODELS,load,sha,source_hashes,validate_protocol,
                               candidate_plan,audited_plan,verify_evidence,backend)


def freeze(audit_path,evidence,out):
    protocol=load(HERE/'confirmation_protocol.json');validate_protocol(protocol)
    audit=load(audit_path);plan=audited_plan(load(HERE/'candidate_plan.json'),audit)
    proofs={m:verify_evidence(evidence,m) for m in MODELS}
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(HERE/'confirmation_protocol.json',out/'confirmation_protocol.json')
    backend.write_json(out/'plan.json',plan);shutil.copy2(audit_path,out/'exposure_audit.json')
    lock={'version':1,'protocol_scope':'Focused confirmation selected on the reported discovery pilot; all held-out outcomes unobserved',
          'source_sha256':source_hashes(),'backend_source_sha256':backend.source_hashes(),
          'backend_protocol_sha256':sha((BACKEND/'protocol.json').read_bytes()),
          'confirmation_protocol_sha256':sha((out/'confirmation_protocol.json').read_bytes()),
          'plan_sha256':sha((out/'plan.json').read_bytes()),'exposure_audit_sha256':sha((out/'exposure_audit.json').read_bytes()),
          'pilot_and_engineering_proofs':proofs,'planned_units_per_model':len(plan['tasks'])*8,
          'planned_tasks':len(plan['tasks']),'primary_tests_total':4,'outcomes_observed_before_freeze':False}
    backend.write_json(out/'lock.json',lock)
    print('Frozen prospective confirmation:',out,'tasks:',len(plan['tasks']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=HERE/'evidence')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();freeze(a.audit,a.evidence,a.out)
