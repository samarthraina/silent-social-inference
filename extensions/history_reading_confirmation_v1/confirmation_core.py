"""Prospective focused confirmation using the byte-identical, GPU-tested v3 backend."""
from pathlib import Path
import hashlib
import json
import sys
import collections
import random

HERE=Path(__file__).resolve().parent
BACKEND=HERE.parent/'generation_path_v1'
sys.path.insert(0,str(BACKEND))
import protocol as backend
from run import gate_proof
from analyze import verify_run as verify_pilot

ARMS=('baseline','native_baseline','generated_block','control_block')
MODELS=('qwen','llama')
ENDPOINTS=('history_gap_reduction','history_vs_control_specificity')


def load(p):return json.loads(Path(p).read_text())
def sha(data):return hashlib.sha256(data).hexdigest()
def canonical(obj):return backend.canonical(obj)


def source_hashes():
    files=sorted(list(HERE.glob('*.py'))+list(HERE.glob('*.sh')))
    return {p.name:sha(p.read_bytes()) for p in files}


def validate_protocol(p):
    expected={'version':1,'arms':list(ARMS),'models':list(MODELS),'primary_endpoints':list(ENDPOINTS),
              'primary_tests_total':4,'family_alpha':.05,'bootstrap_draws':100000,'bootstrap_seed':20261031,
              'generation_backend_protocol_version':3,'max_new_tokens':512,'seed':42,
              'reader_policy':'generated','engineering_recheck':True}
    if p!=expected:raise ValueError('Confirmation protocol differs from its predeclared definition')
    b=load(BACKEND/'protocol.json');backend.validate_protocol(b)
    if b['free_max_new']!=p['max_new_tokens'] or b['seed']!=p['seed']:raise ValueError('Backend decoding differs')


def units(tasks,engineering=False):
    return [dict(task=t['base_task_id'],kind='engineering' if engineering else 'factorial',
                 condition=c,arm=a) for t in tasks for c in backend.CONDS
            for a in (('checks',) if engineering else ARMS)]


def ordered_units(tasks):
    result=[]
    for task in tasks:
        batch=units([task]);random.Random('history-reading-confirm-v1|'+task['base_task_id']).shuffle(batch)
        result+=batch
    return result


def verify_evidence(directory,model):
    d=Path(directory)
    pilot=d/f'{model}_pilot';engineering=d/f'{model}_engineering'
    meta=load(pilot/'metadata.json')
    if meta['source_sha256']!=backend.source_hashes():raise ValueError('GPU-tested backend source changed')
    if meta['protocol_sha256']!=sha((BACKEND/'protocol.json').read_bytes()):raise ValueError('Backend protocol changed')
    if meta['plan_sha256']!=sha((BACKEND/'frozen/plan.json').read_bytes()):raise ValueError('Backend task split changed')
    proof=dict(engineering=list(gate_proof(engineering,model=model,stage='engineering',binding=meta)),
               pilot=list(gate_proof(pilot,model=model,stage='pilot',binding=meta)))
    verify_pilot(pilot) # Full native-vs-explicit/identity/score/unit checks, not a summary-only gate.
    return proof


def candidate_plan():
    old=load(BACKEND/'frozen/plan.json')
    original_split=backend.DEPENDENCY/'split.json'
    peer_split=BACKEND/'frozen/peer_splits.json'
    if sha(original_split.read_bytes())!=old['own_split_sha256'] or sha(peer_split.read_bytes())!=old['peer_split_sha256']:
        raise ValueError('Frozen exposure inputs changed')
    for rel,h in load(original_split)['input_sha256'].items():
        if sha((backend.DEPENDENCY/'reference'/rel).read_bytes())!=h:
            raise ValueError('Original frozen input changed: '+rel)
    rebuilt=backend.build_plan(load(backend.DEPENDENCY/'split.json'),load(BACKEND/'frozen/peer_splits.json'))
    for k,v in rebuilt.items():
        if old[k]!=v:raise ValueError('Task plan does not reproduce from the recorded exposure manifests')
    return {'version':1,'selection_inputs':['recorded mechanistic memberships','task ID','exact-question hash'],
            'outcomes_used_for_task_selection':False,'pilot_used_for_endpoint_selection':True,
            'backend_plan_sha256':sha((BACKEND/'frozen/plan.json').read_bytes()),
            'peer_split_sha256':sha((BACKEND/'frozen/peer_splits.json').read_bytes()),
            'tasks':old['validation'],'engineering_tasks':old['engineering'],
            'dataset_counts':dict(collections.Counter(t['dataset'] for t in old['validation'])),
            'scope':'Candidate tasks without recorded mechanistic exposure; additional team exposure must be audited. '
                    'Not a balanced benchmark sample or necessarily behaviorally unseen/semantically novel.'}


def verify_documented_audit(candidate,audit):
    path=HERE/'documented_exposure_audit.json'
    if audit.get('documented_exposure_report_sha256')!=sha(path.read_bytes()):
        raise ValueError('Documented-source audit report changed')
    report=load(path)
    if report['source_commits']!={'original':'f8010273445adcf276c094cf5b228fc7bd0acf2a','teammate':'630b1c6813a9b7578d9a7825f0ef1dd17a49faa8'}:
        raise ValueError('Source audit refers to different repository revisions')
    registry=backend.DEPENDENCY/'reference/code/data/candidates/final_olmo_answers.json'
    if report['answer_registry_sha256']!=sha(registry.read_bytes()):raise ValueError('Audited answer registry changed')
    if report['unknown_recorded_task_ids']:raise ValueError('Recorded unknown task identities require further review')
    if audit.get('claim_scope')!='held_out_relative_to_documented_sources':raise ValueError('Documented audit cannot assert global non-exposure')
    required={'original mechanistic','original context/patching','teammate post-review','own completed runs'}
    if {row['scope'] for row in report['files']}!=required:raise ValueError('Required source groups are missing')
    exposed={task for row in report['files'] for task in row['task_ids']}
    if sorted(exposed)!=report['documented_mechanistic_or_conservatively_excluded_task_ids']:
        raise ValueError('Per-file identities disagree with the exposure union')
    question={f"{t['dataset']}/{t['source_task_id']}":t['question_sha256'] for t in load(registry)['accepted']}
    q={question[t] for t in exposed}
    if sorted(q)!=report['recorded_exposure_question_sha256']:raise ValueError('Exposure question hashes do not reproduce')
    candidates={t['base_task_id'] for t in candidate['tasks']}
    if report['candidate_question_hashes']!={t['base_task_id']:t['question_sha256'] for t in candidate['tasks']}:
        raise ValueError('Audited candidate identities differ')
    if sorted(candidates&exposed)!=report['candidate_overlap']:
        raise ValueError('Recorded task-ID overlap does not reproduce')
    if [t['base_task_id'] for t in candidate['tasks'] if t['question_sha256'] in q]!=report['candidate_exact_question_overlap']:
        raise ValueError('Recorded exact-question overlap does not reproduce')
    return exposed,q


def audited_plan(candidate,audit):
    from datetime import datetime
    if candidate!=candidate_plan():raise ValueError('Candidate exposure plan changed')
    if audit.get('candidate_plan_sha256')!=sha(canonical(candidate)):raise ValueError('Exposure review binds another candidate plan')
    method=audit.get('review_method','author_exposure_review')
    documented_ids,documented_questions=set(),set()
    if method=='documented_source_audit':
        documented_ids,documented_questions=verify_documented_audit(candidate,audit)
    elif method=='author_exposure_review':
        if audit.get('author_confirms_remaining_tasks_not_used_mechanistically') is not True:
            raise ValueError('Missing author information about additional mechanistic exposure')
    else:raise ValueError('Unknown exposure-review method')
    if not audit.get('reviewer_name') or not audit.get('reviewed_utc') or not audit.get('exposure_statement'):
        raise ValueError('Exposure review needs name, date and the actual statement')
    stamp=datetime.fromisoformat(audit['reviewed_utc'].replace('Z','+00:00'))
    if stamp.tzinfo is None:raise ValueError('Exposure review date requires a time zone')
    extra=set(audit.get('additional_exposed_task_ids',[]))|documented_ids;q=set(audit.get('additional_exposed_question_sha256',[]))|documented_questions
    answers=load(backend.DEPENDENCY/'reference/code/data/candidates/final_olmo_answers.json')['accepted']
    known={f"{t['dataset']}/{t['source_task_id']}":t for t in answers}
    if extra-set(known) and not q:raise ValueError('Unknown additional task IDs require their question hashes as well')
    q|={known[t]['question_sha256'] for t in extra if t in known}
    for h in q:
        if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Invalid question SHA-256')
    remaining=[t for t in candidate['tasks'] if t['base_task_id'] not in extra and t['question_sha256'] not in q]
    if not remaining:raise ValueError('No eligible confirmation tasks remain')
    return dict(candidate,tasks=remaining,dataset_counts=dict(collections.Counter(t['dataset'] for t in remaining)),
                excluded_by_additional_audit=[t['base_task_id'] for t in candidate['tasks'] if t not in remaining],
                exposure_audit_sha256=sha(canonical(audit)))


def verify_lock(lockdir):
    d=Path(lockdir);protocol=load(d/'confirmation_protocol.json');validate_protocol(protocol)
    lock=load(d/'lock.json');plan=load(d/'plan.json');audit=load(d/'exposure_audit.json')
    if lock['source_sha256']!=source_hashes() or lock['backend_source_sha256']!=backend.source_hashes():raise ValueError('Locked source changed')
    if lock['confirmation_protocol_sha256']!=sha((d/'confirmation_protocol.json').read_bytes()):raise ValueError('Protocol snapshot changed')
    if lock['plan_sha256']!=sha((d/'plan.json').read_bytes()):raise ValueError('Plan snapshot changed')
    if lock['exposure_audit_sha256']!=sha((d/'exposure_audit.json').read_bytes()):raise ValueError('Audit snapshot changed')
    if plan!=audited_plan(load(HERE/'candidate_plan.json'),audit):raise ValueError('Locked exposure plan does not reproduce')
    return lock,plan,protocol


def verify_row(r,model):
    from study import RN
    import engine
    if r['model']!=model or r['kind']!='factorial' or r['arm'] not in ARMS:raise ValueError('Off-plan record')
    if not r['ids'] or engine.digest_ids(r['ids'])!=r['token_sha256']:raise ValueError('Token identity checksum differs')
    p=load(BACKEND/'protocol.json');n=len(r['ids']);layers=len(backend.model_blocks(p,model))
    masked=r['arm'] in ('generated_block','control_block')
    keys=r['control_keys'] if r['arm']=='control_block' else r['history_keys']
    if len(r['control_keys'])!=len(r['history_keys']) or set(r['control_keys'])&set(r['history_keys']):raise ValueError('Content-control keys differ')
    if r['events']['mask_hooks']!=n*layers*int(masked) or r['events']['blocked_edges']!=(n-1)*len(keys)*layers*int(masked):raise ValueError('Reader/mask event counts differ')
    if r['events']['patch_writes']!=0:raise ValueError('No residual patches are part of this confirmation')
    if r['mask_blocks']!=(backend.model_blocks(p,model) if masked else []):raise ValueError('Wrong masked block set')
    if r['arm']=='native_baseline' and r.get('computation_path')!='hf_native_cached':raise ValueError('Native baseline used another path')
    computed=RN.payload_from_generation(dict(r,text=r['text']),None,512)
    for k in ('analysis_score','analysis_valid','analysis_validity'):
        if r[k]!=computed[k]:raise ValueError('Score does not reproduce: '+k)


def verify_pairs(records,tasks):
    for t in tasks:
        for c in backend.CONDS:
            def get(a):return records.get(backend.unit_key(dict(task=t['base_task_id'],kind='factorial',condition=c,arm=a)))
            if get('baseline') is not None and get('generated_block') is not None:
                if get('baseline')['ids'][0]!=get('generated_block')['ids'][0]:raise ValueError('Generated-reader block changed the first prediction')
