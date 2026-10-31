from pathlib import Path
import json,hashlib,statistics,collections,sys,tarfile
import argparse
ap=argparse.ArgumentParser(description='Verify a downloaded confirmation backup and rebuild its paired-task analysis. No GPU inference.')
ap.add_argument('--backup-root',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
root=args.backup_root.resolve()
project=Path(__file__).resolve().parents[1]/'history_reading_confirmation_v1'
sys.path.insert(0,str(project))
import confirmation_core as core
out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
import subprocess
subprocess.run([sys.executable,str(project/'summarize_confirmation.py'),'--run',str(root/'history-confirmation-results/qwen'),'--run',str(root/'history-confirmation-results/llama'),'--out',str(out/'recomputed_summary.json')],check=True)
saved=json.loads((root/'history-confirmation-results/confirmation_summary.json').read_text())
assert saved==json.loads((out/'recomputed_summary.json').read_text())
archive=root/'history-confirmation-backup.tar.gz';assert hashlib.sha256(archive.read_bytes()).hexdigest()==(root/(archive.name+'.sha256')).read_text().split()[0]
with tarfile.open(archive,'r:gz') as tar:
 files=[m for m in tar if m.isfile()]
 for member in files:assert hashlib.sha256(tar.extractfile(member).read()).digest()==hashlib.sha256((root/member.name).read_bytes()).digest()
report={'archive_sha256_verified':True,'extracted_files_verified':len(files),'saved_summary_exactly_reproduced':True,
        'frozen_before_run_lock_sha256':hashlib.sha256((project/'preregistered/lock.json').read_bytes()).hexdigest(),
        'models':{},'interpretation':'Held out relative to documented mechanistic experiments; all 44 tasks now have confirmation exposure and must not be reused as a new tuning holdout.'}
exports=[]
for model in ['llama','qwen']:
 d=root/'history-confirmation-results'/model;meta=json.loads((d/'metadata.json').read_text());rows=[json.loads(l) for l in (d/'raw.jsonl').read_text().splitlines()]
 assert hashlib.sha256((d/'lock.json').read_bytes()).hexdigest()==report['frozen_before_run_lock_sha256']
 assert meta['lock_sha256']==report['frozen_before_run_lock_sha256']
 for name in ['plan.json','confirmation_protocol.json','exposure_audit.json']:
  assert (d/name).read_bytes()==(project/'preregistered'/name).read_bytes(),name
 for name,h in meta['backend_source_sha256'].items():assert hashlib.sha256((project.parent/name).read_bytes()).hexdigest()==h,name
 for name,h in meta['source_sha256'].items():assert hashlib.sha256((project/name).read_bytes()).hexdigest()==h,name
 erows=[json.loads(l) for l in (d/'engineering_recheck.jsonl').read_text().splitlines()]
 assert len(erows)==6
 for r in erows:core.backend.verify_engineering_record(r,json.loads((core.BACKEND/'protocol.json').read_text()),model)
 tasks=sorted({r['task'] for r in rows});lookup={(r['task'],r['condition'],r['arm']):r for r in rows}
 arms=core.ARMS;conds=core.backend.CONDS
 common=[t for t in tasks if all(lookup[t,c,a]['analysis_valid'] for c in conds for a in arms)]
 means={a:{'high':statistics.mean(lookup[t,conds[0],a]['analysis_score'] for t in common),
           'low':statistics.mean(lookup[t,conds[1],a]['analysis_score'] for t in common),
           'gap':statistics.mean(lookup[t,conds[0],a]['analysis_score']-lookup[t,conds[1],a]['analysis_score'] for t in common)} for a in arms}
 reductions=[(lookup[t,conds[0],'baseline']['analysis_score']-lookup[t,conds[1],'baseline']['analysis_score'])-
             (lookup[t,conds[0],'generated_block']['analysis_score']-lookup[t,conds[1],'generated_block']['analysis_score']) for t in common]
 for task in tasks:
  row={'model':model,'task':task,'all_arm_complete':task in common}
  for arm in arms:
   h,l=[lookup[task,c,arm] for c in conds]
   row[arm+'_high']=h['analysis_score'];row[arm+'_low']=l['analysis_score']
   row[arm+'_gap']=h['analysis_score']-l['analysis_score'] if h['analysis_valid'] and l['analysis_valid'] else None
  exports.append(row)
 info={'status':meta['status'],'records':len(rows),'engineering_rechecks':len(erows),'source_hashes_match':True,
       'prospective_plan_protocol_audit_lock_exact_match':True,'elapsed_minutes':meta['accumulated_seconds']/60,
       'planned_tasks':len(tasks),'common_complete_tasks':len(common),'paired_arm_means':means,
       'paired_gap_reduction_percent':100*(means['baseline']['gap']-means['generated_block']['gap'])/means['baseline']['gap'],
       'task_gap_reduced':sum(v>0 for v in reductions),'task_gap_unchanged':sum(v==0 for v in reductions),'task_gap_increased':sum(v<0 for v in reductions),
       'validity_counts':dict(collections.Counter(r['analysis_validity'] for r in rows)),
       'invalid_tasks':sorted({r['task'] for r in rows if not r['analysis_valid']}),
       'first_token_boundary_checks':sum(lookup[t,c,'baseline']['ids'][0]==lookup[t,c,'generated_block']['ids'][0] for t in tasks for c in conds)}
 groups={}
 for r in rows:groups.setdefault((r['task'],r['condition']),[]).append(r)
 for key,group in groups.items():
  assert len(group)==4 and len({str(r['input_token_ids']) for r in group})==1,key
  assert all(r['max_new_tokens']==512 for r in group),key
 info['identical_prompt_token_checks_across_all_four_arms']=len(groups)
 report['models'][model]=info
(out/'integrity_and_paired_results.json').write_text(json.dumps(report,indent=2)+'\n')
import csv
with (out/'task_level_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(exports[0]),lineterminator='\n');w.writeheader();w.writerows(exports)
registry={'study':'completed focused history-reading confirmation','task_ids':tasks,
          'task_id_to_question_sha256':{t['base_task_id']:t['question_sha256'] for t in json.loads((project/'preregistered/plan.json').read_text())['tasks']},
          'status':'These tasks are now mechanistically exposed. Do not call them a new untouched confirmation set in later tuning.'}
(out/'post_confirmation_exposure_registry.json').write_text(json.dumps(registry,indent=2)+'\n')
print(json.dumps({m:{k:v for k,v in r.items() if k not in ['paired_arm_means']} for m,r in report['models'].items()},indent=2))
