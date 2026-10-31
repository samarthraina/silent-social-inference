from pathlib import Path
import json,csv,hashlib,collections,re
import argparse
ap=argparse.ArgumentParser(description='Extract task identities only from documented mechanistic sources; never selects tasks using outcomes.')
ap.add_argument('--original-repo',type=Path,required=True)
ap.add_argument('--teammate-repo',type=Path,required=True)
ap.add_argument('--own-backup-root',type=Path,required=True)
ap.add_argument('--candidate-plan',type=Path,default=Path(__file__).with_name('candidate_plan.json'))
ap.add_argument('--out',type=Path,required=True)
args=ap.parse_args()
paper=args.original_repo;peer=args.teammate_repo;my=args.own_backup_root
plan=json.loads(args.candidate_plan.read_text())
candidates={t['base_task_id'] for t in plan['tasks']}
ID=re.compile(r'^(mtbench|arena_hard|wildbench)/[^\s]+$')
LISTS={'tasks','task_ids','base_task_ids','discovery','validation','D','V','selected','splits'}
SCALARS={'task','task_id','source_task_id','base_task_id'}
def identities(obj,context=None):
 ids=set()
 if isinstance(obj,dict):
  dataset=obj.get('dataset');tid=obj.get('task_id',obj.get('source_task_id'))
  if dataset in {'mtbench','arena_hard','wildbench'} and isinstance(tid,(int,str)):
   ids.add(f'{dataset}/{tid}')
  for k,v in obj.items():
   if k in SCALARS and isinstance(v,str) and ID.fullmatch(v):ids.add(v)
   elif isinstance(v,(dict,list)):ids|=identities(v,k)
 elif isinstance(obj,list):
  for v in obj:
   if isinstance(v,(dict,list)):ids|=identities(v,context)
   elif context in LISTS and isinstance(v,str) and ID.fullmatch(v):ids.add(v)
 return ids
sources=[]
def add(p,scope,label):
 if p.is_file() and p.suffix in ['.json','.jsonl','.csv']:sources.append((p,scope,label))
for sub in ['mechanistic/manifest','results/mechanistic']:
 for p in (paper/sub).rglob('*'):add(p,'original mechanistic',str(p.relative_to(paper)))
for sub in ['extensions/context_controls/matched_history_patching','extensions/context_controls/matched_history_patching_qwen','extensions/context_controls/artifacts/engineering','extensions/context_controls/artifacts/sample','extensions/context_controls/results/raw/study_llama','extensions/context_controls/results/raw/study_qwen']:
 for p in (paper/sub).rglob('*'):add(p,'original context/patching',str(p.relative_to(paper)))
for p in (peer/'results/mech_v2/results').rglob('*.jsonl'):add(p,'teammate post-review',str(p.relative_to(peer)))
add(peer/'data/mech_v2/splits_v1.json','teammate post-review','data/mech_v2/splits_v1.json')
for sub in ['server-results','generation-path-v3-backup/generation-path-results-v2','generation-path-v3-backup/generation-path-results-v3','generation-path-v3-backup/generation-numerics-audit']:
 for p in (my/sub).rglob('*.jsonl'):add(p,'own completed runs',str(p.relative_to(my)))
report={'scope':'Task identity extraction only; no score or intervention outcome is used for task selection.',
 'source_commits':{'original':'f8010273445adcf276c094cf5b228fc7bd0acf2a','teammate':'630b1c6813a9b7578d9a7825f0ef1dd17a49faa8'},
 'candidate_count':len(candidates),'files':[],'candidate_overlap':[]}
union=set();by_scope=collections.defaultdict(set)
for p,scope,label in sorted(set(sources),key=lambda item:(item[1],item[2])):
 data=p.read_bytes();ids=set();n=0
 if p.suffix=='.json':ids|=identities(json.loads(data));n=1
 elif p.suffix=='.jsonl':
  for line in data.splitlines():
   if line.strip():ids|=identities(json.loads(line));n+=1
 else:
  with p.open(newline='') as f:
   for row in csv.DictReader(f):ids|=identities(row);n+=1
 union|=ids;by_scope[scope]|=ids
 report['files'].append({'scope':scope,'path':label,'sha256':hashlib.sha256(data).hexdigest(),'rows_or_documents':n,'task_ids':sorted(ids),'candidate_overlap':sorted(candidates&ids)})
report['documented_mechanistic_or_conservatively_excluded_task_ids']=sorted(union)
report['candidate_overlap']=sorted(candidates&union)
report['unique_tasks_by_scope']={k:len(v) for k,v in by_scope.items()}
report['unexposed_candidate_ids']=sorted(candidates-union)
answers=json.loads((paper/'code/data/candidates/final_olmo_answers.json').read_text())['accepted']
question={f"{t['dataset']}/{t['source_task_id']}":t['question_sha256'] for t in answers}
known_exposed={question[t] for t in union if t in question}
report['candidate_exact_question_overlap']=[t['base_task_id'] for t in plan['tasks'] if t['question_sha256'] in known_exposed]
report['candidate_question_hashes']={t['base_task_id']:t['question_sha256'] for t in plan['tasks']}
report['recorded_exposure_question_sha256']=sorted(known_exposed)
report['unknown_recorded_task_ids']=sorted(union-set(question))
report['answer_registry_sha256']=hashlib.sha256((paper/'code/data/candidates/final_olmo_answers.json').read_bytes()).hexdigest()
report['claim_scope']='Held out relative to identified task records in the original repository, teammate branch and the author backups audited here. No claim about unrecorded external experiments, semantic novelty, behavioral non-exposure or pretraining.'
with args.out.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
print('Audited files',len(report['files']),'recorded task union',len(union),'by scope',report['unique_tasks_by_scope'])
print('Candidate overlaps:',report['candidate_overlap'])
for r in report['files']:
 if r['candidate_overlap']:print(r['scope'],r['path'],r['candidate_overlap'])
