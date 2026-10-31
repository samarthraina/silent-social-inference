"""All paired primary outcomes and native-path sensitivity, including nulls/missingness."""
import argparse
import collections
from pathlib import Path
import json
from confirmation_core import (ENDPOINTS,ARMS,MODELS,backend,load,sha,verify_lock,units,
                               verify_row,verify_pairs,BACKEND)
from analyze import combine,evaluate,interval


def read_run(directory):
    d=Path(directory);lock,plan,protocol=verify_lock(d);meta=load(d/'metadata.json')
    if meta['status']!='complete' or meta['phase']!='focused_confirmation':raise ValueError('Confirmation is incomplete')
    if meta['lock_sha256']!=sha((d/'lock.json').read_bytes()):raise ValueError('Lock hash differs')
    for k in ['source_sha256','backend_source_sha256','plan_sha256','confirmation_protocol_sha256']:
        if meta[k]!=lock[k]:raise ValueError('Metadata binding differs: '+k)
    expected={backend.unit_key(u) for u in units(plan['tasks'])}
    records=backend.read_journal(d/'raw.jsonl',expected)
    if set(records)!=expected:raise ValueError('Raw unit coverage incomplete')
    for row in records.values():verify_row(row,meta['model'])
    verify_pairs(records,plan['tasks'])
    eexpected={backend.unit_key(u) for u in units(plan['engineering_tasks'],engineering=True)}
    recheck=backend.read_journal(d/'engineering_recheck.jsonl',eexpected)
    if set(recheck)!=eexpected:raise ValueError('Engineering recheck incomplete')
    for r in recheck.values():backend.verify_engineering_record(r,load(BACKEND/'protocol.json'),meta['model'])
    return meta,plan,protocol,records


def expr(task,name):
    def value(c,a):return {backend.unit_key(dict(task=task,kind='factorial',condition=c,arm=a)):1.}
    def gap(a):return combine((1,value('self_high_10',a)),(-1,value('self_low_2',a)))
    if name=='history_gap_reduction':return combine((1,gap('baseline')),(-1,gap('generated_block')))
    if name=='history_vs_control_specificity':return combine((1,gap('control_block')),(-1,gap('generated_block')))
    if name=='explicit_minus_native_gap':return combine((1,gap('baseline')),(-1,gap('native_baseline')))
    raise ValueError('Unplanned estimand')


def contrast(tasks,records,name,p,primary=True):
    values=[];strata=[];missing=[];bounds=[]
    for t in tasks:
        value,lo,hi,m=evaluate(expr(t['base_task_id'],name),records);bounds.append((lo,hi))
        if value is not None:values.append(value);strata.append(t['dataset'])
        else:missing.append(dict(task=t['base_task_id'],missing_units=m))
    alpha=p['family_alpha']/p['primary_tests_total'] if primary else .05
    ci=interval(values,strata,draws=p['bootstrap_draws'],alpha=alpha,seed=p['bootstrap_seed'])
    result={'complete_tasks':len(values),'planned_tasks':len(tasks),
            'mean_points':sum(values)/len(values) if values else None,
            'ci':ci,'confidence_level':1-alpha,'family_adjusted':primary,
            'planned_task_missing_bounds':[sum(b[i] for b in bounds)/len(tasks) for i in range(2)],
            'missing_tasks':missing}
    if primary:result['positive_with_ci_and_missingness_bounds']=bool(ci and ci[0]>0 and result['planned_task_missing_bounds'][0]>0)
    return result


def describe(tasks,records):
    output={}
    for arm in ARMS:
        pairs=[];invalid=collections.Counter();datasets={}
        for t in tasks:
            rows=[records[backend.unit_key(dict(task=t['base_task_id'],kind='factorial',condition=c,arm=arm))] for c in backend.CONDS]
            invalid.update(r['analysis_validity'] for r in rows if not r['analysis_valid'])
            if all(r['analysis_valid'] for r in rows):
                pair=[r['analysis_score'] for r in rows];pairs.append(pair);datasets.setdefault(t['dataset'],[]).append(pair[0]-pair[1])
        output[arm]={'complete_pairs':len(pairs),'missing_by_reason':dict(invalid),
          'mean_high':sum(p[0] for p in pairs)/len(pairs) if pairs else None,
          'mean_low':sum(p[1] for p in pairs)/len(pairs) if pairs else None,
          'mean_gap':sum(p[0]-p[1] for p in pairs)/len(pairs) if pairs else None,
          'by_dataset':{d:{'n':len(v),'mean_gap':sum(v)/len(v)} for d,v in datasets.items()}}
    return output


def summarize(directory):
    meta,plan,p,records=read_run(directory)
    primary={name:contrast(plan['tasks'],records,name,p) for name in ENDPOINTS}
    return {'model':meta['model'],'confirmatory':True,'planned_tasks':len(plan['tasks']),
            'dataset_counts':plan['dataset_counts'],'records':len(records),
            'exposure_scope':load(Path(directory)/'lock.json').get('exposure_claim_scope'),
            'primary':primary,'descriptive_arms':describe(plan['tasks'],records),
            'native_path_sensitivity':contrast(plan['tasks'],records,'explicit_minus_native_gap',p,primary=False),
            'both_predeclared_checks_positive':all(r['positive_with_ci_and_missingness_bounds'] for r in primary.values()),
            'interpretation_limits':['Direct access route during generation; no unique layer/head or complete circuit identified.',
             'Mechanistic exposure is audited within the frozen documented/reviewer scope, not proof of global non-exposure, semantic novelty or behavioral non-exposure.',
             'The content-token control can affect grading; inspect absolute scores and raw text for degradation.',
             'Native-path sensitivity is descriptive; an interval spanning zero is not proof of equivalence.',
             'No inference about retaliation versus anchoring/calibration, real agent interactions or frontier models.']}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,action='append',required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if len(a.run)!=2:raise ValueError('Supply one completed run per model')
    metas=[load(d/'metadata.json') for d in a.run]
    if {m['model'] for m in metas}!=set(MODELS):raise ValueError('Both models are required')
    if metas[0]['lock_sha256']!=metas[1]['lock_sha256']:raise ValueError('Runs used different frozen confirmations')
    results=[summarize(d) for d in a.run]
    output={'primary_family':'Four tests: two endpoints x two models; two-sided task-stratified bootstrap, alpha=.05/4.',
            'pilot_selection_disclosed':True,'models':results}
    with a.out.open('x') as f:json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
    lines=['# Focused history-reading confirmation','',output['primary_family'],'',
           '|Model|Baseline gap|History-block gap|Content-control gap|Native gap|Both predeclared checks positive?|',
           '|---|---:|---:|---:|---:|---|']
    for m in results:
        def gap(k):
            v=m['descriptive_arms'][k]['mean_gap'];return f'{v:.3f}' if v is not None else 'missing'
        lines.append('|'+ '|'.join([m['model'],gap('baseline'),gap('generated_block'),gap('control_block'),gap('native_baseline'),str(m['both_predeclared_checks_positive'])])+'|')
    lines+=['','Do not interpret this mechanical table as a full-circuit or native-path-equivalence claim. Inspect primary intervals, missing bounds, benchmark counts and absolute scores in the JSON.']
    a.out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('Confirmation analysis:',a.out)


if __name__=='__main__':main()
