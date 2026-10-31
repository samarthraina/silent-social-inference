"""Paired task-level analysis with matched controls, missingness bounds, and 12-test family."""
import argparse
import collections
import json
from pathlib import Path
from protocol import (load, planned_units, read_journal, unit_key, sha, PRIMARY, validate_protocol, model_blocks)


def combine(*terms):
    result=collections.defaultdict(float)
    for scale,expr in terms:
        for k,v in expr.items():result[k]+=scale*v
    return {k:v for k,v in result.items() if v!=0}


def expressions(task):
    def x(kind,c,arm):return {unit_key(dict(task=task,kind=kind,condition=c,arm=arm)):1.}
    hi,lo='self_high_10','self_low_2'
    def gap(arm):return combine((1,x('factorial',hi,arm)),(-1,x('factorial',lo,arm)))
    def score_gap(arm):return combine((1,x('score',hi,arm)),(-1,x('score',lo,arm)))
    def interaction(patch):
        # Signed donor-directed movement, averaged over high->low and low->high.
        return combine((.5,x('factorial',lo,patch+'_generated_block')),(-.5,x('factorial',lo,'generated_block')),
                       (-.5,x('factorial',lo,patch)),(.5,x('factorial',lo,'baseline')),
                       (.5,x('factorial',hi,'generated_block')),(-.5,x('factorial',hi,patch+'_generated_block')),
                       (-.5,x('factorial',hi,'baseline')),(.5,x('factorial',hi,patch)))
    return {'history_gap_reduction':combine((1,gap('baseline')),(-1,gap('generated_block'))),
            'history_vs_control_specificity':combine((1,gap('control_block')),(-1,gap('generated_block'))),
            'patch_block_interaction':interaction('cross_patch'),
            'patch_interaction_vs_l2':combine((1,interaction('cross_patch')),(-1,interaction('l2_patch'))),
            'neutral_rationale_gap_reduction':combine((1,score_gap('generic')),(-1,score_gap('no_history'))),
            'rationale_donor_effect':combine((.5,x('score',hi,hi)),(-.5,x('score',hi,lo)),
                                            (.5,x('score',lo,hi)),(-.5,x('score',lo,lo)))}


def evaluate(expr,records):
    """Same planned linear estimand for complete cases and worst-case missing bounds."""
    value=0.;lower=0.;upper=0.;missing=[]
    for key,weight in expr.items():
        row=records[key]
        if row['analysis_valid']:
            value+=weight*row['analysis_score'];lower+=weight*row['analysis_score'];upper+=weight*row['analysis_score']
        else:
            missing.append(key)
            lower+=min(0.,10*weight);upper+=max(0.,10*weight)
    return (None if missing else value),lower,upper,missing


def interval(values,strata,*,draws,alpha,seed):
    import numpy as np
    if not values:return None
    rng=np.random.default_rng(seed)
    by={s:[] for s in sorted(set(strata))}
    for v,s in zip(values,strata):by[s].append(v)
    means=np.zeros(draws)
    for group in by.values():
        g=np.asarray(group)
        means+=g[rng.integers(0,len(g),size=(draws,len(g)))].sum(axis=1)
    means/=len(values)
    return np.quantile(means,[alpha/2,1-alpha/2]).tolist()


def verify_run(directory):
    from study import RN
    d=Path(directory);meta=load(d/'metadata.json');plan=load(d/'plan.json');p=load(d/'protocol.json')
    validate_protocol(p)
    if meta['status']!='complete' or meta['stage']=='engineering':raise ValueError('Only complete pilot/validation runs can be analyzed')
    if sha((d/'plan.json').read_bytes())!=meta['plan_sha256'] or sha((d/'protocol.json').read_bytes())!=meta['protocol_sha256']:
        raise ValueError('Run snapshots do not match the frozen hashes')
    tasks=plan[meta['stage']];expected={unit_key(u) for u in planned_units(tasks,meta['stage'])}
    records=read_journal(d/'raw.jsonl',expected)
    if set(records)!=expected:raise ValueError('Incomplete raw unit coverage')
    for r in records.values():
        if r['model']!=meta['model']:raise ValueError('Raw record model differs from metadata')
        if r.get('generation_skipped'):
            if r['kind']!='score' or r['analysis_validity']!='invalid_rationale_set':raise ValueError('Unexpected missing-output placeholder')
            donors=[records[k] for k in r['rationale_dependency_units']]
            n=min([p['rationale_prefix_max_tokens']]+[len(d['rationale_content_token_ids']) for d in donors])
            if all(d['rationale_usable'] for d in donors) and n>=p['rationale_prefix_min_tokens']:
                raise ValueError('A valid rationale set was skipped')
            continue
        from engine import digest_ids
        if digest_ids(r['ids'])!=r['token_sha256']:raise ValueError('Generated token checksum differs')
        if r['kind']=='rationale':
            from protocol import rationale_valid
            if bool(rationale_valid(r['text'],r['emitted_terminator'],r['hit_token_cap']))!=r['rationale_usable']:
                raise ValueError('Rationale eligibility does not reproduce')
        if r['kind'] in ('factorial','score'):
            text=r['text'] if r['kind']=='factorial' else 'Final score: '+r['text']
            rec=RN.payload_from_generation(dict(r,text=text),None,
                    p['free_max_new'] if r['kind']=='factorial' else p['score_max_new'])
            for field in ['analysis_score','analysis_valid','analysis_validity']:
                if rec[field]!=r[field]:raise ValueError('Stored score does not reproduce: '+field)
        if r['kind']=='factorial':
            n=len(r['ids']);masked=r['arm'] in ['generated_block','control_block','cross_patch_generated_block','l2_patch_generated_block']
            nkeys=len(r['control_keys'] if r['arm']=='control_block' else r['history_keys'])
            if len(r['control_keys'])!=len(r['history_keys']) or set(r['control_keys'])&set(r['history_keys']):raise ValueError('Control-key mismatch')
            if r['events']['mask_hooks']!=n*len(model_blocks(p,meta['model']))*int(masked):raise ValueError('Mask hook coverage differs')
            if r['events']['blocked_edges']!=(n-1)*nkeys*len(model_blocks(p,meta['model']))*int(masked):raise ValueError('Wrong blocked reader boundary/edge count')
            patched=r['arm'].startswith(('cross_patch','l2_patch')) or r['arm']=='same_state_patch'
            if r['events']['patch_writes']!=int(patched):raise ValueError('Patch did not fire exactly once')
    for t in tasks:
        nonempty_lengths=set()
        for arm in ['self_high_10','self_low_2','no_history','generic','empty']:
            score_rows=[records[unit_key(dict(task=t['base_task_id'],kind='score',condition=c,arm=arm))] for c in ['self_high_10','self_low_2']]
            if not any(r.get('generation_skipped') for r in score_rows):
                if arm!='empty':nonempty_lengths.update(r['shared_rationale_tokens'] for r in score_rows)
                prefixes=[r['assistant_prefix_token_ids'] for r in score_rows]
                if prefixes[0]!=prefixes[1]:raise ValueError('Rationale prefix changed with target history')
                for r in score_rows:
                    from protocol import canonical
                    if sha(canonical(r['assistant_prefix_token_ids']))!=r['rationale_prefix_sha256']:
                        raise ValueError('Rationale prefix checksum mismatch')
                    if len(r['assistant_prefix_token_ids'])!=r['shared_rationale_tokens']:raise ValueError('Rationale prefix length mismatch')
                    if arm=='empty' and r['shared_rationale_tokens']!=0:raise ValueError('Empty prefix has tokens')
                    if arm in ['self_high_10','self_low_2','no_history']:
                        donor=records[unit_key(dict(task=t['base_task_id'],kind='rationale',condition=arm,arm='donor'))]
                        if r['assistant_prefix_token_ids']!=donor['rationale_content_token_ids'][:r['shared_rationale_tokens']]:
                            raise ValueError('Rationale donor prefix differs')
        if len(nonempty_lengths)>1:raise ValueError('Rationale donors were not matched in token length')
        for c in ['self_high_10','self_low_2']:
            def get(a):return records[unit_key(dict(task=t['base_task_id'],kind='factorial',condition=c,arm=a))]
            if get('baseline')['ids']!=get('same_state_patch')['ids']:raise ValueError('Identity control failed')
            for patch in ['baseline','cross_patch','l2_patch']:
                blocked='generated_block' if patch=='baseline' else patch+'_generated_block'
                if get(patch)['ids'][0]!=get(blocked)['ids'][0]:raise ValueError('Generated-reader block affected first output token')
    return meta,plan,p,tasks,records


def summarize(directory):
    meta,plan,p,tasks,records=verify_run(directory)
    alpha=p['family_alpha']/p['primary_tests_total']
    results={}
    for name in PRIMARY:
        valid=[];strata=[];bounds=[];miss=[]
        for t in tasks:
            v,lo,hi,m=evaluate(expressions(t['base_task_id'])[name],records)
            bounds.append((lo,hi))
            if v is not None:valid.append(v);strata.append(t['dataset'])
            else:miss.append(dict(task=t['base_task_id'],missing_units=m))
        results[name]={'complete_tasks':len(valid),'planned_tasks':len(tasks),
                       'complete_case_mean_points':sum(valid)/len(valid) if valid else None,
                       'ci_family_adjusted':interval(valid,strata,draws=p['bootstrap_draws'],alpha=alpha,seed=p['bootstrap_seed']),
                       'planned_task_mean_bounds':[sum(b[i] for b in bounds)/len(bounds) for i in range(2)],
                       'missing_tasks':miss}
    descriptive={}
    for kind in ['factorial','score']:
        for arm in sorted({r['arm'] for r in records.values() if r['kind']==kind}):
            paired=[];hi_scores=[];lo_scores=[];invalid=collections.Counter()
            for t in tasks:
                rows=[records[unit_key(dict(task=t['base_task_id'],kind=kind,condition=c,arm=arm))] for c in ['self_high_10','self_low_2']]
                if all(r['analysis_valid'] for r in rows):
                    paired.append(rows[0]['analysis_score']-rows[1]['analysis_score'])
                    hi_scores.append(rows[0]['analysis_score']);lo_scores.append(rows[1]['analysis_score'])
                invalid.update(r['analysis_validity'] for r in rows if not r['analysis_valid'])
            descriptive[kind+'|'+arm]={'complete_pairs':len(paired),'mean_high_minus_low':sum(paired)/len(paired) if paired else None,'missing_by_reason':dict(invalid),
                'mean_high_complete_pairs':sum(hi_scores)/len(hi_scores) if hi_scores else None,
                'mean_low_complete_pairs':sum(lo_scores)/len(lo_scores) if lo_scores else None}
    directional={}
    for patch in ['cross_patch','l2_patch']:
        for blocked in [False,True]:
            arm=patch+('_generated_block' if blocked else '')
            baseline='generated_block' if blocked else 'baseline'
            for c,sign in [('self_low_2',1),('self_high_10',-1)]:
                values=[]
                for t in tasks:
                    pair=[records[unit_key(dict(task=t['base_task_id'],kind='factorial',condition=c,arm=a))] for a in [arm,baseline]]
                    if all(r['analysis_valid'] for r in pair):values.append(sign*(pair[0]['analysis_score']-pair[1]['analysis_score']))
                directional[arm+'|'+c]={'n_complete':len(values),
                    'mean_donor_directed_points':sum(values)/len(values) if values else None,
                    'unchanged':sum(v==0 for v in values),'toward_donor':sum(v>0 for v in values),'away_from_donor':sum(v<0 for v in values)}
    return {'model':meta['model'],'stage':meta['stage'],'confirmatory':meta['stage']=='validation',
            'planned_tasks':len(tasks),'dataset_counts':dict(collections.Counter(t['dataset'] for t in tasks)),
            'ci':'Two-sided stratified task-bootstrap percentile intervals; Bonferroni alpha=.05/12 across both models and six endpoints. Pilot results remain exploratory.',
            'caveats':['Conditional-on-observed-tasks inference; no model/run-seed population claim.',
                       'Complete-case intervals do not correct selection from missing outputs; use planned-task bounds.',
                       'Non-history content blocking is a competing-content control, not an inert intervention.',
                       'Rationale continuation scores use a different output protocol; not proof of natural causal mediation.'],
            'primary':results,'descriptive_gaps':descriptive,'descriptive_directional_patch_movement':directional,
            'invalid_rationale_donors':sum(r['kind']=='rationale' and not r['rationale_usable'] for r in records.values())}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,action='append',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if len(a.run)!=2:raise ValueError('Supply exactly two complete runs: one per model')
    metas=[load(d/'metadata.json') for d in a.run]
    if {m['model'] for m in metas}!={'llama','qwen'}:raise ValueError('Both model families are required')
    for k in ['stage','plan_sha256','protocol_sha256','source_sha256']:
        if metas[0][k]!=metas[1][k]:raise ValueError('Runs differ in '+k)
    output={'models':[summarize(d) for d in a.run]}
    with a.out.open('x') as f:json.dump(output,f,indent=2,sort_keys=True);f.write('\n')
    print('Analysis written:',a.out)
