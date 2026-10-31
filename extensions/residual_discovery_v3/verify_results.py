import collections,csv,hashlib,json,statistics
from pathlib import Path
import numpy as np

PKG=Path(__file__).resolve().parent
ROOT=PKG/'results'

OUT=PKG/'analysis';OUT.mkdir(exist_ok=True)
SPLIT=json.loads((PKG/'split.json').read_text())
TASKS=[t['base_task_id'] for t in SPLIT['splits']['discovery']]
for rel, expected_hash in SPLIT['input_sha256'].items():
    assert hashlib.sha256((PKG/'reference'/rel).read_bytes()).hexdigest()==expected_hash, rel
DATASETS=['mtbench','arena_hard','wildbench']

def check_hash(row):
    x=dict(row);checksum=x.pop('payload_sha256')
    assert hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()==checksum

def ci(values,strata,seed=20261031):
    groups=[np.array([v for v,s in zip(values,strata) if s==ds]) for ds in DATASETS]
    groups=[g for g in groups if len(g)]
    rng=np.random.default_rng(seed)
    samples=np.concatenate([g[rng.integers(0,len(g),size=(20000,len(g)))] for g in groups],axis=1)
    return np.quantile(samples.mean(axis=1),[.025,.975]).tolist()

audit={'source':'Packaged completed discovery v3 records','split_sha256':hashlib.sha256((PKG/'split.json').read_bytes()).hexdigest(),
       'models':{},'validation_inference_run':False}
export=[]
for model in ['llama','qwen']:
    d=next(ROOT.glob('discovery_v3_'+model+'_*'))
    m=json.loads((d/'run/metadata.json').read_text())
    saved=json.loads((d/'summary.json').read_text())
    specs=json.loads((PKG/f'discovery_v3_{model}.json').read_text())
    rows=[json.loads(l) for l in (d/'run/raw.jsonl').read_text().splitlines()]
    assert m['status']=='complete' and m['phase']=='discovery' and m['model']==model
    assert m['calls']==m['required_calls']==3492
    assert m['split_sha256']==audit['split_sha256']
    assert m['candidates_sha256']==hashlib.sha256((PKG/f'discovery_v3_{model}.json').read_bytes()).hexdigest()
    assert m['executor_sha256']==hashlib.sha256((PKG/'run_targeted.py').read_bytes()).hexdigest()
    assert m['hooks_sha256']==hashlib.sha256((PKG/'interventions.py').read_bytes()).hexdigest()
    base={(r['task'],r['target']):r for r in rows if r['kind']=='baseline'}
    ids={};conds=['self_high_10','self_low_2']
    for r in rows:
        check_hash(r)
        key=(r['task'],r['kind'],r['target'],json.dumps(r.get('spec'),sort_keys=True))
        assert key not in ids;ids[key]=r
        if r['analysis_valid']:assert 0<=r['analysis_score']<=10
        if r['kind']!='baseline':
            assert r['source']!=r['target']
            assert r['events']['prefill_writes']==1
            assert r['events']['decode_skips']==r['output_tokens']-1
            assert len(r['source_positions'])==len(r['target_positions'])>0
            if r['kind']=='same_state':
                assert r['generated_token_ids_sha256']==base[r['task'],r['target']]['generated_token_ids_sha256']
    expected={(t,'baseline',c,json.dumps(None)) for t in TASKS for c in conds}
    expected|={(t,k,c,json.dumps(s,sort_keys=True)) for t in TASKS for c in conds for s in specs for k in ['same_state','cross_patch']}
    assert set(ids)==expected and len(rows)==1764
    assert all(r['analysis_valid'] for r in base.values())
    gaps=[base[t,'self_high_10']['analysis_score']-base[t,'self_low_2']['analysis_score'] for t in TASKS]
    strata=[t.split('/')[0] for t in TASKS]
    sm=json.loads((d/'smoke/metadata.json').read_text())
    smoke=[json.loads(l) for l in (d/'smoke/raw.jsonl').read_text().splitlines()]
    controls=[json.loads(l) for l in (d/'smoke/positive_controls.jsonl').read_text().splitlines()]
    assert sm['status']=='complete' and sm['calls']==180 and len(smoke)==84 and len(controls)==12
    sb={(r['task'],r['target']):r for r in smoke if r['kind']=='baseline'}
    for r in smoke+controls:check_hash(r)
    assert all(r['analysis_valid'] for r in smoke)
    assert all(r['generated_token_ids_sha256']==sb[r['task'],r['target']]['generated_token_ids_sha256'] for r in smoke if r['kind']=='same_state')
    assert all(r['all_tokens_identical'] and r['events']['prefill_writes']==1 for r in controls)
    assert len({(r['task'],r['condition']) for r in controls})==12
    pre=json.loads((d/'token-preflight.json').read_text())
    assert pre['status']=='pass' and {r['task'] for r in pre['tasks'] if r['phase']=='discovery'}==set(TASKS)
    cells=[]
    for s in specs:
        for source in conds:
            target=next(c for c in conds if c!=source);sign=1 if source=='self_high_10' else -1
            patch=[ids[t,'cross_patch',target,json.dumps(s,sort_keys=True)] for t in TASKS]
            valid=[r for r in patch if r['analysis_valid']]
            effects=[sign*(r['analysis_score']-base[r['task'],target]['analysis_score']) for r in valid]
            invalid=[r for r in patch if not r['analysis_valid']]
            # Tight planned-task bounds use known target score and score scale [0,10].
            lower=sum(effects);upper=sum(effects)
            for r in invalid:
                b=base[r['task'],target]['analysis_score']
                limits=sorted([sign*(0-b),sign*(10-b)])
                lower+=limits[0];upper+=limits[1]
            cell={'model':model,'layer':s['layer'],'span':s['span'],'source':source,'target':target,
                  'n_valid':len(valid),'n_missing':len(invalid),'mean_points':statistics.mean(effects),
                  'median_points':statistics.median(effects),'toward':sum(v>0 for v in effects),
                  'unchanged':sum(v==0 for v in effects),'opposite':sum(v<0 for v in effects),
                  'by_dataset':{ds:statistics.mean(sign*(r['analysis_score']-base[r['task'],target]['analysis_score'])
                      for r in valid if r['task'].startswith(ds+'/')) for ds in DATASETS},
                  'target_aware_planned_mean_bounds':[lower/18,upper/18]}
            savedrow=next(x for x in saved['results'] if x['spec']==s and x['source']==source)
            assert abs(cell['mean_points']-savedrow['movement_points_mean'])<1e-12
            assert cell['n_valid']==savedrow['n_complete']
            cell['exploratory_ci9875']=savedrow['movement_ci9875']
            cells.append(cell);export.append(cell)
    cross=[r for r in rows if r['kind']=='cross_patch' and r['analysis_valid']]
    unchanged=sum(r['analysis_score']==base[r['task'],r['target']]['analysis_score'] for r in cross)
    bounds={}
    for s in specs:
        v=[c for c in cells if c['layer']==s['layer'] and c['span']==s['span']]
        bounds[(s['layer'],s['span'])]=min(c['mean_points'] for c in v)
    best=max(bounds,key=bounds.get)
    info={'directory':d.name,'metadata':m,'sha256_raw':hashlib.sha256((d/'run/raw.jsonl').read_bytes()).hexdigest(),
          'coverage_records':len(rows),'integrity':'all payload hashes, frozen inputs and planned units match',
          'validity_counts':dict(collections.Counter(r['analysis_validity'] for r in rows)),
          'identity_checks':864,'identity_failures':0,'smoke_identity_checks':36,'positive_controls':12,
          'baseline':{'n':18,'mean_high':statistics.mean(base[t,'self_high_10']['analysis_score'] for t in TASKS),
                      'mean_low':statistics.mean(base[t,'self_low_2']['analysis_score'] for t in TASKS),
                      'gap_mean':statistics.mean(gaps),'gap_ci95':ci(gaps,strata),
                      'positive_tasks':sum(g>0 for g in gaps),'median_gap':statistics.median(gaps),
                      'by_dataset':{ds:statistics.mean(g for g,t in zip(gaps,TASKS) if t.startswith(ds+'/')) for ds in DATASETS}},
          'cross_valid':len(cross),'cross_unchanged':unchanged,'cross_unchanged_percent':100*unchanged/len(cross),
          'best_by_smaller_direction_mean':{'layer':best[0],'span':best[1],'smaller_mean_points':bounds[best]},
          'cells':cells}
    audit['models'][model]=info
(OUT/'audit_and_results.json').write_text(json.dumps(audit,indent=2)+'\n')
with (OUT/'all_patch_effects.csv').open('w') as f:
    fields=['model','layer','span','source','target','n_valid','n_missing','mean_points','median_points','toward','unchanged','opposite']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(export)
for model,info in audit['models'].items():
    print(model,json.dumps(info['baseline']), 'unchanged',info['cross_unchanged_percent'],'lead',info['best_by_smaller_direction_mean'])
print('Audit written:',OUT/'audit_and_results.json')
