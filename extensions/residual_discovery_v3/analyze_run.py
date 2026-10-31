"""Analyze a complete new run. Point effects in score units; paired task bootstrap.

For validation: candidate[0] minus control[1], fixed direction by experimental condition,
not by observed per-task score sign. Never divide individual task gaps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics

def interval(values, strata, draws=20000, seed=20261030):
    if not values:
        return None
    groups = {}
    for v,s in zip(values,strata):
        groups.setdefault(s,[]).append(v)
    rng = random.Random(seed)
    means = sorted(statistics.mean([rng.choice(g) for g in groups.values() for _ in g])
                   for _ in range(draws))
    # Four primary tests: 2 models x 2 directions. Familywise alpha <= .05.
    return [means[int(.00625*draws)],means[int(.99375*draws)]]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--split',required=True,type=Path)
    p.add_argument('--candidates',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a = p.parse_args()
    meta = json.loads((a.run/'metadata.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('Incomplete run cannot enter confirmatory analysis')
    for path,key in [(a.split,'split_sha256'),(a.candidates,'candidates_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta[key]:
            raise ValueError('Analysis inputs changed')
    specs = json.loads(a.candidates.read_text())
    tasks = json.loads(a.split.read_text())['splits'][meta['phase']]
    ids = {t['base_task_id'] for t in tasks}
    records = [json.loads(l) for l in (a.run/'raw.jsonl').read_text().splitlines()]
    def spec_key(s):
        return json.dumps(s,sort_keys=True)
    lookup = {}
    for r in records:
        checksum = r.pop('payload_sha256')
        if hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest() != checksum:
            raise ValueError('Payload integrity failure')
        k = (r['task'],r['kind'],r['target'],spec_key(r.get('spec')))
        if k in lookup or r['task'] not in ids:
            raise ValueError('Duplicate or unplanned record')
        lookup[k] = r
    conds = ['self_high_10','self_low_2']
    expected = {(b,'baseline',c,spec_key(None)) for b in ids for c in conds}
    expected |= {(b,k,c,spec_key(s)) for b in ids for c in conds
                 for k in ['same_state','cross_patch'] for s in specs}
    if set(lookup) != expected:
        raise ValueError('Coverage mismatch')
    controls_checked = None
    if meta.get('engineering_protocol') == 'v2_unequal_length_embedding_replay':
        controls=[json.loads(l) for l in (a.run/'positive_controls.jsonl').read_text().splitlines()]
        seen=set()
        for control in controls:
            checksum=control.pop('payload_sha256')
            if hashlib.sha256(json.dumps(control,sort_keys=True).encode()).hexdigest()!=checksum:
                raise ValueError('Positive-control integrity failure')
            key=(control['task'],control['condition'])
            if key in seen or control.get('all_tokens_identical') is not True:
                raise ValueError('Duplicate or failed positive control')
            seen.add(key)
        if seen != {(b,c) for b in ids for c in conds}:
            raise ValueError('Positive-control coverage mismatch')
        controls_checked=len(controls)
    def score(b,kind,c,s=None):
        row = lookup[(b,kind,c,spec_key(s))]
        return row['analysis_score'] if row['analysis_valid'] else None
    output = {'phase':meta['phase'],'model':meta['model'],'planned_tasks':len(tasks),
              'engineering_positive_controls_checked':controls_checked,
              'ci_family':('98.75% percentile stratified paired-task bootstrap; 4 primary tests'
                           if meta['phase']=='validation' else
                           '98.75% exploratory intervals; not adjusted for candidate search; not confirmatory'),
              'results':[]}
    for s in specs:
        for target in conds:
            source = next(c for c in conds if c != target)
            sign = 1 if source == 'self_high_10' else -1
            effects, gaps, strata, differences, dstrata = [],[],[],[],[]
            valid_baselines = []
            for b in sorted(ids):
                src,tgt = score(b,'baseline',source),score(b,'baseline',target)
                val = score(b,'cross_patch',target,s)
                if src is not None and tgt is not None:
                    valid_baselines.append(sign*(src-tgt))
                if None not in (src,tgt,val):
                    effects.append(sign*(val-tgt));gaps.append(sign*(src-tgt));strata.append(b.split('/')[0])
                if len(specs) == 2 and s == specs[0]:
                    ctrl = score(b,'cross_patch',target,specs[1])
                    if None not in (val,ctrl):
                        differences.append(sign*(val-ctrl));dstrata.append(b.split('/')[0])
            missing = len(ids)-len(effects)
            total = sum(effects)
            row = dict(spec=s,source=source,target=target,n_complete=len(effects),missing=missing,
                       baseline_gap_mean=statistics.mean(valid_baselines) if valid_baselines else None,
                       movement_points_mean=statistics.mean(effects) if effects else None,
                       movement_ci9875=interval(effects,strata),
                       movement_missing_bounds=[(total-10*missing)/len(ids),(total+10*missing)/len(ids)],
                       ratio_of_means=sum(effects)/sum(gaps) if gaps and abs(sum(gaps))>1e-12 else None)
            if differences:
                nmiss = len(ids)-len(differences)
                row.update(paired_site_minus_control_n=len(differences),
                           paired_site_minus_control_points=statistics.mean(differences),
                           paired_site_minus_control_ci9875=interval(differences,dstrata),
                           paired_missing_bounds=[(sum(differences)-10*nmiss)/len(ids),
                                                  (sum(differences)+10*nmiss)/len(ids)])
            output['results'].append(row)
    with a.out.open('x') as f:
        json.dump(output,f,indent=2)

if __name__ == '__main__':
    main()
