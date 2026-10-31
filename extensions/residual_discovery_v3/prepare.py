"""Freeze task-ID-only splits. Standard library; never loads weights or scores."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

SEED = 'silent-influence-token-path-v1-20261030'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare(repo):
    excluded = set()
    evidence = {}
    def read(rel):
        p = repo / rel
        evidence[rel] = digest(p)
        return json.loads(p.read_text())
    answers = read('code/data/candidates/final_olmo_answers.json')['accepted']
    for name in ['mech_task_manifest', 'mech_ext72_manifest', 'crossmodel_discovery_12']:
        obj = read(f'mechanistic/manifest/{name}.json')
        for group in ['discovery', 'validation', 'tasks']:
            for t in obj.get(group, []):
                excluded.add(f"{t['dataset']}/{t['task_id']}")
    for t in read('extensions/context_controls/artifacts/sample/sample_manifest_v1.json')['selected']:
        excluded.add(t['base_task_id'])
    rel = 'extensions/context_controls/artifacts/engineering/timing_gate_manifest_v1.jsonl'
    evidence[rel] = digest(repo / rel)
    for line in (repo / rel).read_text().splitlines():
        excluded.add(json.loads(line)['base_task_id'])
    excluded.update(['arena_hard/4c7d8a034c4e40f6', 'arena_hard/2da7e53308a547d6'])
    # Prevent identical questions appearing on both sides of a split, or duplicating
    # a previously intervened-on task under a second ID.
    excluded_questions = {t['question_sha256'] for t in answers
                          if f"{t['dataset']}/{t['source_task_id']}" in excluded}
    seen_questions = set(excluded_questions)
    split = {k: [] for k in ['engineering', 'discovery', 'validation']}
    pools = {}
    for ds in ['mtbench', 'arena_hard', 'wildbench']:
        candidates = sorted([t for t in answers if t['dataset'] == ds],
                            key=lambda t: hashlib.sha256(
                                f"{SEED}|{ds}/{t['source_task_id']}".encode()).hexdigest())
        candidates = [t for t in candidates
                      if f"{ds}/{t['source_task_id']}" not in excluded]
        unique = []
        for t in candidates:
            if t['question_sha256'] not in seen_questions:
                unique.append(t)
                seen_questions.add(t['question_sha256'])
        pools[ds] = len(unique)
        if len(unique) < 32:
            raise ValueError(f'{ds}: need 32 unique eligible questions, got {len(unique)}')
        offset = 0
        for phase, n in [('engineering', 2), ('discovery', 6), ('validation', 24)]:
            for t in unique[offset:offset+n]:
                split[phase].append({
                    'base_task_id': f"{ds}/{t['source_task_id']}",
                    'dataset': ds, 'question_sha256': t['question_sha256'],
                    'answer_sha256': t['answer_sha256'],
                    'prior_behavioral_use': t.get('prior_use_phases', [])})
            offset += n
    return dict(version=1, seed=SEED, source_commit='f8010273445adcf276c094cf5b228fc7bd0acf2a',
                selection_inputs=['task_id', 'dataset', 'question identity', 'prior experiment membership'],
                outcomes_used=False, input_sha256=evidence,
                scope='Mechanistically held out relative to recorded manifests; not necessarily behaviorally unseen. '
                      'External/unrecorded exposure and semantic duplicates require author audit.',
                excluded_ids=sorted(excluded), eligible_pool=pools, splits=split)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    obj = prepare(args.repo)
    data = (json.dumps(obj, indent=2, sort_keys=True) + '\n').encode()
    with args.out.open('xb') as f:
        f.write(data)
    args.out.with_suffix('.sha256').write_text(hashlib.sha256(data).hexdigest() + '\n')
    print(json.dumps({'split_counts': {k: len(v) for k,v in obj['splits'].items()},
                      'eligible_pool': obj['eligible_pool']}))
