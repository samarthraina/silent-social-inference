"""Freeze an outcome-blind joint-exposure task plan; creates a new directory only."""
import argparse
from pathlib import Path
from protocol import DEPENDENCY, build_plan, load, sha, write_json

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--peer-split', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    plan = build_plan(load(DEPENDENCY/'split.json'), load(a.peer_split))
    plan['peer_split_sha256'] = sha(a.peer_split.read_bytes())
    a.out.mkdir(parents=True, exist_ok=False)
    write_json(a.out/'plan.json', plan)
    (a.out/'peer_splits.json').write_bytes(a.peer_split.read_bytes())
    print({k: len(plan[k]) for k in ['engineering', 'pilot', 'validation']})
    print('Validation dataset counts:', plan['validation_dataset_counts'])
