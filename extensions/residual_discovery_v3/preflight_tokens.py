"""No-weight tokenization preflight. Only engineering/discovery tasks may be read."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from interventions import semantic_positions
from run_targeted import encode, span_positions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--model',choices=['llama','qwen'],required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.repo/'extensions/context_controls/src'))
    from rba import calib as C, inputs as I
    from rba.legacy import exact_generation as eg,scaled_conditions as sc
    from transformers import AutoTokenizer
    split_bytes=(a.package/'split.json').read_bytes();split=json.loads(split_bytes)
    for rel,expected in split['input_sha256'].items():
        if hashlib.sha256((a.repo/rel).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen split input changed: '+rel)
    model=C.MODELS[a.model]
    tok=AutoTokenizer.from_pretrained(model['hf_id'],revision=model['revision'])
    tasks={f"{t['dataset']}/{t['source_task_id']}":t for t in I.load_json('answers')['accepted']}
    report={'model':a.model,'model_revision':model['revision'],'weights_loaded':False,
            'split_sha256':hashlib.sha256(split_bytes).hexdigest(),'tasks':[]}
    for phase in ['engineering','discovery']:
        specs=json.loads((a.package/f'{"smoke_v3" if phase=="engineering" else "discovery_v3"}_{a.model}.json').read_text())
        spans=sorted({s['span'] for s in specs})
        for row in split['splits'][phase]:
            t=tasks[row['base_task_id']];ids={};prompts={};offsets={};starts={}
            for cond in ['self_high_10','self_low_2']:
                prompts[cond]=sc.render_prompt(t['question'],t['answer'],sc.resolve(cond))
                native,_,offsets[cond],starts[cond]=encode(eg,tok,C.messages(a.model,prompts[cond]),model['disable_thinking'])
                ids[cond]=native[0].tolist()
            checks=[]
            for source,target in [('self_high_10','self_low_2'),('self_low_2','self_high_10')]:
                for span in spans:
                    pos={c:span_positions(prompts[c],sc.resolve(c),offsets[c],starts[c],span,
                                          question=t['question'],answer=t['answer']) for c in ids}
                    src,dst=semantic_positions(span,ids[source],ids[target],pos[source],pos[target])
                    checks.append({'source':source,'target':target,'span':span,'tokens':len(src)})
            report['tasks'].append({'phase':phase,'task':row['base_task_id'],
                                    'prompt_lengths':{k:len(v) for k,v in ids.items()},'checks':checks})
    report['status']='pass'
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(f"Token preflight PASS: {a.model}, {len(report['tasks'])} engineering/discovery tasks; validation untouched.",flush=True)

if __name__=='__main__':main()
