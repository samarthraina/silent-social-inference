import collections
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import engine
from protocol import (HERE, DEPENDENCY, ARMS, CONDS, canonical, sha, unit_key, load,
                      validate_protocol, build_plan, planned_units, ordered_units,
                      read_journal, rationale_valid, validate_architecture)
from analyze import expressions, evaluate, interval


class PureTests(unittest.TestCase):
    def test_protocol_is_predeclared(self):
        p=load(HERE/'protocol.json');validate_protocol(p)
        p['mask_blocks']=[1]
        with self.assertRaises(ValueError):validate_protocol(p)

    def test_plan_exposure_and_coverage(self):
        own=load(DEPENDENCY/'split.json');peer=load(HERE/'frozen/peer_splits.json')
        p=build_plan(own,peer)
        self.assertEqual([len(p[k]) for k in ['engineering','pilot','validation']],[3,12,44])
        self.assertEqual(p['validation_dataset_counts'].get('mtbench',0),1)
        ids={t['base_task_id'] for t in p['validation']}
        self.assertFalse(ids & (set(peer['splits']['D'])|set(peer['splits']['V'])))
        self.assertFalse(ids & {t['base_task_id'] for k in ['engineering','discovery'] for t in own['splits'][k]})
        units=planned_units(p['pilot'],'pilot')
        self.assertEqual(len(units),348)
        self.assertEqual(len({unit_key(u) for u in units}),348)
        self.assertEqual({unit_key(u) for u in ordered_units(p['pilot'],'pilot')},{unit_key(u) for u in units})
        for task in p['pilot']:
            seq=[u for u in ordered_units([task],'pilot')]
            self.assertLess(max(i for i,u in enumerate(seq) if u['kind']=='rationale'),min(i for i,u in enumerate(seq) if u['kind']=='score'))

    def test_invalid_journal_never_resumes_silently(self):
        u=dict(task='a/1',kind='factorial',condition='self_high_10',arm='baseline',analysis_score=4.)
        u['payload_sha256']=sha(canonical(u));key=unit_key(u)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.jsonl';p.write_text(json.dumps(u)+'\n')
            self.assertEqual(len(read_journal(p,{key})),1)
            p.write_text(json.dumps(u)+'\n'+json.dumps(u)+'\n')
            with self.assertRaises(ValueError):read_journal(p,{key})
            p.write_text(json.dumps(u))
            with self.assertRaises(ValueError):read_journal(p,{key})
            u['analysis_score']=7.;p.write_text(json.dumps(u)+'\n')
            with self.assertRaises(ValueError):read_journal(p,{key})

    def test_rationale_cannot_smuggle_a_rating(self):
        self.assertTrue(rationale_valid('The response explains the main issue clearly.',True,False))
        for text in ['I give this 8/10.','Score: 7','This is 2 out of 10.']:
            self.assertFalse(rationale_valid(text,True,False))
        self.assertFalse(rationale_valid('unfinished',False,True))
        self.assertTrue(rationale_valid('finished on ceiling',True,True))
        self.assertFalse(rationale_valid('',True,False))

    def test_factorial_signs_and_controls(self):
        task='ds/1';records={}
        def put(kind,cond,arm,value):
            records[unit_key(dict(task=task,kind=kind,condition=cond,arm=arm))]={'analysis_valid':True,'analysis_score':value}
        values={'baseline':(9,6),'generated_block':(8,7),'control_block':(8.75,6.25),
                'cross_patch':(8.8,6.2),'cross_patch_generated_block':(6.8,8.2),
                'l2_patch':(8.95,6.05),'l2_patch_generated_block':(7.95,7.05),'same_state_patch':(9,6)}
        for arm,(hi,lo) in values.items():
            for c,v in zip(CONDS,(hi,lo)):put('factorial',c,arm,v)
        for arm,(hi,lo) in {'empty':(9,6),'generic':(9,6),'no_history':(8,7),'self_high_10':(9,8),'self_low_2':(7,6)}.items():
            for c,v in zip(CONDS,(hi,lo)):put('score',c,arm,v)
        expected={'history_gap_reduction':2,'history_vs_control_specificity':1.5,'patch_block_interaction':1,
                  'patch_interaction_vs_l2':1,'neutral_rationale_gap_reduction':2,'rationale_donor_effect':2}
        for k,expr in expressions(task).items():
            v,lo,hi,missing=evaluate(expr,records)
            self.assertAlmostEqual(v,expected[k]);self.assertAlmostEqual(lo,hi);self.assertFalse(missing)
        e=expressions(task)['history_gap_reduction'];key=next(iter(e))
        original=evaluate(e,records)[0];records[key]={'analysis_valid':False,'analysis_score':None}
        v,lo,hi,missing=evaluate(e,records)
        self.assertIsNone(v);self.assertLessEqual(lo,original);self.assertGreaterEqual(hi,original);self.assertEqual(missing,[key])

    def test_bootstrap_is_paired_by_task_and_stratified(self):
        self.assertEqual(interval([1,3],['a','b'],draws=200,alpha=.05,seed=1),[2,2])
        self.assertIsNone(interval([],[],draws=200,alpha=.05,seed=1))

    def test_resume_binding_gate_rejects_failed_checks(self):
        from run import gate_proof
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);binding=dict(plan_sha256='a',protocol_sha256='b',source_sha256={})
            (p/'metadata.json').write_text(json.dumps(dict(binding,status='failed',model='qwen',stage='engineering')))
            with self.assertRaises(ValueError):gate_proof(p,model='qwen',stage='engineering',binding=binding)


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
        cls.torch=torch;torch.set_num_threads(1)
        cls.models=[]
        for config,model in [(LlamaConfig,LlamaForCausalLM),(Qwen3Config,Qwen3ForCausalLM)]:
            torch.manual_seed(23)
            c=config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                     num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                     max_position_embeddings=1024,eos_token_id=63,pad_token_id=63)
            c._attn_implementation='sdpa'
            m=model(c).eval();cls.models.append(m)

    def test_mask_boundary_and_no_future_or_prompt_edits(self):
        t=self.torch;m=engine.causal_mask(t,0,7,7,'cpu',t.float32)
        b,edges=engine.blocked_mask(m,q_start=0,prompt_len=5,keys=[1,2])
        self.assertEqual(edges,4)
        self.assertTrue(t.equal(b[:,:,:5,:],m[:,:,:5,:]))
        self.assertTrue((b[0,0,5:,[1,2]]==t.finfo(t.float32).min).all())
        self.assertTrue(t.equal(m,engine.causal_mask(t,0,7,7,'cpu',t.float32)))
        for keys in [[-1],[5],[1,1]]:
            with self.assertRaises(ValueError):engine.blocked_mask(m,q_start=0,prompt_len=5,keys=keys)

    def test_both_real_architectures_cached_and_uncached(self):
        t=self.torch;ids=t.tensor([[2,5,9,4,8,3]])
        for model in self.models:
            with self.subTest(model=model.__class__.__name__):
                pos=[3];vals=engine.capture(model,ids,1,pos)
                modes=[{},dict(blocks=[0,1],keys=[1,2]),dict(patch=engine.Patch(1,pos,vals)),
                       dict(blocks=[0,1],keys=[1,2],patch=engine.Patch(1,pos,vals+1))]
                for kw in modes:
                    a=engine.greedy(model,ids,prompt_len=6,max_new=5,eos_ids=set(),**kw)
                    b=engine.greedy(model,ids,prompt_len=6,max_new=5,eos_ids=set(),use_cache=False,**kw)
                    self.assertEqual(a['ids'],b['ids'])
                base=engine.greedy(model,ids,prompt_len=6,max_new=5,eos_ids=set())
                identity=engine.greedy(model,ids,prompt_len=6,max_new=5,eos_ids=set(),patch=engine.Patch(1,pos,vals))
                self.assertEqual(base['ids'],identity['ids'])
                block=engine.greedy(model,ids,prompt_len=6,max_new=5,eos_ids=set(),blocks=[0,1],keys=[1,2])
                self.assertTrue(t.equal(base['_first_logits'],block['_first_logits']))
                self.assertEqual(block['events']['blocked_edges'],4*2*2)
                noop=engine.greedy(model,ids,prompt_len=6,max_new=5,eos_ids=set(),blocks=[0,1])
                self.assertEqual(base['ids'],noop['ids'])
                auto=engine.greedy(model,ids,prompt_len=6,max_new=5,eos_ids=set(),explicit=False)
                self.assertEqual(base['ids'],auto['ids'])

    def test_actual_attention_weights_zero_only_on_selected_edges(self):
        t=self.torch;ids=t.tensor([[2,5,9,4,8,3,7]])
        for model in self.models:
            model.config._attn_implementation='eager'
            try:
                mask=engine.causal_mask(t,0,7,7,'cpu',t.float32)
                with t.inference_mode():
                    baseline=model(input_ids=ids,attention_mask=mask,use_cache=False,output_attentions=True)
                    with engine.forward_edits(model,mask,q_start=0,prompt_len=6,keys=[1,2],blocks=[0,1]):
                        cut=model(input_ids=ids,attention_mask=mask,use_cache=False,output_attentions=True)
                for a,b in zip(baseline.attentions,cut.attentions):
                    self.assertTrue(t.equal(a[:,:,:6,:],b[:,:,:6,:]))
                    self.assertTrue(t.equal(b[:,:,6,[1,2]],t.zeros_like(b[:,:,6,[1,2]])))
                    self.assertTrue((a[:,:,6,[1,2]]>0).all())
            finally:model.config._attn_implementation='sdpa'

    def test_hooks_removed_after_failure_and_shape_rejection(self):
        t=self.torch;model=self.models[0];ids=t.tensor([[2,3,4]])
        before=len(model.model.layers[0].self_attn._forward_pre_hooks)
        with self.assertRaises(RuntimeError):
            with engine.forward_edits(model,engine.causal_mask(t,0,3,3,'cpu',t.float32),
                        q_start=0,prompt_len=3,blocks=[0]):raise RuntimeError('injected')
        self.assertEqual(before,len(model.model.layers[0].self_attn._forward_pre_hooks))
        before_patch=len(model.model.layers[0]._forward_hooks)
        with self.assertRaises(ValueError):engine.greedy(model,ids,prompt_len=3,max_new=1,eos_ids=set(),patch=engine.Patch(1,[1],t.zeros(2,32)))
        self.assertEqual(len(model.model.layers[0]._forward_hooks),before_patch)

    def test_eos_on_token_ceiling_is_complete(self):
        t=self.torch;ids=t.tensor([[2,3,4]])
        first=engine.greedy(self.models[0],ids,prompt_len=3,max_new=1,eos_ids=set())['ids'][0]
        g=engine.greedy(self.models[0],ids,prompt_len=3,max_new=1,eos_ids={first})
        self.assertTrue(g['hit_token_cap']);self.assertTrue(g['emitted_terminator'])



class CharacterTokenizer:
    """Local deterministic tokenizer for adapter tests; not used for research inference."""
    eos_token_id=127
    def __call__(self,text,**kwargs):
        out={'input_ids':[ord(c) for c in text]}
        if kwargs.get('return_offsets_mapping'):out['offset_mapping']=[(i,i+1) for i in range(len(text))]
        return out
    def apply_chat_template(self,messages,tokenize=True,**kwargs):
        text='<U>'+messages[-1]['content']+'</U><A>'
        if not tokenize:return text
        import torch
        return {'input_ids':torch.tensor([[ord(c) for c in text]])}
    def decode(self,ids,**kwargs):
        return ''.join(chr(i) for i in ids if i!=127)


class AdapterTests(unittest.TestCase):
    def test_all_arms_and_rationale_dependencies_execute(self):
        import torch
        from unittest.mock import patch
        from study import Study
        tok=CharacterTokenizer();model=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)]))
        scorer=Study(SimpleNamespace(model=model,tokenizer=tok),'qwen',load(HERE/'protocol.json'))
        task={'question':'QUESTION: Are these markers safe? CONTENT TO JUDGE:',
              'answer':'A sufficiently long answer with a QUESTION: marker and a CONTENT TO JUDGE: marker.'}
        calls=[]
        def generate(b,**kw):
            calls.append(kw)
            text=('The response addresses the question clearly.' if kw['maximum']==192 else
                  'Final score: 9/10.' if kw['maximum']==512 else '9/10.')
            ids=[ord(c) for c in text]+[127]
            return {'text':text,'ids':ids,'emitted_terminator':True,'hit_token_cap':False,
                    'output_tokens':len(ids),'token_sha256':engine.digest_ids(ids),
                    'assistant_prefix_token_ids':list(kw.get('prefix_ids',[])),
                    'events':{'mask_hooks':len(ids)*len(kw.get('blocks',[])),
                              'blocked_edges':(len(ids)-1)*len(kw.get('keys',[]))*len(kw.get('blocks',[])),
                              'patch_writes':int(kw.get('patch') is not None)}},None
        def patcher(task,target,layer,same=False):
            return engine.Patch(layer,[0],torch.zeros(1,1)),target if same else next(c for c in CONDS if c!=target)
        journal={};tasks=[dict(base_task_id='ds/1')]
        with patch.object(scorer,'generate',side_effect=generate),patch.object(scorer,'patch',side_effect=patcher):
            for u in ordered_units(tasks,'pilot'):
                r=scorer.run_unit(u,task,journal);journal[unit_key(u)]=dict(u,model='qwen',**r)
                if u['kind']=='factorial':
                    kw=calls[-1]
                    if u['arm']=='control_block':self.assertTrue(kw['keys']);self.assertEqual(len(kw['blocks']),36)
                    if u['arm']=='cross_patch_generated_block':self.assertTrue(kw['keys']);self.assertIsNotNone(kw['patch'])
                    self.assertTrue(r['analysis_valid'])
                if u['kind']=='score':
                    self.assertTrue(r['analysis_valid']);self.assertEqual(r['analysis_score'],9)
                    self.assertTrue(calls[-1]['suffix'].endswith('\n\nFinal score: '))
            self.assertEqual(len(journal),29)
            # Exercise the actual raw verification and bootstrap pipeline on clearly
            # synthetic, temporary adapter data, never on a saved research-results path.
            from analyze import summarize
            p=load(HERE/'protocol.json');p['bootstrap_draws']=20000
            plan={'pilot':[dict(base_task_id='ds/1',dataset='ds')]}
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)
                (out/'plan.json').write_text(json.dumps(plan))
                (out/'protocol.json').write_text(json.dumps(p))
                meta=dict(status='complete',model='qwen',stage='pilot',
                          plan_sha256=sha((out/'plan.json').read_bytes()),
                          protocol_sha256=sha((out/'protocol.json').read_bytes()))
                (out/'metadata.json').write_text(json.dumps(meta))
                with (out/'raw.jsonl').open('w') as f:
                    for rec in journal.values():
                        rec=dict(rec);rec['payload_sha256']=sha(canonical(rec));f.write(json.dumps(rec)+'\n')
                summary=summarize(out)
                self.assertFalse(summary['confirmatory'])
                self.assertEqual(summary['primary']['patch_block_interaction']['complete_tasks'],1)
                self.assertEqual(summary['primary']['patch_block_interaction']['complete_case_mean_points'],0)

            # An invalid donor stays missing; the model must not silently replace it.
            donor=unit_key(dict(task='ds/1',kind='rationale',condition='no_history',arm='donor'))
            journal[donor]['rationale_usable']=False
            u=dict(task='ds/1',kind='score',condition='self_low_2',arm='no_history')
            n=len(calls);r=scorer.run_unit(u,task,journal)
            self.assertFalse(r['analysis_valid']);self.assertTrue(r['generation_skipped']);self.assertEqual(len(calls),n)

    def test_engineering_gate_on_actual_32_and_36_layer_architectures(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
        from study import Study,eg
        task={'question':'What is two plus two?','answer':'Two plus two equals four. This answers the arithmetic question directly.'}
        for name,C,M in [('llama',LlamaConfig,LlamaForCausalLM),('qwen',Qwen3Config,Qwen3ForCausalLM)]:
            torch.manual_seed(42)
            cfg=C(vocab_size=128,hidden_size=8,intermediate_size=16,num_hidden_layers={'llama':32,'qwen':36}[name],
                  num_attention_heads=2,num_key_value_heads=1,head_dim=4,max_position_embeddings=2048,
                  eos_token_id=127,pad_token_id=127)
            cfg._attn_implementation='sdpa';model=M(cfg).eval()
            p=load(HERE/'protocol.json');p['engineering_max_new']=3
            spec=eg.ExactModelSpec(key=name,hf_id='local-tiny-test',revision='local',disable_thinking=name=='qwen')
            validate_architecture(model,p,name)
            wrong=dict(p,model_num_hidden_layers=dict(p['model_num_hidden_layers'],**{name:31}))
            with self.assertRaises(ValueError):validate_architecture(model,wrong,name)
            scorer=Study(SimpleNamespace(model=model,tokenizer=CharacterTokenizer(),spec=spec),name,p)
            result=scorer.engineering(task,'self_low_2')
            self.assertTrue(result['all_checks_pass'],result['checks'])



class BatchTests(unittest.TestCase):
    def test_sequential_batch_resume_and_stop_on_failed_gate(self):
        import os,subprocess,sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binary=root/'bin';binary.mkdir();log=root/'calls.jsonl'
            fake=binary/'python'
            code="""import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['BATCH_TEST_LOG'],'a') as f:f.write(json.dumps(args)+'\\n')
if os.environ.get('FAKE_FAIL_ENGINEERING') and args[0]=='run.py':raise SystemExit(2)
if args[0]=='run.py':
 p=Path(args[args.index('--out')+1]);p.mkdir(parents=True,exist_ok=True);(p/'metadata.json').write_text('{}')
else:Path(args[args.index('--out')+1]).write_text('{}')
"""
            fake.write_text('#!'+sys.executable+'\n'+code);fake.chmod(0o755)
            env=dict(os.environ,PATH=str(binary)+os.pathsep+os.environ['PATH'],BATCH_TEST_LOG=str(log))
            script=HERE/'run_pilot_both.sh'
            subprocess.run(['bash',str(script),str(root/'results')],env=env,check=True,capture_output=True)
            calls=[json.loads(l) for l in log.read_text().splitlines()]
            self.assertEqual([x[x.index('--model')+1] for x in calls[:4]],['qwen','qwen','llama','llama'])
            self.assertEqual([x[x.index('--stage')+1] for x in calls[:4]],['engineering','pilot','engineering','pilot'])
            self.assertEqual(calls[-1][0],'analyze.py')
            log.write_text('')
            subprocess.run(['bash',str(script),str(root/'results')],env=env,check=True,capture_output=True)
            self.assertTrue(all('--resume' in json.loads(l) for l in log.read_text().splitlines()))
            log.write_text('');env['FAKE_FAIL_ENGINEERING']='1'
            result=subprocess.run(['bash',str(script),str(root/'failed-results')],env=env,capture_output=True)
            self.assertNotEqual(result.returncode,0);self.assertEqual(len(log.read_text().splitlines()),1)

if __name__=='__main__':unittest.main()
