import json
from pathlib import Path
import sys
import unittest
import hashlib
import tempfile
import os
from unittest.mock import patch
from types import SimpleNamespace
from interventions import aligned_positions, token_alignment, capture, transplant, embedding_replay_control, semantic_positions
from analyze_run import interval
from run_targeted import span_positions

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

class DesignTests(unittest.TestCase):
    def test_alignment_shift(self):
        self.assertEqual(aligned_positions([1,8,9,3,4],[1,2,3,4],[2,3]),([3,4],[2,3]))
    def test_unequal_span_rejected(self):
        with self.assertRaises(ValueError):
            aligned_positions([1,8,9,3,4],[1,2,3,4],[1])
    def test_equal_length_replacement(self):
        self.assertEqual(token_alignment([1,10,3],[1,2,3]),{0:0,1:1,2:2})
    def test_native_semantic_anchor_after_unequal_rating(self):
        high=[1,7,8,9,3];low=[1,2,9,3]
        self.assertEqual(semantic_positions('history_end',high,low,[3],[2]),([3],[2]))
        self.assertEqual(semantic_positions('answer',high,low,[3,4],[2,3]),([3,4],[2,3]))
        self.assertEqual(semantic_positions('final',high,low,[4],[3]),([4],[3]))
    def test_native_semantic_span_rejects_changed_content(self):
        with self.assertRaises(ValueError):
            semantic_positions('answer',[1,7,3],[1,2,3],[1,2],[1,2])
    def test_zero_effect_bootstrap(self):
        self.assertEqual(interval([0,0,0],['a','b','c'],draws=200),[0,0])
    def test_stratified_bootstrap(self):
        self.assertEqual(interval([1,3],['a','b'],draws=200),[2,2])
    def test_frozen_split_disjoint(self):
        m=json.loads(Path(__file__).with_name('split.json').read_text())
        groups=[{t['base_task_id'] for t in v} for v in m['splits'].values()]
        for i,g in enumerate(groups):
            self.assertFalse(g & set(m['excluded_ids']))
            for other in groups[i+1:]:self.assertFalse(g & other)
        self.assertEqual([len(m['splits'][k]) for k in ['engineering','discovery','validation']],[6,18,72])
        q=[t['question_sha256'] for v in m['splits'].values() for t in v]
        self.assertEqual(len(q),len(set(q)))

    def test_rendered_spans_ignore_markers_in_question(self):
        repo=Path(os.environ.get('REPO',str(Path(__file__).resolve().parents[2]/'work/repository')))
        if not repo.exists():self.skipTest('Run in handoff workspace or set repository path')
        sys.path.insert(0,str(repo/'extensions/context_controls/src'))
        from rba.legacy import scaled_conditions as sc
        question='QUESTION:\nCONTENT TO JUDGE:\nA deliberately embedded marker'
        answer='The answer also contains QUESTION: and CONTENT TO JUDGE:'
        for cond in [sc.resolve('self_high_10'),sc.resolve('self_low_2')]:
            prompt=sc.render_prompt(question,answer,cond)
            offsets=[(i,i+1) for i in range(len(prompt))]
            pos=span_positions(prompt,cond,offsets,0,'answer',question,answer)
            self.assertEqual(''.join(prompt[i] for i in pos),answer)
            pos=span_positions(prompt,cond,offsets,0,'rating',question,answer)
            self.assertEqual(''.join(prompt[i] for i in pos),'10/10' if cond.name=='self_high_10' else '2/10')

    def test_analysis_signed_effect_and_zero_gap_retention(self):
        import analyze_run
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);sp=d/'split.json';cp=d/'candidates.json';out=d/'analysis.json'
            sp.write_text(json.dumps({'splits':{'validation':[{'base_task_id':'a/1'},{'base_task_id':'b/2'}]}}))
            specs=[{'component':'resid','layer':24,'span':'final'},
                   {'component':'resid','layer':2,'span':'final'}]
            cp.write_text(json.dumps(specs))
            meta={'status':'complete','phase':'validation','model':'llama',
                  'split_sha256':hashlib.sha256(sp.read_bytes()).hexdigest(),
                  'candidates_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
            (d/'metadata.json').write_text(json.dumps(meta))
            rows=[]
            for task in ['a/1','b/2']:
                for cond,baseline in [('self_high_10',8),('self_low_2',4 if task=='a/1' else 8)]:
                    rows.append(dict(task=task,kind='baseline',target=cond,analysis_valid=True,analysis_score=baseline))
                    for i,spec in enumerate(specs):
                        for kind in ['same_state','cross_patch']:
                            delta=(1 if cond=='self_low_2' else -1) if kind=='cross_patch' and i==0 else 0
                            rows.append(dict(task=task,kind=kind,target=cond,spec=spec,
                                             analysis_valid=True,analysis_score=baseline+delta))
            for r in rows:r['payload_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()
            (d/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            argv=['analyze_run','--run',str(d),'--split',str(sp),'--candidates',str(cp),'--out',str(out)]
            with patch.object(sys,'argv',argv):analyze_run.main()
            result=json.loads(out.read_text())['results']
            self.assertEqual([r['n_complete'] for r in result],[2,2,2,2])
            self.assertEqual([r['paired_site_minus_control_points'] for r in result[:2]],[1,1])
            self.assertEqual([r['movement_points_mean'] for r in result[:2]],[1,1])
            self.assertEqual([r['ratio_of_means'] for r in result[:2]],[.5,.5])
            # Engineering replay controls are separate from score-transfer rows,
            # and must have complete, hash-checked coverage.
            split_obj=json.loads(sp.read_text())
            split_obj['splits']['engineering']=split_obj['splits']['validation']
            sp.write_text(json.dumps(split_obj))
            meta.update(phase='engineering',engineering_protocol='v2_unequal_length_embedding_replay',
                        split_sha256=hashlib.sha256(sp.read_bytes()).hexdigest())
            (d/'metadata.json').write_text(json.dumps(meta))
            controls=[dict(task=b,condition=c,all_tokens_identical=True)
                      for b in ['a/1','b/2'] for c in ['self_high_10','self_low_2']]
            for c in controls:c['payload_sha256']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
            (d/'positive_controls.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in controls))
            eng_out=d/'engineering_summary.json';argv[-1]=str(eng_out)
            with patch.object(sys,'argv',argv):analyze_run.main()
            self.assertEqual(json.loads(eng_out.read_text())['engineering_positive_controls_checked'],4)

if torch:
    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.o_proj=nn.Linear(4,4,bias=False);self.head_dim=2
            with torch.no_grad():self.o_proj.weight.copy_(torch.eye(4))
        def forward(self,x):return (self.o_proj(x),None)
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.self_attn=Attention();self.mlp=nn.Linear(4,4,bias=False)
        def forward(self,x):
            x=x+self.self_attn(x)[0];return (x+self.mlp(x),)
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();torch.manual_seed(17)
            self.model=nn.Module();self.model.embed_tokens=nn.Embedding(20,4)
            self.model.layers=nn.ModuleList([Block(),Block()])
        def forward(self,input_ids,use_cache=False):
            x=self.model.embed_tokens(input_ids)
            for b in self.model.layers:x=b(x)[0]
            return x

@unittest.skipIf(torch is None,'torch not installed')
class HookTests(unittest.TestCase):
    def setUp(self):
        self.m=Toy().eval();self.ids=torch.tensor([[1,2,3]]);self.src=torch.tensor([[4,5,6]])
    def test_roundtrip_each_component(self):
        baseline=self.m(self.ids)
        for comp,layer in [('embed',0),('resid',1),('attn',1),('mlp',1),('head',1)]:
            v=capture(self.m,self.ids,comp,layer,[2])
            with transplant(self.m,comp,layer,[2],v,3,head=0 if comp=='head' else None) as events:
                actual=self.m(self.ids)
            self.assertTrue(torch.equal(baseline,actual),comp)
            self.assertEqual(events['prefill_writes'],1)
    def test_embedding_positive_control(self):
        v=capture(self.m,self.src,'embed',0,[0,1,2])
        with transplant(self.m,'embed',0,[0,1,2],v,3):
            actual=self.m(self.ids)
        self.assertTrue(torch.equal(actual,self.m(self.src)))
    def test_selected_position_only(self):
        v=capture(self.m,self.src,'resid',2,[1]);before=self.m(self.ids)
        with transplant(self.m,'resid',2,[1],v,3):actual=self.m(self.ids)
        self.assertTrue(torch.equal(actual[0,[0,2]],before[0,[0,2]]))
        self.assertTrue(torch.equal(actual[0,1],v[0]))
    def test_decode_untouched_and_hook_removed(self):
        v=capture(self.m,self.src,'resid',2,[2]);single=torch.tensor([[7]])
        expected=self.m(single)
        with transplant(self.m,'resid',2,[2],v,3) as events:
            self.m(self.ids);actual=self.m(single)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(events['decode_skips'],1)
        self.assertEqual(len(self.m.model.layers[1]._forward_hooks),0)
    def test_cache_off_repeated_prefill_rejected(self):
        v=capture(self.m,self.ids,'resid',1,[2])
        with self.assertRaises(RuntimeError):
            with transplant(self.m,'resid',1,[2],v,3):
                self.m(self.ids);self.m(self.ids)
        self.assertEqual(len(self.m.model.layers[0]._forward_hooks),0)
    def test_head_columns(self):
        v=capture(self.m,self.src,'head',1,[2]);before=capture(self.m,self.ids,'head',1,[2])
        observed=[]
        with transplant(self.m,'head',1,[2],v,3,head=1):
            h=self.m.model.layers[0].self_attn.o_proj.register_forward_pre_hook(
                lambda module,inp: observed.append(inp[0].detach().clone()))
            try:self.m(self.ids)
            finally:h.remove()
        self.assertTrue(torch.equal(observed[0][0,2,:2],before[0,:2]))
        self.assertTrue(torch.equal(observed[0][0,2,2:],v[0,2:]))
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):
            with transplant(self.m,'resid',1,[2],torch.zeros(2,4),3):self.m(self.ids)
    def test_unused_hook_rejected(self):
        with self.assertRaises(RuntimeError):
            with transplant(self.m,'resid',1,[2],torch.zeros(1,4),3):pass

@unittest.skipIf(torch is None,'torch not installed')
class ArchitectureTests(unittest.TestCase):
    def test_tiny_llama_and_qwen_cached_generation(self):
        try:
            from transformers import LlamaConfig,LlamaForCausalLM,Qwen3Config,Qwen3ForCausalLM
        except ImportError:self.skipTest('Install transformers for architecture tests')
        for config_cls,model_cls in [(LlamaConfig,LlamaForCausalLM),(Qwen3Config,Qwen3ForCausalLM)]:
            torch.manual_seed(42)
            cfg=config_cls(vocab_size=32,hidden_size=16,intermediate_size=32,
                           num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                           head_dim=4,attn_implementation='eager',bos_token_id=1,eos_token_id=None,
                           pad_token_id=0)
            m=model_cls(cfg).eval();ids=torch.tensor([[1,3,5,7]]);src=torch.tensor([[1,9,5,7]])
            def gen(x):
                with torch.inference_mode():
                    return m.generate(input_ids=x,attention_mask=torch.ones_like(x),
                                      max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=0)
            base=gen(ids)
            for comp,layer in [('embed',0),('resid',1),('resid',2),('attn',1),('mlp',1),('head',1)]:
                v=capture(m,ids,comp,layer,[3])
                with transplant(m,comp,layer,[3],v,4,head=0 if comp=='head' else None) as events:
                    actual=gen(ids)
                self.assertTrue(torch.equal(base,actual),(model_cls.__name__,comp,layer))
                self.assertEqual(events,{'prefill_writes':1,'decode_skips':3})
            donor=capture(m,src,'embed',0,[0,1,2,3])
            with transplant(m,'embed',0,[0,1,2,3],donor,4):
                patched=gen(ids)
            self.assertTrue(torch.equal(patched[:,4:],gen(src)[:,4:]))

    def test_embedding_replay_accepts_native_prompts_of_different_lengths(self):
        try:
            from transformers import Qwen3Config,Qwen3ForCausalLM
        except ImportError:self.skipTest('Install transformers for architecture tests')
        torch.manual_seed(42)
        m=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=4,
            attn_implementation='eager',bos_token_id=1,eos_token_id=None,pad_token_id=0)).eval()
        kwargs=dict(max_new_tokens=4,do_sample=False,use_cache=True,pad_token_id=0)
        for native in [torch.tensor([[1,2,3,4]]),torch.tensor([[1,5,6,3,4]])]:
            with torch.inference_mode():
                baseline=m.generate(input_ids=native,attention_mask=torch.ones_like(native),**kwargs)
            result=embedding_replay_control(m,native,baseline[0,native.shape[1]:].tolist(),kwargs)
            self.assertTrue(result['all_tokens_identical'])
            self.assertEqual(result['events']['prefill_writes'],1)
            self.assertEqual(result['prompt_tokens'],native.shape[1])

if __name__ == '__main__':unittest.main()
