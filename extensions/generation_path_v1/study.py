"""Pinned prompt/scoring adapter and the three complementary experiment protocols."""
import sys
import time
from pathlib import Path
from protocol import DEPENDENCY, CONDS, DONORS, RATIONALE_REQUEST, SCORE_PREFIX, GENERIC_RATIONALE, rationale_valid, sha, canonical
import engine

sys.path.insert(0, str(DEPENDENCY))
sys.path.insert(0, str(DEPENDENCY/'reference/extensions/context_controls/src'))
from run_targeted import encode, span_positions
from interventions import semantic_positions
from rba import calib as C, runner as RN
from rba.legacy import scaled_conditions as sc, exact_generation as eg


class Study:
    def __init__(self, loaded, model_key, protocol, budget=lambda: None):
        self.loaded, self.key, self.p = loaded, model_key, protocol
        self.model, self.tok = loaded.model, loaded.tokenizer
        self.budget = budget
        self._capture = {}
        self._task_id = None

    def build(self, task, condition, rationale_request=False):
        prompt = sc.render_prompt(task['question'], task['answer'], sc.resolve(condition))
        if rationale_request:
            prompt += RATIONALE_REQUEST
        ids, rendered, offsets, start = encode(eg, self.tok, C.messages(self.key, prompt),
                                              C.MODELS[self.key]['disable_thinking'])
        ids = ids.to(next(self.model.parameters()).device)
        pos = {}
        if condition != 'no_history' and not rationale_request:
            pos = {s: span_positions(prompt, sc.resolve(condition), offsets, start, s,
                      task['question'], task['answer']) for s in ['history','history_end','question','answer']}
            # Exact cardinality and same layer set as the history knockout. This is a
            # competing-content control, not an inert token or a quality-matched memory.
            pool = sorted(set(pos['question']+pos['answer']))
            if len(pool) < len(pos['history']):
                raise ValueError('Not enough non-history content tokens for an equal-key control')
            pos['control'] = pool[:len(pos['history'])]
            if set(pos['control']) & set(pos['history']):
                raise ValueError('Control keys overlap history')
        return {'ids': ids, 'positions': pos, 'prompt': prompt, 'rendered_sha256': sha(rendered.encode())}

    def generate(self, b, *, maximum, suffix='', prefix_ids=(), keys=(), blocks=(), patch=None,
                 explicit=True, use_cache=True):
        import torch
        ids = b['ids']
        if prefix_ids:
            ids = torch.cat((ids, ids.new_tensor([list(prefix_ids)])), dim=1)
        if suffix:
            # Append assistant continuation tokens separately; never retokenize the
            # native chat prompt or silently shift its final token.
            extra = self.tok(suffix, add_special_tokens=False)['input_ids']
            ids = torch.cat((ids, ids.new_tensor([extra])), dim=1)
        start = time.monotonic()
        g = engine.greedy(self.model, ids, prompt_len=b['ids'].shape[1], max_new=maximum,
                         eos_ids=set(eg.terminator_ids(self.loaded)), keys=keys, blocks=blocks,
                         patch=patch, explicit=explicit, use_cache=use_cache, budget=self.budget)
        logits = g.pop('_first_logits')
        text = self.tok.decode(g['ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        g.update(text=text, output_tokens=len(g['ids']), prompt_tokens=ids.shape[1],
                 input_token_ids=ids[0].tolist(), generated_token_ids_sha256=g['token_sha256'],
                 generation_seconds=round(time.monotonic()-start,3), suffix_sha256=sha(suffix.encode()), assistant_prefix_token_ids=list(prefix_ids))
        return g, logits

    def patch(self, task, target, layer, same=False):
        source = target if same else next(c for c in CONDS if c != target)
        a, b = self.build(task, source), self.build(task, target)
        pos_a, pos_b = semantic_positions('history_end', a['ids'][0].tolist(), b['ids'][0].tolist(),
                                         a['positions']['history_end'], b['positions']['history_end'])
        key = (source, layer, tuple(pos_a))
        if key not in self._capture:
            self.budget()
            self._capture[key] = engine.capture(self.model, a['ids'], layer, pos_a)
        return engine.Patch(layer, pos_b, self._capture[key]), source

    def engineering(self, task, condition):
        import torch
        b = self.build(task, condition)
        n = self.p['engineering_max_new']
        base, logits = self.generate(b, maximum=n)
        automatic, _ = self.generate(b, maximum=n, explicit=False)
        no_cache, _ = self.generate(b, maximum=n, use_cache=False)
        noop, _ = self.generate(b, maximum=n, blocks=self.p['mask_blocks'])
        patch, _ = self.patch(task, condition, self.p['patch_sites'][self.key]['layer'], same=True)
        identity, _ = self.generate(b, maximum=n, patch=patch)
        self.budget()
        kwargs = eg.build_generate_kwargs(self.loaded.spec, max_new_tokens=n,
                    pad_token_id=self.tok.eos_token_id, use_cache=True)
        with torch.inference_mode():
            native = self.model.generate(input_ids=b['ids'], attention_mask=torch.ones_like(b['ids']), **kwargs)
        native_ids = eg.trim_to_first_eos(native[0,b['ids'].shape[1]:].tolist(),eg.terminator_ids(self.loaded))
        identities = {name: base['ids'] == other['ids'] for name,other in
                      [('automatic_mask',automatic),('no_cache',no_cache),('noop_hooks',noop),('same_state',identity)]}
        identities['native_generate'] = base['ids'] == native_ids
        # Structural boundary check: first-token logits cannot change when only
        # generated INPUT-token readers are blocked. Then force the same first token
        # for one cached step to exercise the active generated-reader edge.
        blocked, first = self.generate(b, maximum=1, keys=b['positions']['history'], blocks=self.p['mask_blocks'])
        pre, _ = engine.forward(self.model, b['ids'], prompt_len=b['ids'].shape[1])
        post, ev = engine.forward(self.model, b['ids'].new_tensor([[base['ids'][0]]]),
                  prompt_len=b['ids'].shape[1], q_start=b['ids'].shape[1], cache=pre.past_key_values,
                  keys=b['positions']['history'], blocks=self.p['mask_blocks'])
        del pre, post
        cut, positive = engine.forward(self.model,b['ids'],prompt_len=b['ids'].shape[1],
                 keys=b['positions']['history'], blocks=self.p['mask_blocks'],reader_policy='last_prompt')
        difference = float((cut.logits[0,-1].float().cpu()-logits).abs().max())
        del cut
        checks = dict(identities, generated_prefill_identical=torch.equal(logits,first),
                      generated_prefill_edges_zero=blocked['events']['blocked_edges']==0,
                      generated_decode_edges_active=ev['blocked_edges']==len(b['positions']['history'])*32,
                      prompt_cut_positive_control=difference>0)
        return {'checks':checks, 'all_checks_pass':all(checks.values()),
                'positive_control_max_logit_difference':difference,
                'baseline_token_sha256':base['token_sha256'],
                'equivalence_token_sha256':dict(baseline=base['token_sha256'],automatic_mask=automatic['token_sha256'],
                     no_cache=no_cache['token_sha256'],noop_hooks=noop['token_sha256'],same_state=identity['token_sha256'],
                     native_generate=engine.digest_ids(native_ids)),
                'baseline_prefill_logits_sha256':sha(logits.numpy().tobytes()),
                'generated_block_prefill_logits_sha256':sha(first.numpy().tobytes()),
                'active_decode_events':ev, 'prefill_block_events':blocked['events'],
                'history_keys':b['positions']['history'], 'control_keys':b['positions']['control'],
                'native_prompt_tokens':b['ids'].shape[1]}

    def run_unit(self, unit, task, journal):
        from protocol import unit_key
        if self._task_id != unit['task']:
            self._capture.clear();self._task_id=unit['task']
        c, arm, kind = unit['condition'],unit['arm'],unit['kind']
        if kind == 'engineering':
            return self.engineering(task,c)
        b = self.build(task,c, rationale_request=kind=='rationale')
        if kind == 'rationale':
            g, _ = self.generate(b, maximum=self.p['rationale_max_new'])
            return dict(g, rationale_usable=bool(rationale_valid(g['text'],g['emitted_terminator'],g['hit_token_cap'])),
                        rationale_content_token_ids=self.tok(g['text'],add_special_tokens=False)['input_ids'])
        if kind == 'score':
            prefix_ids = []
            dependencies = []
            if arm != 'empty':
                dependencies = [unit_key(dict(task=unit['task'],kind='rationale',condition=d,arm='donor')) for d in DONORS]
                donors = {d:journal[k] for d,k in zip(DONORS,dependencies)}
                n = min([self.p['rationale_prefix_max_tokens']] +
                        [len(r['rationale_content_token_ids']) for r in donors.values()])
                if not all(r['rationale_usable'] for r in donors.values()) or n<self.p['rationale_prefix_min_tokens']:
                    return {'analysis_valid':False,'analysis_score':None,'analysis_validity':'invalid_rationale_set',
                            'rationale_dependency_units':dependencies,'text':None,'generation_skipped':True}
                if arm == 'generic':
                    generic_ids = self.tok(GENERIC_RATIONALE,add_special_tokens=False)['input_ids']
                    prefix_ids = (generic_ids*((n+len(generic_ids)-1)//len(generic_ids)))[:n]
                else:
                    prefix_ids = donors[arm]['rationale_content_token_ids'][:n]
            g, _ = self.generate(b, maximum=self.p['score_max_new'], suffix=SCORE_PREFIX, prefix_ids=prefix_ids)
            # Only newly completed score text is parsed; donor prose cannot supply it.
            g_for_parse = dict(g,text='Final score: '+g['text'])
            return dict(g, **RN.payload_from_generation(g_for_parse,None,self.p['score_max_new']),
                        rationale_dependency_units=dependencies,shared_rationale_tokens=len(prefix_ids),
                        rationale_prefix_sha256=sha(canonical(prefix_ids)),
                        protocol_note='Equal-token rationale-prefix intervention; not ordinary free-generation mediation')
        keys, blocks, patch, source = (),(),None,None
        if 'generated_block' in arm or arm == 'generated_block':
            keys=b['positions']['history'];blocks=self.p['mask_blocks']
        elif arm == 'control_block':
            keys=b['positions']['control'];blocks=self.p['mask_blocks']
        if arm.startswith('cross_patch') or arm.startswith('l2_patch') or arm=='same_state_patch':
            layer = self.p['control_layer'] if arm.startswith('l2_patch') else self.p['patch_sites'][self.key]['layer']
            patch, source = self.patch(task,c,layer,same=arm=='same_state_patch')
        g, _ = self.generate(b,maximum=self.p['free_max_new'],keys=keys,blocks=blocks,patch=patch)
        payload=RN.payload_from_generation(g,None,self.p['free_max_new'])
        return dict(g, **payload, history_keys=b['positions']['history'], control_keys=b['positions']['control'],
                    mask_blocks=list(blocks), source_condition=source,
                    patch_layer=None if patch is None else patch.layer,
                    patch_positions=[] if patch is None else patch.positions)
