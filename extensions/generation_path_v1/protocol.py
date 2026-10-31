"""Pure protocol, journal, and task-selection utilities. No model loading."""
from pathlib import Path
import collections
import hashlib
import json
import random
import re

HERE = Path(__file__).resolve().parent
DEPENDENCY = HERE.parent/'residual_discovery_v3'
CONDS = ('self_high_10', 'self_low_2')
DONORS = CONDS + ('no_history',)
SCORE_ARMS = DONORS + ('generic', 'empty')
GENERIC_RATIONALE = 'An evaluation considers the response in relation to the question. Relevant details and clarity can be considered together. '
ARMS = ('baseline', 'native_baseline', 'generated_block', 'control_block', 'cross_patch',
        'cross_patch_generated_block', 'l2_patch', 'l2_patch_generated_block', 'same_state_patch')
PRIMARY = ('history_gap_reduction', 'history_vs_control_specificity', 'patch_block_interaction',
           'patch_interaction_vs_l2', 'neutral_rationale_gap_reduction', 'rationale_donor_effect')
RATIONALE_REQUEST = ('\n\nFor this step, write a short qualitative evaluation only. Do not state, '
                     'repeat, or suggest any numeric rating. Do not provide a final score.')
SCORE_PREFIX = '\n\nFinal score: '
RATING_MARKER = re.compile(r'\b\d+(?:\.\d+)?\s*(?:/\s*10\b|out\s+of\s+10\b)|'
                           r'\b(?:final\s+|overall\s+)?score\s*(?:is|of|:)?\s*\**\s*\d', re.I)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
    tmp.replace(p)


def source_hashes():
    paths = list(HERE.glob('*.py')) + [DEPENDENCY/'run_targeted.py', DEPENDENCY/'interventions.py']
    paths += list((DEPENDENCY/'reference/extensions/context_controls/src/rba').rglob('*.py'))
    return {str(p.relative_to(HERE.parent)): sha(p.read_bytes()) for p in sorted(paths)}


def validate_protocol(p):
    if p['version'] != 3 or p['primary_tests_total'] != len(PRIMARY)*2 or p.get('model_num_hidden_layers') != {'llama':32,'qwen':36} or p['mask_blocks'] != {'llama':list(range(32)), 'qwen':list(range(36))}:
        raise ValueError('Protocol v3 requires all 32 Llama / 36 Qwen blocks and 12 planned tests')
    if p.get('engineering_policy')!='strict_within_path_invariance_cross_path_diagnostic_v3' or p.get('native_baseline')!='hf_generate_cached_same_native_messages':
        raise ValueError('Protocol v3 requires exact within-path controls and a native baseline arm')
    if p['reader_policy'] != 'generated' or p['control_layer'] != 2 or p['family_alpha'] != .05 or p['primary_endpoints'] != list(PRIMARY) or p.get('seed') != 42:
        raise ValueError('Changed reader policy/control requires a new protocol version')
    if not (1 <= p['free_max_new'] <= 1024 and 1 <= p['rationale_max_new'] <= 512
            and 1 <= p['score_max_new'] <= 128 and 1 <= p['engineering_max_new'] <= 256 and p['bootstrap_draws'] >= 20000):
        raise ValueError('Invalid generation/bootstrap budget')
    if p['rationale_request'] != RATIONALE_REQUEST or p['score_prefix'] != SCORE_PREFIX or p['generic_rationale'] != GENERIC_RATIONALE or p['rationale_prefix_max_tokens'] != 96 or p['rationale_prefix_min_tokens'] != 8:
        raise ValueError('Rationale protocol text differs from the locked implementation')
    if p['patch_sites'] != {'llama': {'layer': 12, 'span': 'history_end'},
                            'qwen': {'layer': 16, 'span': 'history_end'}}:
        raise ValueError('This protocol tests the previously discovered leads, not a new site search')


HARD_ENGINEERING_CHECKS = ('repeat_explicit_tokens','repeat_explicit_logits','noop_hooks','same_state',
    'generated_prefill_identical','generated_prefill_edges_zero','generated_decode_edges_active',
    'prompt_cut_positive_control')


def verify_engineering_record(r, protocol, model):
    if set(r['checks']) != set(HARD_ENGINEERING_CHECKS) or not all(r['checks'].values()) or r.get('all_checks_pass') is not True:
        raise ValueError('A mandatory within-path engineering control failed or is absent')
    h=r['equivalence_token_sha256']
    if len({h[k] for k in ['baseline','repeat_explicit','noop_hooks','same_state']})!=1:
        raise ValueError('Within-path generated-token identity failed')
    if not r['baseline_prediction_logits_sha256'] or r['baseline_prediction_logits_sha256']!=r['repeat_prediction_logits_sha256']:
        raise ValueError('Repeated explicit-path logits are not bit-identical')
    if r['baseline_prefill_logits_sha256']!=r['generated_block_prefill_logits_sha256']:
        raise ValueError('Generated-reader blocking changed prompt logits')
    if r['active_decode_events']['blocked_edges']!=len(r['history_keys'])*len(model_blocks(protocol,model)) or r['prefill_block_events']['blocked_edges']!=0:
        raise ValueError('Reader-boundary check failed')
    if not r['positive_control_max_logit_difference']>0:
        raise ValueError('Active mask control had no measurable effect')
    diagnostic=r['cross_path_diagnostics']
    comparisons={'automatic_mask_vs_explicit':('automatic_mask','baseline'),
                 'native_generate_vs_explicit':('native_generate','baseline'),
                 'recomputed_vs_explicit':('no_cache','baseline'),
                 'automatic_mask_vs_native_generate':('automatic_mask','native_generate')}
    if set(diagnostic)!=set(comparisons):raise ValueError('Cross-path diagnostics are incomplete')
    for key,(a,b) in comparisons.items():
        if diagnostic[key] != (h[a]==h[b]):raise ValueError('Cross-path diagnostic disagrees with saved hashes')


def model_blocks(protocol, model_key):
    return list(protocol['mask_blocks'][model_key])


def validate_model_config(config, protocol, model_key):
    expected = protocol['model_num_hidden_layers'][model_key]
    expected_type = {'llama':'llama','qwen':'qwen3'}[model_key]
    if config.num_hidden_layers != expected or config.model_type != expected_type:
        raise ValueError(f'{model_key}: pinned architecture must be {expected_type} with {expected} blocks; '
                         f'got {config.model_type} / {config.num_hidden_layers}')


def validate_architecture(model, protocol, model_key):
    validate_model_config(model.config, protocol, model_key)
    if len(model.model.layers) != protocol['model_num_hidden_layers'][model_key]:
        raise ValueError('Loaded decoder-block count differs from its validated configuration')


def planned_units(tasks, stage):
    out = []
    for t in tasks:
        task = t['base_task_id']
        if stage == 'engineering':
            out += [dict(task=task, kind='engineering', condition=c, arm='checks') for c in CONDS]
        else:
            out += [dict(task=task, kind='factorial', condition=c, arm=a) for c in CONDS for a in ARMS]
            out += [dict(task=task, kind='rationale', condition=c, arm='donor') for c in DONORS]
            out += [dict(task=task, kind='score', condition=c, arm=d) for c in CONDS
                    for d in SCORE_ARMS]
    return out


def unit_key(u):
    return '|'.join(u[k] for k in ('task', 'kind', 'condition', 'arm'))


def ordered_units(tasks, stage):
    """All rationale donors precede continuations; counterbalance arm order within task."""
    out = []
    for t in tasks:
        units = planned_units([t], stage)
        rng = random.Random('generation-path-v1|'+t['base_task_id'])
        for kind in ('engineering', 'factorial', 'rationale', 'score'):
            group = [u for u in units if u['kind'] == kind]
            rng.shuffle(group)
            out += group
    return out


def read_journal(path, expected):
    """Fail closed on truncation, duplicates, off-plan units, and changed payloads."""
    p = Path(path)
    if not p.exists():
        return {}
    data = p.read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError('Journal has an unfinished tail. Keep a backup and use repair_tail.py first.')
    seen = {}
    for line in data.splitlines():
        r = json.loads(line)
        h = r.pop('payload_sha256')
        if sha(canonical(r)) != h:
            raise ValueError('Journal payload checksum mismatch')
        key = unit_key(r)
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unplanned journal unit')
        r['payload_sha256'] = h
        seen[key] = r
    return seen


def rationale_valid(text, ended, capped):
    return bool(text.strip()) and (ended or not capped) and not RATING_MARKER.search(text)


def build_plan(own, peer):
    """Task IDs/question hashes only. Never inspects any score or intervention outcome."""
    blocked_ids = set(own['excluded_ids'])
    blocked_ids |= {t['base_task_id'] for phase in ['engineering', 'discovery'] for t in own['splits'][phase]}
    blocked_ids |= set(peer['splits']['D']) | set(peer['splits']['V'])
    # Exclude exact-question duplicates under alternate IDs as well.
    answers = load(DEPENDENCY/'reference/code/data/candidates/final_olmo_answers.json')['accepted']
    blocked_questions = {t['question_sha256'] for t in answers
                         if f"{t['dataset']}/{t['source_task_id']}" in blocked_ids}
    valid = [t for t in own['splits']['validation'] if t['base_task_id'] not in blocked_ids
             and t['question_sha256'] not in blocked_questions]
    pilot, engineering = [], []
    for ds in ('mtbench', 'arena_hard', 'wildbench'):
        pilot += [t for t in own['splits']['discovery'] if t['dataset'] == ds][:4]
        engineering += [t for t in own['splits']['engineering'] if t['dataset'] == ds][:1]
    return {'version': 1, 'outcomes_consulted': False, 'engineering': engineering,
            'pilot': pilot, 'validation': valid,
            'validation_dataset_counts': dict(collections.Counter(t['dataset'] for t in valid)),
            'scope': 'Unexposed to the recorded mechanistic runs by ID and exact-question hash; '
                     'not necessarily behaviorally unseen. External/unrecorded exposure requires author audit. '
                     'Only one MT-Bench task remains; this is not a balanced three-benchmark confirmation set.',
            'own_split_sha256': sha((DEPENDENCY/'split.json').read_bytes())}
