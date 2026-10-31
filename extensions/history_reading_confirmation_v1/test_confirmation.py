import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import confirmation_core as core
from confirmation_core import HERE,backend,load,canonical,sha
from summarize_confirmation import expr,contrast
from analyze import evaluate


def valid_audit():
    return {'candidate_plan_sha256':sha(canonical(core.candidate_plan())),
            'reviewer_name':'SYNTHETIC TEST AUTHOR','reviewed_utc':'2026-10-31T00:00:00Z',
            'exposure_statement':'Synthetic temporary test only; not an author attestation for research.',
            'additional_exposed_task_ids':[],'additional_exposed_question_sha256':[],
            'author_confirms_remaining_tasks_not_used_mechanistically':True}


def row(task,condition,arm,value=8.):
    ids=[5,6];masked=arm in ['generated_block','control_block'];n=2;layers=36
    return {'task':task,'kind':'factorial','condition':condition,'arm':arm,'model':'qwen',
            'ids':ids,'token_sha256':__import__('engine').digest_ids(ids),
            'text':f'Final score: {value}/10.','analysis_score':value,'analysis_valid':True,'analysis_validity':'valid',
            'output_tokens':2,'hit_token_cap':False,'emitted_terminator':True,
            'events':{'mask_hooks':n*layers*int(masked),'blocked_edges':(n-1)*2*layers*int(masked),'patch_writes':0},
            'history_keys':[1,2],'control_keys':[4,5],
            'mask_blocks':list(range(36)) if masked else [],
            'computation_path':'hf_native_cached' if arm=='native_baseline' else 'explicit'}


class ProtocolTests(unittest.TestCase):
    def test_exact_four_arm_four_test_protocol(self):
        p=load(HERE/'confirmation_protocol.json');core.validate_protocol(p)
        for key,value in [('primary_tests_total',2),('max_new_tokens',32),('arms',['baseline'])]:
            bad=dict(p,**{key:value})
            with self.assertRaises(ValueError):core.validate_protocol(bad)
        plan=core.candidate_plan();self.assertEqual(len(plan['tasks']),44)
        self.assertEqual(plan['dataset_counts'],{'mtbench':1,'arena_hard':21,'wildbench':22})
        self.assertEqual(len(core.units(plan['tasks'])),352)
        self.assertEqual(len({backend.unit_key(u) for u in core.units(plan['tasks'])}),352)
        self.assertEqual(set(backend.unit_key(u) for u in core.ordered_units(plan['tasks'])),set(backend.unit_key(u) for u in core.units(plan['tasks'])))

    def test_missing_exposure_information_blocks_freeze(self):
        from freeze_confirmation import freeze
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'audit.json';p.write_text(json.dumps(load(HERE/'exposure_review.template.json')))
            with self.assertRaises(ValueError):freeze(p,HERE/'evidence',root/'locks')
            self.assertFalse((root/'locks').exists())

    def test_documented_source_review_is_scoped_and_hash_verified(self):
        audit=load(HERE/'source_exposure_review.json')
        self.assertFalse(audit['author_confirms_remaining_tasks_not_used_mechanistically'])
        self.assertEqual(audit['claim_scope'],'held_out_relative_to_documented_sources')
        plan=core.audited_plan(core.candidate_plan(),audit)
        self.assertEqual(len(plan['tasks']),44)
        bad=dict(audit,documented_exposure_report_sha256='0'*64)
        with self.assertRaises(ValueError):core.audited_plan(core.candidate_plan(),bad)
        bad=dict(audit,claim_scope='globally_unseen')
        with self.assertRaises(ValueError):core.audited_plan(core.candidate_plan(),bad)

    def test_exposure_filter_removes_ids_and_question_hashes(self):
        candidate=core.candidate_plan();audit=valid_audit()
        audit['additional_exposed_task_ids']=[candidate['tasks'][0]['base_task_id']]
        audit['additional_exposed_question_sha256']=[candidate['tasks'][1]['question_sha256']]
        plan=core.audited_plan(candidate,audit)
        self.assertEqual(len(plan['tasks']),42)
        self.assertEqual(len(plan['excluded_by_additional_audit']),2)
        self.assertFalse(set(plan['excluded_by_additional_audit'])&{t['base_task_id'] for t in plan['tasks']})
        audit['reviewed_utc']='2026-10-31T00:00:00'
        with self.assertRaises(ValueError):core.audited_plan(candidate,audit)

    def test_frozen_lock_detects_changes_before_inference(self):
        from freeze_confirmation import freeze
        # Verified real proof files; attestation is synthetic and stored only in a
        # temporary directory. No persistent research lock is created by this test.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit=root/'audit.json';audit.write_text(json.dumps(valid_audit()))
            freeze(audit,HERE/'evidence',root/'locks')
            lock,plan,p=core.verify_lock(root/'locks')
            self.assertEqual(lock['planned_units_per_model'],352)
            with self.assertRaises(FileExistsError):freeze(audit,HERE/'evidence',root/'locks')
            path=root/'locks/confirmation_protocol.json';value=load(path);value['max_new_tokens']=32;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):core.verify_lock(root/'locks')

    def test_pilot_evidence_matches_immutable_backend(self):
        for model in core.MODELS:
            proof=core.verify_evidence(HERE/'evidence',model)
            self.assertEqual(len(proof['pilot']),2);self.assertEqual(len(proof['engineering']),2)


class AnalysisAndRecoveryTests(unittest.TestCase):
    def test_signs_and_missing_bounds(self):
        records={}
        for arm,hi,lo in [('baseline',9,6),('native_baseline',8.8,6),('generated_block',8,7),('control_block',8.5,6)]:
            for c,v in zip(backend.CONDS,[hi,lo]):
                r=row('ds/1',c,arm,v);records[backend.unit_key(r)]=r
        self.assertAlmostEqual(evaluate(expr('ds/1','history_gap_reduction'),records)[0],2)
        self.assertAlmostEqual(evaluate(expr('ds/1','history_vs_control_specificity'),records)[0],1.5)
        self.assertAlmostEqual(evaluate(expr('ds/1','explicit_minus_native_gap'),records)[0],.2)
        task={'base_task_id':'ds/1','dataset':'ds'}
        p=load(HERE/'confirmation_protocol.json')
        result=contrast([task],records,'history_gap_reduction',p)
        self.assertEqual(result['confidence_level'],.9875)
        self.assertEqual(result['ci'],[2,2])
        self.assertTrue(result['positive_with_ci_and_missingness_bounds'])
        key=next(iter(expr('ds/1','history_gap_reduction')))
        records[key]['analysis_valid']=False;records[key]['analysis_score']=None
        result=contrast([task],records,'history_gap_reduction',p)
        self.assertFalse(result['positive_with_ci_and_missingness_bounds'])
        self.assertLessEqual(result['planned_task_missing_bounds'][0],2)
        self.assertGreaterEqual(result['planned_task_missing_bounds'][1],2)

    def test_row_guard_rejects_wrong_reader_counts_and_scores(self):
        r=row('ds/1','self_high_10','generated_block');core.verify_row(r,'qwen')
        bad=copy.deepcopy(r);bad['events']['blocked_edges']+=1
        with self.assertRaises(ValueError):core.verify_row(bad,'qwen')
        bad=copy.deepcopy(r);bad['analysis_score']=1.
        with self.assertRaises(ValueError):core.verify_row(bad,'qwen')
        bad=row('ds/1','self_high_10','native_baseline');bad['computation_path']='custom'
        with self.assertRaises(ValueError):core.verify_row(bad,'qwen')

    def test_complete_summary_and_incomplete_coverage_rejection(self):
        from summarize_confirmation import summarize
        import shutil
        task={'base_task_id':'SYNTHETIC/1','dataset':'SYNTHETIC'}
        plan={'tasks':[task],'engineering_tasks':core.candidate_plan()['engineering_tasks'],'dataset_counts':{'SYNTHETIC':1}}
        protocol=load(HERE/'confirmation_protocol.json')
        lock={'source_sha256':core.source_hashes(),'backend_source_sha256':backend.source_hashes(),
              'plan_sha256':'synthetic-plan','confirmation_protocol_sha256':'synthetic-protocol'}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'lock.json').write_text(json.dumps(lock))
            meta=dict(lock,status='complete',phase='focused_confirmation',model='qwen',lock_sha256=sha((root/'lock.json').read_bytes()))
            (root/'metadata.json').write_text(json.dumps(meta))
            shutil.copy2(HERE/'evidence/qwen_engineering/raw.jsonl',root/'engineering_recheck.jsonl')
            lines=[]
            for arm,hi,lo in [('baseline',9.,6.),('native_baseline',8.8,6.),('generated_block',8.,7.),('control_block',8.5,6.)]:
                for c,value in zip(backend.CONDS,[hi,lo]):
                    r=row(task['base_task_id'],c,arm,value);r['payload_sha256']=sha(canonical(r));lines.append(json.dumps(r)+'\n')
            raw=root/'raw.jsonl';raw.write_text(''.join(lines))
            with patch('summarize_confirmation.verify_lock',return_value=(lock,plan,protocol)):
                result=summarize(root)
                self.assertTrue(result['both_predeclared_checks_positive'])
                self.assertAlmostEqual(result['native_path_sensitivity']['mean_points'],.2)
                raw.write_text(''.join(lines[:-1]))
                with self.assertRaises(ValueError):summarize(root)

    def test_interrupted_task_resumes_exactly_once(self):
        from confirm import execute_validation
        task={'base_task_id':'ds/1','dataset':'ds','question_sha256':sha(b'question'),'answer_sha256':sha(b'answer')}
        lookup={'ds/1':{'question':'question','answer':'answer'}}
        class FakeStudy:
            key='qwen'
            def __init__(self):self.calls=[]
            def run_unit(self,u,t,j):
                self.calls.append(backend.unit_key(u));r=row(u['task'],u['condition'],u['arm']);
                return {k:v for k,v in r.items() if k not in u and k!='model'}
        study=FakeStudy();records={};checkpoints=[];count=[0]
        def budget():
            count[0]+=1
            if count[0]==4:raise RuntimeError('synthetic interruption')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'raw.jsonl'
            with p.open('a') as f:
                with self.assertRaises(RuntimeError):execute_validation(study,[task],lookup,records,f,budget,lambda:checkpoints.append(len(records)))
            self.assertEqual(len(records),3)
            expected={backend.unit_key(u) for u in core.units([task])}
            restored=backend.read_journal(p,expected)
            with p.open('a') as f:execute_validation(study,[task],lookup,restored,f,lambda:None,lambda:None)
            self.assertEqual(len(restored),8);self.assertEqual(len(set(study.calls)),8);self.assertEqual(len(study.calls),8)
            core.verify_pairs(restored,[task])
            anyrow=next(iter(restored.values()));wrong=copy.deepcopy(anyrow);wrong['ids'][0]=99
            # Pair guard compares the unchanged first prediction, regardless of later tokens.
            c='self_high_10'
            restored[backend.unit_key(dict(task='ds/1',kind='factorial',condition=c,arm='generated_block'))]['ids'][0]=99
            with self.assertRaises(ValueError):core.verify_pairs(restored,[task])


if __name__=='__main__':unittest.main()
