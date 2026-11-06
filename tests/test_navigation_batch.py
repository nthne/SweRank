"""Regression tests for multi-repo aggregation, dedup and leakage guards."""
import json
import subprocess
from unittest.mock import patch
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'/'collect'))
from aggregate_navigation_batch import aggregate
from build_navigation_dataset import get_checkout


class TestBatchAggregation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_repo(self, name, cases=None, hashval='same', rejected=None):
        p=self.root/'data'/('nav-repo-'+name.replace('/','__'))
        d=p/'dataset'
        d.mkdir(parents=True)
        cases=cases or []
        rejected=rejected or []
        def dump(name, rows):
            (d/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        dump('navigation_cases.jsonl',cases)
        dump('rejected_cases.jsonl',rejected)
        (d/'summary.json').write_text(json.dumps({'cases':len(cases),'eligible_issues':len(cases)+len(rejected),'rejected_cases':len(rejected),'input_rows':len(cases)+len(rejected),'manifest_sha256':hashval,'case_distribution':{'direct_retrieval':len(cases)}}))
        (d/'case_hop_statistics.json').write_text(json.dumps({'deep_hop_diagnostic_paths':sum(len(x.get('deep_hop_probes', [])) for x in cases),'deep_hop_distribution':{}}))
        return p

    def case(self, repo, n, hops=2):
        return {'instance_id':f'{repo}-{n}','repo':repo,'case_type':'retrieval_to_target_path',
                'accepted_targets':{'src/main.py/fix':'llm_pseudo'},
                'gold_status':'static_shortest_path_proxy_not_causal_truth',
                'gold_paths':[{'hops':hops}],
                'deep_hop_probes':[{'hops':hops,'oracle_only':True}]}

    def test_merge_multi_repo_paths(self):
        self.make_repo('owner/first',[self.case('owner/first',1,2)])
        self.make_repo('owner/second',[self.case('owner/second',2,4)],rejected=[{'repo':'owner/second','reason':'no_eligible_targets'}])
        x=aggregate(self.root/'data',self.root/'out',9)
        self.assertEqual(x['completed_repository_count'],2)
        self.assertEqual(x['accepted_cases'],2)
        self.assertEqual(x['rejected_cases'],1)
        self.assertEqual(x['selected_proxy_path_hops'],{'2':1,'4':1})
        self.assertEqual(x['deep_probe_path_hops'],{'2':1,'4':1})
        self.assertEqual(x['llm_pseudo_targets'],2)
        self.assertEqual(x['human_verified_targets'],0)


    def test_pinned_worktrees_reuse_single_clone(self):
        source = self.root / 'mock-remote'
        source.mkdir()
        subprocess.run(['git', 'init', '-q', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'config', 'user.email', 'test@example.invalid'], check=True)
        subprocess.run(['git', '-C', str(source), 'config', 'user.name', 'Test'], check=True)
        (source / 'main.py').write_text('def before(): pass\n')
        subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(source), 'commit', '-qm', 'first'], check=True)
        one = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],text=True).strip()
        (source / 'main.py').write_text('def after(): pass\n')
        subprocess.run(['git', '-C', str(source), 'commit', '-qam', 'second'], check=True)
        two = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],text=True).strip()
        real_run=subprocess.run
        clones=[]
        def reroute(cmd, **kwargs):
            if cmd[:2] == ['git', 'clone']:
                clones.append(cmd)
                cmd=list(cmd)
                cmd[-2]=str(source)
            return real_run(cmd, **kwargs)
        with patch('build_navigation_dataset.subprocess.run', side_effect=reroute):
            old=get_checkout('owner/example',one,self.root/'checkouts')
            new=get_checkout('owner/example',two,self.root/'checkouts')
        self.assertEqual(len(clones),1)
        self.assertIn('before', (old/'main.py').read_text())
        self.assertIn('after', (new/'main.py').read_text())

    def test_manifest_mismatch_fails(self):
        self.make_repo('owner/one',hashval='a')
        self.make_repo('owner/two',hashval='b')
        with self.assertRaisesRegex(ValueError,'exclusions changed'):
            aggregate(self.root/'data',self.root/'out')

    def test_repo_mismatch_fails(self):
        self.make_repo('owner/one',[self.case('owner/two',1)])
        with self.assertRaisesRegex(ValueError,'Artifact repo'):
            aggregate(self.root/'data',self.root/'out')

    def test_no_artifacts_fails(self):
        with self.assertRaisesRegex(ValueError,'No per-repository'):
            aggregate(self.root/'data',self.root/'out')


if __name__=='__main__':
    unittest.main()