"""Offline tests: no GPU, network, API key, or repository execution needed."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

COLLECT = Path(__file__).resolve().parents[1] / 'src' / 'collect'
sys.path.insert(0, str(COLLECT))
from navigation_graph import (adjacency, bm25_rank, collect_repo, distance_to_targets,
                              shortest_paths)
from build_navigation_dataset import build_case, classify, read_manifest, features, build_dataset
from train_navigation_scorer import fit, sigmoid, dot
from infer_navigation import infer
from export_navigation_review import summarize


def case_row(target, match=True, confidence='high'):
    return {'instance_id': 'demo__repo-42', 'repo': 'demo/repo',
            'base_commit': 'a' * 40, 'query': 'request does not render the response correctly from handler middleware when processing inbound web requests',
            'issue_url': 'https://github.com/demo/repo/issues/42',
            'positive_id': target, 'llm_relevance_label': 'fix_target',
            'llm_judge_status': 'completed_unverified', 'llm_confidence': confidence,
            'issue_pr_match': match}


class TestNavigationDataset(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        src = self.root / 'src' / 'demo'
        src.mkdir(parents=True)
        (src / '__init__.py').write_text('')
        (src / 'engine.py').write_text('''\
from .utils import finish

def response_handler(x):
    return finish(x)

def unrelated_handler(x):
    return x

class Server:
    def process_request(self, x):
        return self.forward(x)
    def forward(self, x):
        return response_handler(x)
''')
        (src / 'utils.py').write_text('''\
def finish(x):
    return x

def unrelated(x):
    return x
''')
        self.symbols, self.edges, self.stats = collect_repo(self.root)

    def node(self, suffix):
        return next(k for k in self.symbols if k.endswith(suffix))

    def test_static_edges_at_base_commit(self):
        method = self.node('/Server/process_request')
        forward = self.node('/Server/forward')
        handler = self.node('/response_handler')
        finish = self.node('/finish')
        links = {(e['source'], e['target']) for e in self.edges}
        self.assertIn((method, forward), links)
        self.assertIn((forward, handler), links)
        self.assertIn((handler, finish), links)
        self.assertNotIn((self.node('/unrelated_handler'), finish), links)
        self.assertEqual(shortest_paths(method, finish, adjacency(self.edges)[0])[0],
                         [method, forward, handler, finish])

    def test_three_retrieval_cases(self):
        finish = self.node('/finish')
        handler = self.node('/response_handler')
        unrelated = self.node('/unrelated_handler')
        names = sorted(self.symbols)

        def ranking_with_first(node):
            order = [node] + [x for x in names if x != node]
            return [(key, float(len(order) - index)) for index, key in enumerate(order)]

        with patch('build_navigation_dataset.bm25_rank', return_value=ranking_with_first(finish)):
            direct, err = build_case([case_row(finish)], self.symbols, self.edges,
                                     self.stats, top_k=1)
        self.assertIsNone(err)
        self.assertEqual(direct[0]['case_type'], 'direct_retrieval')
        self.assertEqual(direct[0]['target_hops_from_selected_seed'][finish], 0)

        with patch('build_navigation_dataset.bm25_rank', return_value=ranking_with_first(handler)):
            linked, err = build_case([case_row(finish)], self.symbols, self.edges,
                                     self.stats, top_k=1)
        self.assertIsNone(err)
        self.assertEqual(linked[0]['case_type'], 'retrieval_to_target_path')
        self.assertEqual(linked[0]['target_hops_from_selected_seed'][finish], 1)

        with patch('build_navigation_dataset.bm25_rank', return_value=ranking_with_first(unrelated)):
            reverse, err = build_case([case_row(finish)], self.symbols, self.edges,
                                      self.stats, top_k=1, reverse_hops=5)
        self.assertIsNone(err)
        self.assertEqual(reverse[0]['case_type'], 'reverse_oracle_probe')
        self.assertTrue(reverse[0]['oracle_rescue_used_for_training'])
        self.assertGreater(reverse[0]['reverse_probe_best_rank'], 1)
        self.assertTrue(reverse[0]['oracle_not_available_at_inference'])
        self.assertNotIn('teacher_distance', next(iter(reverse[1]))['features'])

    def test_reject_bad_pr_and_weak_labels(self):
        finish = self.node('/finish')
        self.assertIsNone(classify(case_row(finish, match=False)))
        invalid = case_row(finish)
        invalid['llm_relevance_label'] = 'incidental_change'
        self.assertIsNone(classify(invalid))
        self.assertIsNone(build_case([invalid], self.symbols, self.edges, self.stats)[0])
        valid = case_row(finish)
        valid['reviewer_label'] = 'verified_fix_target'
        self.assertEqual(classify(valid), 'human_verified')

    def test_all_shortest_paths_and_reverse_distance(self):
        outgoing = {'a': {'b', 'c'}, 'b': {'d'}, 'c': {'d'}}
        incoming = {'d': {'b', 'c'}, 'b': {'a'}, 'c': {'a'}}
        self.assertEqual(shortest_paths('a', 'd', outgoing),
                         [['a', 'b', 'd'], ['a', 'c', 'd']])
        self.assertEqual(distance_to_targets(['d'], incoming)['a'], 2)

    def test_end_to_end_dataset_and_no_oracle_inference(self):
        finish = self.node('/finish')
        row = case_row(finish)
        p = self.root / 'audited.jsonl'
        p.write_text(json.dumps(row) + '\n')
        manifest = self.root / 'benchmarks.json'
        manifest.write_text(json.dumps({'sources': {
            'princeton-nlp/SWE-bench': {'instances': 2294},
            'czlll/Loc-Bench_V1': {'instances': 560}},
            'repositories': [f'official/eval{i}' for i in range(12)]}))
        with patch('build_navigation_dataset.get_checkout', return_value=self.root):
            stats = build_dataset(p, self.root / 'dataset', manifest, self.root / 'cache',
                                  top_k=1, retriever='bm25')
        self.assertEqual(stats['cases'], 1)
        self.assertTrue((self.root / 'dataset' / 'navigation_cases.jsonl').exists())
        self.assertTrue((self.root / 'dataset' / 'stop_training.jsonl').exists())
        model = {'edge_model': {'weights': [0] * 10, 'bias': 0.0},
                 'stop_model': {'weights': [0] * 10, 'bias': 1.0}}
        result = infer(row['query'], self.symbols, self.edges, model,
                       top_k=2, budget=5, stop_threshold=.7)
        self.assertLessEqual(result['expanded_nodes'], 5)
        self.assertEqual(result['llm_calls'], 0)
        self.assertFalse(result['model_uses_oracle_targets'])
        self.assertTrue(result['marked_functions'])

    def test_benchmark_exclusion_fail_closed(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({'sources': {'princeton-nlp/SWE-bench': {'instances': 2294}},
                                    'repositories': ['demo/repo']}))
        with self.assertRaises(ValueError):
            read_manifest(path)
        path.write_text(json.dumps({'sources': {
            'princeton-nlp/SWE-bench': {'instances': 2294},
            'czlll/Loc-Bench_V1': {'instances': 560}},
            'repositories': [f'a/repo{i}' for i in range(10)]}))
        self.assertEqual(len(read_manifest(path)[0]), 10)

    def test_review_summary_handles_unreached_multiple_targets(self):
        cases = [{'case_type': 'reverse_oracle_probe', 'accepted_targets': {'target_a': 'llm_pseudo', 'target_b': 'llm_pseudo'},
                  'gold_paths': [{'target': 'target_a', 'seed': 'seed', 'nodes': ['seed', 'target_a'], 'hops': 1}]}]
        result = summarize(cases, [])
        self.assertEqual(result['unreached_accepted_target_count'], 1)
        self.assertEqual(result['hop_distribution_min_per_issue'], {'1': 1})
        self.assertEqual(result['hop_distribution_min_per_target'], {'1': 1})

    def test_scorer_no_teacher_feature_leakage(self):
        feat = {k: .2 for k in __import__('build_navigation_dataset').FEATURES}
        samples = [{'features': {**feat, 'candidate_relevance': .95}, 'label': 1, 'repo': 'train/repo'},
                   {'features': {**feat, 'candidate_relevance': .05}, 'label': 0, 'repo': 'train/repo'}]
        model = fit(samples, epochs=300)
        self.assertGreater(sigmoid(dot(model['weights'], samples[0]['features'], model['bias'])),
                           sigmoid(dot(model['weights'], samples[1]['features'], model['bias'])))
        self.assertNotIn('teacher_distance', model['features'])


if __name__ == '__main__':
    unittest.main()