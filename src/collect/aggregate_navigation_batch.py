#!/usr/bin/env python3
"""Merge independent benchmark-disjoint repository navigation outputs.

Uses each repository's *real* outputs; never substitutes synthetic paths or
promotes LLM pseudo-targets to human verified ground truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

FILES = ('navigation_cases.jsonl', 'rejected_cases.jsonl', 'navigation_steps.jsonl',
         'seed_training.jsonl', 'stop_training.jsonl')


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as dest:
        for row in rows:
            dest.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def aggregate(artifacts_root, output_dir, expected_repos=None):
    root = Path(artifacts_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = sorted(root.glob('*/dataset/summary.json'))
    if not summaries:
        raise ValueError('No per-repository summary.json was found: ' + str(root))
    merged = {name: [] for name in FILES}
    reviewed = []
    seen = {name: set() for name in FILES}
    per_repo = {}
    cases_hist = Counter()
    hops_hist = Counter()
    deep_hops_hist = Counter()
    selected_paths = 0
    deep_oracle = 0
    reviewed_targets = Counter()
    audit_summaries = {}
    all_manifests = set()
    for summary_path in summaries:
        repo_dir = summary_path.parent
        artifact_name = summary_path.parents[1].name
        repo = artifact_name.removeprefix('nav-repo-').replace('__', '/')
        if repo in per_repo:
            raise ValueError('Duplicate repository artifact: ' + repo)
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        all_manifests.add(summary.get('manifest_sha256'))
        if not summary.get('manifest_sha256'):
            raise ValueError('Missing benchmark manifest hash: ' + repo)
        repo_cases = read_jsonl(repo_dir / 'navigation_cases.jsonl')
        # Refuse mislabeled/cross-repo inputs rather than silently corrupt the dataset.
        if any(row.get('repo') != repo for row in repo_cases):
            raise ValueError('Artifact repo does not match row repo: ' + repo)
        if summary['cases'] != len(repo_cases):
            raise ValueError('Case-count mismatch for ' + repo)
        if any((row.get('gold_status') != 'static_shortest_path_proxy_not_causal_truth')
               for row in repo_cases):
            raise ValueError('Unexpected gold status in ' + repo)
        for name in FILES:
            for row in read_jsonl(repo_dir / name):
                key = json.dumps(row, ensure_ascii=False, sort_keys=True)
                if key not in seen[name]:
                    merged[name].append(row)
                    seen[name].add(key)
        for case in repo_cases:
            cases_hist[case['case_type']] += 1
            reviewed_targets.update(case.get('accepted_targets', {}).values())
            for record in case.get('gold_paths', []):
                hops_hist[str(record['hops'])] += 1
                selected_paths += 1
            for record in case.get('deep_hop_probes', []):
                deep_hops_hist[str(record['hops'])] += 1
                if record.get('oracle_only'):
                    deep_oracle += 1
        review_csv = repo_dir / 'manual_path_review.csv'
        if review_csv.exists():
            with review_csv.open(encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f):
                    reviewed.append({'repo': repo, **row})
        stats_path = repo_dir / 'case_hop_statistics.json'
        if stats_path.exists():
            extra_stats = json.loads(stats_path.read_text(encoding='utf-8'))
        else:
            extra_stats = {}
        audit_path = summary_path.parents[1] / 'audit' / 'enrichment_metadata.json'
        if audit_path.exists():
            audit_summaries[repo] = json.loads(audit_path.read_text(encoding='utf-8'))
        per_repo[repo] = {'issues': summary.get('eligible_issues', 0), 'accepted_cases': len(repo_cases),
                          'rejected_cases': summary.get('rejected_cases', 0),
                          'weak_functions': summary.get('input_rows', 0),
                          'case_distribution': summary.get('case_distribution', {}),
                          'deep_hop_diagnostic_paths': extra_stats.get('deep_hop_diagnostic_paths', 0),
                          'max_hop': max([int(k) for k in extra_stats.get('deep_hop_distribution', {})], default=0),
                          'benchmark_manifest_sha256': summary['manifest_sha256']}
    if len(all_manifests) != 1:
        raise ValueError('Benchmark exclusions changed across repository jobs')
    for name, rows in merged.items():
        write_jsonl(out / ('combined_' + name), rows)
    fields = list(dict.fromkeys(k for row in reviewed for k in row))
    with (out / 'combined_manual_path_review.csv').open('w', newline='', encoding='utf-8') as f:
        if fields:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(reviewed)
    n = len(merged['navigation_cases.jsonl'])
    kinds = ['direct_retrieval', 'retrieval_to_target_path', 'reverse_oracle_probe']
    case_distribution = {k: {'count': cases_hist[k], 'percentage': round(cases_hist[k]*100/max(1,n), 2)} for k in kinds}
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'dataset_quality': 'llm_pseudo_static_shortest_path_proxy_not_verified_gold',
              'benchmark_manifest_sha256': next(iter(all_manifests)),
              'expected_repository_count': expected_repos,
              'completed_repository_count': len(per_repo),
              'repositories': per_repo,
              'total_issues': sum(v['issues'] for v in per_repo.values()),
              'accepted_cases': n,
              'rejected_cases': len(merged['rejected_cases.jsonl']),
              'weak_function_pairs': sum(v['weak_functions'] for v in per_repo.values()),
              'case_distribution': case_distribution,
              'selected_proxy_path_hops': dict(sorted(hops_hist.items(), key=lambda x:int(x[0]))),
              'deep_probe_path_hops': dict(sorted(deep_hops_hist.items(),key=lambda x:int(x[0]))),
              'selected_proxy_path_count': selected_paths,
              'deep_oracle_probe_paths': deep_oracle,
              'human_verified_targets': reviewed_targets.get('human_verified',0),
              'llm_pseudo_targets': reviewed_targets.get('llm_pseudo',0),
              'navigation_steps': len(merged['navigation_steps.jsonl']),
              'seed_samples': len(merged['seed_training.jsonl']),
              'stop_samples': len(merged['stop_training.jsonl']),
              'manual_review_rows': len(reviewed),
              'audit_usage': {repo: a.get('llm_usage', {}) for repo,a in audit_summaries.items()},
              'note': 'Paths are static shortest-path proxies, not verified causal ground truth. Deep probes can be oracle-only.'}
    (out / 'batch_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# Graph Navigation — Multi-repository batch', '',
             f"- Repositories completed: {len(per_repo)}",
             f"- Issue candidates: {result['total_issues']}",
             f"- Accepted cases: {n}",
             f"- Rejected cases: {result['rejected_cases']}",
             f"- Weak function pairs: {result['weak_function_pairs']}",
             f"- Selected proxy paths: {selected_paths}",
             f"- Deep diagnostic paths (2–5 hops): {sum(deep_hops_hist.values())}",
             '', '| Repo | Issues | Cases | Rejected | Deep paths |',
             '|---|---:|---:|---:|---:|']
    for repo, v in sorted(per_repo.items()):
        lines.append(f"| {repo} | {v['issues']} | {v['accepted_cases']} | {v['rejected_cases']} | {v['deep_hop_diagnostic_paths']} |")
    lines.extend(['', '**Label notice:** all graph paths are static proxies; an LLM pseudo-label is not a human verification.'])
    (out / 'BATCH_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--expected-repos', type=int)
    args = p.parse_args()
    result = aggregate(args.artifacts_root, args.output_dir, args.expected_repos)
    print(json.dumps({k:v for k,v in result.items() if k!='repositories'}, indent=2, ensure_ascii=False))
    if not result['accepted_cases']:
        raise SystemExit('No accepted cases collected; see per-repo audit artifacts')


if __name__ == '__main__':
    main()