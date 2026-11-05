#!/usr/bin/env python3
"""Export manual path-review queue and case/hop statistics from actual graph cases.

All computed paths are graph proxies, NEVER confirmed causal/functional ground truth.
Unreachable/rejected issues remain explicit so manual reviewers can audit selection bias.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def read_jsonl(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def summarize(cases, rejected):
    counts = Counter(c.get('case_type', 'unknown') for c in cases)
    n = len(cases)
    paths = Counter()
    best_target_hops = Counter()
    best_case_hops = Counter()
    missing = 0
    for c in cases:
        allpaths = c.get('gold_paths', [])
        for p in allpaths:
            paths[str(p['hops'])] += 1
        by_target = defaultdict(list)
        for p in allpaths:
            by_target[p['target']].append(p['hops'])
        for t in c.get('accepted_targets', {}):
            if by_target[t]:
                best_target_hops[str(min(by_target[t]))] += 1
            else:
                missing += 1
        available_hops = [min(v) for v in by_target.values() if v]
        if available_hops:
            best_case_hops[str(min(available_hops))] += 1
        else:
            best_case_hops['unreached'] += 1
    return {
        'accepted_issue_count': n,
        'rejected_issue_count': len(rejected),
        'case_distribution': {k: {'count': counts[k], 'percentage': round(100 * counts[k] / n, 2) if n else 0.0}
                              for k in ['direct_retrieval', 'retrieval_to_target_path', 'reverse_oracle_probe']},
        'hop_distribution_proxy_paths': dict(sorted(paths.items(), key=lambda v: int(v[0]))),
        'hop_distribution_min_per_target': dict(sorted(best_target_hops.items(), key=lambda v: int(v[0]))),
        'hop_distribution_min_per_issue': dict(sorted(best_case_hops.items(), key=lambda v: -1 if v[0] == 'unreached' else int(v[0]))),
        'unreached_accepted_target_count': missing,
        'note': 'Proxy graph shortest paths, NOT causally verified path ground truth. In reverse case, path starts from teacher-only oracle rescue; retrieval originally failed.'
    }


def code_url(repo, sha, symbol, graph_nodes):
    node = graph_nodes.get(symbol, {})
    file = node.get('path')
    if not file and '.py/' in symbol:
        file = symbol.split('.py/')[0] + '.py'
    if not file:
        return ''
    loc = '#L' + str(node['start_line']) if node.get('start_line') else ''
    return f'https://github.com/{repo}/blob/{sha}/{file}{loc}'


def export(cases_path, rejected_path, graph_dir, audit_path, output_dir):
    cases = read_jsonl(cases_path)
    rejected = read_jsonl(rejected_path)
    audit = read_jsonl(audit_path) if audit_path else []
    audits = defaultdict(list)
    for row in audit:
        audits[(row.get('instance_id'), row.get('base_commit'))].append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields = [
        'instance_id', 'issue_url', 'pr_urls', 'repo', 'base_commit', 'case_type',
        'retriever', 'top_k', 'target', 'target_quality', 'llm_label', 'llm_confidence',
        'llm_evidence', 'seed', 'retrieval_seed_rank', 'target_retrieval_rank',
        'hops', 'path_status', 'path_nodes', 'path_edge_types',
        'seed_source_url', 'target_source_url', 'oracle_rescue_teacher_only',
        'target_review', 'path_review', 'reviewer_notes'
    ]
    rows = []
    for c in cases:
        instance = c['instance_id']
        repo = c['repo']
        sha = c['base_commit']
        filename = repo.replace('/', '__') + '_' + sha[:12] + '.json'
        graph_file = Path(graph_dir) / filename
        graph = json.loads(graph_file.read_text(encoding='utf-8')) if graph_file.exists() else {}
        nodes = {n['id']: n for n in graph.get('nodes', [])}
        etypes = {(e['source'], e['target']): e.get('type', '') for e in graph.get('edges', [])}
        evaluations = {x['positive_id']: x for x in audits.get((instance, sha), [])}
        pr_urls = sorted({x.get('pr_url', '') for x in audits.get((instance, sha), []) if x.get('pr_url')})
        seedranks = {s.get('node'): s.get('rank') for s in c.get('seed_scores', [])}
        for candidate in c.get('reverse_probe_top', []):
            seedranks.setdefault(candidate['node'], candidate['retrieval_rank'] + 1)
        listed = list(c.get('gold_paths', []))
        alltargets = c.get('accepted_targets', {})
        for target in sorted(alltargets):
            matching = [p for p in listed if p['target'] == target]
            if not matching:
                matching = [{'target': target, 'seed': '', 'nodes': [], 'hops': None}]
            judgement = evaluations.get(target, {})
            for p in matching:
                seq = p['nodes']
                rows.append({
                    'instance_id': instance, 'issue_url': c.get('issue_url', ''),
                    'pr_urls': ' | '.join(pr_urls), 'repo': repo,
                    'base_commit': sha, 'case_type': c['case_type'],
                    'retriever': c.get('retriever', ''), 'top_k': c.get('retrieval_k', ''),
                    'target': target, 'target_quality': alltargets[target],
                    'llm_label': judgement.get('llm_relevance_label', ''),
                    'llm_confidence': judgement.get('llm_confidence', ''),
                    'llm_evidence': judgement.get('llm_evidence', ''),
                    'seed': p['seed'], 'retrieval_seed_rank': seedranks.get(p['seed'], ''),
                    'target_retrieval_rank': seedranks.get(target, ''),
                    'hops': p['hops'] if p['hops'] is not None else '',
                    'path_status': 'static_shortest_path_proxy' if seq else 'no_static_path',
                    'path_nodes': ' -> '.join(seq),
                    'path_edge_types': ' -> '.join(etypes.get((a, b), 'unknown')
                                                    for a, b in zip(seq, seq[1:])),
                    'seed_source_url': code_url(repo, sha, p['seed'], nodes) if p['seed'] else '',
                    'target_source_url': code_url(repo, sha, target, nodes),
                    'oracle_rescue_teacher_only': c['case_type'] == 'reverse_oracle_probe',
                    'target_review': '', 'path_review': '', 'reviewer_notes': ''
                })
    with (output / 'manual_path_review.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # Separate diagnostic for minimum-length 2-5-hop ancestors; never mix with
    # the actual retrieval-seeded path distribution or claim these as gold paths.
    deep_rows = []
    for c in cases:
        repo, sha = c['repo'], c['base_commit']
        graph_file = Path(graph_dir) / (repo.replace('/', '__') + '_' + sha[:12] + '.json')
        graph = json.loads(graph_file.read_text(encoding='utf-8')) if graph_file.exists() else {}
        nodes = {n['id']: n for n in graph.get('nodes', [])}
        etypes = {(e['source'], e['target']): e.get('type', '') for e in graph.get('edges', [])}
        for p in c.get('deep_hop_probes', []):
            seq = p['nodes']
            deep_rows.append({
                'instance_id': c['instance_id'], 'issue_url': c.get('issue_url', ''),
                'repo': repo, 'base_commit': sha, 'target': p['target'],
                'target_quality': c.get('accepted_targets', {}).get(p['target'], ''),
                'retriever': c.get('retriever', ''), 'top_k': c.get('retrieval_k'),
                'ancestor': p['ancestor'], 'ancestor_retrieval_rank': p['ancestor_retrieval_rank'],
                'in_initial_top_k': p['in_initial_top_k'],
                'hops': p['hops'], 'path_nodes': ' -> '.join(seq),
                'path_edge_types': ' -> '.join(etypes.get((a,b), 'unknown') for a,b in zip(seq,seq[1:])),
                'ancestor_source_url': code_url(repo, sha, p['ancestor'], nodes),
                'target_source_url': code_url(repo, sha, p['target'], nodes),
                'path_status': 'offline_oracle_probe_shortest_static_path',
                'is_actual_retrieval_navigation_path': p['in_initial_top_k'],
                'manual_review': '', 'reviewer_notes': ''
            })
    deep_fields = ['instance_id', 'issue_url', 'repo', 'base_commit', 'target', 'target_quality',
                   'retriever', 'top_k', 'ancestor', 'ancestor_retrieval_rank',
                   'in_initial_top_k', 'hops', 'path_nodes', 'path_edge_types',
                   'ancestor_source_url', 'target_source_url', 'path_status',
                   'is_actual_retrieval_navigation_path', 'manual_review', 'reviewer_notes']
    with (output / 'deep_hop_path_review.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=deep_fields)
        writer.writeheader()
        writer.writerows(deep_rows)
    stats = summarize(cases, rejected)
    stats['manual_review_rows'] = len(rows)
    stats['deep_hop_diagnostic_paths'] = len(deep_rows)
    stats['deep_hop_distribution'] = dict(sorted(Counter(str(x['hops']) for x in deep_rows).items()))
    stats['deep_hop_from_actual_topk'] = sum(x['in_initial_top_k'] for x in deep_rows)
    (output / 'case_hop_statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'rejected_issue_summary.json').write_text(json.dumps([
        {'instance_id': x.get('instance_id'), 'reason': x.get('reason'), 'repo': x.get('repo')}
        for x in rejected], indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', required=True)
    p.add_argument('--rejected', required=True)
    p.add_argument('--graphs', required=True)
    p.add_argument('--audit', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    print(json.dumps(export(args.cases, args.rejected, args.graphs, args.audit, args.output_dir), indent=2))


if __name__ == '__main__':
    main()