#!/usr/bin/env python3
"""Build issue-conditioned multi-hop navigation training data from reviewed SWELOC rows.

Targets are used ONLY offline as teacher labels, never as inference features.
Only call graphs over the repository's PR base commit are considered.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from navigation_graph import (adjacency, bm25_rank, collect_repo, distance_to_targets,
                              shortest_paths, swerank_rank)

REPO_PATTERN = re.compile(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$')
SHA_PATTERN = re.compile(r'^[0-9a-f]{40}$')
FEATURES = ['current_relevance', 'candidate_relevance', 'candidate_rank_fraction',
            'candidate_out_degree', 'candidate_in_degree', 'hop_fraction',
            'edge_lexical', 'edge_method', 'edge_imported', 'candidate_in_top_k']


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def read_manifest(path):
    """Fail closed: both benchmark sources must be represented and checked."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'princeton-nlp/SWE-bench', 'czlll/Loc-Bench_V1'}
    sources = manifest.get('sources', {})
    if not required.issubset(sources):
        raise ValueError('Benchmark manifest must contain SWE-bench and Loc-Bench_V1')
    for key in required:
        if int(sources[key].get('instances', 0)) < 100:
            raise ValueError('Incomplete benchmark source: ' + key)
    excluded = {str(s).lower() for s in manifest.get('repositories', [])}
    if len(excluded) < 10:
        raise ValueError('Manifest must enumerate all evaluation repositories')
    return excluded, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_checkout(repo, base, root):
    """Reuse one blobless bare clone per repository and pin worktrees to base SHAs.

    This avoids cloning the full Git history once for *every* issue/PR.
    No repository code or hooks are executed.
    """
    if not REPO_PATTERN.fullmatch(repo) or not SHA_PATTERN.fullmatch(base):
        raise ValueError('Invalid repo or full commit SHA')
    root = Path(root).resolve()
    slug = repo.replace('/', '__')
    checkout = root / slug / base
    if (checkout / '.git').exists():
        return checkout
    bare = root / '.object-cache' / (slug + '.git')
    bare.parent.mkdir(parents=True, exist_ok=True)
    if not bare.exists():
        subprocess.run(['git', 'clone', '--quiet', '--bare', '--filter=blob:none',
                        'https://github.com/' + repo + '.git', str(bare)],
                       check=True, timeout=360)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    # `worktree add` uses a pinned commit from the shared bare repository.
    # An unavailable SHA is a build error, not a reason to fall back to HEAD.
    subprocess.run(['git', '-C', str(bare), 'worktree', 'add', '--quiet',
                    '--detach', str(checkout), base], check=True, timeout=240)
    return checkout

def classify(row):
    """LLM is provisional, reviewer evidence can elevate a target to verified."""
    label = (row.get('reviewer_label') or '').strip().lower()
    if label in ('verified_fix_target', 'verified_target', 'fix_target_verified'):
        return 'human_verified'
    if label not in ('', 'unreviewed', 'pending'):
        return None
    if row.get('issue_pr_match') is False or row.get('llm_relevance_label') != 'fix_target':
        return None
    if row.get('llm_judge_status') not in ('completed_unverified', 'cache_hit_unverified'):
        return None
    if row.get('llm_confidence') in ('high', 'medium'):
        return 'llm_pseudo'
    return None


def normalized_ranks(ranking):
    scores = [score for _, score in ranking]
    lo, hi = min(scores, default=0), max(scores, default=0)
    return {key: (score - lo) / (hi - lo) if hi > lo else 0.0
            for key, score in ranking}


def features(current, candidate, hop, edge_type, relevance, ranks, outgoing, incoming, top_k, size, max_hops):
    r = ranks.get(candidate, size)
    return {
        'current_relevance': round(relevance.get(current, 0.0), 6),
        'candidate_relevance': round(relevance.get(candidate, 0.0), 6),
        'candidate_rank_fraction': round(1.0 - r / max(1, size), 6),
        'candidate_out_degree': min(1.0, len(outgoing[candidate]) / 20.0),
        'candidate_in_degree': min(1.0, len(incoming[candidate]) / 20.0),
        'hop_fraction': round(min(1.0, hop / max(1, max_hops)), 6),
        'edge_lexical': int(edge_type == 'lexical_call'),
        'edge_method': int(edge_type == 'method_call'),
        'edge_imported': int(edge_type in ('imported_call', 'module_call')),
        'candidate_in_top_k': int(r < top_k)
    }


def build_case(rows, symbols, edges, repo_stats, top_k=20, max_hops=8,
               reverse_hops=5, retriever='bm25'):
    first = rows[0]
    issue = first['query']
    candidates = {}
    rejected_targets = []
    for row in rows:
        target = row['positive_id']
        level = classify(row)
        if level and target in symbols:
            if target not in candidates or level == 'human_verified':
                candidates[target] = level
        else:
            rejected_targets.append({'id': target, 'label': row.get('llm_relevance_label'),
                                     'issue_pr_match': row.get('issue_pr_match'),
                                     'reason': 'unverified_or_not_in_base_graph'})
    if not candidates:
        return None, {'instance_id': first['instance_id'], 'repo': first['repo'],
                      'base_commit': first['base_commit'], 'reason': 'no_eligible_targets',
                      'rejected_targets': rejected_targets}
    ranking = (swerank_rank(issue, symbols) if retriever == 'swerank'
               else bm25_rank(issue, symbols))
    if not ranking:
        return None, {'instance_id': first['instance_id'], 'reason': 'empty_graph'}
    relevance = normalized_ranks(ranking)
    ids = [key for key, _ in ranking]
    ranks = {key: i for i, key in enumerate(ids)}
    seeds = ids[:top_k]
    targets = sorted(candidates)
    outgoing, incoming = adjacency(edges)
    target_distance = distance_to_targets(targets, incoming, max_hops=max(len(symbols), max_hops))
    direct = sorted(set(seeds) & set(targets))
    retrievable = {(seed, target): shortest_paths(seed, target, outgoing, max_hops=max_hops, max_paths=8)
                   for seed in seeds for target in targets}
    reaches = {(seed, target): paths for (seed, target), paths in retrievable.items() if paths}
    fallback = []
    if direct:
        case_type = 'direct_retrieval'
    elif reaches:
        case_type = 'retrieval_to_target_path'
    else:
        case_type = 'reverse_oracle_probe'
        for key, hop in target_distance.items():
            if 1 <= hop <= reverse_hops and key not in seeds:
                fallback.append({'node': key, 'retrieval_rank': ranks.get(key),
                                 'distance_to_target': hop,
                                 'retrieval_score': dict(ranking).get(key)})
        fallback.sort(key=lambda r: (r['retrieval_rank'], r['distance_to_target'], r['node']))

    # For research/manual review, expose genuinely minimal multi-hop caller chains.
    # This is OFFLINE oracle-only supervision, NEVER a candidate generated at inference.
    # Do not manufacture extra hops by extending a shorter path.
    deep_hop_probes = []
    for target in targets:
        reverse_dists = distance_to_targets([target], incoming, max_hops=reverse_hops)
        for distance in range(2, reverse_hops + 1):
            ancestors = sorted((ranks[node], node) for node, hop in reverse_dists.items()
                               if hop == distance and node in ranks)
            for rank, ancestor in ancestors[:5]:
                paths = shortest_paths(ancestor, target, outgoing,
                                       max_hops=reverse_hops, max_paths=1)
                if paths and len(paths[0]) - 1 == distance:
                    deep_hop_probes.append({
                        'target': target, 'ancestor': ancestor,
                        'ancestor_retrieval_rank': rank + 1,
                        'in_initial_top_k': rank < top_k,
                        'hops': distance, 'nodes': paths[0],
                        'oracle_only': True,
                        'is_actual_retrieval_navigation_path': rank < top_k,
                        'status': 'static_shortest_path_proxy_not_verified_causal_path'
                    })

    # Oracle fallback is diagnostic and never a claimed inference-time action.
    rescue_seeds = [f['node'] for f in fallback[:1]] if fallback else []
    training_seeds = seeds if case_type != 'reverse_oracle_probe' else rescue_seeds
    gold_paths, chosen_seed = [], {}
    for target in targets:
        possibilities = [(target, [[target]])] if target in seeds else []
        for seed in ([] if target in seeds else training_seeds):
            paths = shortest_paths(seed, target, outgoing, max_hops=max_hops)
            if paths:
                possibilities.append((seed, paths))
        if possibilities:
            # Choose a seed by issue relevance first, then use shortest graph paths.
            seed, paths = min(possibilities, key=lambda x: (-relevance[x[0]], len(x[1][0]), x[0]))
            chosen_seed[target] = seed
            gold_paths.extend({'target': target, 'seed': seed, 'nodes': path,
                               'hops': len(path) - 1} for path in paths)
    gold_edges = {(a, b) for rec in gold_paths for a, b in zip(rec['nodes'], rec['nodes'][1:])}
    edge_types = {(e['source'], e['target']): e['type'] for e in edges}
    training = []
    for a in sorted({start for start, _ in gold_edges}):
        # Independent candidate action supervision: all valid shortest-path edges are positives.
        hop = min((p['nodes'].index(a) for p in gold_paths if a in p['nodes'][:-1]), default=0)
        for candidate in sorted(outgoing[a]):
            valid = (a, candidate) in gold_edges
            reaches_target = candidate in target_distance
            if valid:
                label, weight = 1, 1.0
            elif reaches_target:
                label, weight = None, 0.0  # Alternate reachable routes are NOT hard negatives.
            else:
                label, weight = 0, 1.0
            training.append({'instance_id': first['instance_id'], 'repo': first['repo'],
                             'base_commit': first['base_commit'], 'query': issue,
                             'current': a, 'candidate': candidate, 'hop': hop,
                             'edge_type': edge_types.get((a, candidate), 'unknown'),
                             'features': features(a, candidate, hop, edge_types.get((a, candidate)),
                                                  relevance, ranks, outgoing, incoming,
                                                  top_k, len(ids), max_hops),
                             'label': label, 'sample_weight': weight,
                             'teacher_distance': target_distance.get(candidate),
                             'target_quality': 'human_verified' if all(v == 'human_verified'
                                                for v in candidates.values()) else 'llm_pseudo_mixed',
                             'candidate_code': symbols[candidate].source[:1400]})

    stop_examples = []
    seen_stops = set()
    for path in gold_paths:
        for hop, node in enumerate(path['nodes']):
            key = (node, hop)
            if key in seen_stops:
                continue
            seen_stops.add(key)
            stop_examples.append({'instance_id': first['instance_id'], 'repo': first['repo'],
                                  'base_commit': first['base_commit'], 'query': issue,
                                  'current': node, 'label': int(node in candidates),
                                  'features': features(node, node, hop, None, relevance,
                                                       ranks, outgoing, incoming,
                                                       top_k, len(ids), max_hops),
                                  'stop_meaning': 'mark_node_as_fix_target_not_end_multi_target_search'})
    seed_examples = []
    selected = set(chosen_seed.values())
    for seed in seeds:
        if seed in selected:
            label = 1
        elif seed not in target_distance:
            label = 0
        else:
            label = None
        seed_examples.append({'instance_id': first['instance_id'], 'repo': first['repo'],
                              'base_commit': first['base_commit'], 'query': issue,
                              'seed': seed, 'label': label, 'relevance': relevance[seed],
                              'rank': ranks[seed] + 1,
                              'teacher_distance': target_distance.get(seed),
                              'code': symbols[seed].source[:1400]})
    sorted_seed_scores = []
    for seed in seeds:
        dist = target_distance.get(seed)
        score = 0 if dist is None else .7 / (dist + 1) + .3 * relevance[seed]
        sorted_seed_scores.append({'node': seed, 'rank': ranks[seed] + 1,
                                   'issue_relevance': round(relevance[seed], 6),
                                   'teacher_score': round(score, 6),
                                   'teacher_distance': dist,
                                   'label': ('selected_seed' if seed in selected else
                                             'unreachable_in_static_graph' if dist is None else
                                             'reachable_alternative')})

    status = ('human_verified_proxy_paths' if all(v == 'human_verified' for v in candidates.values())
              else 'llm_pseudo_proxy_paths')
    case = {'instance_id': first['instance_id'], 'repo': first['repo'],
            'base_commit': first['base_commit'], 'issue_url': first.get('issue_url'),
            'query': issue, 'case_type': case_type, 'retriever': retriever,
            'retrieval_k': top_k, 'retrieval_seeds': seeds,
            'raw_modified_targets': sorted(set(r['positive_id'] for r in rows)),
            'accepted_targets': candidates, 'rejected_targets': rejected_targets,
            'target_quality': status, 'gold_paths': gold_paths,
            'gold_nodes': sorted({n for p in gold_paths for n in p['nodes']}),
            'gold_edges': [{'source': a, 'target': b} for a, b in sorted(gold_edges)],
            'target_hops_from_selected_seed': {t: min((x['hops'] for x in gold_paths if x['target'] == t),
                                                      default=None) for t in targets},
            'unreached_targets': sorted(set(targets) - set(chosen_seed)),
            'reverse_probe_top': fallback[:50],
            'deep_hop_probes': deep_hop_probes,
            'reverse_probe_best_rank': fallback[0]['retrieval_rank'] + 1 if fallback else None,
            'oracle_rescue_used_for_training': case_type == 'reverse_oracle_probe' and bool(rescue_seeds),
            'oracle_rescue_seed': rescue_seeds[0] if rescue_seeds else None,
            'oracle_not_available_at_inference': True,
            'seed_scores': sorted_seed_scores,
            'graph_stats': {**repo_stats, 'nodes': len(symbols), 'edges': len(edges)},
            'gold_status': 'static_shortest_path_proxy_not_causal_truth'}
    return (case, training, seed_examples, stop_examples), None


def build_dataset(input_path, output_dir, benchmark_manifest, checkouts, top_k=20,
                  max_hops=8, reverse_hops=5, retriever='bm25', store_graphs=False):
    excluded, manifest_sha = read_manifest(benchmark_manifest)
    rows = read_jsonl(input_path)
    groups = defaultdict(list)
    for row in rows:
        if row['repo'].lower() in excluded:
            continue
        if len((row.get('query') or '').strip()) < 80:
            continue
        groups[(row['repo'], row['base_commit'], row['instance_id'])].append(row)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases, training, seed_training, stop_training, rejected, graph_cache = [], [], [], [], [], {}
    for (repo, sha, instance_id), issue_rows in sorted(groups.items()):
        try:
            graph_id = (repo, sha)
            if graph_id not in graph_cache:
                checkout = get_checkout(repo, sha, checkouts)
                graph_cache[graph_id] = collect_repo(checkout)
            symbols, edges, stats = graph_cache[graph_id]
            value, error = build_case(issue_rows, symbols, edges, stats, top_k,
                                      max_hops, reverse_hops, retriever)
            if error:
                rejected.append(error)
            else:
                case, edges_training, seeds_training, stops_training = value
                case['benchmark_manifest_sha256'] = manifest_sha
                cases.append(case)
                training.extend(edges_training)
                seed_training.extend(seeds_training)
                stop_training.extend(stops_training)
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            rejected.append({'instance_id': instance_id, 'repo': repo,
                             'base_commit': sha, 'reason': 'build_error', 'details': str(e)[:400]})
    dump_jsonl(out / 'navigation_cases.jsonl', cases)
    dump_jsonl(out / 'navigation_steps.jsonl', training)
    dump_jsonl(out / 'seed_training.jsonl', seed_training)
    dump_jsonl(out / 'stop_training.jsonl', stop_training)
    dump_jsonl(out / 'rejected_cases.jsonl', rejected)
    if store_graphs:
        graph_dir = out / 'graphs'
        graph_dir.mkdir(exist_ok=True)
        for (repo, sha), (symbols, edges, stats) in graph_cache.items():
            name = repo.replace('/', '__') + '_' + sha[:12]
            (graph_dir / (name + '.json')).write_text(json.dumps({
                'repo': repo, 'base_commit': sha,
                'nodes': [s.public() for s in symbols.values()], 'edges': edges,
                'stats': stats}, ensure_ascii=False), encoding='utf-8')
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'manifest_sha256': manifest_sha, 'excluded_repository_count': len(excluded),
               'input_rows': len(rows), 'eligible_issues': len(groups),
               'cases': len(cases), 'rejected_cases': len(rejected),
               'training_steps': len(training), 'supervised_steps': sum(t['label'] is not None for t in training),
               'seed_examples': len(seed_training), 'stop_examples': len(stop_training),
               'case_distribution': {kind: sum(c['case_type'] == kind for c in cases)
                                     for kind in ('direct_retrieval', 'retrieval_to_target_path',
                                                  'reverse_oracle_probe')},
               'warning': 'Proxy paths and pseudo targets are NOT verified causal ground truth.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='enriched_sweloc.jsonl')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--benchmark-manifest', required=True, help='full repo-disjoint benchmark manifest')
    p.add_argument('--checkout-root', default='.cache/navigation-checkouts')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--max-hops', type=int, default=8)
    p.add_argument('--reverse-hops', type=int, default=5)
    p.add_argument('--retriever', choices=['bm25', 'swerank'], default='swerank')
    p.add_argument('--store-graphs', action='store_true')
    args = p.parse_args()
    if min(args.top_k, args.max_hops, args.reverse_hops) < 1:
        p.error('top-k, max-hops and reverse-hops must be >= 1')
    result = build_dataset(args.input, args.output_dir, args.benchmark_manifest,
                           args.checkout_root, args.top_k, args.max_hops,
                           args.reverse_hops, args.retriever, args.store_graphs)
    print(json.dumps(result, indent=2))
    if not result['cases']:
        raise SystemExit('No training cases. Inspect rejected_cases.jsonl and benchmark guard.')


if __name__ == '__main__':
    main()