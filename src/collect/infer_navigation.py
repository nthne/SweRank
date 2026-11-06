#!/usr/bin/env python3
"""Best-first graph traversal with ONLY observable features and trained scorers.
No target, PR diff, gold path, reverse-oracle seed, or LLM is used at inference.
"""
import argparse
import heapq
import json
from pathlib import Path
from navigation_graph import adjacency, bm25_rank, collect_repo, swerank_rank
from build_navigation_dataset import features, normalized_ranks
from train_navigation_scorer import dot, sigmoid


def score(model, feat):
    return sigmoid(dot(model['weights'], feat, model['bias']))


def infer(query, symbols, edges, models, top_k=20, budget=100, max_hops=8,
          retriever='bm25', stop_threshold=.65):
    ranking = swerank_rank(query, symbols) if retriever == 'swerank' else bm25_rank(query, symbols)
    ids = [key for key, _ in ranking]
    if not ids:
        return {'marked_functions': [], 'visited': [], 'expanded_nodes': 0}
    relevance = normalized_ranks(ranking)
    rank_index = {key: i for i, key in enumerate(ids)}
    outgoing, incoming = adjacency(edges)
    edge_types = {(e['source'], e['target']): e['type'] for e in edges}
    edge_model = models.get('edge_model') or models
    stop_model = models.get('stop_model')
    queue, best = [], {}
    for key in ids[:top_k]:
        priority = .5 + .5 * relevance[key]
        best[key] = priority
        heapq.heappush(queue, (-priority, key, 0, [key]))
    visited, marked = [], []
    expanded = set()
    while queue and len(expanded) < budget:
        neg, node, hop, path = heapq.heappop(queue)
        if node in expanded or -neg < best.get(node, 0) - 1e-10:
            continue
        expanded.add(node)
        stop_features = features(node, node, hop, None, relevance, rank_index,
                                 outgoing, incoming, top_k, len(ids), max_hops)
        mark_probability = score(stop_model, stop_features) if stop_model else None
        visited.append({'node': node, 'hop': hop, 'priority': round(-neg, 6),
                        'target_probability': round(mark_probability, 6) if mark_probability is not None else None})
        if mark_probability is not None and mark_probability >= stop_threshold:
            marked.append({'node': node, 'path': path, 'confidence': round(mark_probability, 6)})
        if hop == max_hops:
            continue
        for neighbor in sorted(outgoing[node]):
            if neighbor in expanded or neighbor in path:
                continue
            cand_features = features(node, neighbor, hop, edge_types.get((node, neighbor)),
                                     relevance, rank_index, outgoing, incoming,
                                     top_k, len(ids), max_hops)
            probability = score(edge_model, cand_features)
            priority = (.35 * relevance[neighbor] + .65 * probability) * (.99 ** (hop + 1))
            if priority > best.get(neighbor, -1):
                best[neighbor] = priority
                heapq.heappush(queue, (-priority, neighbor, hop + 1, path + [neighbor]))
    return {'retriever': retriever, 'retrieval_seeds': ids[:top_k],
            'marked_functions': marked, 'visited': visited,
            'expanded_nodes': len(expanded), 'model_uses_oracle_targets': False,
            'llm_calls': 0, 'stop_model_available': stop_model is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-checkout', required=True, help='local pre-fix git checkout')
    p.add_argument('--query-file', required=True, help='plain issue description')
    p.add_argument('--model', required=True, help='trained edge_scorer.json')
    p.add_argument('--output', required=True)
    p.add_argument('--retriever', choices=['bm25', 'swerank'], default='bm25')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--budget', type=int, default=100)
    p.add_argument('--max-hops', type=int, default=8)
    p.add_argument('--stop-threshold', type=float, default=.65)
    a = p.parse_args()
    if min(a.top_k, a.budget, a.max_hops) < 1:
        p.error('budgets must be >= 1')
    syms, edges, _ = collect_repo(a.repo_checkout)
    result = infer(Path(a.query_file).read_text(encoding='utf-8'), syms, edges,
                   json.loads(Path(a.model).read_text(encoding='utf-8')),
                   a.top_k, a.budget, a.max_hops, a.retriever, a.stop_threshold)
    Path(a.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Expanded', result['expanded_nodes'], 'nodes; marked', len(result['marked_functions']))


if __name__ == '__main__':
    main()