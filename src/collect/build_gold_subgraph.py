#!/usr/bin/env python3
"""Multi-target shortest-path proxy graph from verified targets, offline ONLY.

Input: {"instance_id": "...", "seeds": [...], "targets": [...],
        "edges": [{"source": "...", "target": "...", "type": "..."}]}
Not a proven causal explanation; true targets must NOT be used at inference.
"""
import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

def shortest_path_subgraph(case, reverse_edges=False):
    seeds = sorted(set(case.get("seeds", [])))
    targets = sorted(set(case.get("targets", [])))
    adj = defaultdict(set)
    for edge in case.get("edges", []):
        a, b = edge["source"], edge["target"]
        if a == b:
            continue
        adj[a].add(b)
        if reverse_edges:
            adj[b].add(a)
    dist = {x: 0 for x in seeds}
    parents = defaultdict(set)
    todo = deque(seeds)
    while todo:
        a = todo.popleft()
        for b in sorted(adj[a]):
            d = dist[a] + 1
            if b not in dist:
                dist[b] = d
                todo.append(b)
            if dist[b] == d:
                parents[b].add(a)
    gold_nodes, gold_edges = set(), set()
    pending = [t for t in targets if t in dist]
    done = set()
    while pending:
        b = pending.pop()
        if b in done:
            continue
        done.add(b)
        gold_nodes.add(b)
        for a in parents[b]:
            gold_edges.add((a, b))
            pending.append(a)
    return {
        "instance_id": case.get("instance_id"), "seeds": seeds, "targets": targets,
        "target_hops": {t: dist[t] for t in targets if t in dist},
        "unreachable_targets": [t for t in targets if t not in dist],
        "gold_nodes": sorted(gold_nodes),
        "gold_edges": [{"source": a, "target": b} for a, b in sorted(gold_edges)],
        "gold_status": "shortest_path_proxy_not_causal_proof"
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reverse-edges", action="store_true")
    args = p.parse_args()
    result = shortest_path_subgraph(json.loads(Path(args.input).read_text(encoding="utf-8")),
                                    reverse_edges=args.reverse_edges)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Gold nodes={len(result['gold_nodes'])}, edges={len(result['gold_edges'])}, unreachable={len(result['unreachable_targets'])}")

if __name__ == "__main__":
    main()
