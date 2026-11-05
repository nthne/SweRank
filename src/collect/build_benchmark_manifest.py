#!/usr/bin/env python3
"""Fetch evaluation-repository exclusions from the OFFICIAL benchmark tables.

Output is intentionally repository-disjoint, stricter than exact issue matching.
Fail closed if either benchmark is missing/incomplete.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

DATASETS = [('princeton-nlp/SWE-bench', 'test', 2000),
            ('czlll/Loc-Bench_V1', 'test', 500)]


def get_repos(load_dataset):
    sources, repos = {}, set()
    for dataset, split, minimum in DATASETS:
        data = load_dataset(dataset, split=split)
        if len(data) < minimum:
            raise ValueError(f'Benchmark unexpectedly small: {dataset}: {len(data)}')
        values = {str(row['repo']).strip().lower() for row in data}
        if not values or any('/' not in v for v in values):
            raise ValueError(f'Invalid repository names in {dataset}')
        repos.update(values)
        sources[dataset] = {'split': split, 'instances': len(data), 'repositories': len(values)}
    return {'generated_utc': datetime.now(timezone.utc).isoformat(),
            'policy': 'exclude_entire_repositories', 'sources': sources,
            'repositories': sorted(repos)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from datasets import load_dataset
    manifest = get_repos(load_dataset)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('Excluded repositories:', len(manifest['repositories']))


if __name__ == '__main__':
    main()