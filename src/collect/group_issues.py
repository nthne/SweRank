#!/usr/bin/env python3
"""Aggregate weak labels at the issue/base-commit level; keep provisional judge signals."""
import argparse, json
from pathlib import Path
def group(rows):
    buckets={}
    for r in rows:
        key=(r['instance_id'],r['base_commit'])
        if key not in buckets:
            buckets[key]=dict(instance_id=r['instance_id'],repo=r['repo'],base_commit=r['base_commit'],
              issue_url=r['issue_url'],query=r['query'],pr_urls=set(),weak_targets=set(),
              weak_source_targets=set(),modified_test_functions=set(),review_flags=set(),
              candidate_evaluations=[],groundtruth_status='unverified_multi_target_candidates')
        d=buckets[key];target=r['positive_id']
        d['pr_urls'].add(r['pr_url']);d['weak_targets'].add(target)
        if target.startswith('tests/') or '/tests/' in target or target.startswith('test_'):
            d['modified_test_functions'].add(target)
        else:d['weak_source_targets'].add(target)
        d['review_flags'].update(r.get('review_flags',[]))
        if 'semantic_similarity' in r or 'llm_relevance_label' in r:
            d['candidate_evaluations'].append(dict(positive_id=target,
              semantic_similarity=r.get('semantic_similarity'),
              semantic_status=r.get('semantic_status'),
              llm_relevance_label=r.get('llm_relevance_label'),
              llm_judge_status=r.get('llm_judge_status'),
              llm_evidence=r.get('llm_evidence')))
    return [{k:sorted(v) if isinstance(v,set) else v for k,v in buckets[key].items()} for key in sorted(buckets)]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='audit_outputs/sample_sweloc.jsonl')
    p.add_argument('--output',default='audit_outputs/issue_level.jsonl')
    a=p.parse_args()
    with open(a.input,encoding='utf-8') as f:rows=[json.loads(x) for x in f if x.strip()]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for row in group(rows):f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print('Aggregated',len(rows),'labels')
if __name__=='__main__':main()
