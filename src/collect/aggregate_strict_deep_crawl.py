#!/usr/bin/env python3
"""Strict Case1/2/3 collection report. Never mix oracle paths into localization success."""
import argparse
from collections import Counter,defaultdict
import csv,json,pathlib

CASES=('direct_retrieval','retrieval_to_target_path','reverse_oracle_probe')
def load(path):
    p=pathlib.Path(path)
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []
def write(path,rows):
    with open(path,'w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
def aggregate(root,out,expected_repos):
    root,out=pathlib.Path(root),pathlib.Path(out)
    out.mkdir(parents=True,exist_ok=True)
    sources=list(sorted(root.glob('*/dataset/summary.json')))
    if len(sources)!=expected_repos:raise ValueError(f'Expected {expected_repos} artifacts, found {len(sources)}')
    cases,rejects,teacher,localize,rows=[],[],[],[],[]
    perrepo={}
    all_sha=set()
    for path in sources:
        parent=path.parent;summary=json.loads(path.read_text(encoding='utf-8'))
        name=path.parent.parent.name
        actual_repo=name.removeprefix('nav-deep-').replace('__','/')
        if actual_repo in perrepo:raise ValueError('Duplicate repository: '+actual_repo)
        accepted=load(parent/'navigation_cases.jsonl')
        rejected=load(parent/'rejected_cases.jsonl')
        if len(accepted)!=summary['cases']:raise ValueError('Summary count mismatch: '+actual_repo)
        if any(c['repo'].lower()!=actual_repo.lower() or c['retriever']!='swerank' for c in accepted):
            raise ValueError('Wrong repo or retriever: '+actual_repo)
        all_sha.add(summary['manifest_sha256'])
        hist=Counter()
        for c in accepted:
            kind=c['case_type'];direct=set(c['retrieval_seeds'])&set(c['accepted_targets'])
            gp=c.get('gold_paths',[]);op=c.get('oracle_paths',[])
            if kind not in CASES:raise ValueError('Unknown case')
            if kind==CASES[0]:
                if not direct or any(x['hops']!=0 for x in gp) or op:
                    raise ValueError('Case1 must have 0-hop localization only')
                hist['0']+=1
                for x in gp:localize.append({'repo':actual_repo,'issue':c['instance_id'],'kind':kind,**x})
            elif kind==CASES[1]:
                if direct or not gp or op or any(x['hops']<1 for x in gp):
                    raise ValueError('Case2 must use retrieval-seeded graph paths only')
                h=min(x['hops'] for x in gp)
                hist[str(h)]+=1
                for x in gp:localize.append({'repo':actual_repo,'issue':c['instance_id'],'kind':kind,**x})
            else:
                if direct or gp or c.get('reported_localization_hops') is not None:
                    raise ValueError('Case3 must not have localization paths')
                hist['unreachable']+=1
                for x in op:teacher.append({'repo':actual_repo,'issue':c['instance_id'],'kind':kind,**x})
                for x in c.get('deep_hop_probes',[]):
                    rows.append({'repo':actual_repo,'issue':c['instance_id'],'target':x['target'],
                                 'seed':x['ancestor'],'hops':x['hops'],'seed_rank':x['ancestor_retrieval_rank'],
                                 'path_nodes':' -> '.join(x['nodes']),'oracle_only':True,
                                 'localization_success':False,'manual_review':''})
        perrepo[actual_repo]={'accepted':len(accepted),'rejected':len(rejected),
           'cases':dict(Counter(x['case_type'] for x in accepted)),
           'issue_localization_hops':dict(hist),
           'case3_reverse_deep_2_5':len([x for x in rows if x['repo']==actual_repo])}
        cases.extend(accepted);rejects.extend(rejected)
    if len(all_sha)!=1:raise ValueError('Benchmark exclusions differed')
    write(out/'combined_navigation_cases.jsonl',cases)
    write(out/'combined_rejected_cases.jsonl',rejects)
    write(out/'retrieval_localization_paths.jsonl',localize)
    write(out/'case3_oracle_teacher_paths.jsonl',teacher)
    fields=['repo','issue','target','seed','hops','seed_rank','path_nodes','oracle_only',
            'localization_success','manual_review']
    with (out/'case3_only_deep_paths.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    cases_count=Counter(c['case_type'] for c in cases)
    primary=Counter('0' if c['case_type']==CASES[0] else
                    str(c['reported_localization_hops']) if c['case_type']==CASES[1]
                    else 'unreachable' for c in cases)
    oracle_hist=Counter(str(x['hops']) for x in rows)
    summary={'repositories':perrepo,'repo_count':len(perrepo),'issues':len(cases)+len(rejects),
             'accepted':len(cases),'rejected':len(rejects),
             'cases':dict(cases_count),'case_localization_hops':dict(primary),
             'case3_oracle_deep_hops':dict(oracle_hist),
             'case3_teacher_paths':len(teacher),'case3_2_to_5_oracle_probes':len(rows),
             'retrieval_localization_paths':len(localize),
             'retriever':'Salesforce/SweRankEmbed-Small',
             'manifest_sha256':next(iter(all_sha)),
             'important':'All Case3 paths are teacher-only proxies; Case1 always 0 hops; only Case2 genuine TopK-to-target navigation. Pseudo targets not human-verified.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    report=['# Strict Case1/2/3 — SweRankEmbed new crawl','',
           f"Repositories: {summary['repo_count']}; accepted: {summary['accepted']}; rejected: {summary['rejected']}.",
           '','| Case | Count |','|---|---:|']
    report += [f'| {k} | {cases_count[k]} |' for k in CASES]
    report+=['','| Localization minimum hops (per issue) | Count |','|---|---:|']
    report += [f'| {k} | {v} |' for k,v in sorted(primary.items())]
    report+=['','| Case3 reverse-only diagnostic hop | Count |','|---|---:|']
    report += [f'| {k} | {v} |' for k,v in sorted(oracle_hist.items())]
    report+=['','Every reverse diagnostic is **oracle-only**; not a successful navigation from the original Top-K.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts-root',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--expected-repos',type=int,required=True)
    a=p.parse_args()
    print(json.dumps(aggregate(a.artifacts_root,a.output_dir,a.expected_repos),indent=2))
