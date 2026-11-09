#!/usr/bin/env python3
"""Frozen 47-case BM25 vs SweRankEmbed evaluation from original per-repo artifacts."""
import argparse, collections, csv, json, pathlib, types
from navigation_graph import bm25_rank, swerank_rank, adjacency, distance_to_targets
from build_navigation_dataset import build_case
MODEL='Salesforce/SweRankEmbed-Small'
KVALS=(1,5,10,20,50)
KINDS=('direct_retrieval','retrieval_to_target_path','reverse_oracle_probe')
RUN='42153623735'
def read(path):
    return [json.loads(x) for x in pathlib.Path(path).read_text(encoding='utf8').splitlines() if x.strip()]
def write(path,seq):
    with open(path,'w',encoding='utf8') as f:
        for x in seq: f.write(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n')
def graph_for(root,repo,sha):
    for p in (root/'dataset'/'graphs').glob('*.json'):
        if sha[:12] not in p.name: continue
        g=json.loads(p.read_text())
        if g['repo']==repo and g['base_commit']==sha: return g
    raise ValueError('Missing immutable pinned graph: '+repo+'@'+sha)
def audit_ranking(ranking,targets,incoming):
    ordered=[x[0] for x in ranking]; rank={x:i+1 for i,x in enumerate(ordered)}
    distances={t:distance_to_targets([t],incoming,max_hops=8) for t in targets}
    metrics={}
    for k in KVALS:
        seeds=ordered[:k]
        direct=sum(t in seeds for t in targets)
        hops={t:min((distances[t].get(s,999999) for s in seeds),default=999999) for t in targets}
        reachable=sum(v<=8 for v in hops.values())
        nontrivial=sum(any(0<distances[t].get(s,999999)<=8 for s in seeds) for t in targets)
        metrics[str(k)]={'direct':direct,'reachable':reachable,'nontrivial':nontrivial,
           'issue_direct':int(direct>0),'issue_reachable':int(reachable>0),
           'all_targets_reachable':int(reachable==len(targets)),
           'target_hops':{t:(None if n>8 else n) for t,n in hops.items()}}
    x=metrics['10']
    kind=KINDS[0] if x['issue_direct'] else KINDS[1] if x['issue_reachable'] else KINDS[2]
    return {'case':kind,'target_ranks':{t:rank[t] for t in targets},'top50':ordered[:50],'metrics':metrics}
def summarise(rows):
    n=len(rows);nt=sum(len(x['targets']) for x in rows)
    out={'issues':n,'targets':nt,'cases':{},'metrics':{},'hop_distribution_k10':{}}
    for name in ['bm25','swerank']:
        out['cases']=dict(out['cases'],**{name:{k:sum(x[name]['case']==k for x in rows) for k in KINDS}})
        out['metrics'][name]={}
        for k in KVALS:
            samples=[x[name]['metrics'][str(k)] for x in rows]
            def avg(key,d):return round(sum(s[key] for s in samples)/max(1,d),6)
            out['metrics'][name][str(k)]={
                'issue_recall_at_k':avg('issue_direct',n),'target_recall_at_k':avg('direct',nt),
                'issue_path_coverage_at_k':avg('issue_reachable',n),
                'target_path_coverage_at_k':avg('reachable',nt),
                'all_target_issue_path_coverage_at_k':avg('all_targets_reachable',n),
                'nontrivial_target_path_coverage_at_k':avg('nontrivial',nt)}
        hist=collections.Counter('unreachable' if h is None else str(h)
             for x in rows for h in x[name]['metrics']['10']['target_hops'].values())
        out['hop_distribution_k10'][name]=dict(hist)
    return out
def per_repo(args):
    src=pathlib.Path(args.input);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    old=read(src/'dataset'/'navigation_cases.jsonl')
    origin={x['instance_id']:x for x in read(args.reference) if x['repo']==args.repo}
    if len(origin)!=len(old) or set(origin)!={x['instance_id'] for x in old}:
        raise ValueError('Frozen issue ID count or set changed')
    cache={}; data=[]; cases=[];steps=[];seeds=[];stops=[];reviews=[]
    for i,c in enumerate(old):
        b=origin[c['instance_id']]
        if any(b[k]!=c[k] for k in ('query','base_commit','accepted_targets','repo','instance_id')):
            raise ValueError('Target labels, base commit, or issue mutated: '+c['instance_id'])
        if c['retriever']!='bm25' or c['retrieval_k']!=10:raise ValueError('Not original BM25 Top10')
        gkey=c['base_commit']
        if gkey not in cache:
            graph=graph_for(src,args.repo,gkey)
            syms={x['id']:types.SimpleNamespace(**x) for x in graph['nodes']}
            cache[gkey]=(syms,graph['edges'],graph['stats'])
        syms,edges,stats=cache[gkey]
        targets=c['accepted_targets']
        if not set(targets).issubset(syms): raise ValueError('Missing target in original graph')
        rank_b=bm25_rank(c['query'],syms)
        if [x for x,v in rank_b[:10]]!=c['retrieval_seeds']:raise ValueError('Original BM25 Top-10 drift')
        rank_e=swerank_rank(c['query'],syms)
        if len(rank_e)!=len(rank_b):raise ValueError('Retrieval candidates changed')
        incoming=adjacency(edges)[1]
        bm=audit_ranking(rank_b,targets,incoming)
        em=audit_ranking(rank_e,targets,incoming)
        if bm['case']!=c['case_type']:raise ValueError('Original case mismatch')
        pseudo=[dict(instance_id=c['instance_id'],repo=c['repo'],base_commit=c['base_commit'],query=c['query'],
            issue_url=c['issue_url'],positive_id=t,issue_pr_match=True,llm_relevance_label='fix_target',
            llm_judge_status='completed_unverified',llm_confidence='high',
            reviewer_label=('verified_fix_target' if level=='human_verified' else ''))
            for t,level in sorted(targets.items())]
        built,error=build_case(pseudo,syms,edges,stats,top_k=10,max_hops=8,reverse_hops=5,retriever='swerank')
        if error:raise ValueError('Replay failed '+str(error))
        new,st,sd,sp=built
        if new['accepted_targets']!=targets or new['retrieval_seeds']!=em['top50'][:10] or new['case_type']!=em['case']:
            raise ValueError('Rerun label, rank or case mismatch')
        data.append(dict(instance_id=c['instance_id'],repo=args.repo,base_commit=gkey,issue_url=c['issue_url'],
            query=c['query'],targets=targets,bm25=bm,swerank=em,
            selected_proxy_paths=new['gold_paths'],oracle_deep_paths=new['deep_hop_probes']))
        cases.append(new);steps+=st;seeds+=sd;stops+=sp
        for target in targets:
            reviews.append(dict(repo=args.repo,issue=c['instance_id'],issue_url=c['issue_url'],
                target=target,label=targets[target],bm25_rank=bm['target_ranks'][target],
                swerank_rank=em['target_ranks'][target],
                bm25_hops_k10=bm['metrics']['10']['target_hops'][target],
                swerank_hops_k10=em['metrics']['10']['target_hops'][target],
                bm25_case=bm['case'],swerank_case=em['case']))
        print(f'{args.repo} {i+1}/{len(old)} {c["instance_id"]}: {bm["case"]} -> {em["case"]}',flush=True)
    for path,rows in [('case_comparison.jsonl',data),('navigation_cases_swerank.jsonl',cases),
       ('navigation_steps_swerank.jsonl',steps),('seed_training_swerank.jsonl',seeds),('stop_training_swerank.jsonl',stops)]:
        write(out/path,rows)
    with (out/'target_review.csv').open('w',newline='',encoding='utf8') as f:
        cols=['repo','issue','issue_url','target','label','bm25_rank','swerank_rank',
             'bm25_hops_k10','swerank_hops_k10','bm25_case','swerank_case']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(reviews)
    summary=summarise(data);original=json.loads((src/'dataset'/'summary.json').read_text())
    summary.update(repo=args.repo,model=MODEL,source_run_id=RUN,labels_frozen=True,
       manifest_sha256=original['manifest_sha256'],no_llm_calls=True,
       note='Static proxy graph; oracle reverse probes excluded from reachability and path coverage.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary
def merge(args):
    root=pathlib.Path(args.input);out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    parts=sorted(root.glob('*/summary.json'))
    if len(parts)!=9:raise ValueError(f'Expected 9 repo jobs, got {len(parts)}')
    entries=[];cases=[];steps=[];seeds=[];stops=[];reviews=[];sha=set();repo=set()
    for p in parts:
        path=p.parent;summary=json.loads(p.read_text());sha.add(summary['manifest_sha256'])
        if summary['repo'] in repo:raise ValueError('Duplicate repo')
        repo.add(summary['repo'])
        entries+=read(path/'case_comparison.jsonl')
        for filename,arr in [('navigation_cases_swerank.jsonl',cases),
          ('navigation_steps_swerank.jsonl',steps),('seed_training_swerank.jsonl',seeds),
          ('stop_training_swerank.jsonl',stops)]:
            arr+=read(path/filename)
        with (path/'target_review.csv').open(encoding='utf8',newline='') as f:reviews+=list(csv.DictReader(f))
    ref=read(args.reference)
    if len(entries)!=47 or len(ref)!=47 or len(sha)!=1:raise ValueError('47-case or manifest failure')
    original={x['instance_id']:x for x in ref}
    if set(original)!={x['instance_id'] for x in entries}:raise ValueError('Issue ID mismatch')
    if any(x['targets']!=original[x['instance_id']]['accepted_targets'] for x in entries):
        raise ValueError('Target label mismatch')
    for filename,arr in [('combined_case_comparison.jsonl',entries),
      ('combined_navigation_cases_swerank.jsonl',cases),
      ('combined_navigation_steps_swerank.jsonl',steps),
      ('combined_seed_training_swerank.jsonl',seeds),('combined_stop_training_swerank.jsonl',stops)]:
        write(out/filename,arr)
    with (out/'combined_target_review.csv').open('w',encoding='utf8',newline='') as f:
        if reviews:
            w=csv.DictWriter(f,fieldnames=list(reviews[0]));w.writeheader();w.writerows(reviews)
    summary=summarise(entries)
    summary.update(repo_count=9,source_run_id=RUN,model=MODEL,labels_frozen=True,no_llm_calls=True,
       manifest_sha256=next(iter(sha)),
       caveat='164 LLM pseudo-target instances, NOT verified gold. Path coverage = reachable within 8 static caller->callee edges from Top-K, includes hop0, excludes oracle fallback.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    lines=['# Frozen 47-case BM25 versus SweRankEmbed','',
        'Same 47 issue descriptions, pinned code graph snapshots and 164 accepted pseudo targets.',
        'No LLM calls; retrieval success excludes teacher-only reverse probes.','',
        '| Metric | BM25 | SweRankEmbed |','|---|---:|---:|']
    for k in KVALS:
        for name in ('issue_recall_at_k','target_recall_at_k','issue_path_coverage_at_k',
                     'target_path_coverage_at_k','all_target_issue_path_coverage_at_k',
                     'nontrivial_target_path_coverage_at_k'):
            b=summary['metrics']['bm25'][str(k)][name]
            e=summary['metrics']['swerank'][str(k)][name]
            lines.append(f'| {name}@{k} | {b:.3f} | {e:.3f} |')
    lines+=['','## Cases','', '| Case | BM25 | SweRankEmbed |','|---|---:|---:|']
    for kind in KINDS:
        lines.append(f'| {kind} | {summary["cases"]["bm25"][kind]} | {summary["cases"]["swerank"][kind]} |')
    lines+=['','## Min hops from Top-10 to targets','', '| Hops | BM25 | SweRankEmbed |','|---|---:|---:|']
    hb=summary['hop_distribution_k10']['bm25'];he=summary['hop_distribution_k10']['swerank']
    for hop in sorted(set(hb)|set(he),key=lambda h:int(h) if h.isdigit() else 99):
        lines.append(f'| {hop} | {hb.get(hop,0)} | {he.get(hop,0)} |')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['repo','merge'])
    p.add_argument('--input',required=True);p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--repo')
    args=p.parse_args()
    result=per_repo(args) if args.mode=='repo' else merge(args)
    print(json.dumps({'issues':result['issues'],'targets':result['targets'],
        'case_distribution':result['cases']},indent=2))
if __name__=='__main__':main()
