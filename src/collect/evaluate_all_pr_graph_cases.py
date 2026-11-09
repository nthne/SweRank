#!/usr/bin/env python3
"""Full Issue→SweRankEmbed→Graph→Case1/2/3 flow; all actually modified PR functions are weak targets.

No LLM. Function additions absent at base commit remain labeled not_in_base_commit.
Graph paths are static proxies, NOT validated causal paths. Case3 reverse paths are oracle-only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
from navigation_graph import collect_repo, adjacency, shortest_paths, distance_to_targets

BUCKETS = ("11-20", "21-50", "51-100", ">100", "no_incoming_edge", "not_in_base_commit", "other_unreachable")
def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]
def save(path, rows):
    with Path(path).open("w", encoding="utf8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False,sort_keys=True)+"\n")
def bucket(rank, in_degree, exists):
    if not exists: return "not_in_base_commit"
    if rank is None: return "no_incoming_edge" if not in_degree else "other_unreachable"
    return "11-20" if rank <= 20 else "21-50" if rank <= 50 else "51-100" if rank <= 100 else ">100"
def best_path(seeds,target,outgoing,order,max_hops):
    choices=[]
    for seed in seeds:
        for path in shortest_paths(seed,target,outgoing,max_hops=max_hops,max_paths=6):
            choices.append((len(path)-1,order[seed],seed,path))
    if not choices:return None
    h,rank,seed,path=min(choices)
    return dict(target=target,seed=seed,hops=h,nodes=path,seed_rank=rank,oracle_only=False)
def analyze_issue(row,ranking,nodes,edges,top_k,max_hops,reverse_hops):
    ordered=ranking["ranked_functions"]; rank={x:i+1 for i,x in enumerate(ordered)}
    if len(rank)!=len(ordered):raise ValueError("Duplicate rank candidate")
    if set(ordered)!=set(nodes):
        missing=sorted(set(nodes)-set(ordered))
        extra=sorted(set(ordered)-set(nodes))
        raise ValueError(f"Graph/index mismatch: {len(missing)} graph-only, {len(extra)} index-only; examples {missing[:2]}, {extra[:2]}")
    raw_targets=sorted(set(row["all_modified_functions"]))
    if not raw_targets:raise ValueError("No changed function for "+row["instance_id"])
    seeds=ordered[:top_k]
    out,inc=adjacency(edges)
    known=[t for t in raw_targets if t in nodes]
    direct=set(seeds)&set(known)
    per_target=[]
    graph_paths=[]
    for t in raw_targets:
        exists=t in nodes
        p=best_path(seeds,t,out,rank,max_hops) if exists else None
        if p:graph_paths.append(p)
        indeg=len(inc[t]) if exists else 0
        outdeg=len(out[t]) if exists else 0
        ancestors=(distance_to_targets([t],inc,max_hops=reverse_hops) if exists else {})
        ancestor_hits=sorted((rank[a],d,a) for a,d in ancestors.items()
                             if 1<=d<=reverse_hops and a!=t and a in rank)
        rstar,closest,ancestor=ancestor_hits[0] if ancestor_hits else (None,None,None)
        per_target.append(dict(target=t,in_base_commit=exists,in_retrieval_index=t in rank,
           retrieval_rank=rank.get(t),in_degree=indeg,out_degree=outdeg,
           best_forward_hops=p["hops"] if p else None,
           best_forward_seed=p["seed"] if p else None,
           best_ancestor_rank=rstar,best_ancestor_node=ancestor,best_ancestor_hops=closest,
           ancestor_count_1_to_5=len(ancestor_hits),
           rank_bucket=bucket(rstar,indeg,exists),
           target_kind=("test" if t in row.get("modified_test_functions",[]) else "source")))
    if direct:case="case1_direct";issue_hops=0
    elif graph_paths:case="case2_forward";issue_hops=min(p["hops"] for p in graph_paths)
    else:case="case3_no_topk_path";issue_hops=None
    # Strict reporting: never present any 2..5 hop alternative from Case1 as localization.
    selected = ([p for p in graph_paths if p["hops"]==0] if case=="case1_direct"
                else graph_paths if case=="case2_forward" else [])
    oracle=[]
    if case=="case3_no_topk_path":
        for t in per_target:
            if not t["best_ancestor_node"]:continue
            for path in shortest_paths(t["best_ancestor_node"],t["target"],out,
                                       max_hops=reverse_hops,max_paths=5):
                oracle.append(dict(target=t["target"],seed=t["best_ancestor_node"],
                  seed_rank=t["best_ancestor_rank"],hops=len(path)-1,nodes=path,
                  oracle_only=True,localization_success=False))
    # Target-specific status preserves the truth about incomplete multi-target recovery.
    return dict(instance_id=row["instance_id"],repo=row["repo"],issue_id=row.get("issue_id"),
        pr_number=row["pr_number"],pr_url=row["pr_url"],issue_url=row["issue_url"],
        base_commit=row["base_commit"],query=row["query"],
        retrieval_method=ranking["model"],retrieval_k=top_k,
        graph_nodes=len(nodes),graph_edges=len(edges),case=case,
        issue_localization_hops=issue_hops,all_modified_targets=raw_targets,
        modified_source_functions=row.get("modified_source_functions",[]),
        modified_test_functions=row.get("modified_test_functions",[]),
        targets_not_in_base=[t for t in raw_targets if t not in nodes],
        complete_extraction=row["complete_extraction"],
        top_k_seeds=seeds,target_details=per_target,
        localization_paths=selected,case3_teacher_paths=oracle,
        teacher_oracle_never_used_for_inference=True,
        groundtruth_status="all_modified_PR_functions_weak_unverified")
def graph_checkout(repo,sha,dircache):
    import re
    if not re.fullmatch(r"[0-9a-f]{40}",sha):raise ValueError("unsafe commit")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",repo):raise ValueError("unsafe repository")
    bare=dircache/(repo.replace("/","__")+".git")
    checkout=dircache/"checkout"/sha
    if not bare.is_dir():
        subprocess.run(["git","clone","-q","--bare","--filter=blob:none",
             "https://github.com/"+repo+".git",str(bare)],check=True,timeout=400)
    if not checkout.exists():
        checkout.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(["git","-C",str(bare),"worktree","add","--quiet","--detach",str(checkout),sha],
               check=True,timeout=400)
    return checkout
def generate_training(case,ranked,edges):
    ranks={name:i+1 for i,name in enumerate(ranked)}
    out,inc=adjacency(edges)
    targets=set(case["all_modified_targets"])
    known=[t for t in targets if t in ranks]
    dists=distance_to_targets(known,inc,max_hops=max(1,len(ranks))) if known else {}
    paths=case["localization_paths"] if case["case"]!="case3_no_topk_path" else case["case3_teacher_paths"]
    oracle=case["case"]=="case3_no_topk_path"
    steps=[];stops=[];seeds=[]
    for seed in case["top_k_seeds"]:
        if seed in targets:lab=1
        elif seed in dists:lab=None
        else:lab=0
        seeds.append(dict(instance_id=case["instance_id"],node=seed,rank=ranks[seed],
                          label=lab,teacher_distance=dists.get(seed)))
    for path in paths:
        for pos,node in enumerate(path["nodes"]):
            stops.append(dict(instance_id=case["instance_id"],node=node,hop=pos,
                             label=int(node in targets),oracle_only_teacher=oracle))
            if pos+1==len(path["nodes"]):continue
            true_next=path["nodes"][pos+1]
            for candidate in sorted(out[node]):
                target_distance=dists.get(candidate)
                label=1 if candidate==true_next else None if target_distance is not None else 0
                steps.append(dict(instance_id=case["instance_id"],current=node,
                   candidate=candidate,hop=pos,score_features=dict(
                   rank_fraction=1-ranks[candidate]/max(len(ranks),1),
                   current_rank_fraction=1-ranks[node]/max(len(ranks),1),
                   candidate_in_degree=len(inc[candidate]),candidate_out_degree=len(out[candidate]),
                   hop=pos),label=label,teacher_distance=target_distance,
                   oracle_only_teacher=oracle))
    return steps,stops,seeds
def run(args):
    gold=load(args.gold);rankings=load(args.rankings)
    if len({x["instance_id"] for x in rankings})!=len(rankings):raise ValueError("Duplicate rankings")
    ranked={x["instance_id"]:x for x in rankings}
    valid=[r for r in gold if "collection_error" not in r and r.get("all_modified_functions")]
    if not valid:raise ValueError("No valid sample")
    if any(not r["complete_extraction"] for r in valid):raise ValueError("Partial extraction: abort")
    if set(r["instance_id"] for r in valid)!=set(ranked):raise ValueError("Sample IDs differ")
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    results=[];all_steps=[];all_stops=[];all_seeds=[]
    graph_cache={}
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        for i,row in enumerate(valid):
            sha=row["base_commit"];repo=row["repo"]
            if sha not in graph_cache:
                checkout=graph_checkout(repo,sha,work)
                symbols,edges,stats=collect_repo(checkout)
                graph_cache[sha]=(symbols,edges,stats)
            symbols,edges,stats=graph_cache[sha]
            item=analyze_issue(row,ranked[row["instance_id"]],symbols,edges,args.top_k,args.max_hops,args.reverse_hops)
            steps,stops,seeds=generate_training(item,ranked[row["instance_id"]]["ranked_functions"],edges)
            all_steps+=steps;all_stops+=stops;all_seeds+=seeds;results.append(item)
            print(f"{i+1}/{len(valid)} {row['instance_id']} {item['case']} hops={item['issue_localization_hops']} targets={len(item['all_modified_targets'])}",flush=True)
    for name,rows in [("issue_cases.jsonl",results),("navigation_steps.jsonl",all_steps),
                      ("stop_training.jsonl",all_stops),("seed_training.jsonl",all_seeds)]:
        save(root/name,rows)
    issues=Counter(x["case"] for x in results)
    target_status=Counter()
    target_kind=Counter()
    for x in results:
        for target in x["target_details"]:
            target_kind[target["target_kind"]]+=1
            target_status["not_in_base" if not target["in_base_commit"] else
                "direct" if target["best_forward_hops"]==0 else
                "forward_path" if target["best_forward_hops"] else
                "not_reachable_from_top_k"]+=1
    def cnt_hops(field):
        return dict(Counter(str(p["hops"]) for x in results for p in x[field]))
    buckets=Counter(x["rank_bucket"] for c in results if c["case"]=="case3_no_topk_path"
                    for x in c["target_details"])
    case_hops=Counter("unreachable" if x["issue_localization_hops"] is None else
                      str(x["issue_localization_hops"]) for x in results)
    summary=dict(issue_pr_samples=len(results),repos=sorted(set(x["repo"] for x in results)),
        weak_modified_targets=sum(len(x["all_modified_targets"]) for x in results),
        weak_source_targets=target_kind["source"],weak_test_targets=target_kind["test"],
        cases=dict(issues),issue_localization_hops=dict(case_hops),
        case3_target_rank_buckets=dict(buckets),target_status=dict(target_status),
        retrieval_localization_path_hops=cnt_hops("localization_paths"),
        case3_teacher_only_path_hops=cnt_hops("case3_teacher_paths"),
        retrieval="Salesforce/SweRankEmbed-Small",top_k=args.top_k,
        max_hops=args.max_hops,reverse_hops=args.reverse_hops,
        no_llm=True,all_modified_functions_are_weak_targets=True)
    (root/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    # Full per-target review (all weak functions, even if they did not exist at base commit).
    columns=["instance_id","issue_url","pr_url","repo","case","issue_hops","target","target_kind",
             "in_base_commit","retrieval_rank","best_forward_hops","best_forward_seed",
             "in_degree","out_degree","best_ancestor_rank","best_ancestor_hops","rank_bucket"]
    with (root/"manual_target_review.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for c in results:
            for t in c["target_details"]:
                w.writerow(dict(instance_id=c["instance_id"],issue_url=c["issue_url"],
                   pr_url=c["pr_url"],repo=c["repo"],case=c["case"],issue_hops=c["issue_localization_hops"],
                   **{k:t.get(k) for k in columns[6:]}))
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gold",required=True);p.add_argument("--rankings",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--max-hops",type=int,default=8)
    p.add_argument("--reverse-hops",type=int,default=5)
    run(p.parse_args())
