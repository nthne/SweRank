#!/usr/bin/env python3
"""Re-evaluate frozen 198 ET1 samples with the enhanced graph only.

The exact original SweRankEmbed top-10 and target rank are reused verbatim:
this isolates graph changes from model/corpus drift. Re-encode with the same
SweRankEmbed model only for r-star when no top-10 path exists.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import tempfile

from et1_single_root_paths import code_checkout, is_source, MODEL
from navigation_graph import adjacency, shortest_paths, distance_to_targets, swerank_rank
from typed_call_graph import collect_repo_v2


def read_rows(file):
    return [json.loads(x) for x in Path(file).read_text(encoding="utf8").splitlines() if x.strip()]


def save_rows(file,records):
    file=Path(file)
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in records),encoding="utf8")


def case_for_issue(row, old, symbols, edges, stats, top_k, max_hops, full_rstar):
    target=row["target"]
    out,inc=adjacency(edges)
    topk=[x["function"] for x in old.get("retrieved_topk",[])[:top_k]]
    # These rankings were actually computed with Salesforce/SweRankEmbed-Small
    # from pre-fix code, not oracle target labels.
    # A tiny repository can have fewer functions than the requested top-k.
    # Preserve its complete frozen ranking instead of rejecting valid samples.
    expected=min(top_k, old.get("index_size", len(symbols)))
    if not topk or len(topk)!=expected:
        raise ValueError("Frozen top-k length mismatch: "+row["instance_id"]+
                         " expected "+str(expected)+" got "+str(len(topk)))
    if target not in symbols:
        return dict(case="target_missing_in_base",paths=[],selected_path=None,
                    selected_nontrivial_hops=None,best_ancestor_bin="not_in_base_commit",
                    has_incoming_static_edge=False)
    missing=set(topk)-set(symbols)
    if missing:raise ValueError("Frozen retriever/graph candidate set changed: "+str(sorted(missing)[:4]))
    rank=old.get("target_rank")
    if not isinstance(rank,int):raise ValueError("No verified SweRank target rank")
    paths=[]
    for i,entry in enumerate(topk,1):
        if entry==target:continue
        for seq in shortest_paths(entry,target,out,max_hops=max_hops,max_paths=16):
            types=[]
            for u,v in zip(seq,seq[1:]):
                evidence=[x for x in edges if x["source"]==u and x["target"]==v]
                types.append({"type":evidence[0].get("type"),"confidence":evidence[0].get("confidence","legacy"),
                              "evidence":evidence[0].get("evidence") if evidence else None})
            paths.append({"entry":entry,"entry_rank":i,"hops":len(seq)-1,"nodes":seq,
                          "edges":types,"oracle_only":False})
    paths.sort(key=lambda x:(x["hops"],x["entry_rank"],x["nodes"]))
    case=("case1_direct" if rank<=top_k else
          "case2_forward" if paths else "case3_no_topk_path")
    rstar=None;ancestor=None;ahops=None
    # Avoid recomputing tens of thousands of dense embeddings when unnecessary.
    if case=="case3_no_topk_path" and full_rstar and inc.get(target):
        scores=swerank_rank(row["query"],symbols,MODEL)
        ranks={name:i+1 for i,(name,score) in enumerate(scores)}
        d=distance_to_targets([target],inc,max_hops=max_hops)
        candidates=sorted((ranks[x],length,x) for x,length in d.items()
                          if x!=target and 1<=length<=max_hops and x in ranks)
        if candidates:rstar,ahops,ancestor=candidates[0]
    elif case=="case3_no_topk_path" and not full_rstar and inc.get(target):
        # Reliable lower bound: no top-10 ancestor; exact rank requires embeddings.
        rstar=None
    if case!="case3_no_topk_path":bucket="not_case3"
    elif not inc.get(target):bucket="no_incoming_edge"
    elif not full_rstar:bucket="rstar_not_computed"
    elif rstar is None:bucket="other_unreachable"
    elif rstar<=10:bucket="1-10"
    elif rstar<=20:bucket="11-20"
    elif rstar<=50:bucket="21-50"
    elif rstar<=100:bucket="51-100"
    else:bucket=">100"
    return dict(case=case,paths=paths[:32],
                selected_path=paths[0]["nodes"] if paths else ([target] if case=="case1_direct" else None),
                selected_nontrivial_hops=paths[0]["hops"] if paths else None,
                best_ancestor_rank=rstar,best_ancestor_hops=ahops,best_ancestor=ancestor,
                best_ancestor_bin=bucket,has_incoming_static_edge=bool(inc.get(target)))


def run(args):
    rows=read_rows(args.issues)
    previous=read_rows(args.previous)
    prev={r["instance_id"]:r for r in previous}
    if len(rows)!=198 or len(prev)!=198 or len(set(prev))!=198:
        raise SystemExit("Frozen benchmark mismatch. Expected 198 exact samples")
    for r in rows:
        if r["instance_id"] not in prev:raise SystemExit("Missing previous rank "+r["instance_id"])
        if prev[r["instance_id"]]["target"]!=r["target"] or prev[r["instance_id"]]["base_commit"]!=r["base_commit"]:
            raise SystemExit("Target/base_commit drift "+r["instance_id"])
    repos=sorted({r["repo"] for r in rows})
    selected={repo for i,repo in enumerate(repos) if i%args.shard_count==args.shard_index}
    chosen=[r for r in rows if r["repo"] in selected]
    by_revision=defaultdict(list)
    for r in chosen:by_revision[(r["repo"],r["base_commit"])].append(r)
    results=[];failure_revisions=[]
    with tempfile.TemporaryDirectory(prefix="et1-graphv2-") as tmp:
        for i,((repo,sha),group) in enumerate(by_revision.items(),1):
            try:
                checkout=code_checkout(repo,sha,Path(tmp))
                symbols,all_edges,stats=collect_repo_v2(checkout)
                symbols={key:s for key,s in symbols.items() if is_source(s.path)}
                edges=[e for e in all_edges if e["source"] in symbols and e["target"] in symbols]
                if len(symbols)>args.max_functions:
                    raise ValueError("Source corpus larger than configured limit")
                for row in group:
                    old=prev[row["instance_id"]]
                    new=case_for_issue(row,old,symbols,edges,stats,args.top_k,args.max_hops,args.full_rstar)
                    results.append({
                        "instance_id":row["instance_id"],"repo":repo,"target":row["target"],
                        "base_commit":sha,"issue_url":row["issue_url"],"pr_url":row["pr_url"],
                        "target_rank":old["target_rank"],"retrieved_topk":old["retrieved_topk"],
                        "previous_case":old["case"],"previous_best_ancestor_bin":old.get("best_ancestor_bin"),
                        "graph_v1_edges":old.get("graph_edges"),"graph_v2_edges":len(edges),
                        "graph_v2_new_types":stats.get("v2_new_edge_counts",{}),
                        **new})
                print("GRAPH_V2",args.shard_index,i,"/",len(by_revision),repo,
                      "old",[prev[r["instance_id"]]["case"] for r in group],
                      "new",[results[-len(group)+j]["case"] for j in range(len(group))],
                      "new_edges",stats.get("v2_new_edges",0),flush=True)
            except Exception as exc:
                failure_revisions.append(dict(repo=repo,sha=sha,error=str(exc)[:450],
                                              instances=[x["instance_id"] for x in group]))
                print("GRAPH_V2_ERROR",repo,str(exc)[:450],flush=True)
            save_rows(Path(args.output)/"results.jsonl",results)
            save_rows(Path(args.output)/"errors.jsonl",failure_revisions)
    counts=Counter(r["case"] for r in results)
    before=Counter(r["previous_case"] for r in results)
    hops=Counter(str(r["selected_nontrivial_hops"]) for r in results
                 if r.get("selected_nontrivial_hops") is not None)
    case2_hops=Counter(str(r["selected_nontrivial_hops"]) for r in results
                       if r["case"]=="case2_forward")
    transition=Counter(r["previous_case"]+" -> "+r["case"] for r in results)
    bins=Counter(r["best_ancestor_bin"] for r in results if r["case"]=="case3_no_topk_path")
    summary=dict(shard=args.shard_index,repo_count=len(selected),samples_requested=len(chosen),
                 samples_evaluated=len(results),failures=len(failure_revisions),
                 old_cases=dict(before),new_cases=dict(counts),transitions=dict(transition),
                 nontrivial_hops=dict(hops),case2_hops=dict(case2_hops),case3_bins=dict(bins),
                 source_ranking="original_SweRankEmbed_frozen_top10",rstar_recomputed=args.full_rstar,
                 max_hops=args.max_hops,top_k=args.top_k)
    Path(args.output).mkdir(parents=True,exist_ok=True)
    (Path(args.output)/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("GRAPH_V2_SUMMARY",json.dumps(summary),flush=True)
    if failure_revisions:raise SystemExit("Frozen rerun incomplete: "+str(len(failure_revisions))+" revision errors")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--issues",required=True)
    p.add_argument("--previous",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--shard-count",type=int,default=10)
    p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--max-hops",type=int,default=5)
    p.add_argument("--max-functions",type=int,default=9000)
    p.add_argument("--full-rstar",action="store_true",
                   help="Recompute rank of all ancestors using real SweRankEmbed")
    args=p.parse_args()
    run(args)


if __name__=="__main__":
    main()
