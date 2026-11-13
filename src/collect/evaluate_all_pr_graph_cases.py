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
def best_paths(seeds, target, outgoing, ranks, max_hops, limit=16):
    """Keep ALL equally-shortest target-specific paths, not one per issue."""
    found=[]
    for seed in seeds:
        for nodes in shortest_paths(seed,target,outgoing,max_hops=max_hops,max_paths=limit):
            found.append(dict(target=target,seed=seed,seed_rank=ranks[seed],
                              nodes=nodes,hops=len(nodes)-1,oracle_only=False,
                              localization_success=True))
    if not found:return []
    m=min(p["hops"] for p in found)
    return sorted((p for p in found if p["hops"]==m),
                  key=lambda p:(p["seed_rank"],p["seed"],p["nodes"]))[:limit]


def analyze_issue(row, ranking, nodes, edges, top_k, max_hops, reverse_hops):
    ordered=ranking["ranked_functions"]
    ranks={x:i+1 for i,x in enumerate(ordered)}
    if len(ranks)!=len(ordered):raise ValueError("Duplicate ranked candidate")
    if set(ordered)!=set(nodes):
        raise ValueError("Graph/retrieval candidate sets differ. Recreate index on the identical base commit.")
    targets=sorted(set(row["all_modified_functions"]))
    if not targets:raise ValueError("No function-level target: "+row["instance_id"])
    seeds=ordered[:top_k]
    outgoing,incoming=adjacency(edges)
    details, selected_paths, oracle_paths=[],[],[]
    test_set=set(row.get("modified_test_functions",[]))
    for target in targets:
        exists=target in nodes
        found=best_paths(seeds,target,outgoing,ranks,max_hops) if exists else []
        dist=found[0]["hops"] if found else None
        # A function added after the base snapshot is NOT an unreachable target.
        label=("not_in_base_commit" if not exists else
               "case1_direct" if dist==0 else
               "case2_forward" if dist is not None else "case3_no_topk_path")
        incoming_count=len(incoming[target]) if exists else 0
        outgoing_count=len(outgoing[target]) if exists else 0
        ancestor_dists=(distance_to_targets([target],incoming,max_hops=reverse_hops) if exists else {})
        ancestors=sorted((ranks[anc],length,anc) for anc,length in ancestor_dists.items()
                         if 1<=length<=reverse_hops and anc!=target and anc in ranks)
        rstar,ancestor_hops,ancestor=ancestors[0] if ancestors else (None,None,None)
        if label in ("case1_direct","case2_forward"):
            selected_paths.extend(found)
        elif label=="case3_no_topk_path" and ancestor:
            paths=shortest_paths(ancestor,target,outgoing,max_hops=reverse_hops,max_paths=8)
            oracle_paths.extend(dict(target=target,seed=ancestor,seed_rank=rstar,
                nodes=nodes_,hops=len(nodes_)-1,oracle_only=True,
                localization_success=False) for nodes_ in paths)
        details.append(dict(target=target,pair_id=row["instance_id"]+"::"+target,
            case=label,in_base_commit=exists,in_retrieval_index=exists,
            retrieval_rank=ranks.get(target),
            target_kind="test" if target in test_set else "source",
            in_degree=incoming_count,out_degree=outgoing_count,
            best_forward_hops=dist,best_forward_seed=found[0]["seed"] if found else None,
            num_shortest_paths=len(found),
            best_ancestor_rank=rstar,best_ancestor_node=ancestor,
            best_ancestor_hops=ancestor_hops,ancestor_count_1_to_5=len(ancestors),
            rank_bucket=bucket(rstar,incoming_count,exists)))
    counts=Counter(t["case"] for t in details)
    # An issue is a collection of target-pair samples, never a single Case1/2/3.
    return dict(instance_id=row["instance_id"],repo=row["repo"],issue_id=row.get("issue_id"),
        pr_number=row["pr_number"],pr_url=row["pr_url"],issue_url=row["issue_url"],
        base_commit=row["base_commit"],query=row["query"],retrieval_method=ranking["model"],
        retrieval_k=top_k,graph_nodes=len(nodes),graph_edges=len(edges),
        case="multi_target_issue",issue_localization_hops=None,
        target_case_counts=dict(counts),all_modified_targets=targets,
        modified_source_functions=row.get("modified_source_functions",[]),
        modified_test_functions=row.get("modified_test_functions",[]),
        targets_not_in_base=[t for t in targets if t not in nodes],
        complete_extraction=row["complete_extraction"],top_k_seeds=seeds,
        target_details=details,localization_paths=selected_paths,
        case3_teacher_paths=oracle_paths,
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
def generate_training(issue, ranked, edges):
    """Every training record is conditioned on ONE (issue, target) pair."""
    ranks={name:i+1 for i,name in enumerate(ranked)}
    outgoing,incoming=adjacency(edges)
    steps,stops,seeds=[],[],[]
    path_index=defaultdict(list)
    for p in issue["localization_paths"]+issue["case3_teacher_paths"]:
        path_index[p["target"]].append(p)
    for target in issue["target_details"]:
        t=target["target"];case=target["case"]
        if case=="not_in_base_commit":
            continue
        pair_id=target["pair_id"]
        target_dist=distance_to_targets([t],incoming,max_hops=max(1,len(ranks)))
        selected_seeds={p["seed"] for p in path_index[t] if not p.get("oracle_only")}
        for seed in issue["top_k_seeds"]:
            if seed in selected_seeds:label=1
            elif seed in target_dist:label=None   # a reachable alternative is not a hard negative
            else:label=0
            seeds.append(dict(instance_id=issue["instance_id"],target=t,pair_id=pair_id,
                pair_case=case,node=seed,rank=ranks[seed],label=label,
                teacher_distance=target_dist.get(seed)))
        seen_edges=set();seen_stops=set()
        for p in path_index[t]:
            oracle=bool(p["oracle_only"])
            for i,node in enumerate(p["nodes"]):
                stop_key=(node,i,oracle)
                if stop_key not in seen_stops:
                    stops.append(dict(instance_id=issue["instance_id"],target=t,pair_id=pair_id,
                        pair_case=case,node=node,hop=i,label=int(node==t),
                        oracle_only_teacher=oracle))
                    seen_stops.add(stop_key)
                if i+1>=len(p["nodes"]):continue
                current=node
                # Merge equally-short paths: ALL next nodes on a best path are positive.
                correct={q["nodes"][i+1] for q in path_index[t]
                         if len(q["nodes"])>i+1 and q["nodes"][i]==current
                         and bool(q["oracle_only"])==oracle}
                for candidate in sorted(outgoing[current]):
                    key=(current,candidate,i,oracle)
                    if key in seen_edges:continue
                    seen_edges.add(key)
                    label=1 if candidate in correct else None if candidate in target_dist else 0
                    steps.append(dict(instance_id=issue["instance_id"],target=t,pair_id=pair_id,
                        pair_case=case,current=current,candidate=candidate,hop=i,
                        score_features=dict(
                           rank_fraction=1-ranks[candidate]/max(len(ranks),1),
                           current_rank_fraction=1-ranks[current]/max(len(ranks),1),
                           candidate_in_degree=len(incoming[candidate]),
                           candidate_out_degree=len(outgoing[candidate]),hop=i),
                        label=label,teacher_distance=target_dist.get(candidate),
                        oracle_only_teacher=oracle))
    return steps,stops,seeds

def run(args):
    gold=load(args.gold);rankings=load(args.rankings)
    if len({x["instance_id"] for x in rankings})!=len(rankings):raise ValueError("Duplicate rankings")
    ranked={x["instance_id"]:x for x in rankings}
    valid=[r for r in gold if "collection_error" not in r and r.get("all_modified_functions")]
    if not valid:
        root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
        for name in ("issue_cases.jsonl","navigation_steps.jsonl","stop_training.jsonl","seed_training.jsonl"):
            (root/name).write_text("",encoding="utf8")
        (root/"summary.json").write_text(json.dumps({"issue_pr_samples":0,"cases":{},
            "excluded_no_modified_function_targets":sum(not r.get("all_modified_functions")
                   for r in gold if "collection_error" not in r),
            "reason":"No actual Python function change in selected issue-linked PRs",
            "no_llm":True},indent=2)+"\n")
        (root/"manual_target_review.csv").write_text("instance_id,target,case\n")
        print("No function-level target; preserving zero-target issue-PR as excluded rather than inventing Case3")
        return
    if any(not r["complete_extraction"] for r in valid):raise ValueError("Partial extraction: abort")
    # PRs that edit only metadata/docs or add module-level code may have no
    # changed Python function target. The retriever ranks them, but they must
    # be explicitly excluded from function-localization evaluation.
    valid_ids={r["instance_id"] for r in valid}
    ranked_ids=set(ranked)
    if valid_ids - ranked_ids:
        raise ValueError("Missing rankings for valid issue-PR IDs: "+str(sorted(valid_ids-ranked_ids)))
    skipped_empty=[r for r in gold if "collection_error" not in r and not r.get("all_modified_functions")]
    skipped_errors=[r for r in gold if "collection_error" in r]
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
            print(f"{i+1}/{len(valid)} {row['instance_id']} pair_cases={item['target_case_counts']} targets={len(item['all_modified_targets'])}",flush=True)
    for name,rows in [("issue_cases.jsonl",results),("navigation_steps.jsonl",all_steps),
                      ("stop_training.jsonl",all_stops),("seed_training.jsonl",all_seeds)]:
        save(root/name,rows)
    all_pairs=[]
    for issue in results:
        for t in issue["target_details"]:
            all_pairs.append(dict(instance_id=issue["instance_id"],issue_url=issue["issue_url"],
                pr_url=issue["pr_url"],repo=issue["repo"],pr_number=issue["pr_number"],
                base_commit=issue["base_commit"],**t))
    save(root/"target_pair_cases.jsonl",all_pairs)
    path_records=[]
    for issue in results:
        for p in issue["localization_paths"]+issue["case3_teacher_paths"]:
            path_records.append(dict(instance_id=issue["instance_id"],
                pair_id=issue["instance_id"]+"::"+p["target"],
                repo=issue["repo"],pair_case=("case3_no_topk_path" if p["oracle_only"]
                    else "case1_direct" if p["hops"]==0 else "case2_forward"),**p))
    save(root/"target_pair_paths.jsonl",path_records)
    counts=Counter(t["case"] for t in all_pairs)
    kinds=Counter(t["target_kind"] for t in all_pairs)
    real_hops=Counter(str(t["best_forward_hops"]) for t in all_pairs
                      if t["case"] in ("case1_direct","case2_forward"))
    rank_buckets=Counter(t["rank_bucket"] for t in all_pairs if t["case"]=="case3_no_topk_path")
    case_mix=Counter("+".join(sorted(set(x["target_case_counts"]))) for x in results)
    summary=dict(issue_pr_samples=len(results),repos=sorted(set(x["repo"] for x in results)),
        all_modified_target_pairs=len(all_pairs),
        evaluable_pairs=sum(counts[key] for key in ("case1_direct","case2_forward","case3_no_topk_path")),
        weak_source_targets=kinds["source"],weak_test_targets=kinds["test"],
        pair_cases=dict(counts),pair_minimum_forward_hops=dict(real_hops),
        issue_case_combinations=dict(case_mix),
        case3_target_rank_buckets=dict(rank_buckets),
        localization_path_count=sum(len(i["localization_paths"]) for i in results),
        case3_teacher_only_path_count=sum(len(i["case3_teacher_paths"]) for i in results),
        step_training_pairs=len(all_steps),seed_training_pairs=len(all_seeds),
        stop_training_pairs=len(all_stops),
        retrieval="Salesforce/SweRankEmbed-Small",top_k=args.top_k,
        max_hops=args.max_hops,reverse_hops=args.reverse_hops,
        sample_unit="issue_pr__target_function",
        issue_case_is_not_defined=True,no_llm=True,
        all_modified_functions_are_weak_targets=True,
        excluded_no_modified_function_targets=len(skipped_empty),
        excluded_collection_errors=len(skipped_errors))
    (root/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    cols=["instance_id","issue_url","pr_url","repo","pr_number","pair_id","target",
          "case","target_kind","in_base_commit","retrieval_rank","best_forward_hops",
          "best_forward_seed","num_shortest_paths","in_degree","out_degree",
          "best_ancestor_rank","best_ancestor_hops","rank_bucket"]
    with (root/"manual_target_review.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for t in all_pairs:w.writerow({field:t.get(field) for field in cols})
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gold",required=True);p.add_argument("--rankings",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--max-hops",type=int,default=8)
    p.add_argument("--reverse-hops",type=int,default=5)
    run(p.parse_args())
