#!/usr/bin/env python3
"""Re-evaluate frozen 10-repo multi-target Variant B using typed static graph V2.

No target filtering or LLM: EVERY actually modified PR function from old crawl.
Frozen SweRankEmbed rankings prevent retriever/corpus/model drift in Case1/2/3.
Evaluation unit remains a (issue-PR, modified-function) pair, not an issue.
Added functions absent at base commit are tracked but NOT counted as Case3.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tempfile

from et1_single_root_paths import code_checkout
from typed_call_graph import collect_repo_v2
from evaluate_all_pr_graph_cases import analyze_issue


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf8")


def check_baseline(gold,rankings,pairs):
    original={(x["instance_id"],x["target"]):x for x in pairs}
    if len(original)!=len(pairs):
        raise ValueError("Duplicate baseline target pair")
    rankmap={x["instance_id"]:x for x in rankings}
    if len(rankmap)!=len(rankings):
        raise ValueError("Duplicate SweRankEmbed ranking IDs")
    valid=[r for r in gold if "collection_error" not in r and r.get("all_modified_functions")]
    if any(not x.get("complete_extraction",False) for x in valid):
        raise ValueError("Incomplete PR source extraction")
    expected={(x["instance_id"],target) for x in valid for target in set(x["all_modified_functions"])}
    if expected!=set(original):
        missing=sorted(expected-set(original))[:5]
        extra=sorted(set(original)-expected)[:5]
        raise ValueError("Frozen pair mismatch: missing="+str(missing)+" extra="+str(extra))
    if set(x["instance_id"] for x in valid)!=set(rankmap):
        raise ValueError("Frozen rankings do not match issue sample IDs")
    for x in valid:
        rank=rankmap[x["instance_id"]]
        if x["base_commit"]!=rank["base_commit"] or x["repo"]!=rank["repo"]:
            raise ValueError("Buggy snapshot changed: "+x["instance_id"])
    return valid,rankmap,original


def compare_repo(args):
    root=Path(args.input)
    gold=jsonl(root/"all_pr_functions.jsonl")
    rankings=jsonl(root/"swerankembed_rankings.jsonl")
    baseline=jsonl(root/"graph_navigation"/"target_pair_cases.jsonl")
    issues,rankmap,orig=check_baseline(gold,rankings,baseline)

    group=defaultdict(list)
    for issue in issues:
        group[issue["base_commit"]].append(issue)
    repo=issues[0]["repo"] if issues else ""
    if not repo or any(x["repo"]!=repo for x in issues):
        raise ValueError("Each job must contain one nonempty repository")

    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    paths=[]
    issues_report=[]
    revisions=[]
    with tempfile.TemporaryDirectory(prefix="graph-v2-multi-") as tmp:
        for i,(commit,items) in enumerate(group.items(),1):
            checkout=code_checkout(repo,commit,Path(tmp))
            nodes,edges,stats=collect_repo_v2(checkout)
            before_edges=stats.get("v2_new_edges",0)
            revisions.append(dict(repo=repo,base_commit=commit,issues=len(items),
                                  graph_nodes=len(nodes),graph_edges=len(edges),
                                  graph_v2_added_edges=before_edges,
                                  graph_v2_edge_types=stats.get("v2_new_edge_counts",{}),
                                  external_entrypoints=len(stats.get("v2_external_entrypoints",{})),
                                  parse_errors=stats.get("parse_errors",0)))
            for issue in items:
                ranking=rankmap[issue["instance_id"]]
                # analyze_issue verifies exact equality of graph nodes with frozen
                # SweRankEmbed corpus, avoiding an accidental retrieval denominator change.
                item=analyze_issue(issue,ranking,nodes,edges,args.top_k,args.max_hops,args.reverse_hops)
                issues_report.append(dict(instance_id=issue["instance_id"],repo=repo,
                    issue_url=issue["issue_url"],pr_url=issue["pr_url"],
                    base_commit=commit,target_pairs=len(item["target_details"]),
                    cases=item["target_case_counts"]))
                evidence_map={(e["source"],e["target"]):e for e in edges}
                for p in item["localization_paths"] + item["case3_teacher_paths"]:
                    enriched=dict(p,instance_id=issue["instance_id"],repo=repo,
                                  pair_id=issue["instance_id"]+"::"+p["target"],
                                  edge_evidence=[evidence_map.get((a,b),{})
                                                 for a,b in zip(p["nodes"],p["nodes"][1:])])
                    paths.append(enriched)
                for d in item["target_details"]:
                    old=orig[(issue["instance_id"],d["target"])]
                    # target status must remain identical: absent from historical
                    # source must NOT accidentally become Case3.
                    if old["in_base_commit"]!=d["in_base_commit"]:
                        raise ValueError("Historical target membership changed "+old["pair_id"])
                    if old.get("retrieval_rank")!=d.get("retrieval_rank"):
                        raise ValueError("SweRankEmbed target rank changed "+old["pair_id"])
                    pair=dict(
                        repo=repo,
                        pair_id=d["pair_id"],
                        instance_id=issue["instance_id"],
                        issue_url=issue["issue_url"],pr_url=issue["pr_url"],
                        base_commit=commit,target=d["target"],
                        target_kind=d["target_kind"],
                        original_case=old["case"],
                        v2_case=d["case"],
                        transition=old["case"]+" -> "+d["case"],
                        original_incoming_edges=old.get("in_degree"),
                        v2_incoming_edges=d["in_degree"],
                        original_best_ancestor_rank=old.get("best_ancestor_rank"),
                        v2_best_ancestor_rank=d["best_ancestor_rank"],
                        original_best_ancestor_bin=old.get("rank_bucket"),
                        v2_best_ancestor_bin=d["rank_bucket"],
                        original_forward_hops=old.get("best_forward_hops"),
                        v2_forward_hops=d["best_forward_hops"],
                        num_shortest_paths=d["num_shortest_paths"],
                        topk_seeds=item["top_k_seeds"],
                        retrieval_rank=d["retrieval_rank"])
                    results.append(pair)
            print("MULTI_GRAPH_V2",i,"/",len(group),repo,commit[:10],
                  "issues",len(items),"nodes",len(nodes),
                  "edges",len(edges),"added_edges",before_edges,flush=True)
            write_jsonl(output/"target_pair_comparison.jsonl",results)
            write_jsonl(output/"paths_with_edge_evidence.jsonl",paths)
            write_jsonl(output/"graph_revisions.jsonl",revisions)
            write_jsonl(output/"issue_cases.jsonl",issues_report)

    old=Counter(x["original_case"] for x in results)
    new=Counter(x["v2_case"] for x in results)
    transitions=Counter(x["transition"] for x in results)
    c3_before=sum(x["original_case"]=="case3_no_topk_path" for x in results)
    c3_after=sum(x["v2_case"]=="case3_no_topk_path" for x in results)
    zero_before=sum(x["original_case"]=="case3_no_topk_path" and not x["original_incoming_edges"] for x in results)
    zero_after=sum(x["v2_case"]=="case3_no_topk_path" and not x["v2_incoming_edges"] for x in results)
    c3_bins=Counter(x["v2_best_ancestor_bin"] for x in results if x["v2_case"]=="case3_no_topk_path")
    h=Counter(str(x["v2_forward_hops"]) for x in results if x["v2_case"] in ("case1_direct","case2_forward"))
    kind_c3=Counter(x["target_kind"] for x in results if x["v2_case"]=="case3_no_topk_path")
    summary=dict(repo=repo,issue_pr_samples=len(issues),all_modified_target_pairs=len(results),
                 evaluable_pairs=sum(new[k] for k in ("case1_direct","case2_forward","case3_no_topk_path")),
                 old_cases=dict(old),v2_cases=dict(new),transitions=dict(transitions),
                 original_case3=c3_before,v2_case3=c3_after,case3_reduced=c3_before-c3_after,
                 original_case3_no_incoming=zero_before,v2_case3_no_incoming=zero_after,
                 v2_case3_bins=dict(c3_bins),v2_forward_hops=dict(h),
                 case3_target_kinds=dict(kind_c3),
                 v2_path_records=len(paths),graph_revisions=len(revisions),
                 new_edge_type_occurrences=dict(sum((Counter(x["graph_v2_edge_types"]) for x in revisions),Counter())),
                 fixed_swerankembed_ranking=True,no_llm=True,
                 sample_unit="issue_pr_target_function",max_hops=args.max_hops,
                 reverse_hops=args.reverse_hops,top_k=args.top_k)
    (output/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("MULTI_GRAPH_V2_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True,help="Unpacked original GitHub Actions artifact")
    ap.add_argument("--output",required=True)
    ap.add_argument("--top-k",type=int,default=10)
    ap.add_argument("--max-hops",type=int,default=8,
                    help="Keep old 10-repo benchmark budget 8")
    ap.add_argument("--reverse-hops",type=int,default=5)
    compare_repo(ap.parse_args())
