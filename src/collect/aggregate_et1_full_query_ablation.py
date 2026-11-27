#!/usr/bin/env python3
"""Aggregate full-query SweRankEmbed-Small ablation over frozen 198 ET1 samples.

The unchanged 163 short issues inherit the *exact* V2 results.
Only the 35 long queries are re-embedded; both V2 graph and code docs remain fixed.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from urllib.parse import quote


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]


def write(path, rows):
    Path(path).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf8")


def node_url(repo, sha, sym):
    filename=sym.split(".py/")[0]+".py" if ".py/" in sym else sym
    return "https://github.com/"+repo+"/blob/"+sha+"/"+quote(filename,safe="/")


def score(records, rank_key, case_key):
    ranks=[x[rank_key] for x in records]
    n=len(ranks)
    return {
        "n":n,"mrr":sum(1/x for x in ranks)/n,
        **{"recall_at_"+str(k):sum(x<=k for x in ranks)/n for k in [1,5,10,20,50,100]},
        "cases":dict(Counter(x[case_key] for x in records)),
        "graph_recall_at_10_max5":sum(x[case_key]!="case3_no_topk_path" for x in records)/n,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--issues",required=True)
    ap.add_argument("--baseline-v2",required=True)
    ap.add_argument("--shards",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    issues=read(args.issues)
    old=read(args.baseline_v2)
    baseline={x["instance_id"]:x for x in old}
    assert len(issues)==len(old)==len(baseline)==198
    affected={x["instance_id"] for x in issues if len(x["query"])>3000}
    assert len(affected)==35
    new_rows=[]
    shard_directories=list(Path(args.shards).glob("*/"))
    errors=[]
    for path in Path(args.shards).glob("*/affected_results.jsonl"):
        new_rows+=read(path)
    for path in Path(args.shards).glob("*/errors.jsonl"):
        errors+=read(path)
    updates={x["instance_id"]:x for x in new_rows}
    assert len(updates)==len(new_rows),"duplicate experiment results"
    assert set(updates)==affected,(len(updates),len(affected),sorted(affected-set(updates))[:10])
    assert not errors,errors
    full=[];issues_by_id={x["instance_id"]:x for x in issues}
    for orig in old:
        key=orig["instance_id"]
        row=issues_by_id[key]
        replacement=updates.get(key)
        full.append({
          "instance_id":key,"repo":orig["repo"],"issue_url":orig["issue_url"],"pr_url":orig["pr_url"],
          "base_commit":orig["base_commit"],"target":orig["target"],"query_characters":len(row["query"]),
          "query_changed":replacement is not None,
          "tokens_in_full_query":replacement.get("query_tokens_including_prompt") if replacement else None,
          "model_query_truncated":replacement.get("query_truncated_by_model") if replacement else None,
          "rank_baseline":orig["target_rank"],"case_baseline":orig["case"],
          "rank_full_query":replacement["target_rank"] if replacement else orig["target_rank"],
          "case_full_query":replacement["case"] if replacement else orig["case"],
          "top10_baseline":orig["retrieved_topk"],
          "top10_full_query":replacement["retrieved_topk"] if replacement else orig["retrieved_topk"],
          "paths_baseline":orig["paths"],
          "paths_full_query":replacement["paths"] if replacement else orig["paths"],
          "selected_baseline":orig["selected_path"],
          "selected_full_query":replacement["selected_path"] if replacement else orig["selected_path"],
          "ancestor_rank_baseline":orig.get("best_ancestor_rank"),
          "ancestor_rank_full_query":replacement.get("best_ancestor_rank") if replacement else orig.get("best_ancestor_rank"),
          "ancestor_bucket_baseline":orig.get("best_ancestor_bin"),
          "ancestor_bucket_full_query":replacement.get("best_ancestor_bin") if replacement else orig.get("best_ancestor_bin"),
        })
    before=score(full,"rank_baseline","case_baseline")
    after=score(full,"rank_full_query","case_full_query")
    transition=Counter(x["case_baseline"]+" -> "+x["case_full_query"] for x in full)
    changed=[x for x in full if x["query_changed"]]
    changed_transitions=Counter(x["case_baseline"]+" -> "+x["case_full_query"] for x in changed)
    rank_change=Counter("improved" if x["rank_full_query"]<x["rank_baseline"] else
                        "worsened" if x["rank_full_query"]>x["rank_baseline"] else "same" for x in changed)
    summary={
      "status":"completed" if len(full)==198 and len(updates)==35 and not errors else "incomplete",
      "model":"Salesforce/SweRankEmbed-Small",
      "dataset":"frozen_single_root_198",
      "issues":len(full),"repositories":len({x["repo"] for x in full}),
      "queries_over_3000_chars":len(changed),
      "unchanged_queries":len(full)-len(changed),
      "baseline":"3000_char_query_graph_v2","experiment":"full_query_graph_v2",
      "document_input":"path + qualname + original 3000 character source",
      "max_hops":5,"topk":10,
      "baseline_metrics":before,"full_query_metrics":after,
      "case_transitions":dict(transition),
      "affected_case_transitions":dict(changed_transitions),
      "affected_rank_changes":dict(rank_change),
      "model_truncated_issues":sum(x.get("query_truncated_by_model",False) for x in new_rows),
      "max_chars_in_dataset":max(x["query_characters"] for x in full),
      "warning":"call graph V2 is static; paths are candidates, not proven causal traces",
    }
    (dest/"overall_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    write(dest/"full_query_vs_3000_198.jsonl",full)
    case_columns=["instance_id","repo","issue_url","pr_url","target","target_url","base_commit",
       "query_characters","query_changed","tokens_in_full_query","model_query_truncated",
       "rank_baseline","rank_full_query","rank_delta_full_minus_baseline",
       "case_baseline","case_full_query","transition","ancestor_rank_baseline",
       "ancestor_rank_full_query","ancestor_bucket_baseline","ancestor_bucket_full_query",
       "top10_baseline_json","top10_full_query_json","selected_baseline_json","selected_full_query_json"]
    with (dest/"cases_198.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=case_columns);writer.writeheader()
        for x in full:
            writer.writerow({
                **{k:x.get(k) for k in case_columns},
                "target_url":node_url(x["repo"],x["base_commit"],x["target"]),
                "rank_delta_full_minus_baseline":x["rank_full_query"]-x["rank_baseline"],
                "transition":x["case_baseline"]+" -> "+x["case_full_query"],
                "top10_baseline_json":json.dumps(x["top10_baseline"]),
                "top10_full_query_json":json.dumps(x["top10_full_query"]),
                "selected_baseline_json":json.dumps(x["selected_baseline"]),
                "selected_full_query_json":json.dumps(x["selected_full_query"]),
            })
    paths_columns=["instance_id","repo","issue_url","pr_url","graph_version",
        "query_mode","case","target_function","target_url","entry","entry_url",
        "entry_rank","hops","path_type","path_functions","path_function_urls",
        "edge_types","edge_evidence"]
    with (dest/"paths_v2_full_query_and_baseline.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=paths_columns);writer.writeheader()
        for x in full:
            for mode in ["baseline","full_query"]:
                paths=x["paths_"+mode]
                case=x["case_"+mode]
                if case=="case1_direct":
                    # Primary direct path, always 0-hop (other paths are supplementary).
                    paths=[{"entry":x["target"],"rank":x["rank_"+mode],
                            "hops":0,"nodes":[x["target"]],"edge_types":[],"edge_evidence":[],"path_type":"DIRECT_0_HOP"}]+[
                            {**p,"path_type":"TOP10_ALTERNATIVE"} for p in paths]
                elif case=="case2_forward":
                    paths=[{**p,"path_type":"TOP10_FORWARD"} for p in paths]
                else:
                    paths=[{"entry":None,"rank":None,"hops":None,"nodes":[],
                            "path_type":"NO_PATH_FROM_TOP10"}]
                for p in paths:
                    nodes=p.get("nodes",[])
                    writer.writerow({
                        "instance_id":x["instance_id"],"repo":x["repo"],
                        "issue_url":x["issue_url"],"pr_url":x["pr_url"],
                        "graph_version":"GraphV2","query_mode":mode,
                        "case":case,"target_function":x["target"],
                        "target_url":node_url(x["repo"],x["base_commit"],x["target"]),
                        "entry":p.get("entry"),
                        "entry_url":node_url(x["repo"],x["base_commit"],p["entry"]) if p.get("entry") else "",
                        "entry_rank":p.get("rank"),"hops":p.get("hops"),
                        "path_type":p.get("path_type","TOP10_FORWARD"),
                        "path_functions":" -> ".join(nodes),
                        "path_function_urls":json.dumps([node_url(x["repo"],x["base_commit"],n) for n in nodes]),
                        "edge_types":json.dumps(p.get("edge_types",[])),
                        "edge_evidence":json.dumps(p.get("edge_evidence",[])),
                    })
    with (dest/"summary_by_repo.csv").open("w",encoding="utf-8-sig",newline="") as f:
        cols=["repo","issues","queries_changed","old_case1","old_case2","old_case3",
              "full_case1","full_case2","full_case3","improved_rank","worsened_rank"]
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for repo in sorted({x["repo"] for x in full}):
            subset=[x for x in full if x["repo"]==repo]
            writer.writerow(dict(repo=repo,issues=len(subset),
                queries_changed=sum(x["query_changed"] for x in subset),
                old_case1=sum(x["case_baseline"]=="case1_direct" for x in subset),
                old_case2=sum(x["case_baseline"]=="case2_forward" for x in subset),
                old_case3=sum(x["case_baseline"]=="case3_no_topk_path" for x in subset),
                full_case1=sum(x["case_full_query"]=="case1_direct" for x in subset),
                full_case2=sum(x["case_full_query"]=="case2_forward" for x in subset),
                full_case3=sum(x["case_full_query"]=="case3_no_topk_path" for x in subset),
                improved_rank=sum(x["rank_full_query"]<x["rank_baseline"] for x in subset),
                worsened_rank=sum(x["rank_full_query"]>x["rank_baseline"] for x in subset)))
    print("FULL_QUERY_EXPERIMENT_RESULT",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
