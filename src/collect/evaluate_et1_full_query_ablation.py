#!/usr/bin/env python3
"""ET1 198-sample ablation: SweRankEmbed-Small full query vs 3000 characters.

Re-encode ONLY the 35 inputs where query[:3000] differs. For the remaining
163, the exact input embeddings, rankings and V2 graph paths are unchanged.
Source-function snippets (3000 chars), buggy commits and V2 edges are fixed.
No developer patch/target is injected into the query.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np
from sentence_transformers import SentenceTransformer

from et1_single_root_paths import code_checkout, is_source, MODEL
from navigation_graph import adjacency, shortest_paths, distance_to_targets
from typed_call_graph import collect_repo_v2


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")


def ordered_rank(ids, scores):
    return sorted(zip(ids, np.asarray(scores).tolist()),
                  key=lambda x:(-float(x[1]), x[0]))


def retrieve_pair(model, symbols, query):
    """Exactly match navigation_graph.swerank_rank except omit query[:3000]."""
    ids=tuple(sorted(symbols))
    documents=[symbols[k].path+"\n"+symbols[k].qualname+"\n"+symbols[k].source for k in ids]
    doc_vec=np.asarray(model.encode(documents,batch_size=16,
                                    normalize_embeddings=True,show_progress_bar=False))
    query_vec=np.asarray(model.encode([query[:3000],query],prompt_name="query",
                                      normalize_embeddings=True,show_progress_bar=False))
    old_rank=ordered_rank(ids,doc_vec@query_vec[0])
    full_rank=ordered_rank(ids,doc_vec@query_vec[1])
    return old_rank,full_rank


def token_count(model, text):
    prompt=model.prompts.get("query","")
    return len(model.tokenizer(prompt+text,add_special_tokens=True,
                               truncation=False)["input_ids"])


def describe_paths(target, ranked, edges, topk=10, max_hops=5):
    rank={name:i+1 for i,(name,_) in enumerate(ranked)}
    outgoing,incoming=adjacency(edges)
    edge_map={(e["source"],e["target"]):e for e in edges}
    candidates=[]
    for i,(entry,score) in enumerate(ranked[:topk],1):
        if entry==target: continue
        for path in shortest_paths(entry,target,outgoing,max_hops=max_hops,max_paths=16):
            candidates.append({
                "entry":entry,"rank":i,"hops":len(path)-1,"nodes":path,
                "edge_types":[edge_map[(a,b)].get("type","unknown")
                              for a,b in zip(path,path[1:])],
                "edge_evidence":[edge_map[(a,b)].get("evidence")
                                 for a,b in zip(path,path[1:])],
                "oracle_only":False
            })
    candidates.sort(key=lambda p:(p["hops"],p["rank"],p["nodes"]))
    case=("case1_direct" if rank[target]<=topk else
          "case2_forward" if candidates else "case3_no_topk_path")
    ancestors=distance_to_targets([target],incoming,max_hops=max_hops)
    eligible=sorted((rank[sym],dist,sym) for sym,dist in ancestors.items()
                    if sym!=target and 1<=dist<=max_hops and sym in rank)
    rstar,hstar,anode=eligible[0] if eligible else (None,None,None)
    if not incoming.get(target):
        bucket="no_incoming_edge"
    elif rstar is None:
        bucket="other_unreachable"
    elif rstar<=10:bucket="1-10"
    elif rstar<=20:bucket="11-20"
    elif rstar<=50:bucket="21-50"
    elif rstar<=100:bucket="51-100"
    else:bucket=">100"
    return {
        "case":case,"target_rank":rank[target],
        "retrieved_topk":[{"function":s,"score":float(score),"rank":i}
                          for i,(s,score) in enumerate(ranked[:topk],1)],
        "paths":candidates[:32],
        "selected_path":([target] if case=="case1_direct" else
                         candidates[0]["nodes"] if candidates else None),
        "selected_nontrivial_hops":candidates[0]["hops"] if candidates else None,
        "best_ancestor_rank":rstar,"best_ancestor_hops":hstar,
        "best_ancestor":anode,"best_ancestor_bin":bucket,
        "has_incoming_static_edge":bool(incoming.get(target)),
        "rank_at_1":rank[target]<=1,"rank_at_5":rank[target]<=5,
        "rank_at_10":rank[target]<=10,"rank_at_20":rank[target]<=20,
        "rank_at_50":rank[target]<=50,"rank_at_100":rank[target]<=100,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--issues",required=True)
    ap.add_argument("--baseline-v1",required=True)
    ap.add_argument("--baseline-v2",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--shard-index",type=int,default=0)
    ap.add_argument("--shard-count",type=int,default=5)
    args=ap.parse_args()
    issue_rows=load_jsonl(args.issues)
    v1_rows=load_jsonl(args.baseline_v1)
    v2_rows=load_jsonl(args.baseline_v2)
    v1={r["instance_id"]:r for r in v1_rows}
    v2={r["instance_id"]:r for r in v2_rows}
    assert len(issue_rows)==len(v1)==len(v2)==198, "Frozen sample count differs"
    assert {x["instance_id"] for x in issue_rows}==set(v1)==set(v2)
    assert len({x["instance_id"] for x in issue_rows})==198
    for r in issue_rows:
        key=r["instance_id"]
        assert r["target"]==v1[key]["target"]==v2[key]["target"]
        assert r["base_commit"]==v1[key]["base_commit"]==v2[key]["base_commit"]
        assert v1[key]["target_rank"]==v2[key]["target_rank"]
    affected=[r for r in issue_rows if len(r["query"])>3000]
    assert len(affected)==35, "Input drift: expected exactly 35 clipped queries"
    repos=sorted({r["repo"] for r in affected})
    selected={repo for i,repo in enumerate(repos) if i%args.shard_count==args.shard_index}
    selected_rows=[r for r in affected if r["repo"] in selected]
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    model=SentenceTransformer(MODEL,trust_remote_code=True)
    print("LOADED_MODEL",MODEL,"max_seq_length",model.max_seq_length,
          "query_prompt",repr(model.prompts.get("query","")),flush=True)
    results=[];errors=[]
    with tempfile.TemporaryDirectory(prefix="et1-full-query-") as temp:
        for i,r in enumerate(selected_rows,1):
            try:
                source=code_checkout(r["repo"],r["base_commit"],Path(temp))
                symbols,all_edges,stats=collect_repo_v2(source)
                symbols={k:v for k,v in symbols.items() if is_source(v.path)}
                edges=[e for e in all_edges if e["source"] in symbols and e["target"] in symbols]
                assert r["target"] in symbols, "Root cause not present at base commit"
                assert len(symbols)==v1[r["instance_id"]]["index_size"], "Indexed function set changed"
                old,full=retrieve_pair(model,symbols,r["query"])
                gold=r["target"];idx=r["instance_id"]
                old_rank={k:n for n,(k,_) in enumerate(old,1)}
                old_top=[x["function"] for x in v1[idx]["retrieved_topk"]]
                reproduced=(old_rank[gold]==v1[idx]["target_rank"] and
                             [x[0] for x in old[:len(old_top)]]==old_top)
                if not reproduced:
                    raise ValueError("Baseline 3000-character ranking not reproduced: "
                                     +str(old_rank[gold])+" vs "+
                                     str(v1[idx]["target_rank"])+" index="+str(len(symbols)))
                changed=describe_paths(gold,full,edges,topk=10,max_hops=5)
                tok=token_count(model,r["query"])
                result={
                    "instance_id":idx,"repo":r["repo"],"issue_url":r["issue_url"],
                    "pr_url":r["pr_url"],"base_commit":r["base_commit"],
                    "target":gold,"query_chars":len(r["query"]),
                    "query_tokens_including_prompt":tok,
                    "query_tokens_used":min(tok,model.max_seq_length),
                    "query_truncated_by_model":tok>model.max_seq_length,
                    "old_3000_target_rank":v2[idx]["target_rank"],
                    "old_3000_case":v2[idx]["case"],
                    "old_3000_case1":v2[idx]["target_rank"]<=10,
                    "baseline_reproduced":True,
                    "graph_edges":len(edges),"graph_kind":"typed_call_graph_v2",
                    "model":MODEL,
                    "query_mode":"full_input_model_token_truncation_only",
                    **changed
                }
                results.append(result)
                print("FULL_QUERY",args.shard_index,i,"/",len(selected_rows),
                      idx,"rank",v2[idx]["target_rank"],"->",result["target_rank"],
                      "case",v2[idx]["case"],"->",result["case"],
                      "tokens",tok,flush=True)
            except Exception as exc:
                errors.append(dict(instance_id=r["instance_id"],repo=r["repo"],error=str(exc)[:500]))
                print("FULL_QUERY_ERROR",r["instance_id"],repr(exc),flush=True)
            dump_jsonl(out/"affected_results.jsonl",results)
            dump_jsonl(out/"errors.jsonl",errors)
    summary=dict(shard=args.shard_index,affected_issues=len(selected_rows),
                 processed=len(results),errors=len(errors),
                 transitions=dict(Counter(v2[x["instance_id"]]["case"]+" -> "+x["case"]
                                          for x in results)),
                 full_query_model_truncations=sum(x["query_truncated_by_model"] for x in results),
                 change_rank_improved=sum(x["target_rank"]<x["old_3000_target_rank"] for x in results),
                 change_rank_worsened=sum(x["target_rank"]>x["old_3000_target_rank"] for x in results))
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")
    print("FULL_QUERY_SUMMARY",json.dumps(summary),flush=True)
    if errors:raise SystemExit("Incomplete affected full-query experiment")

if __name__=="__main__":main()
