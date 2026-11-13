#!/usr/bin/env python3
"""Evaluate ranked function IDs against ALL modified source functions (no LLM labels).

Input rankings JSONL: {"instance_id":"...","ranked_functions":["a.py/f",...]}
All targets are WEAK labels; test targets are reported separately and not
included in source-code retrieval metrics.
"""
import argparse
import json
from pathlib import Path

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def evaluate(gold, rankings, ks=(1,3,5,10,20,50,100)):
    by_id={row["instance_id"]:row for row in rankings}
    results=[]
    for row in gold:
        if "collection_error" in row:
            continue
        targets=set(row["modified_source_functions"])
        if not targets:
            continue
        retrieved=by_id.get(row["instance_id"],{}).get("ranked_functions",[])
        ranks=list(dict.fromkeys(retrieved))
        item={"instance_id":row["instance_id"],"repo":row["repo"],
              "pr_number":row["pr_number"],"target_count":len(targets),
              "test_target_count":len(row["modified_test_functions"]),
              "extraction_complete":row["complete_extraction"],
              "has_ranking":row["instance_id"] in by_id}
        for k in ks:
            found=targets.intersection(ranks[:k])
            item[f"recall@{k}"]=len(found)/len(targets)
            item[f"all_target_acc@{k}"]=int(found==targets)
            item[f"hit@{k}"]=int(bool(found))
        item["unretrieved_targets"]=sorted(targets-set(ranks))
        results.append(item)
    summary={"samples":len(results),"rankings_provided":sum(r["has_ranking"] for r in results),
             "total_weak_source_targets":sum(r["target_count"] for r in results)}
    for k in ks:
        for metric in ("recall","all_target_acc","hit"):
            summary[f"macro_{metric}@{k}"]=round(sum(r[f"{metric}@{k}"] for r in results)/len(results),6) if results else None
    summary["label_warning"]="All modified functions are unverified weak labels, not proven causal fixes"
    return summary,results

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold",required=True)
    p.add_argument("--rankings",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    summary,items=evaluate(load(args.gold),load(args.rankings))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"summary":summary,"instances":items},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
