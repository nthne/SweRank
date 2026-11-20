#!/usr/bin/env python3
"""ET1 one-root SweRankEmbed retrieval and directed static diagnostic path candidates.

This extends existing SweRank sample_sweloc, collect_all_pr_functions, navigation_graph.
No target/test/fix patch is passed into SweRankEmbed. Paths are offline static proxies.
"""
import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from sample_sweloc import api, pages, FIX
from collect_all_pr_functions import source_at, spans, diff_lines
from navigation_graph import collect_repo, swerank_rank, adjacency, shortest_paths, distance_to_targets

MODEL = "Salesforce/SweRankEmbed-Small"
NON_CODE = re.compile(r"(?:^|/)(?:docs?|examples?|\.github)/|\.(?:md|rst|txt)$|^(?:LICENSE|CHANGELOG|README)", re.I)

def is_test(filename):
    parts = Path(filename.lower()).parts
    return any(p in ("test","tests","testing") for p in parts) or parts[-1].startswith("test_") or parts[-1].endswith("_test.py") or parts[-1]=="conftest.py"

def is_source(filename):
    return filename.endswith(".py") and not is_test(filename) and not NON_CODE.search(filename)

def assign(lines, spans_by_name):
    found=set(); missing=set()
    for line in lines:
        matches=[(name,end-start) for name,(start,end) in spans_by_name.items() if start<=line<=end]
        if matches: found.add(min(matches,key=lambda x:x[1])[0])
        else: missing.add(line)
    return found, missing

def single_root(files, before_fn, after_fn):
    """Accept only if every non-test, non-documentation edit is within one OLD function."""
    changed=set()
    for file in files:
        name=file["filename"]
        if is_test(name) or NON_CODE.search(name): continue
        if not is_source(name): return None,"extra_production_file"
        if file.get("status")!="modified" or not file.get("patch"): return None,"incomplete_source_patch"
        if file.get("previous_filename",name)!=name: return None,"renamed_source"
        before=before_fn(name); after=after_fn(name)
        old_lines,new_lines=diff_lines(file["patch"])
        if not old_lines and not new_lines: return None,"empty_edit"
        oldsp,newsp=spans(before,name),spans(after,name)
        old,badold=assign(old_lines,oldsp); new,badnew=assign(new_lines,newsp)
        if badold or badnew: return None,"edit_outside_function"
        if new-set(oldsp) or old-set(newsp): return None,"new_or_deleted_function"
        changed.update(old | new)
    if len(changed)!=1: return None,"function_count_"+str(len(changed))
    return next(iter(changed)),"accepted"

def fixing_issues(pr, repo):
    result=set()
    for m in FIX.finditer(pr.get("body") or ""):
        if not m.group(1) or m.group(1).lower()==repo.lower():
            result.add(int(m.group(2)))
    return sorted(result)

def save(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")

def crawl(repo, args, exclusions):
    report={"repo":repo,"status":"unknown","issues":0,"prs_inspected":0}
    if repo.lower() in exclusions:
        report["status"]="benchmark_overlap"; return [],report
    meta=api("/repos/"+repo,os.environ["GITHUB_TOKEN"])
    if (meta.get("archived") or meta.get("fork") or meta.get("disabled")
        or not meta.get("has_issues") or int(meta.get("stargazers_count",0))<args.min_stars
        or int(meta.get("size",0))>args.max_size_kb):
        report["status"]="quality_gate";return [],report
    prs=pages("/repos/"+repo+"/pulls?state=closed&sort=updated&direction=desc",os.environ["GITHUB_TOKEN"])
    results=[];counts=Counter()
    for pr in prs:
        if report["prs_inspected"]>=args.max_prs or len(results)>=args.max_issues_per_repo: break
        numbers=fixing_issues(pr,repo)
        if not pr.get("merged_at") or not numbers: continue
        report["prs_inspected"]+=1
        try:
            files=pages(f"/repos/{repo}/pulls/{pr['number']}/files",os.environ["GITHUB_TOKEN"])
            if len(files)>=300:
                counts["oversize_pr"]+=1;continue
            target,reason=single_root(files,
                lambda filename:source_at(repo,pr["base"]["sha"],filename,os.environ["GITHUB_TOKEN"]),
                lambda filename:source_at(repo,pr["head"]["sha"],filename,os.environ["GITHUB_TOKEN"]))
            if not target:
                counts[reason]+=1;continue
            for number in numbers:
                if len(results)>=args.max_issues_per_repo:break
                issue=api(f"/repos/{repo}/issues/{number}",os.environ["GITHUB_TOKEN"])
                if issue.get("state")!="closed" or "pull_request" in issue:
                    counts["not_closed_issue"]+=1;continue
                query=((issue.get("title") or "")+"\n"+(issue.get("body") or "")).strip()
                if not query:
                    counts["empty_issue"]+=1;continue
                results.append(dict(
                    instance_id=f"{repo.replace('/','__')}-{number}-pr{pr['number']}",
                    repo=repo,issue_number=number,pr_number=pr["number"],
                    pr_url=pr["html_url"],issue_url=issue["html_url"],
                    base_commit=pr["base"]["sha"],target=target,query=query,
                    target_count=1,label_rule="one_existing_production_function_edited",
                    patch_only_used_offline=True))
        except Exception as exc:
            counts["pr_exception"]+=1
            print("PR ERROR",repo,pr["number"],str(exc)[:180],flush=True)
    report.update(status="eligible" if results else "no_single_root_issue",
                  issues=len(results),rejections=dict(counts))
    return results,report

def code_checkout(repo,sha,root):
    path=root/repo.replace("/","__")
    if not (path/".git").exists():
        subprocess.run(["git","clone","-q","--filter=blob:none","--no-checkout",
             f"https://github.com/{repo}.git",str(path)],check=True,timeout=450)
    p=subprocess.run(["git","-C",str(path),"checkout","-q","--force",sha],capture_output=True,timeout=240)
    if p.returncode:
        subprocess.run(["git","-C",str(path),"fetch","-q","origin",sha],check=True,timeout=450)
        subprocess.run(["git","-C",str(path),"checkout","-q","--force",sha],check=True,timeout=240)
    return path

def evaluate(rows,args,root):
    bysha=defaultdict(list)
    for row in rows:bysha[row["base_commit"]].append(row)
    results=[]
    for sha,items in bysha.items():
        try:
            checkout=code_checkout(items[0]["repo"],sha,root)
            symbols,edges,stats=collect_repo(checkout)
            symbols={k:v for k,v in symbols.items() if is_source(v.path)}
            edges=[e for e in edges if e["source"] in symbols and e["target"] in symbols]
            if len(symbols)>args.max_functions:raise ValueError("source index exceeds max_functions")
            outgoing,incoming=adjacency(edges)
            for row in items:
                answer={k:row[k] for k in ("instance_id","repo","target","base_commit","issue_url","pr_url")}
                answer.update(retriever=MODEL,index_size=len(symbols),graph_edges=len(edges),graph_stats=stats,
                              path_type="static_call_proxy_not_causal_proof",
                              target_rank=None,paths=[],selected_path=None)
                if row["target"] not in symbols:
                    answer["case"]="target_missing_in_base"
                else:
                    ranked=swerank_rank(row["query"],symbols,MODEL)
                    rank={name:i+1 for i,(name,_) in enumerate(ranked)}
                    target=row["target"];target_rank=rank[target]
                    answer["target_rank"]=target_rank
                    answer["retrieved_topk"]=[{"function":name,"score":float(score),"rank":i}
                                             for i,(name,score) in enumerate(ranked[:args.top_k],1)]
                    options=[]
                    for entry,_score in ranked[:args.top_k]:
                        if entry==target:continue
                        for path in shortest_paths(entry,target,outgoing,max_hops=args.max_hops,max_paths=8):
                            options.append({"entry":entry,"rank":rank[entry],"hops":len(path)-1,
                                            "nodes":path,"oracle_only":False})
                    options.sort(key=lambda x:(x["hops"],x["rank"],x["nodes"]))
                    d=distance_to_targets([target],incoming,max_hops=args.max_hops)
                    ancestors=sorted((rank[a],dist,a) for a,dist in d.items()
                                     if a!=target and 1<=dist<=args.max_hops and a in rank)
                    rstar,hstar,node=ancestors[0] if ancestors else (None,None,None)
                    bin_name=("no_incoming_edge" if not incoming.get(target) else "other_unreachable") if rstar is None else (
                        "1-10" if rstar<=10 else "11-20" if rstar<=20 else "21-50" if rstar<=50 else
                        "51-100" if rstar<=100 else ">100")
                    case=("case1_direct" if target_rank<=args.top_k else
                          "case2_forward" if options else "case3_no_topk_path")
                    answer.update(case=case,paths=options[:32],
                        selected_path=options[0]["nodes"] if options else ([target] if case=="case1_direct" else None),
                        selected_nontrivial_hops=options[0]["hops"] if options else None,
                        best_ancestor_rank=rstar,best_ancestor_hops=hstar,
                        best_ancestor=node,best_ancestor_bin=bin_name,
                        has_incoming_static_edge=bool(incoming.get(target)))
                results.append(answer)
        except Exception as exc:
            print("EVAL ERROR",items[0]["repo"],str(exc)[:250],flush=True)
            for row in items:
                results.append(dict(instance_id=row["instance_id"],repo=row["repo"],
                                    target=row["target"],case="runtime_error",error=str(exc)[:300]))
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repos-file",required=True);p.add_argument("--exclusions",required=True)
    p.add_argument("--output",required=True);p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--shard-count",type=int,default=1);p.add_argument("--max-prs",type=int,default=12)
    p.add_argument("--max-issues-per-repo",type=int,default=3)
    p.add_argument("--min-stars",type=int,default=100);p.add_argument("--max-size-kb",type=int,default=150000)
    p.add_argument("--top-k",type=int,default=10);p.add_argument("--max-hops",type=int,default=5)
    p.add_argument("--max-functions",type=int,default=9000)
    args=p.parse_args()
    if not os.getenv("GITHUB_TOKEN"):raise SystemExit("GITHUB_TOKEN required")
    if not 0<=args.shard_index<args.shard_count:raise SystemExit("Invalid shard")
    exclusion=json.loads(Path(args.exclusions).read_text(encoding="utf8"))
    if exclusion.get("policy")!="fail_closed_repository_level_exclusion" or not exclusion.get("sources"):
        raise SystemExit("Incomplete benchmark exclusion manifest")
    excluded=set(exclusion["excluded_repos"])
    seeds=[s.strip() for s in Path(args.repos_file).read_text().splitlines() if s.strip() and not s.startswith("#")]
    seeds=[s for i,s in enumerate(seeds) if i%args.shard_count==args.shard_index]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    records=[];audit=[];paths=[]
    with tempfile.TemporaryDirectory(prefix="et1-swerank-") as tmp:
        for i,repo in enumerate(seeds,1):
            try:
                got,status=crawl(repo,args,excluded);records.extend(got);audit.append(status)
                if got:paths.extend(evaluate(got,args,Path(tmp)))
                print(f"SHARD {args.shard_index}: {i}/{len(seeds)} {repo}: "
                      f"{status['status']} issue_count={len(got)}",flush=True)
            except Exception as exc:
                audit.append(dict(repo=repo,status="repo_error",error=str(exc)[:250]))
                print("REPO ERROR",repo,str(exc)[:250],flush=True)
            save(out/"issues_single_root.jsonl",records)
            save(out/"repo_audit.jsonl",audit)
            save(out/"et1_path_candidates.jsonl",paths)
    counts=Counter(x["case"] for x in paths)
    hops=Counter(str(x["selected_nontrivial_hops"]) for x in paths if x.get("selected_nontrivial_hops") is not None)
    bins=Counter(x.get("best_ancestor_bin") for x in paths if x["case"]=="case3_no_topk_path")
    summary=dict(shard=args.shard_index,candidate_repos_checked=len(audit),
                 quality_repos=sum(x["status"] in ("eligible","no_single_root_issue") for x in audit),
                 repos_with_single_root_issues=sum(x["status"]=="eligible" for x in audit),
                 single_root_issue_pr_samples=len(records),paths_evaluated=len(paths),
                 cases=dict(counts),nontrivial_hops=dict(hops),case3_ancestor_bins=dict(bins),
                 model=MODEL,top_k=args.top_k,max_hops=args.max_hops,no_llm=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("SUMMARY",json.dumps(summary),flush=True)

if __name__=="__main__":main()
