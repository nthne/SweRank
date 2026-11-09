#!/usr/bin/env python3
"""Collect ALL modified Python functions from merged, issue-linked PRs.

No LLM, no semantic threshold, no selection of functions. All extracted
functions are weak labels, not confirmed causally necessary fixes.
Python standard library only; network required when running the crawler.
"""
import argparse
import ast
import base64
import json
import os
from pathlib import Path
from urllib.parse import quote
from sample_sweloc import api, pages, linked_issues
from build_benchmark_exclusions import canonical

def spans(source, filename):
    result = {}
    tree = ast.parse(source)
    def walk(nodes, ancestors=()):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, ancestors + (node.name,))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                path = filename + "/" + "/".join(ancestors + (node.name,))
                # Include decorators in function span, but not unrelated definitions.
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                result[path] = (start, node.end_lineno or start)
                walk(node.body, ancestors + (node.name,))
    walk(tree.body)
    return result

def diff_lines(patch):
    """Return changed line numbers on the OLD and NEW side independently."""
    import re
    old, new = set(), set()
    a = b = None
    for line in (patch or "").splitlines():
        match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if match:
            a, b = int(match[1]), int(match[3])
        elif a is not None and line.startswith(" "):
            a += 1; b += 1
        elif a is not None and line.startswith("-") and not line.startswith("---"):
            old.add(a); a += 1
        elif a is not None and line.startswith("+") and not line.startswith("+++"):
            new.add(b); b += 1
    return old, new

def select_functions(source, filename, changed_lines):
    if not source:
        return set()
    symbols = spans(source, filename)
    changed = set()
    for line in changed_lines:
        matches = [(symbol, end - start) for symbol, (start, end) in symbols.items()
                   if start <= line <= end]
        if matches:
            changed.add(min(matches, key=lambda item:item[1])[0])
    return changed

def source_at(repo, sha, filename, token):
    data = api(f"/repos/{repo}/contents/{quote(filename,safe='/')}?ref={quote(sha,safe='')}",token)
    return base64.b64decode(data["content"]).decode("utf-8", "replace")

def modified_symbols(filename, patch, before="", after=""):
    old_lines, new_lines = diff_lines(patch)
    old_names = select_functions(before,filename,old_lines)
    new_names = select_functions(after,filename,new_lines)
    return old_names | new_names, old_names, new_names

def manifest_repos(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not data.get("sources") or data.get("policy") != "fail_closed_repository_level_exclusion":
        raise ValueError("Exclusion manifest missing verified sources or policy")
    excluded = data.get("excluded_repos")
    if not excluded or not isinstance(excluded,list):
        raise ValueError("Exclusion manifest empty")
    return set(excluded)

def collect(repo, max_prs, token, excluded):
    name = canonical(repo)
    if not name or name != repo.lower():
        raise ValueError("Require canonical owner/name")
    if name in excluded:
        raise ValueError(f"Excluded benchmark repository: {repo}")
    # GitHub PR list is sorted by latest update, but merged_at may be absent.
    prs=pages(f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc",token)
    outputs=[]
    considered=0
    for pr in prs:
        if considered>=max_prs:
            break
        if not pr.get("merged_at"):
            continue
        linked=linked_issues(pr.get("body"),repo)
        if not linked:
            continue
        considered+=1
        pr_no=pr["number"]
        base=pr["base"]["sha"]
        files=pages(f"/repos/{repo}/pulls/{pr_no}/files",token)
        targets=set(); source_targets=set(); test_targets=set(); skipped=[]
        for file in files:
            name=file.get("filename","")
            if not name.endswith(".py"):
                continue
            status=file.get("status")
            if not file.get("patch"):
                skipped.append({"file":name,"reason":"missing_patch"})
                continue
            previous=file.get("previous_filename",name)
            try:
                before="" if status=="added" else source_at(repo,base,previous,token)
                after="" if status=="removed" else source_at(repo,pr["head"]["sha"],name,token)
                old, new=diff_lines(file["patch"])
                old_names=select_functions(before,previous,old)
                new_names=select_functions(after,name,new)
                # Old and new paths both retained to represent moves/renames.
                symbols=old_names|new_names
                targets|=symbols
                if "test" in name.lower() or "test" in previous.lower():
                    test_targets|=symbols
                else:
                    source_targets|=symbols
            except (ValueError,SyntaxError,RuntimeError,KeyError,UnicodeError) as exc:
                skipped.append({"file":name,"reason":str(exc)[:250]})
        for issue_no in linked:
            try:
                issue=api(f"/repos/{repo}/issues/{issue_no}",token)
                if "pull_request" in issue:
                    continue
                query=((issue.get("title") or "")+"\n"+(issue.get("body") or "")).strip()
                if not query:
                    continue
                outputs.append({
                    "instance_id":f"{repo.replace('/','__')}-{issue_no}-pr{pr_no}",
                    "issue_id":f"{repo.replace('/','__')}-{issue_no}",
                    "repo":repo,"issue_number":issue_no,"pr_number":pr_no,
                    "issue_url":issue["html_url"],"pr_url":pr["html_url"],
                    "base_commit":base,"head_commit":pr["head"]["sha"],
                    "query":query,
                    "all_modified_functions":sorted(targets),
                    "modified_source_functions":sorted(source_targets),
                    "modified_test_functions":sorted(test_targets),
                    "unparsed_files":skipped,"complete_extraction":not skipped,
                    "groundtruth_status":"weak_all_pr_modified_functions_unverified",
                    "selection_policy":"all_modified_no_llm_no_similarity_filter"
                })
            except (RuntimeError,ValueError,KeyError) as exc:
                outputs.append({"repo":repo,"pr_number":pr_no,"issue_number":issue_no,
                                "collection_error":str(exc)[:300]})
    return outputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",action="append",required=True)
    p.add_argument("--exclusions",required=True)
    p.add_argument("--max-prs",type=int,default=20)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    excluded=manifest_repos(args.exclusions)
    if args.max_prs < 1:
        p.error("--max-prs must be positive")
    rows=[]
    for repo in args.repo:
        rows+=collect(repo,args.max_prs,os.environ.get("GITHUB_TOKEN"),excluded)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row,ensure_ascii=False)+"\n")
    valid=[r for r in rows if "collection_error" not in r]
    print(f"Saved {len(valid)} issue/PR instances; {sum(len(r['all_modified_functions']) for r in valid)} weak labels, {len(rows)-len(valid)} errors")
    if not valid or any(not r["complete_extraction"] for r in valid):
        raise SystemExit("Incomplete or empty extraction; inspect output. No silently accepted instances.")

if __name__=="__main__":
    main()
