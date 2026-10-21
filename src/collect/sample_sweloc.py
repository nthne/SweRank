#!/usr/bin/env python3
"""Create a small audit dataset from merged, issue-linked GitHub PRs.

NOT original SWELOC. Changed functions are WEAK labels, not confirmed bugs.
CPU-only, Python standard library.
"""
import argparse
import ast
import base64
import csv
import json
import os
import re
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

FIXES = re.compile(r"(?i)\b(?:fix(?:es|ed)?|clos(?:es|ed)?|resolv(?:es|ed)?)\s*:?\s*(?:issue\s+)?(?:([\w.-]+/[\w.-]+))?#(\d+)\b")
ISSUE_URL = re.compile(r"https://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)")
HUNK = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
WORDS = re.compile(r"[A-Za-z_][A-Za-z_0-9]{2,}")
STOP = set("the and for from with that this when does not can into have but are has was function file error issue fix bug test python return self none true false".split())

def api(url, token):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "SweRank-groundtruth-audit"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = Request("https://api.github.com" + url, headers=headers)
    try:
        with urlopen(req, timeout=35) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"GitHub API {exc.code}: {url}") from exc

def pages(path, token):
    result = []
    for page in range(1, 4):
        sep = "&" if "?" in path else "?"
        batch = api(f"{path}{sep}per_page=100&page={page}", token)
        if not isinstance(batch, list):
            raise RuntimeError(f"Expected a list: {path}")
        result.extend(batch)
        if len(batch) < 100:
            break
    return result

def linked_issues(body, repo):
    out = set()
    for m in FIXES.finditer(body or ""):
        if not m.group(1) or m.group(1).lower() == repo.lower():
            out.add(int(m.group(2)))
    for m in ISSUE_URL.finditer(body or ""):
        if m.group(1).lower() == repo.lower():
            out.add(int(m.group(2)))
    return sorted(out)

def changed_old_lines(patch):
    result, old = set(), None
    for line in (patch or "").splitlines():
        match = HUNK.match(line)
        if match:
            old = int(match.group(1))
            continue
        if old is None or line.startswith("\\"):
            continue
        if line.startswith(" "):
            old += 1
        elif line.startswith("-"):
            result.add(old)
            old += 1
        elif line.startswith("+"):
            result.add(max(1, old - 1))
    return result

def modified_functions(code, path, patch):
    tree = ast.parse(code)
    spans = []
    def walk(nodes, outer=()):
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                walk(node.body, outer + (node.name,))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                spans.append((node.lineno, node.end_lineno or node.lineno, outer + (node.name,)))
                walk(node.body, outer + (node.name,))
    walk(tree.body)
    result = set()
    for line in changed_old_lines(patch):
        matches = [span for span in spans if span[0] <= line <= span[1]]
        if matches:
            span = min(matches, key=lambda v: (v[1] - v[0], -len(v[2])))
            result.add(f"{path}/{'/'.join(span[2])}")
    return sorted(result)

def overlap(issue, function_id):
    def tokens(txt):
        txt = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", txt).replace("_", " ")
        return {w.lower() for w in WORDS.findall(txt) if w.lower() not in STOP}
    q, f = tokens(issue), tokens(function_id)
    return round(len(q & f) / len(f), 3) if f else 0

def audit_pr(repo, pr, token, need_test=True):
    if not pr.get("merged_at"):
        return []
    links = linked_issues(pr.get("body"), repo)
    if not links:
        return []
    files = pages(f"/repos/{repo}/pulls/{pr['number']}/files", token)
    has_test = any("test" in f.get("filename", "").lower().split("/")[-1]
                   or "/tests/" in "/" + f.get("filename", "").lower() for f in files)
    if need_test and not has_test:
        return []
    targets, missing = set(), []
    for item in files:
        name = item.get("filename", "")
        if not name.endswith(".py") or item.get("status") == "added":
            continue
        if not item.get("patch"):
            missing.append(name)
            continue
        try:
            path = f"/repos/{repo}/contents/{quote(name, safe='/')}?ref={quote(pr['base']['sha'])}"
            data = api(path, token)
            code = base64.b64decode(data["content"]).decode("utf-8", "replace")
            targets.update(modified_functions(code, name, item["patch"]))
        except (ValueError, SyntaxError, RuntimeError, UnicodeError, KeyError) as exc:
            missing.append(f"{name}: {exc}")
    rows = []
    for issue_no in links:
        if not targets:
            break
        issue = api(f"/repos/{repo}/issues/{issue_no}", token)
        if "pull_request" in issue:
            continue
        query = ((issue.get("title") or "") + "\n" + (issue.get("body") or "")).strip()
        if not query:
            continue
        for target in sorted(targets):
            sim = overlap(query, target)
            flags = (["low_identifier_overlap"] if sim == 0 else [])
            if len(targets) > 1:
                flags.append("multi_function_pr")
            if missing:
                flags.append("incomplete_patch_or_parse")
            rows.append({
                "instance_id": f"{repo.replace('/', '__')}-{issue_no}",
                "repo": repo, "pr_number": pr["number"],
                "issue_url": issue["html_url"], "pr_url": pr["html_url"],
                "base_commit": pr["base"]["sha"], "query": query,
                "positive_id": target, "positive_source": "modified_function_in_merged_pr",
                "groundtruth_status": "weak_label_unverified",
                "semantic_overlap": sim, "review_flags": flags,
                "other_modified_functions": sorted(targets - {target}),
                "unparsed_files": missing
            })
    return rows

def collect(repos, prs, max_prs, token, need_test=True):
    selected = []
    for spec in prs:
        repo, number = spec.rsplit("#", 1)
        selected.append((repo, api(f"/repos/{repo}/pulls/{int(number)}", token)))
    for repo in repos:
        candidates = pages(f"/repos/{repo}/pulls?state=closed&sort=updated&direction=desc", token)
        eligible = [p for p in candidates if p.get("merged_at") and linked_issues(p.get("body"), repo)]
        selected.extend((repo, p) for p in eligible[:max_prs])
    seen, rows = set(), []
    for repo, pr in selected:
        identity = (repo, pr["number"])
        if identity in seen:
            continue
        seen.add(identity)
        try:
            rows.extend(audit_pr(repo, pr, token, need_test))
        except (RuntimeError, ValueError, KeyError, URLError) as exc:
            print(f"Skip {repo}#{pr['number']}: {exc}", file=sys.stderr)
    return rows

def write(rows, outdir):
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "sample_sweloc.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    columns = ["instance_id", "issue_url", "pr_url", "positive_id", "semantic_overlap",
               "review_flags", "reviewer_label", "reviewer_evidence"]
    with (path / "review_queue.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda x: x["semantic_overlap"]):
            writer.writerow({**{k: row.get(k, "") for k in columns},
                            "review_flags": ";".join(row["review_flags"]),
                            "reviewer_label": "unreviewed", "reviewer_evidence": ""})
    summary = ("# SWELOC mini-audit\n\n"
               f"Issue instances: {len({r['instance_id'] for r in rows})}\n\n"
               f"Weak function labels: {len(rows)}\n\n"
               "**Not a verified ground truth**. Flags indicate review priority only.\n")
    (path / "README.md").write_text(summary, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[], help="owner/name")
    parser.add_argument("--pr", action="append", default=[], help="owner/name#number")
    parser.add_argument("--max-prs", type=int, default=5)
    parser.add_argument("--output-dir", default="audit_outputs")
    parser.add_argument("--allow-no-test", action="store_true")
    args = parser.parse_args()
    if not args.repo and not args.pr:
        parser.error("Specify --repo or --pr")
    records = collect(args.repo, args.pr, args.max_prs,
                      os.getenv("GITHUB_TOKEN"), not args.allow_no_test)
    write(records, args.output_dir)
    print(f"Saved {len(records)} weak labels in {args.output_dir}")

if __name__ == "__main__":
    main()
