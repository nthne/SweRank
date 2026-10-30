#!/usr/bin/env python3
"""Group modified-function weak labels per issue/base commit; NOT verified targets."""
import argparse
import json
from pathlib import Path

def group(rows):
    buckets = {}
    for row in rows:
        key = (row["instance_id"], row["base_commit"])
        if key not in buckets:
            buckets[key] = {
                "instance_id": row["instance_id"], "repo": row["repo"],
                "base_commit": row["base_commit"], "issue_url": row["issue_url"],
                "query": row["query"], "pr_urls": set(), "weak_targets": set(),
                "weak_source_targets": set(), "modified_test_functions": set(),
                "review_flags": set(), "groundtruth_status": "unverified_multi_target_candidates"}
        item = buckets[key]
        target = row["positive_id"]
        item["pr_urls"].add(row["pr_url"])
        item["weak_targets"].add(target)
        if target.startswith("tests/") or "/tests/" in target or target.startswith("test_"):
            item["modified_test_functions"].add(target)
        else:
            item["weak_source_targets"].add(target)
        item["review_flags"].update(row.get("review_flags", []))
    return [{k: sorted(v) if isinstance(v, set) else v
             for k, v in buckets[key].items()} for key in sorted(buckets)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="audit_outputs/sample_sweloc.jsonl")
    p.add_argument("--output", default="audit_outputs/issue_level.jsonl")
    args = p.parse_args()
    with Path(args.input).open(encoding="utf-8") as f:
        rows = [json.loads(s) for s in f if s.strip()]
    grouped = group(rows)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in grouped:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Grouped {len(rows)} weak labels into {len(grouped)} issue snapshots at {path}")

if __name__ == "__main__":
    main()
