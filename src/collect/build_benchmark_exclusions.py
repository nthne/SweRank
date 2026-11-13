#!/usr/bin/env python3
"""Build a fail-closed repository exclusion manifest from benchmark datasets.

This is repository-level exclusion, not merely instance-level de-duplication.
Installation: pip install datasets huggingface_hub
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BENCHMARKS = (
    "SWE-bench/SWE-bench",
    "SWE-bench/SWE-bench_Lite",
    "SWE-bench/SWE-bench_Verified",
    "SWE-bench/SWE-bench_Multilingual",
    "SWE-bench/SWE-bench_Multimodal",
    "czlll/Loc-Bench_V1",
)
URL_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+)", re.I)
NAME_RE = re.compile(r"^([\w.-]+)__([\w.-]+)-\d+$")


def canonical(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    match = URL_RE.search(value)
    if match:
        value = match.group(1)
    value = value.removeprefix("https://github.com/").strip("/")
    if value.count("/") == 1 and all(re.fullmatch(r"[\w.-]+", x) for x in value.split("/")):
        return value.lower().removesuffix(".git")
    return None


def repository_from_row(row):
    for key in ("repo", "repository", "repo_full_name", "repository_full_name", "repo_url"):
        if key in row:
            name = canonical(row[key])
            if name:
                return name
    for a, b in (("repo_owner", "repo_name"), ("owner", "name")):
        if row.get(a) and row.get(b):
            name = canonical(str(row[a]) + "/" + str(row[b]))
            if name:
                return name
    for key in ("issue_url", "pr_url", "pull_url", "pull_request_url"):
        name = canonical(row.get(key))
        if name:
            return name
    instance = row.get("instance_id")
    if isinstance(instance, str):
        match = NAME_RE.match(instance)
        if match:
            return (match.group(1) + "/" + match.group(2)).lower()
    return None


def collect_manifest(datasets=BENCHMARKS, extras=()):
    """Every dataset must be readable and yield recognizable repos; otherwise fail."""
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
    from huggingface_hub import HfApi

    all_repos = set()
    sources = []
    hub = HfApi()
    for dataset in datasets:
        revision = hub.dataset_info(dataset).sha
        configs = get_dataset_config_names(dataset, revision=revision)
        source_repos = set()
        seen_rows = 0
        for config in configs:
            splits = get_dataset_split_names(dataset, config_name=config, revision=revision)
            for split in splits:
                rows = load_dataset(dataset, name=config, split=split, revision=revision, streaming=True)
                for row in rows:
                    seen_rows += 1
                    repo = repository_from_row(row)
                    if repo:
                        source_repos.add(repo)
        if seen_rows == 0 or not source_repos:
            raise RuntimeError(f"Unsafe to proceed: {dataset} returned no identifiable repositories")
        all_repos.update(source_repos)
        sources.append({"dataset": dataset, "revision": revision, "rows": seen_rows,
                        "repositories": len(source_repos)})
    for file in extras:
        path = Path(file)
        data = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else path.read_text(encoding="utf-8").splitlines()
        if isinstance(data, dict):
            data = data.get("excluded_repos", data.get("repositories", []))
        extra_repos = {canonical(item) for item in data}
        if None in extra_repos:
            raise ValueError(f"Invalid repository name in additional exclusions: {file}")
        all_repos.update(extra_repos)
        sources.append({"manual_file": str(path), "repositories": len(extra_repos)})
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "excluded_repos": sorted(all_repos), "sources": sources,
            "policy": "fail_closed_repository_level_exclusion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", help="HF benchmark dataset, repeatable; defaults to six")
    parser.add_argument("--extra", action="append", default=[], help="Additional JSON or plain-text repo list")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = collect_manifest(tuple(args.dataset) if args.dataset else BENCHMARKS, args.extra)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Excluded {len(manifest['excluded_repos'])} repositories from {len(manifest['sources'])} manifests")


if __name__ == "__main__":
    main()
