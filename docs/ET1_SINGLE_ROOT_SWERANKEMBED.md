# ET1 single-root SweRankEmbed golden path candidates

Experiment branch: `feature/et1-single-root-100repos`, based on existing
`experiment/all-pr-graph-navigation-no-llm` (real SweRankEmbed graph pipeline).

## Dataset and leakage policy

1. Search merged PRs that explicitly fix/close/resolve a CLOSED GitHub issue.
2. Exactly ONE existing Python production function is modified across the PR.
   Disallow standalone global/config/source changes and other changed functions.
   Test and documentation changes are allowed.
3. Retrieve with real `Salesforce/SweRankEmbed-Small` on **buggy base commit
   source functions** (no BM25, no PR patch or target passed to retriever).
4. Build directed conservative Python AST call graph on the SAME base commit.
5. Case 1: target function appears in top 10 (0 hop).
   Case 2: target not in top 10 but some top-10 candidate reaches it in 1-5 hops.
   Case 3: neither condition. Keep all cases, not just successful ones.
6. Compute best-ranked reachable ancestor and bucket ranks 11-20, 21-50,
   51-100, >100. Retain nontrivial path candidates even for Case 1.
7. Only source functions are used for retrieval/graph. Tests excluded
   from both. No LLM filtering or implicit target oracle at retrieval.
8. `build_benchmark_exclusions.py` blocks SWE-bench and LocBench
   repositories and fails closed if benchmark datasets are inaccessible.

**Important:** Exactly one modified function is operational ground truth of
one root-cause *target*, not formal proof of full causality. AST graph paths
are **candidate diagnostic paths**, not verified runtime causality.

## GitHub Actions

Workflow: `.github/workflows/et1-single-root-100repos.yml` on feature branch.
128 candidate repositories from `datasets/et1_repo_candidates.txt`, sharded
10 ways and quality filtered at runtime. Target is ~100 quality-qualified
repositories; actual count must be reported, never assumed.

Artifacts:
- `et1-exclusions`: benchmark exclusion manifest
- `et1-shard-0` ... `et1-shard-9`: per-shard issues, paths, audits and summary
- `et1-combined-100repo-results`: combined JSONL and metrics

Files: `issues_single_root.jsonl`, `et1_path_candidates.jsonl`,
`repo_audit.jsonl`, `summary.json`.

## Local reproduction

```bash
python -m pip install 'datasets>=3,<5' 'huggingface_hub>=0.25,<2' \
  'sentence-transformers==3.4.1' 'transformers==4.51.3' einops numpy
export GITHUB_TOKEN=...
python src/collect/build_benchmark_exclusions.py --output out/benchmark_exclusions.json
python src/collect/et1_single_root_paths.py \
  --repos-file datasets/et1_repo_candidates.txt \
  --exclusions out/benchmark_exclusions.json \
  --output out/et1 \
  --shard-index 0 --shard-count 10 --max-prs 12 --top-k 10 --max-hops 5
```

The existing graph engine is the same as in earlier SweRank experiments,
but new filter criteria explicitly enforce a single modified production target.
