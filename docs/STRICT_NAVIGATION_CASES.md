# Strict Case-1/2/3 navigation and hop reporting

## Definitions (evaluated in this order, at issue level)

Given issue `q`, pinned base-commit function graph `G=(V,E)`, Top-K retrieval seeds `A`, and pseudo/verified target set `T`:

1. **Case 1** when `A ∩ T ≠ ∅`: report **0 localization hops**. Additional alternative/reverse paths must not enter localization or Case 2/3 counts.
2. **Case 2** only when `A ∩ T = ∅`, but at least one `a ∈ A` has a directed path to `t ∈ T` of length 1..8: report minimum forward path length from Top-K (primary, not teacher-recovered).
3. **Case 3** only when both tests fail: **report localization unsuccessful / hop null**, optionally generate candidate paths by reversing incoming edges from the known target (teacher-only) within 1..5 hops. This is offline supervision, **not inference navigation success**.

For multi-target issues, always report per-target direct-hit and reachability coverage separately, because an issue-level Case-1 may match only a subset of targets.

## Previously collected BM25 batch (47 accepted issue samples)

The original 9-repository batch has:
- Case 1: 33 issues, strictly 0 localization hops.
- Case 2: 2 issues, both with a 1-hop forward path from Top-10.
- Case 3: 12 issues, no retrieval-localization path. Of these, only `Textualize__rich-3176` contains extracted reverse diagnostic shortest paths of 2, 2 and 3 hops.
- No observed 4- or 5-hop **Case-2/Case-3** path in this frozen BM25 batch. Previous 4/5-hop figures mixed in oracle probes from Case-1 issues and must not be reported as Case-2/3 gains.

## New batch, true SweRankEmbed

Workflow `.github/workflows/navigation-strict-deep-crawl.yml` processes up to 30 issue-linked PRs per repo on Rich, Werkzeug, Datasette and ItsDangerous, explicitly excludes evaluation repositories from SWE-bench and LocBench, calls the offline LLM judge with a 24-request/repo cap, and builds graph and retrieval with `Salesforce/SweRankEmbed-Small` at each base commit.

Output is split into:

- `retrieval_localization_paths.jsonl`: **Case 1 zero-hop and Case 2 real Top-K-to-target paths only.**
- `case3_oracle_teacher_paths.jsonl`: teacher-only reverse-recovered paths.
- `case3_only_deep_paths.csv`: shortest-path diagnostics of length 2–5 **only for Case 3**; include rank and manual-review fields.
- `summary.json`: case counts and separate hop distributions for localization and Case-3 diagnostics.
- Per-repository Actions artifacts: full graph snapshots, enriched LLM pseudo labels, and rejected cases.

**Research caveat:** LLM-accepted functions remain pseudo-target labels; a static caller-to-callee shortest path is not human-verified causal ground truth.
