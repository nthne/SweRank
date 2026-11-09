# End-to-end experiment: ALL changed functions are targets (NO LLM)

Branch: `experiment/all-pr-graph-navigation-no-llm`

## Dataset policy

For each merged issue-linked GitHub PR, determine changed function/method spans from **actual +/- diff lines** on both the base and head snapshots, rather than hunk context. **Every changed function** is a weak positive label without any LLM judgement or semantic filter. The dataset includes production and test functions. Removed, renamed and newly-added functions are retained; when a function does not exist at the base commit it has `in_base_commit=false` and **cannot** receive a valid retrieval rank / graph path. It must not be labeled 'isolated'.

Source issue–PR pairs can share the same PR. Their modified functions are counted once per issue–PR sample, as separate localization queries.

## Pipeline

`collect_all_pr_functions.py` → `retrieve_with_swerankembed.py` (includes all Python functions, even test functions) → `evaluate_all_pr_graph_cases.py`.

The call graph is conservatively resolved using static Python AST. Full SweRankEmbed-Small ranking considers all indexed functions, without access to the target labels. For each issue–PR:

- **Case 1**: at least one target is in Top-10 → issue reports **0 hop**. Do not present alternative 2–5-hop paths as localization success.
- **Case 2**: no target direct hit, but at least one Top-10 candidate has a directed path to a target within 8 hops → issue reports shortest forward path.
- **Case 3**: neither of the above → localization reports **unreached**. An offline teacher may reverse-traverse callers up to 5 hops to create **oracle-only** training data. It is never counted as successful retrieval/navigation.
- Per-target status is ALWAYS preserved in addition to the issue-level case. Issue-level Case1 may find 1 of 30 targets; that is **not** full patch coverage.

## Outputs

- `out/all_pr_functions.jsonl`: every changed function for each issue–PR with source/test distinction.
- `out/swerankembed_rankings.jsonl`: full base-commit ranking and `target_missing_from_index`.
- `out/graph_navigation/issue_cases.jsonl`: strict issue-level cases and per-target outcomes.
- `out/graph_navigation/manual_target_review.csv`: per-target rank, min forward hops, in-degree, 1–5-hop ancestor, `r^*(T)`, and missing-base flag.
- `out/graph_navigation/navigation_steps.jsonl`, `seed_training.jsonl`, `stop_training.jsonl`: training supervision with oracle-only provenance preserved.
- `out/graph_navigation/summary.json`: case distribution and hop distribution, separating teacher-only paths.

## Limitations

1. These are **weak** PR-edit targets, not proven root causes. This ablation explicitly tests the consequences of retaining all functions.
2. The graph misses dynamic dispatch, callbacks and property reads; `no_incoming_edge` means only missing from the extracted graph.
3. The code model uses `trust_remote_code=True`. Pin and audit the model's remote revision for a production training pipeline.
4. A PR with zero modified Python functions is not a localization case; it is saved with an explicit exclusion reason, not fabricated as Case3.
5. GitHub Actions run: https://github.com/nthne/SweRank/actions/workflows/all-pr-graph-navigation-no-llm.yml
