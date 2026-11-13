# Target pair-level evaluation (2026-11-13)

Deprecated: issue-level Case1/2/3 based on ANY target being retrieved. That falsely turns the entire multi-target issue into a 0-hop sample.

Every (issue-PR, target-function) is one independent sample, keyed by pair_id. A single issue can contain all three cases.

- Case1: exact target in Top-K; 0 hop.
- Case2: target not in Top-K, reachable via directed Top-K-to-target path in up to 8 hops. Report minimum hops and preserve all equally-shortest paths (up to per-target cap).
- Case3: no Top-K-to-target path within the budget. Reverse target-to-ancestor searches are oracle-only and NEVER count as retrieved paths.
- not_in_base_commit: function added by PR or missing in pre-fix source index. Keep in raw PR targets but exclude from Case1/2/3 graph-localization denominator.

Each pair produces target-specific seed/next-hop/stop supervision, with pair_id, target, pair_case, and oracle_only_teacher; do not treat alternate reachable next hops as negatives.

Outputs: target_pair_cases.jsonl; target_pair_paths.jsonl; issue_cases.jsonl (container with target_case_counts, not one case); manual_target_review.csv; summary.json; training JSONLs.

The ten-repository 2026-11-13 completed run has 43 issue-PR samples and 719 raw (issue-PR, function) targets; 466 targets were absent from base commit. Its original training JSONLs were generated with OLD issue-level logic. Rebuild training rows with this corrected code.

Available regression tests: tests/test_all_pr_graph_cases.py; GitHub Actions Validate Pair-level Graph Navigation.
