# Frozen 47-case BM25 versus SweRankEmbed

Same 47 issue descriptions, pinned code graph snapshots and 164 accepted pseudo targets.
No LLM calls; retrieval success excludes teacher-only reverse probes.

| Metric | BM25 | SweRankEmbed |
|---|---:|---:|
| issue_recall_at_k@1 | 0.362 | 0.489 |
| target_recall_at_k@1 | 0.104 | 0.140 |
| issue_path_coverage_at_k@1 | 0.426 | 0.553 |
| target_path_coverage_at_k@1 | 0.128 | 0.171 |
| all_target_issue_path_coverage_at_k@1 | 0.319 | 0.426 |
| nontrivial_target_path_coverage_at_k@1 | 0.024 | 0.030 |
| issue_recall_at_k@5 | 0.617 | 0.702 |
| target_recall_at_k@5 | 0.238 | 0.293 |
| issue_path_coverage_at_k@5 | 0.660 | 0.766 |
| target_path_coverage_at_k@5 | 0.323 | 0.360 |
| all_target_issue_path_coverage_at_k@5 | 0.468 | 0.681 |
| nontrivial_target_path_coverage_at_k@5 | 0.116 | 0.122 |
| issue_recall_at_k@10 | 0.702 | 0.766 |
| target_recall_at_k@10 | 0.348 | 0.372 |
| issue_path_coverage_at_k@10 | 0.745 | 0.830 |
| target_path_coverage_at_k@10 | 0.488 | 0.451 |
| all_target_issue_path_coverage_at_k@10 | 0.617 | 0.745 |
| nontrivial_target_path_coverage_at_k@10 | 0.213 | 0.201 |
| issue_recall_at_k@20 | 0.809 | 0.872 |
| target_recall_at_k@20 | 0.524 | 0.482 |
| issue_path_coverage_at_k@20 | 0.809 | 0.872 |
| target_path_coverage_at_k@20 | 0.671 | 0.537 |
| all_target_issue_path_coverage_at_k@20 | 0.702 | 0.809 |
| nontrivial_target_path_coverage_at_k@20 | 0.274 | 0.220 |
| issue_recall_at_k@50 | 0.809 | 0.915 |
| target_recall_at_k@50 | 0.780 | 0.701 |
| issue_path_coverage_at_k@50 | 0.830 | 0.936 |
| target_path_coverage_at_k@50 | 0.878 | 0.787 |
| all_target_issue_path_coverage_at_k@50 | 0.766 | 0.851 |
| nontrivial_target_path_coverage_at_k@50 | 0.366 | 0.335 |

## Cases

| Case | BM25 | SweRankEmbed |
|---|---:|---:|
| direct_retrieval | 33 | 36 |
| retrieval_to_target_path | 2 | 3 |
| reverse_oracle_probe | 12 | 8 |

## Min hops from Top-10 to targets

| Hops | BM25 | SweRankEmbed |
|---|---:|---:|
| 0 | 57 | 61 |
| 1 | 13 | 9 |
| 2 | 7 | 3 |
| 3 | 2 | 1 |
| 4 | 1 | 0 |
| unreachable | 84 | 90 |
