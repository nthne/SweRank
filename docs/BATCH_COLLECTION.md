# Benchmark-disjoint, multi-repository Graph Navigation batch

The workflow `.github/workflows/navigation-batch.yml` creates a dataset from nine
Python repos outside both SWE-bench and LocBench benchmark repository sets.
The fresh, official benchmark repository manifest is built first and
**each** repository is checked before any data is sampled. Any overlap is a
failure, not a warning.

## Batch settings

- Repositories: `simonw/datasette`, `Textualize/rich`, `pypa/packaging`,
  `pallets/itsdangerous`, `django-commons/django-debug-toolbar`,
  `encode/httpx`, `pallets/werkzeug`, `pypa/build`, `fastapi/fastapi`.
- Per repository: up to 12 recent merged issue-linked PRs (not 12 guaranteed
  valid issues), Top-10 retrieval, max 8 forward hops, max 5 reverse hops.
- Offline target audit: capped at 24 LLM calls per repo, 8 functions per batch.
  The existing GitHub Actions `OPENAI_API_KEY` secret is reused; no key is
  stored in git. No LLM is required for inference.
- Initial large-scale retrieval: BM25 over all function nodes at the PR's
  **base commit**. This is an intentional cheap first-pass baseline; these
  batch scores **are not SweRankEmbed scores**. Run SweRankEmbed on the most
  useful cases in a later, separate evaluation.
- Base-commit code graph: AST-resolved static call edges. Multiple PR base
  commits for one repo reuse a shared, blobless bare repository clone with
  pinned worktrees rather than re-cloning full history repeatedly.
- Results of every repository are stored in **separate GitHub Actions artifacts**.
  If a job fails, the other jobs are retained; investigate partial/missing
  repositories before claiming complete collection.
- The aggregation job collects every completed repository's issues, positive
  and rejected targets, navigation examples, deep-hop oracle diagnostics, and
  review CSVs. It writes combined JSONL + CSV and a per-repo case/hop summary
  under `navigation_results/YYYY-MM-DD/multi_repo_batch`.

**Gold policy:** PR-modified functions are weak labels. Fix-target candidates
with LLM medium/high confidence are *pseudo-labels*, not human-verified fix
locations. Paths are *static shortest-path proxies*. Neither teacher-only
reverse recovery nor a diagnostic 2–5-hop path demonstrates that the model
navigated to the target. Record this distinction in any paper/report.

## Notes on data volume

A repository may have no eligible linked PRs in the recent crawl window,
no valid Python code changes, or no LLM-accepted targets. Thus 9 repos ×
12 PRs is **a collection budget**, not a promised sample count. The batch
summary includes rejected cases and each repo's eligible count. The
artifact preserves raw/enriched audit records for later human verification.

## Review outputs

- `batch_summary.json`: source repo counts, case distribution, hop histograms,
  accepted/rejected case totals, budgeted API call usage.
- `combined_navigation_cases.jsonl`: accepted cases with target labels and
  selected/diagnostic paths.
- `combined_rejected_cases.jsonl`: no acceptable target or parse/checkout
  errors, for bias and dataset-quality audits.
- `combined_navigation_steps.jsonl`, `combined_seed_training.jsonl`,
  `combined_stop_training.jsonl`: candidate model training samples.
- `combined_manual_path_review.csv`: review queue joined from per-repo exports.
- `BATCH_REPORT.md`: readable per-repo scoreboard.

The full per-repo graph snapshots and raw PR/LLM metadata are available in
workflow artifacts, not silently discarded if the combination fails.