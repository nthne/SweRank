# SWELOC weak-label audit (experimental fork)

**Purpose:** Check the assumption that all functions modified in a merged PR are true issue ground truths. This repo adds a small CPU-only sample collector; it is **not** the published SWELOC corpus. Every extracted patched function has status `weak_label_unverified` until inspected by a human.

## Fast experiment (standard Python 3.11, no GPU)

```bash
export GITHUB_TOKEN=YOUR_TOKEN   # optional for public data, recommended
python -m unittest discover -s tests -v
python src/collect/sample_sweloc.py --pr pallets/flask#6096 --pr psf/requests#7315 --output-dir audit_outputs
```

Or collect recent merged PRs with linked same-repo issues and test changes:

```bash
python src/collect/sample_sweloc.py --repo pallets/flask --max-prs 12
```

Review `audit_outputs/review_queue.csv`, using linked issue, PR patch and regression tests. Prioritize zero-identifier-overlap or multi-function PRs; **neither flag proves label error**. Record `confirmed_relevant`, `confirmed_incidental` or `uncertain` only with evidence. Missing/truncated patches or add-only functions may not be mapped. A function with no lexical overlap may still be a genuine multi-hop target.

## Rebuild original SWELOC pipeline (heavier, GPU/Ray needed for mining)

```bash
export GITHUB_TOKEN=YOUR_TOKEN
MAX_NUM_REPOS=10 bash src/collect/collect_data.sh
bash src/run_negative_mining.sh
(cd src && bash get_reranker_data.sh)
```

Keep the raw and premined outputs to inspect false positives and hard multi-hop cases removed by semantic filtering. Fixed upstream argument mismatches, incorrect mining script continuation, and directory/file matching in filter_dataset. Paper top-20 is zero-based ranks 0–19. The full original PyPI scraper also needs Selenium and Chrome, dependency setup, repository cloning, and GPU for retrieval mining.

## GitHub Actions

Use **Actions > SWELOC sample audit > Run workflow**, then download artifact `sweloc-mini-audit`. The workflow runs tests and samples, not GPU model training.
