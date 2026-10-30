# SWELOC enrichment flow

This is a SWELOC-like **mini-audit**, not the official training dataset. All PR-modified functions remain weak labels.

## GitHub Actions

Open **Actions → SWELOC sample audit**, run against a public repo; a push to main also triggers a 4-PR test. Output is committed under `audit_results/YYYY-MM-DD/owner__repo/` (Bangkok timezone). Re-running the same repository on the same date updates that folder. No run-ID directories.

Files include original `sample_sweloc.jsonl`, `issue_level.jsonl`, enrichment `enriched_sweloc.jsonl`, `issue_level_enriched.jsonl`, `review_queue_enriched.csv`, `judge_reviews.jsonl`, `enrichment_metadata.json` and `metadata.json`.

## Cosine score

The script `src/collect/enrich_sweloc.py` retrieves the **base-commit version** of each positive function via GitHub REST, extracts source with Python AST and embeds the issue text and code separately using `Salesforce/SweRankEmbed-Small`. It computes cosine from returned vectors, stores `semantic_similarity`, `semantic_model`, `semantic_status` and source resolution status. It never copies lexical `semantic_overlap` as a semantic score. Embedded vectors are not ground-truth correctness.

The GitHub runner installs the embedding model for CPU; the first run downloads model weights. If model download or loading fails, the job **fails before committing** rather than claiming it computed similarity. Overlong functions and unresolved symbols are marked, not silently fabricated.

## LLM relevance judgment

Set repository **Settings → Secrets and variables → Actions → New repository secret**, name `OPENAI_API_KEY`. When set, the workflow sends groups of up to 8 functions from the same issue/PR to the Responses API using a reasoning model (`gpt-6.1-sol` by default), with issue description, base source, code diff and test patches. Default limit is 16 pairs to cap spend. Label choices: `fix_target`, `necessary_helper`, `test_change`, `incidental_change`, `wrong_issue_pr_link`, `uncertain`. Judge evidence and confidence are recorded. The LLM makes **provisional recommendations only**, never automatically replaces verified labels.

Without the secret, LLM scoring is skipped and `llm_judge_status=not_requested`; this is NOT an LLM validation. API failures are recorded as `judge_error`. Note: the issue/patch content is sent to an external provider when LLM review is enabled. Use only approved public code.

Run locally:
```bash
python src/collect/sample_sweloc.py --repo pallets/flask --max-prs 4 --output-dir audit_outputs
python src/collect/enrich_sweloc.py --input audit_outputs/sample_sweloc.jsonl --output-dir audit_outputs --embedding --judge
python src/collect/group_issues.py --input audit_outputs/enriched_sweloc.jsonl --output audit_outputs/issue_level_enriched.jsonl
```

## Known limitations

- A modified function is not necessarily a required fix.
- The first function's base-commit source may not exist (added/moved symbol) and is left unresolved.
- Code graph and verified multi-target shortest-path supervision are separate steps.
- Do not use LLM confidence as calibrated probability.
