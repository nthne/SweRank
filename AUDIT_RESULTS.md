# SWELOC mini-audit — run 39588939576

- Workflow: https://github.com/nthne/SweRank/actions/runs/39588939576 (success, 3/3 unit tests passed)
- Artifact: https://github.com/nthne/SweRank/actions/runs/39588939576/artifacts/12341123232
- Revision: 8f75ecdf508bedec51bf8d33e2f3256657d5a0e3
- Mini-sample: 8 issue/PR pairs, 106 *weak* modified-function positives, 31 test-function positives, 16 zero identifier-overlap labels, 105 multi-function PR labels.

**Not the released SWELOC dataset.** This tests a SWELOC-like weak-label construction rule on public merged PRs; it has not gone through SWELOC top-K filtering. Results cannot establish which examples appear in the official training data.

## Example 1: confirmed wrong issue–PR association

- Issue: https://github.com/pallets/flask/issues/3193 — "Flask Session variable not available in threads" (closed in 2019).
- PR: https://github.com/pallets/flask/pull/6133 — "add `app.query` route decorator". Its body says `closes #3193`.
- Extracted sample: `pallets__flask-3193` → `tests/test_basic.py/test_method_route` (identifier overlap 0).
- Patch: adds an HTTP `QUERY` route decorator in `src/flask/sansio/scaffold.py` and expands the method-route test. No session/thread fix.
- **Finding:** erroneous issue–PR link and therefore an incorrect issue–function training pair. A text-based `closes` matching rule is insufficient.

## Example 2: auxiliary test change vs. faulty production function

- Issue: https://github.com/pallets/flask/issues/5916 — "`provide_automatic_options` is weird".
- PR: https://github.com/pallets/flask/pull/5917 — "fix provide_automatic_options override".
- Extracted sample: `pallets__flask-5916` → `tests/test_blueprints.py/test_default_static_max_age`.
- Main behavior fix: `src/flask/sansio/app.py/App.add_url_rule`, correcting automatic OPTIONS handling.
- The test function adjusts blueprint/test configuration for static-file cache max age; it is **not the primary defective production function**.
- **Finding:** auxiliary test change. Do not label "completely unrelated": the issue also mentions collateral test/duplicate-route cleanups.

## Dataset-quality implications

1. Independently validate the issue–PR link; an old issue can be erroneously marked as closed by a future PR.
2. Separate source-code bug targets from test maintenance and other necessary patch edits.
3. Never infer causal relevance from lexical overlap alone.
4. Label statuses should include `confirmed_relevant`, `wrong_issue_pr_link`, `auxiliary_test_change`, and `uncertain`.
5. Use raw prefiltered examples as well as filtered data to study hard multi-hop cases. Further reconstruction is required to assess *official SWELOC* false positives.
