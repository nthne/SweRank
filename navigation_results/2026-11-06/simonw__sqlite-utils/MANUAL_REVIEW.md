# SWELOC Graph Navigation — real repository audit

**Repository:** `simonw/sqlite-utils`  
**Run:** https://github.com/nthne/SweRank/actions/runs/42121342738  
**Review date:** 2026-11-06 (Asia/Bangkok)  
**Retriever:** SweRankEmbed-Small · Top-K=10 · forward<=8 hops · reverse<=5 hops  
**All paths are static shortest-path proxies, not confirmed causal ground truth.**

## Case distribution

| Case | Count | Rate among accepted |
|---|---:|---:|
| Case 1: direct retrieval | 2 | 66.67% |
| Case 2: forward reachable | 0 | 0.0% |
| Case 3: reverse diagnostic | 1 | 33.33% |

Accepted issues: 3. Rejected issues: 2.
LLM labels: 51/51; LLM API calls: 8; input tokens: 14199; output tokens: 2139.

## Hop distribution

Proxy path hops: `{'0': 2, '1': 1}`; unreachable targets: 1.

## Manual review paths

### 1. simonw__sqlite-utils-705 — static_shortest_path_proxy

- **Issue:** https://github.com/simonw/sqlite-utils/issues/705
- **PR:** https://github.com/simonw/sqlite-utils/pull/776
- **Base SHA:** `02281f77ed7bfb56ff74a2f09a00aa03298e3268`
- **Case:** `direct_retrieval`; **target source:** `llm_pseudo`
- **Target:** `sqlite_utils/db.py/Database/ensure_autocommit_off`
- **Seed rank:** 1; **hops:** 0
- **Path:** `sqlite_utils/db.py/Database/ensure_autocommit_off`
- **Edge kinds:** `none`
- **Seed code:** https://github.com/simonw/sqlite-utils/blob/02281f77ed7bfb56ff74a2f09a00aa03298e3268/sqlite_utils/db.py#L613
- **Target code:** https://github.com/simonw/sqlite-utils/blob/02281f77ed7bfb56ff74a2f09a00aa03298e3268/sqlite_utils/db.py#L613
- **LLM evidence:** Its misleading name and docstring are the reported defect, corrected by renaming and clarification.
- [ ] Verify issue–PR correspondence
- [ ] Confirm target actually needs changing
- [ ] Confirm each directed edge represents a valid call relation
- [ ] Confirm path reflects meaningful issue context (not just a static shortcut)
- [ ] Mark accepted / rejected / uncertain and add evidence

### 2. simonw__sqlite-utils-594 — static_shortest_path_proxy

- **Issue:** https://github.com/simonw/sqlite-utils/issues/594
- **PR:** https://github.com/simonw/sqlite-utils/pull/770
- **Base SHA:** `623331b3f4ee78e1ed2618ccfc2e18e0bd77cd36`
- **Case:** `direct_retrieval`; **target source:** `llm_pseudo`
- **Target:** `sqlite_utils/db.py/Table/foreign_keys`
- **Seed rank:** 2; **hops:** 0
- **Path:** `sqlite_utils/db.py/Table/foreign_keys`
- **Edge kinds:** `none`
- **Seed code:** https://github.com/simonw/sqlite-utils/blob/623331b3f4ee78e1ed2618ccfc2e18e0bd77cd36/sqlite_utils/db.py#L1909
- **Target code:** https://github.com/simonw/sqlite-utils/blob/623331b3f4ee78e1ed2618ccfc2e18e0bd77cd36/sqlite_utils/db.py#L1909
- **LLM evidence:** Reads PRAGMA foreign_key_list and produces the output incorrectly representing compound constraints as separate keys.
- [ ] Verify issue–PR correspondence
- [ ] Confirm target actually needs changing
- [ ] Confirm each directed edge represents a valid call relation
- [ ] Confirm path reflects meaningful issue context (not just a static shortcut)
- [ ] Mark accepted / rejected / uncertain and add evidence

### 3. simonw__sqlite-utils-769 — static_shortest_path_proxy

- **Issue:** https://github.com/simonw/sqlite-utils/issues/769
- **PR:** https://github.com/simonw/sqlite-utils/pull/779
- **Base SHA:** `adc10df98102c76c86c77108615792ba238a0ae3`
- **Case:** `reverse_oracle_probe`; **target source:** `llm_pseudo`
- **Target:** `sqlite_utils/cli.py/insert_upsert_implementation`
- **Seed rank:** 50; **hops:** 1
- **Path:** `sqlite_utils/cli.py/bulk -> sqlite_utils/cli.py/insert_upsert_implementation`
- **Edge kinds:** `lexical_call`
- **Seed code:** https://github.com/simonw/sqlite-utils/blob/adc10df98102c76c86c77108615792ba238a0ae3/sqlite_utils/cli.py#L1449
- **Target code:** https://github.com/simonw/sqlite-utils/blob/adc10df98102c76c86c77108615792ba238a0ae3/sqlite_utils/cli.py#L990
- **LLM evidence:** Diff fixes table existence checking so view errors reach intended error handling.
- [ ] Verify issue–PR correspondence
- [ ] Confirm target actually needs changing
- [ ] Confirm each directed edge represents a valid call relation
- [ ] Confirm path reflects meaningful issue context (not just a static shortcut)
- [ ] Mark accepted / rejected / uncertain and add evidence

### 4. simonw__sqlite-utils-769 — no_static_path

- **Issue:** https://github.com/simonw/sqlite-utils/issues/769
- **PR:** https://github.com/simonw/sqlite-utils/pull/779
- **Base SHA:** `adc10df98102c76c86c77108615792ba238a0ae3`
- **Case:** `reverse_oracle_probe`; **target source:** `llm_pseudo`
- **Target:** `sqlite_utils/migrations.py/Migrations/apply`
- **Seed rank:** not retrieved; **hops:** unreached
- **Path:** `NO STATIC PATH`
- **Edge kinds:** `none`
- **Seed code:** 
- **Target code:** https://github.com/simonw/sqlite-utils/blob/adc10df98102c76c86c77108615792ba238a0ae3/sqlite_utils/migrations.py#L90
- **LLM evidence:** Apply changes reject already-applied stop_before migrations before applying pending migrations.
- [ ] Verify issue–PR correspondence
- [ ] Confirm target actually needs changing
- [ ] Confirm each directed edge represents a valid call relation
- [ ] Confirm path reflects meaningful issue context (not just a static shortcut)
- [ ] Mark accepted / rejected / uncertain and add evidence

## Methodology limitations

- Ground truth targets are `llm_pseudo`, not `human_verified`.
- Case 3 rescue uses target oracle ONLY during offline dataset construction; it is not a legitimate inference-time path from Top-K.
- A missing edge may indicate parser limitations, dynamic dispatch, or a genuinely disconnected function.
- Percentages refer to accepted navigation cases only; reporting 2 rejected cases avoids survivorship bias.
- The GitHub run failed in report export because one target had an empty path, after successfully producing all graph and case files; CSV and statistics were regenerated locally from the original artifact using fixed exporter.
