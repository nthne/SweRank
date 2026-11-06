# Issue-conditioned Function Graph Navigation Dataset (v0.1)

This is a **new task**, not a claim that the original SWELoc patch labels are gold paths. The old `sweloc-sample-audit.yml` workflow is not changed.

## Objective

Given an issue `q`, rank **all** functions in a pre-fix checkout; retrieve a seed set `A` of `K` functions, and learn a cheap edge-selection policy that navigates the directed static code graph to one or more truly relevant fix targets. Targets are visible to the **offline teacher only**. Inference must use only issue text, source code, retrieval outputs and observable graph structure, without repeated LLM calls.

```
issue + PR base_commit + reviewed modified-function candidates
    |  benchmark repo leakage guard (SWE-bench AND LocBench)
    v
AST function graph G(base_commit)  +  full-repository BM25 or SweRankEmbed ranking
    |                                       |
    +------------------- A top K ----------+
                         |
             ┌───────────┼────────────────┐
       (1) A has target  (2) A reaches T   (3) A cannot reach T
           0 hops         shortest paths    reverse oracle inspection (<=5 hops)
             └───────────┼────────────────┘
                         v
      navigation_cases + seed_training + navigation_steps
                         v
          low-cost trained edge scorer (baseline)
```

## Ground-truth policy

* `raw_modified_targets` is **not** positive ground truth.
* `reviewer_label=verified_fix_target` creates a human-verified target (assuming a real human review; the program cannot verify the reviewer).
* `llm_relevance_label=fix_target` + `medium/high` confidence + `issue_pr_match != false` + completed judgement creates **only** a provisional `llm_pseudo` target.
* `incidental_change`, `wrong_issue_pr_link`, `uncertain`, `necessary_helper`, `test_change`, missing source and low-confidence labels are **excluded from fix targets**. They remain in `rejected_targets` for audit. A necessary helper may be on a navigation path, but is not automatically a fix target.
* If no eligible targets survive, write the sample to `rejected_cases.jsonl` rather than fabricate labels.
* All gold paths are **static shortest-path proxies**, not proven causality. Missing static connections mean "unreachable in extracted graph", NOT "no runtime/dependency relationship exists".

## Three cases and their meaning

1. `direct_retrieval`: a validated target is already in top-K. Its path is length zero; do not navigate unnecessarily.
2. `retrieval_to_target_path`: a target is not directly retrieved but is forward-reachable from at least one seed through static calls, within `max_hops`. The issue-relevant seed is chosen first, then all tied shortest paths from that seed are recorded (max 32 per target). For multiple targets, retain each branch and union them.
3. `reverse_oracle_probe`: no target is reachable from top-K within the hop budget. Starting from verified/pseudo targets, reverse-search **callers**, up to 5 hops, and inspect full-repository ranking positions. Choose the highest-ranked matching ancestor as a *teacher-only recovery seed*. Record the original retrieval failure; this rescue is **not allowed at inference**. If no bridge is available, retain `unreached_targets` and an empty path without inventing one.

**Hop definition:** a directed edge between two function/method nodes in the pre-fix code graph. The issue is not a code node, and issue→seed retrieval is not counted as a hop.

## Graph extraction and fidelity

`navigation_graph.py` parses Python AST without importing, executing or testing the sampled repository. Each function/method is a node identified as `path/to/file.py/Class/method`, consistent with existing audit IDs. It creates only resolvable function-to-function call edges (`lexical_call`, `method_call`, `imported_call`, `module_call`) to avoid broad ambiguous edges. Module imports alone do **not** imply a function-call edge. Dynamic dispatch, polymorphism, decorators, re-exports, reflection, frameworks and language boundaries are incomplete. Graph coverage, parse failures, and unresolved calls are recorded. To support more edge types, add explicit confidence tiers and validate precision rather than converting every import into a false directed call.

## Labeling and scoring

Offline seed teacher for a retrieved function `v`:

`score*(v,q,T) = 0` if no accepted target is reachable in the extracted graph; otherwise

`score*(v,q,T) = 0.7 / (1 + d_G(v,T)) + 0.3 * normalized_relevance(q,v)`.

`d_G(v,T)` is shortest path length to any accepted target, computed with reverse BFS **only for training labels**. This weight is a **heuristic hyperparameter** that must be ablated, not a discovered optimal formula. The distance component increases toward target, but the total score need not be strictly monotonic because issue relevance varies. Off-path but reachable nodes are `reachable_alternative` or `label=null` (ambiguous), not arbitrary false negatives. High-similarity unreachable nodes are graph-proxy hard negatives. One cannot infer that a static path proves issue relevance.

`navigation_steps.jsonl` contains supervised neighbor actions with `features` and `label` (1 for chosen shortest-path edge; 0 for graph-proxy unreachable dead end; `null` for a possibly valid alternative). `teacher_distance`, target labels, and gold paths are explicit **teacher fields**, NEVER consumed by `train_navigation_scorer.py` as features.

The trained `edge_scorer.json` provides edge and (when labels are sufficient) fix-target marking heads as reproducible logistic-regression **CPU baselines**, not an RLCD implementation. `infer_navigation.py` performs budgeted best-first traversal with these heads and no oracle or LLM. It scores visible candidate features without an LLM at inference. Production extensions should train a graph-conditioned encoder on issue and code embeddings, refine multi-target completion/stopping (distinct from local target marking), and perform imitation learning followed by RLCD-style preference/trajectory optimization. Do not claim RLCD or navigation success until the corresponding training and rollout evaluation are implemented.

## Benchmark isolation

`build_benchmark_manifest.py` loads the official test partitions of `princeton-nlp/SWE-bench` and `czlll/Loc-Bench_V1`, records all repository names and counts, then excludes **entire repositories** (stricter than overlapping issue IDs). The graph builder refuses to run without both complete sources, with at least 100 samples each. This prevents training on Flask and other repositories appearing in either benchmark. The isolation snapshot SHA256 is attached to every case for audit. If either source cannot be loaded, **fail closed**; do not silently weaken the rule.

Other benchmark families, forks of evaluation repos, copied functions, code-history leaks and pretrained model contamination are **not covered**. Before publishable comparisons, freeze a benchmark snapshot and add repository/fork aliases, issue/PR IDs and near-duplicate hashing to leakage checks.

## Run

In GitHub Actions, select **Graph Navigation Dataset → Run workflow**, input a public repository absent from benchmark manifest. The job validates exclusions **before crawling**. It then crawls linked merged PRs, enriches with SweRankEmbed + one-time LLM auditing (requires repository secret `OPENAI_API_KEY`), checks out each PR base commit and creates the issue-conditioned graph dataset. The LLM is used only offline. New samples are saved at `navigation_results/YYYY-MM-DD/owner__repo`; graph snapshots and audit metadata also appear in the Actions artifact.

To run locally on already reviewed function-level data:

```bash
python src/collect/build_benchmark_manifest.py --output benchmark_manifest.json
python src/collect/build_navigation_dataset.py \
  --input audit_outputs/enriched_sweloc.jsonl \
  --output-dir navigation_outputs --benchmark-manifest benchmark_manifest.json \
  --retriever swerank --top-k 20 --max-hops 8 --reverse-hops 5 --store-graphs
python src/collect/train_navigation_scorer.py \
  --input navigation_outputs/navigation_steps.jsonl \
  --stop-input navigation_outputs/stop_training.jsonl \
  --output navigation_outputs/edge_scorer.json
python -m unittest discover -s tests -p test_navigation_dataset.py -v
```

`--retriever bm25` runs on the Python standard library and is for reproducible smoke tests; `swerank` requires `sentence-transformers`, `torch` and the public `Salesforce/SweRankEmbed-Small` model. The pretrained model's download/remote code must be trusted and pinned for scientific reproduction.

Outputs:

| File | Contents |
|---|---|
| `navigation_cases.jsonl` | Issue, case 1/2/3, top-K, target decisions, gold proxy branches, seed scoring and reverse diagnostic ranks |
| `navigation_steps.jsonl` | Issue-conditioned candidate neighbor transitions, observable features, positive/negative/ambiguous teacher labels |
| `seed_training.jsonl` | Seed selection ranking supervision |
| `stop_training.jsonl` | Observed-only features and labels for marking a fix target while continuing multi-target search |
| `rejected_cases.jsonl` | Issues without eligible fix targets or graph build errors |
| `summary.json` | Coverage, counts, leakage-manifest SHA and limitations |
| `graphs/*.json` (optional) | AST nodes, code and directed call edges at base commit |
| `edge_scorer.json` (optional) | Learned baseline weights and training metrics |

## Evaluation protocol (not yet claimed)

Split by repository (and verify no PR/issue duplicate) rather than by function rows. Compare BM25, SweRankEmbed, an issue-conditioned reranker, shortest-path heuristics and the learned scorer; ablate relevance-only vs distance-only vs combined supervision, graph direction, top-K and search budget. Evaluate Target Recall@K, Path Recall / Edge F1, Navigation Success@B, first error depth, recall of **all** multi-target fixes, oracle-rescue frequency, graph coverage, hops, latency and number of model/LLM calls. Report results separately for case 1/2/3 and quality of target labels (human vs provisional). Do not compute "path ground-truth" accuracy against the same heuristic that built the paths without clearly labeling it proxy fidelity.