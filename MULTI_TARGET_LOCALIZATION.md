# Multi-target issue localization — proposed dataset and scoring

**Research proposal:** This is our adaptation of ideas from RLCD and Graph-R1, not an algorithm stated in those papers. SWELOC-like modified functions are weak positives, not verified causal bugs.

## Persist dataset in this repository

The GitHub Actions workflow saves audit_results/latest/ and audit_results/<run_id>/, and also keeps the artifact backup. The run folders contain sample_sweloc.jsonl (function-level weak labels), issue_level.jsonl (issue-level candidate target sets), review_queue.csv, README.md, metadata.json. Snapshots must be nonempty and <= 5 MB. The workflow force-adds only audit results because the original gitignore excludes JSONL, CSV and *results* paths. GitHub Actions needs main-branch write permission.

## How to obtain reliable training labels

1. Verify issue–PR links using issue description, patch, regression tests, not only a matching closes/fixes #number. Flask #3193 / PR #6133 is the concrete false link in AUDIT_RESULTS.md.
2. Pin base_commit, patch/PR SHA, repository, issue URLs and function IDs; build graph and retrieval source from the pre-fix base commit.
3. Annotate each changed function as verified_fix_target, necessary_helper_change, test_change, incidental_refactor, wrong_issue_pr_link, or uncertain, with rationale and reviewer identity. Keep production and tests separate.
4. Where feasible require fail-before/pass-after tests or more focused patch-ablation evidence. Do not make all unmodified functions hard negatives: many are legitimate graph navigation nodes.
5. Retain difficult zero-similarity / distant multi-hop cases. Split by repository/time to avoid data leakage. Evaluate on held-out independently reviewed issues.

## Multiple targets imply a gold shortest-path SUBGRAPH, not one path

Example: Issue -> retrieved A -> B -> target T1 and A -> C -> target T2.

The union of all shortest paths from retriever seeds to ALL verified targets is a useful OFFLINE navigation proxy. It is generally a DAG (branches/merges), not necessarily a tree. It is not proven causal explanation; static code graphs may miss dynamic dependencies. The shortest path is not necessarily the only correct investigation.

Build offline with Python 3.11:

```bash
python src/collect/build_gold_subgraph.py --input my_graph.json --output gold_subgraph.json
```

Input JSON: {"instance_id":"example","seeds":["A"],"targets":["T1","T2"],"edges":[{"source":"A","target":"B","type":"calls"}]}

Output: gold_nodes, gold_edges, target_hops, unreachable_targets. Paths are compressed into one shortest-path DAG without enumerating all possible paths. Targets and distances MUST NOT be inputs to the model at inference. The script does not construct the code graph: it needs a separate base-commit graph extractor.

## Learning graph search and multiple targets

Train an issue-conditioned navigation action scorer Q(issue,state,edge), a separate target classifier P(needs_fix | issue,function,state), and optionally a STOP score. The state tracks visited functions, global unexpanded frontier, reported target candidates, budget and issue context.

RLCD-inspired pairwise training prefers actions reaching *new undiscovered* verified targets over genuinely irrelevant actions. Do not use all non-shortest paths as negatives, since valid alternatives exist. Graph-R1-inspired multi-step reward gives positive credit for each newly covered verified target, penalizes false target reports, repeated nodes and extra hops; this is an adaptation, not their exact reward.

After finding T1 on branch B, keep other queued branch C in the GLOBAL priority frontier. Do not stop or stay on the current branch. If target T2 is disconnected, retrieve/reseed more function candidates. Avoid revisiting visited nodes.

Do NOT stop merely when the next local score decreases, and do NOT exhaust the entire repository graph by default. Stop when all frontier actions have low calibrated expected marginal gain relative to expansion cost, or a hard budget is reached; the true number of targets is unknown at inference. Train/evaluate STOP against verified held-out multi-target issues and report whether stopping was threshold- or budget-based.

Evaluate all-target Recall@budget, target-set precision/recall/F1, expansions, false positives, unreachable targets and stop calibration. Also track shortest-path-subgraph coverage as a supplementary metric, never the sole goal.
