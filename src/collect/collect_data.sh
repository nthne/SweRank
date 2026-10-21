#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
: "$GITHUB_TOKEN"
MAX_NUM_REPOS="$(printenv MAX_NUM_REPOS || echo 10)"
mkdir -p datasets prs tasks
python get_top_pypi.py --max-repos "$MAX_NUM_REPOS"
python get_valid_urls.py --repo_info_path pypi_rankings.jsonl --out_path valid_top_pypi_gitrepos.json
python generate_tasks.py --repo_info_path valid_top_pypi_gitrepos.json \
 --prs_path prs --tasks_path tasks --git_token "$GITHUB_TOKEN" \
 --start_idx 0 --num_repos_to_process "$MAX_NUM_REPOS"
python build_dataset_ft.py --instances_path tasks --output_dir datasets
echo "Raw dataset saved to $SCRIPT_DIR/datasets/SWE_PRS_FT_DATASET.jsonl"
