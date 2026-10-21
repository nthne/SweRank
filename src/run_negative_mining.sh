#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
RAW="collect/datasets/SWE_PRS_FT_DATASET.jsonl"
if [[ ! -f "$RAW" ]]; then
  echo "Missing $RAW; run src/collect/collect_data.sh first" >&2
  exit 1
fi
python get_train_by_repo.py --data_path "$RAW" --save_file_name repo_contrastive_premined.jsonl
python repo_negative_mining.py --data_path repo_contrastive_premined.jsonl \
 --num_workers_per_gpu 1 --save_file_name repo_contrastive_mined.jsonl
python collect/filter_dataset.py --file_path "$SCRIPT_DIR"
echo "Preserved raw/premined/mined and filtered records in $SCRIPT_DIR"
