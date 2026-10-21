"""Consistency-filter mined SWELOC records; keep the unfiltered data for audit."""
import os
import fire
import ujson
import numpy as np
from tqdm import tqdm

NEGATIVE_INDICES = range(3, 26)

def filter_dataset(rows, max_positive_rank=19):
    # Zero-based ranks 0..19 = paper's top-20. Not a causal-label verifier!
    kept = []
    for row in tqdm(rows):
        rank = row.get("positive_code_rank", 99999)
        if not 0 <= rank <= max_positive_rank:
            continue
        ids = [i for i, r in enumerate(row.get("negative_code_rank", [])) if r in NEGATIVE_INDICES]
        for key, value in list(row.items()):
            if key.startswith("negative_") and isinstance(value, list):
                row[key] = np.array(value, dtype=object)[ids].tolist()
        kept.append(row)
    return kept

def main(file_path=".", max_positive_rank=19, file_prefix="repo_contrastive_mined"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    if os.path.isfile(file_path):
        paths = [file_path]
        folder = os.path.dirname(os.path.abspath(file_path))
    else:
        folder = file_path
        paths = [os.path.join(folder, name) for name in sorted(os.listdir(folder))
                 if name.startswith(file_prefix) and name.endswith(".jsonl")
                 and "filtered" not in name and "premined" not in name]
    if not paths:
        raise ValueError("No mined JSONL files found")
    result = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            result.extend(filter_dataset([ujson.loads(line) for line in f if line.strip()], max_positive_rank))
    dest = os.path.join(folder, "repo_contrastive_mined_filtered.jsonl")
    with open(dest, "w", encoding="utf-8") as f:
        for row in result:
            f.write(ujson.dumps(row) + "\n")
    print(f"Saved {len(result)} examples to {dest}")

if __name__ == "__main__":
    fire.Fire(main)
