import json
import fire
from tqdm import tqdm

def main(repo_info_path="pypi_rankings.jsonl", out_path="valid_top_pypi_gitrepos.json"):
    with open(repo_info_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    repos = set()
    for record in tqdm(records):
        url = record.get("github") or ""
        if "github.com/" not in url:
            continue
        chunks = url.split("github.com/", 1)[1].strip("/").split("/")
        if len(chunks) >= 2:
            repos.add(f"{chunks[0]}/{chunks[1].removesuffix('.git')}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted(repos), f, indent=2)
    print(f"Saved {len(repos)} repos: {out_path}")

if __name__ == "__main__":
    fire.Fire(main)
