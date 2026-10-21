import json
import os
import fire
from get_tasks_pipeline import main as get_tasks_pipeline
from tqdm import tqdm

def main(repo_info_path, prs_path, tasks_path, git_token,
         start_idx=0, num_repos_to_process=-1):
    if not git_token:
        raise ValueError("Pass --git_token or set GITHUB_TOKEN")
    os.environ["GITHUB_TOKENS"] = git_token
    with open(repo_info_path, encoding="utf-8") as f:
        repos = json.load(f)
    selected = repos[start_idx:] if num_repos_to_process < 0 else repos[start_idx:start_idx + num_repos_to_process]
    os.makedirs(prs_path, exist_ok=True)
    os.makedirs(tasks_path, exist_ok=True)
    print(f"Collecting {len(selected)} of {len(repos)} repos")
    for repo in tqdm(selected):
        get_tasks_pipeline(repos=[repo], path_prs=prs_path, path_tasks=tasks_path)

if __name__ == "__main__":
    fire.Fire(main)
