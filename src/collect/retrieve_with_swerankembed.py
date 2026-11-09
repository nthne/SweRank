#!/usr/bin/env python3
"""REAL SweRankEmbed-Small retrieval on *pre-fix* repository functions.

No target labels are passed into retrieval. Every Python function (including test functions)
available at base_commit is indexed (no oracle candidate reduction). The
retrieved rankings can be evaluated separately with evaluate_all_pr_targets.py.
"""
import argparse
import ast
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

MODEL_NAME="Salesforce/SweRankEmbed-Small"

def git(*args,cwd=None):
    return subprocess.check_output(["git",*args],cwd=cwd,text=True,stderr=subprocess.PIPE).strip()

def source_functions(workdir):
    files=git("ls-files","--","*.py",cwd=workdir).splitlines()
    corpus={}
    skipped=[]
    for path in files:
        # All actual modified functions are targets, including test functions.
        # Index the same pre-fix source candidates as the call graph.
        if any(part in {"__pycache__","venv",".venv","site-packages","node_modules","build","dist"} for part in Path(path).parts):
            continue
        try:
            code=(Path(workdir)/path).read_text(encoding="utf-8")
            tree=ast.parse(code)
            lines=code.splitlines(keepends=True)
            def walk(nodes,parents=()):
                for node in nodes:
                    if isinstance(node,ast.ClassDef):
                        walk(node.body,parents+(node.name,))
                    elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        name=path+"/"+"/".join(parents+(node.name,))
                        start=min([node.lineno]+[d.lineno for d in node.decorator_list])-1
                        corpus[name]=(name+"\n"+"".join(lines[start:node.end_lineno]))[:10000]
                        walk(node.body,parents+(node.name,))
            walk(tree.body)
        except (SyntaxError,UnicodeError,ValueError,OSError) as exc:
            skipped.append({"file":path,"error":str(exc)[:200]})
    return corpus,skipped

def fetch_repositories(rows,root):
    repositories=sorted(set(row["repo"] for row in rows))
    directories={}
    for repo in repositories:
        path=root/repo.replace("/","__")
        git("clone","-q","--filter=blob:none",f"https://github.com/{repo}.git",str(path))
        directories[repo]=path
    return directories

def retrieve(rows,clone_root,model_name=MODEL_NAME,model=None):
    if model is None:
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(model_name,trust_remote_code=True,device="cpu")
        model.max_seq_length=512
    repositories=fetch_repositories(rows,clone_root)
    groups=defaultdict(list)
    for row in rows:
        if "collection_error" not in row:
            groups[(row["repo"],row["base_commit"])].append(row)
    rankings=[]
    metadata=[]
    for (repo,sha),items in groups.items():
        workdir=repositories[repo]
        git("checkout","-q","--force",sha,cwd=workdir)
        corpus,skipped=source_functions(workdir)
        if not corpus:
            raise RuntimeError(f"Empty code index for {repo} @ {sha}")
        names=list(sorted(corpus))
        # Embedding is computed from the entire pre-fix source index, not target lists.
        docs=model.encode([corpus[name] for name in names],normalize_embeddings=True,
                          batch_size=16,show_progress_bar=False)
        queries=model.encode([row["query"][:3000] for row in items],prompt_name="query",
                             normalize_embeddings=True,batch_size=8,show_progress_bar=False)
        import numpy as np
        scores=np.asarray(queries)@np.asarray(docs).T
        for i,row in enumerate(items):
            order=np.argsort(-scores[i],kind="stable")
            ranking=[names[j] for j in order]
            targets=set(row["all_modified_functions"])
            rankings.append({"instance_id":row["instance_id"],"repo":repo,"base_commit":sha,
                             "model":model_name,"candidate_count":len(names),
                             "ranked_functions":ranking,"target_found_in_index":sorted(targets&set(names)),
                             "target_missing_from_index":sorted(targets-set(names))})
        metadata.append({"repo":repo,"base_commit":sha,"issue_count":len(items),
                         "candidate_count":len(names),"source_parse_errors":skipped})
        print(f"[RETRIEVED] {repo}@{sha[:9]} issues={len(items)} candidates={len(names)} "
              f"parse_errors={len(skipped)}",flush=True)
    return rankings,metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--metadata",required=True)
    p.add_argument("--model",default=MODEL_NAME)
    args=p.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    with tempfile.TemporaryDirectory() as tmp:
        rankings,metadata=retrieve(rows,Path(tmp),args.model)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rankings),encoding="utf-8")
    Path(args.metadata).write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("Saved",len(rankings),"real SweRankEmbed rankings",flush=True)
if __name__=="__main__":main()
