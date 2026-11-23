#!/usr/bin/env python3
"""Audit old 'no_incoming_edge' Case3 on exact buggy commits, without inventing graph edges.

No model needed: comparison is AST coverage for 15 original ET1 graph failures.
Every potential caller is recorded with evidence, not assumed to be an actual edge.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import tempfile

from navigation_graph import collect_repo
from et1_single_root_paths import is_source

def write_jsonl(path,rows):
    Path(path).write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf8")

def checkout(repo,sha,root):
    dest=root/repo.replace("/","__")
    subprocess.run(["git","clone","-q","--filter=blob:none","--no-checkout",
                    "https://github.com/"+repo+".git",str(dest)],check=True,timeout=540)
    p=subprocess.run(["git","-C",str(dest),"checkout","-q","--force",sha],capture_output=True,timeout=300)
    if p.returncode:
        subprocess.run(["git","-C",str(dest),"fetch","-q","origin",sha],check=True,timeout=540)
        subprocess.run(["git","-C",str(dest),"checkout","-q","--force",sha],check=True,timeout=300)
    return dest

def caller_for_line(path,lineno,symbols):
    spans=[(s.end_line-s.start_line,k) for k,s in symbols.items()
           if s.path==path and s.start_line<=lineno<=s.end_line]
    return min(spans)[1] if spans else "(module-level) "+path

def get_target_decorators(code,target_id):
    try: tree=ast.parse(code)
    except (SyntaxError,ValueError):return [],[]
    qual=target_id.split("/")
    out=[];same_name=[]
    def visit(stmts,parents=()):
        for node in stmts:
            if isinstance(node,ast.ClassDef):
                visit(node.body,parents+(node.name,))
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                name="/".join(parents+(node.name,))
                if name=="/".join(qual):
                    out.extend(ast.unparse(x) for x in node.decorator_list)
                if node.name==qual[-1]:
                    same_name.append(name)
                visit(node.body,parents+(node.name,))
    visit(tree.body)
    return out,same_name

def audit_case(row,root):
    repo,sha,target=row["repo"],row["base_commit"],row["target"]
    path=target[:target.find(".py/")+3]
    relative=target[len(path)+1:]
    name=target.rsplit("/",1)[-1]
    checkout_path=checkout(repo,sha,root)
    symbols,graph,graph_stats=collect_repo(checkout_path)
    symbols={k:v for k,v in symbols.items() if is_source(v.path)}
    graph=[x for x in graph if x["source"] in symbols and x["target"] in symbols]
    original_incoming=[e for e in graph if e["target"]==target]
    source_file=checkout_path/path
    decorators,same_name=get_target_decorators(source_file.read_text(encoding="utf8"),relative)
    counts=Counter();hits=[]
    parsed_files=0;failed_files=0
    for f in checkout_path.rglob("*.py"):
        relative_path=f.relative_to(checkout_path).as_posix()
        if not is_source(relative_path) or any(part in {".venv","venv","site-packages","build","dist",".git"}
                                               for part in f.relative_to(checkout_path).parts):continue
        try:
            code=f.read_text(encoding="utf8")
            tree=ast.parse(code)
        except (UnicodeError,ValueError,SyntaxError,OSError):
            failed_files+=1;continue
        parsed_files+=1
        lines=code.splitlines()
        call_funcs={id(n.func):n for n in ast.walk(tree) if isinstance(n,ast.Call)}
        for n in ast.walk(tree):
            kind=None;receiver=""
            if isinstance(n,ast.Call):
                c=n.func
                if isinstance(c,ast.Name) and c.id==name:
                    kind="direct_call"
                elif isinstance(c,ast.Attribute) and c.attr==name:
                    kind="attribute_call"
                    try:receiver=ast.unparse(c.value)[:110]
                    except Exception:pass
            elif isinstance(n,ast.Attribute) and n.attr==name and id(n) not in call_funcs:
                kind="attribute_read"
                try:receiver=ast.unparse(n.value)[:110]
                except Exception:pass
            elif isinstance(n,ast.Name) and n.id==name and id(n) not in call_funcs:
                kind="name_reference"
            if not kind:continue
            caller=caller_for_line(relative_path,n.lineno,symbols)
            if caller==target:continue
            counts[kind]+=1
            if len(hits)<100:
                example=lines[n.lineno-1].strip()[:180] if 0<n.lineno<=len(lines) else ""
                hits.append(dict(file=relative_path,line=n.lineno,kind=kind,receiver=receiver,
                                 caller=caller,source=example))
    callback_decorators=[d for d in decorators if any(w in d.lower() for w in
                       ("route","property","hook","register","event","abstractmethod","cached_property"))]
    flag=("property_access" if any(d in ("property","cached_property") for d in decorators)
          else "registered_route_or_hook" if callback_decorators or name.startswith("pytest_")
          else "unresolved_object_method_call" if counts["attribute_call"]
          else "non_call_reference_only" if counts["attribute_read"] or counts["name_reference"]
          else "no_python_source_reference")
    result=dict(repo=repo,instance_id=row["instance_id"],issue_url=row["issue_url"],
        base_commit=sha,target=target,target_rank=row.get("target_rank"),
        original_graph_incoming=len(original_incoming),graph_nodes=len(symbols),graph_edges=len(graph),
        graph_unresolved_calls=graph_stats["unresolved_calls"],graph_parse_errors=graph_stats["parse_errors"],
        audit_parsed_source_files=parsed_files,audit_parse_failures=failed_files,
        decorators=decorators,class_method_name_duplicates=same_name,
        evidence_count=dict(counts),evidence=hits,
        hypothesis=flag,
        conclusion="old_zero_in_degree_does_not_imply_unreachable_in_python",
        edges_added=0)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--shard-count",type=int,default=5)
    args=p.parse_args()
    rows=[json.loads(x) for x in Path(args.input).read_text(encoding="utf8").splitlines() if x.strip()]
    targets=[x for x in rows if x.get("case")=="case3_no_topk_path"
             and x.get("best_ancestor_bin")=="no_incoming_edge"]
    if len(targets)!=15:raise SystemExit("Expected 15 no-incoming Case3 samples, got "+str(len(targets)))
    shard=[x for i,x in enumerate(targets) if i%args.shard_count==args.shard_index]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    audited=[];failures=[]
    with tempfile.TemporaryDirectory(prefix="ast-audit-") as tmp:
        for index,row in enumerate(shard,1):
            try:
                result=audit_case(row,Path(tmp))
                audited.append(result)
                print("AST_AUDIT",index,"/",len(shard),result["repo"],
                      result["hypothesis"],result["evidence_count"],
                      "old_incoming",result["original_graph_incoming"],flush=True)
            except Exception as exc:
                failures.append(dict(repo=row["repo"],target=row["target"],error=str(exc)[:400]))
                print("AST_AUDIT_ERROR",row["repo"],str(exc)[:300],flush=True)
            write_jsonl(out/"audited.jsonl",audited)
            write_jsonl(out/"failures.jsonl",failures)
    report=dict(shard=args.shard_index,expected=len(shard),audited=len(audited),failed=len(failures),
                flagged= dict(Counter(x["hypothesis"] for x in audited)),
                total_evidence_counts=dict(sum((Counter(x["evidence_count"]) for x in audited),Counter())))
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("AST_AUDIT_SUMMARY",json.dumps(report),flush=True)
    if failures:raise SystemExit("Audit incomplete: "+str(len(failures))+" errors")

if __name__=="__main__":main()
