#!/usr/bin/env python3
"""Base-commit AST caller->callee graph; Case1/2/3 and 5-hop r-star audit.
No target information in graph construction. No LLM.
"""
import ast,csv,json,subprocess,tempfile,argparse
from pathlib import Path
from collections import defaultdict,deque,Counter

def git(*a,cwd=None):
    return subprocess.check_output(["git",*a],cwd=cwd,stderr=subprocess.PIPE,text=True).strip()

def load(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]

def file_and_functions(root):
    funcs={};modules={};errors=[]
    for path in git("ls-files","--","*.py",cwd=root).splitlines():
        parts=Path(path).parts
        if any(x in ("tests","test","testing","__pycache__","venv",".venv") for x in parts) or parts[-1].startswith("test_") or parts[-1].endswith("_test.py"):continue
        try:
            tree=ast.parse((root/path).read_text(encoding="utf-8"))
            module=path[:-3].replace("/",".").removeprefix("src.")
            if module.endswith(".__init__"):module=module[:-9]
            modules[module]=(path,tree)
            def walk(nodes,parents=(),classes=()):
                for n in nodes:
                    if isinstance(n,ast.ClassDef):walk(n.body,parents+(n.name,),classes+(n.name,))
                    elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                        key=path+"/"+"/".join(parents+(n.name,))
                        funcs[key]=(path,module,n,parents,classes)
                        walk(n.body,parents+(n.name,),classes)
            walk(tree.body)
        except (OSError,ValueError,UnicodeError,SyntaxError) as e:errors.append((path,str(e)))
    return funcs,modules,errors

def imports(tree,module,path):
    pkg=module if path.endswith("/__init__.py") else module.rsplit(".",1)[0]
    mapping={}
    for n in tree.body:
        if isinstance(n,ast.Import):
            for x in n.names:mapping[x.asname or x.name.split(".")[0]]=x.name if x.asname else x.name.split(".")[0]
        elif isinstance(n,ast.ImportFrom):
            base=n.module or ""
            if n.level:
                p=pkg.split(".") if pkg else []
                base=".".join([x for x in (".".join(p[:max(0,len(p)-n.level+1)]),base) if x])
            for x in n.names:
                if x.name!="*":mapping[x.asname or x.name]=".".join(y for y in (base,x.name) if y)
    return mapping

def dotted(node):
    if isinstance(node,ast.Name):return node.id
    if isinstance(node,ast.Attribute):
        a=dotted(node.value)
        return a+"."+node.attr if a else None
    return None

def graph(root,ranking):
    funcs,modules,parse_errors=file_and_functions(root)
    nodes=set(funcs)
    if nodes!=set(ranking):
        raise RuntimeError("graph/index mismatch graph=%d index=%d missing=%d extra=%d"%(len(nodes),len(ranking),len(set(ranking)-nodes),len(nodes-set(ranking))))
    aliases={path:imports(tree,module,path) for module,(path,tree) in modules.items()}
    mods=sorted(modules,key=len,reverse=True)
    def lookup(qualified):
        for mod in mods:
            if qualified.startswith(mod+"."):
                n=modules[mod][0]+"/"+qualified[len(mod)+1:].replace(".","/")
                if n in nodes:return n
                for c in ("/__init__","/__new__"):
                    if n+c in nodes:return n+c
        return None
    class Calls(ast.NodeVisitor):
        def __init__(self):self.found=[]
        def visit_Call(self,node):self.found.append(node.func);self.generic_visit(node)
        def visit_FunctionDef(self,node):pass
        def visit_AsyncFunctionDef(self,node):pass
        def visit_ClassDef(self,node):pass
    edges=set();unresolved=0;dynamic=0
    for source,(path,module,node,parents,classes) in funcs.items():
        visitor=Calls()
        for expr in node.body:visitor.visit(expr)
        for call in visitor.found:
            d=dotted(call)
            if d is None:dynamic+=1;continue
            rootname,*tail=d.split(".")
            q=[]
            if rootname in ("self","cls") and classes:
                q=[module+"."+".".join(classes+tuple(tail))]
            elif rootname in aliases[path]:
                q=[aliases[path][rootname]+("."+(".".join(tail)) if tail else "")]
            else:
                q=[module+"."+".".join(parents[:n]+tuple([rootname]+tail)) for n in range(len(parents),-1,-1)]
                q.append(d)
            dst=next((n for p in q if (n:=lookup(p))),None)
            if dst and dst!=source:edges.add((source,dst))
            elif not dst:unresolved+=1
    return sorted(edges),parse_errors,{"dynamic_calls":dynamic,"unresolved_or_external_calls":unresolved}

def ancestor_paths(t,rev,maxh=5):
    q=deque([t]);paths={t:[t]}
    while q:
        at=q.popleft()
        if len(paths[at])-1>=maxh:continue
        for p in sorted(rev[at]):
            if p not in paths:
                paths[p]=[p]+paths[at];q.append(p)
    paths.pop(t,None)
    return paths

def casetype(targets,ranking,edges):
    if targets.intersection(ranking[:10]):return "Case 1"
    forward=defaultdict(set)
    for a,b in edges:forward[a].add(b)
    for seed in ranking[:10]:
        q=deque([(seed,0)]);seen={seed}
        while q:
            a,h=q.popleft()
            if 0<h<=8 and a in targets:return "Case 2"
            if h==8:continue
            for b in forward[a]:
                if b not in seen:seen.add(b);q.append((b,h+1))
    return "Case 3"

def bucket(rank):
    if rank is None:return "no_ranked_ancestor_within_5_hops"
    if rank<=10:return "1-10"
    if rank<=20:return "11-20"
    if rank<=50:return "21-50"
    if rank<=100:return "51-100"
    return ">100"

def audit(gold,rankings):
    by_id={r["instance_id"]:r for r in rankings}
    out=[];gmeta=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        grouped=defaultdict(list)
        for row in gold:
            if "collection_error" not in row:
                if row["instance_id"] not in by_id:raise RuntimeError("Missing ranking for "+row["instance_id"])
                grouped[(row["repo"],row["base_commit"])].append(row)
        for (repo,sha),items in grouped.items():
            local=root/repo.replace("/","__")
            if not local.exists():git("clone","-q","--filter=blob:none","https://github.com/"+repo+".git",str(local))
            git("checkout","-q","--force",sha,cwd=local)
            names=by_id[items[0]["instance_id"]]["ranked_functions"]
            edges,errors,unresolved=graph(local,names)
            rev=defaultdict(set)
            for a,b in edges:rev[b].add(a)
            gmeta.append({"repo":repo,"base_commit":sha,"nodes":len(names),"edges":len(edges),"parse_errors":errors,"unresolved_calls":unresolved})
            print("GRAPH",repo,sha[:9],"nodes",len(names),"edges",len(edges),flush=True)
            for row in items:
                ranklist=by_id[row["instance_id"]]["ranked_functions"]
                rank={n:i+1 for i,n in enumerate(ranklist)}
                if len(rank)!=len(ranklist) or set(rank)!=set(names):raise ValueError("Non-matching ranking")
                ts=set(row["modified_source_functions"])
                present=ts & set(names)
                which=casetype(present,ranklist,edges) if ts else "No source target"
                targets=[]
                for t in sorted(ts):
                    exists=t in rank
                    paths=ancestor_paths(t,rev,5) if exists else {}
                    best=min(paths,key=lambda p:(rank[p],len(paths[p])-1,p)) if paths else None
                    r=rank[best] if best else None
                    targets.append({
                        "target":t,"exists_at_base":exists,"direct_rank":rank.get(t),
                        "incoming_edge_count":len(rev[t]) if exists else None,
                        "r_star":r,"r_star_bucket":bucket(r) if exists else "missing_at_base",
                        "best_ancestor":best,"best_hops":len(paths[best])-1 if best else None,
                        "best_path":paths[best] if best else [],"ancestor_count_5":len(paths) if exists else None
                    })
                viable=[t for t in targets if t["r_star"] is not None]
                best=min(viable,key=lambda t:(t["r_star"],t["best_hops"])) if viable else None
                out.append({
                    "instance_id":row["instance_id"],"issue_url":row["issue_url"],"pr_url":row["pr_url"],
                    "repo":repo,"base_commit":sha,"case":which,
                    "target_count":len(ts),"present_targets":len(present),"missing_at_base":len(ts-present),
                    "r_star":best["r_star"] if best else None,
                    "r_star_bucket":bucket(best["r_star"] if best else None),
                    "r_star_target":best["target"] if best else None,
                    "r_star_path":best["best_path"] if best else [],
                    "targets":targets
                })
    cases=Counter(x["case"] for x in out)
    c3=[x for x in out if x["case"]=="Case 3"]
    t3=[t for x in c3 for t in x["targets"]]
    summary={
        "issue_total":len(out),"cases":dict(cases),"case3_issue_count":len(c3),
        "case3_issue_rank_buckets":dict(Counter(x["r_star_bucket"] for x in c3)),
        "case3_target_rank_buckets":dict(Counter(x["r_star_bucket"] for x in t3)),
        "case3_targets":len(t3),"case3_missing_at_base":sum(not t["exists_at_base"] for t in t3),
        "case3_existing_zero_incoming":sum(t["incoming_edge_count"]==0 for t in t3 if t["exists_at_base"]),
        "all_existing_zero_incoming":sum(t["incoming_edge_count"]==0 for x in out for t in x["targets"] if t["exists_at_base"]),
        "all_missing_at_base":sum(x["missing_at_base"] for x in out),
        "graph_definition":"conservative static AST-resolved direct calls caller -> callee, at pre-fix base commit",
        "case_definition":"Case 1 target Top10; Case 2 no direct hit and forward reachable from Top10 <=8 hops; Case 3 otherwise",
        "r_star_definition":"minimum retrieval rank of ancestor at directed distance 1..5 to any present target",
        "warning":"zero static incoming is NOT proof that no runtime/dynamic incoming edges exist; missing base nodes are separate; targets are weak PR labels"
    }
    return {"summary":summary,"issues":out,"graphs":gmeta}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gold",required=True);p.add_argument("--rankings",required=True)
    p.add_argument("--output",required=True);p.add_argument("--csv",required=True)
    a=p.parse_args()
    result=audit(load(a.gold),load(a.rankings))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    fields=["instance_id","case","target","exists_at_base","direct_rank","incoming_edge_count","r_star","r_star_bucket","best_ancestor","best_hops","best_path","ancestor_count_5"]
    with open(a.csv,"w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for issue in result["issues"]:
            for target in issue["targets"]:
                row={"instance_id":issue["instance_id"],"case":issue["case"],**target}
                writer.writerow({k:" -> ".join(row[k]) if k=="best_path" else row.get(k) for k in fields})
    print(json.dumps(result["summary"],indent=2,ensure_ascii=False))
if __name__=="__main__":main()
