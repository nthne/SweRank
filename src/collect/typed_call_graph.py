"""High-precision additions to the Python static call graph.

Keep old AST graph for comparability, but infer only uniquely resolved object,
property, super and callback callsites. Unknown receivers are NOT name-matched.
No imports or project code executed; built from buggy source snapshot only.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from pathlib import Path

from navigation_graph import collect_repo, module_name, SKIP_DIRS


class SourceCalls(ast.NodeVisitor):
    """Traverse the entire callable expression; never enter nested functions."""
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.attrs = []

    def visit_FunctionDef(self, node):
        if node is self.root:
            for stmt in node.body:
                self.visit(stmt)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        pass

    def visit_Lambda(self, node):
        pass

    def visit_Call(self, node):
        self.calls.append(node)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.attrs.append(node)
        self.generic_visit(node)


def _decorators(node):
    return {ast.unparse(x).rsplit(".",1)[-1] for x in node.decorator_list}


def collect_repo_v2(root, max_source_chars=3000):
    """Return symbols, typed edges, stats. All novel edges include evidence."""
    root = Path(root)
    symbols, legacy, stats = collect_repo(root, max_source_chars)
    classes = {}
    functions = {}
    module_imports = {}
    files = {}
    # Parse only sources indexed by the original graph.
    for path in sorted({s.path for s in symbols.values()}):
        if any(p in SKIP_DIRS for p in Path(path).parts):
            continue
        try:
            tree = ast.parse((root/path).read_text(encoding="utf8"), filename=path)
        except (SyntaxError, ValueError, OSError, UnicodeError):
            continue
        files[path] = tree
        module_imports[path] = next((s.imports for s in symbols.values() if s.path == path), {})
        def walk(items, prefix=()):
            for node in items:
                if isinstance(node, ast.ClassDef):
                    qual = ".".join(prefix+(node.name,))
                    classes[(path,qual)] = dict(node=node, qual=qual, path=path,
                                                module=module_name(path), bases=node.bases,
                                                methods={}, fields={}, properties={})
                    walk(node.body,prefix+(node.name,))
                elif isinstance(node, (ast.FunctionDef,ast.AsyncFunctionDef)):
                    fnid = path+"/"+"/".join(prefix+(node.name,))
                    if fnid in symbols:
                        functions[fnid] = node
                    walk(node.body,prefix+(node.name,))
        walk(tree.body)

    class_mod = defaultdict(list)
    class_name = defaultdict(list)
    for key, cl in classes.items():
        class_mod[(cl["module"],cl["qual"])].append(key)
        class_name[(cl["path"],cl["qual"])].append(key)
    def only(xs):
        return xs[0] if len(xs)==1 else None
    def import_class(expr, path, parents=()):
        """Resolve only a known class by lexical or explicit import identity."""
        if isinstance(expr, ast.Subscript):
            return import_class(expr.value,path,parents)
        if isinstance(expr, ast.Name):
            for i in range(len(parents),-1,-1):
                key=only(class_name.get((path,".".join(parents[:i]+(expr.id,))),[]))
                if key: return key
            imports=module_imports.get(path,{})
            spec=imports.get(expr.id)
            if spec and spec[0]=="symbol":
                return only(class_mod.get((spec[1],spec[2]),[]))
            return None
        if isinstance(expr, ast.Attribute):
            if isinstance(expr.value,ast.Name):
                imports=module_imports.get(path,{})
                spec=imports.get(expr.value.id)
                if spec:
                    if spec[0]=="module":
                        return only(class_mod.get((spec[1],expr.attr),[]))
                    if spec[0]=="symbol":
                        return only(class_mod.get((spec[1]+"."+spec[2],expr.attr),[]))
            return None
        return None

    for key, cl in classes.items():
        for item in cl["node"].body:
            if isinstance(item,(ast.FunctionDef,ast.AsyncFunctionDef)):
                symid=cl["path"]+"/"+cl["qual"].replace(".","/")+"/"+item.name
                if symid in functions:
                    cl["methods"][item.name]=symid
                    decorators=_decorators(item)
                    if "property" in decorators or "cached_property" in decorators:
                        cl["properties"][item.name] = symid
    base_cache={}
    def bases(key):
        if key in base_cache:return base_cache[key]
        cl=classes[key]
        parent=tuple(cl["qual"].split("."))[:-1]
        result=[]
        for node in cl["bases"]:
            b=import_class(node,cl["path"],parent)
            if b and b!=key:result.append(b)
        base_cache[key]=result
        return result

    def find_method(key,name,skip_current=False,seen=None):
        if not key:return None
        if seen is None:seen=set()
        if key in seen:return None
        seen.add(key)
        if not skip_current and name in classes[key]["methods"]:
            return classes[key]["methods"][name]
        found=[find_method(b,name,seen=seen) for b in bases(key)]
        found=[x for x in found if x]
        return found[0] if len(set(found))==1 and found else None

    def annotation_type(n,path,parents):
        if n is None:return None
        if isinstance(n,ast.Constant) and isinstance(n.value,str):
            try: return annotation_type(ast.parse(n.value,mode="eval").body,path,parents)
            except SyntaxError:return None
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.BitOr):
            vals={annotation_type(part,path,parents) for part in (n.left,n.right)}
            vals.discard(None)
            return next(iter(vals)) if len(vals)==1 else None
        if isinstance(n,ast.Subscript):
            value=ast.unparse(n.value).split(".")[-1]
            if value in ("Optional","Annotated","ClassVar","Final"):
                x=n.slice
                return annotation_type(x.elts[0] if isinstance(x,ast.Tuple) else x,path,parents)
            return None
        return import_class(n,path,parents)

    # Type inference only when known from explicit type / constructor.
    def init_field_types():
        for key,cl in classes.items():
            for node in cl["node"].body:
                if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
                    typ=annotation_type(node.annotation,cl["path"],tuple(cl["qual"].split(".")))
                    if typ:cl["fields"][node.target.id]=typ
            init=functions.get(cl["methods"].get("__init__"))
            if not init:continue
            fnpars={n.arg:annotation_type(n.annotation,cl["path"],tuple(cl["qual"].split(".")))
                    for n in init.args.posonlyargs+init.args.args+init.args.kwonlyargs}
            for stmt in ast.walk(init):
                if isinstance(stmt,ast.AnnAssign) and isinstance(stmt.target,ast.Attribute):
                    if isinstance(stmt.target.value,ast.Name) and stmt.target.value.id=="self":
                        typ=annotation_type(stmt.annotation,cl["path"],tuple(cl["qual"].split(".")))
                        if typ:cl["fields"][stmt.target.attr]=typ
                if isinstance(stmt,ast.Assign):
                    if len(stmt.targets)!=1:continue
                    t=stmt.targets[0]
                    if not isinstance(t,ast.Attribute) or not isinstance(t.value,ast.Name) or t.value.id!="self":continue
                    typ=(import_class(stmt.value.func,cl["path"]) if isinstance(stmt.value,ast.Call)
                         else fnpars.get(stmt.value.id) if isinstance(stmt.value,ast.Name) else None)
                    if typ:cl["fields"][t.attr]=typ
    init_field_types()

    def field_type(key,name,seen=None):
        if not key:return None
        if seen is None:seen=set()
        if key in seen:return None
        seen.add(key)
        cl=classes[key]
        if name in cl["fields"]:return cl["fields"][name]
        candidates=[field_type(base,name,seen) for base in bases(key)]
        candidates=[x for x in candidates if x]
        return candidates[0] if len(set(candidates))==1 and candidates else None

    # Return type of uniquely-identified property/method (annotation or safe constructor return).
    def result_type(method_id):
        if not method_id:return None
        node=functions.get(method_id)
        if not node:return None
        sym=symbols[method_id]
        qual=tuple(sym.class_name.split(".")) if sym.class_name else ()
        return annotation_type(node.returns,sym.path,qual)

    def prop_type(key,attr):
        symid=find_method(key,attr)
        if not symid:return None
        declaring=(symbols[symid].path,symbols[symid].class_name)
        if symid not in classes.get(declaring,{}).get("properties",{}).values():return None
        return result_type(symid)

    # Avoid guessing on ambiguous assignments. Names are only inferred if all assignments agree.
    def name_types(node,path,parents):
        env={}
        for p in node.args.posonlyargs+node.args.args+node.args.kwonlyargs:
            typ=annotation_type(p.annotation,path,parents)
            if typ:env[p.arg]=typ
        assigned=defaultdict(set)
        for stmt in ast.walk(node):
            if isinstance(stmt,(ast.FunctionDef,ast.AsyncFunctionDef)) and stmt is not node:
                continue
            if isinstance(stmt,ast.Assign):
                typ=import_class(stmt.value.func,path,parents) if isinstance(stmt.value,ast.Call) else None
                for t in stmt.targets:
                    if isinstance(t,ast.Name):
                        assigned[t.id].add(typ)
            elif isinstance(stmt,ast.AnnAssign) and isinstance(stmt.target,ast.Name):
                assigned[stmt.target.id].add(annotation_type(stmt.annotation,path,parents))
        for name,poss in assigned.items():
            if len(poss)==1 and None not in poss:env[name]=next(iter(poss))
            else:env.pop(name,None)
        return env

    def class_from_expr(expr,path,cls,env,depth=0):
        if depth>5:return None
        if isinstance(expr,ast.Name):
            if expr.id in ("self","cls"):return cls
            return env.get(expr.id)
        if isinstance(expr,ast.Call):
            if isinstance(expr.func,ast.Name) and expr.func.id=="super":return None
            return import_class(expr.func,path)
        if isinstance(expr,ast.Attribute):
            receiver=class_from_expr(expr.value,path,cls,env,depth+1)
            if not receiver:return None
            return field_type(receiver,expr.attr) or prop_type(receiver,expr.attr)
        return None

    def method_of_attr(expr,path,cls,env):
        if not isinstance(expr,ast.Attribute):return None,None
        if isinstance(expr.value,ast.Call) and isinstance(expr.value.func,ast.Name) and expr.value.func.id=="super":
            if not cls:return None,None
            found=[find_method(b,expr.attr) for b in bases(cls)]
            found=[v for v in found if v]
            return (found[0],"super_method") if len(set(found))==1 and found else (None,None)
        receiver=class_from_expr(expr.value,path,cls,env)
        if not receiver:return None,None
        found=find_method(receiver,expr.attr)
        if not found:return None,None
        if isinstance(expr.value,ast.Call):kind="constructed_object_method"
        elif isinstance(expr.value,ast.Name) and expr.value.id in ("self","cls"):kind="self_or_inherited_method"
        elif isinstance(expr.value,ast.Name):kind="typed_local_method"
        else:kind="typed_field_or_property_method"
        return found,kind

    # Identify only provably callable helper parameters (for callback argument flow).
    callable_params={}
    for symid,node in functions.items():
        params={a.arg:i for i,a in enumerate(node.args.posonlyargs+node.args.args)}
        visitor=SourceCalls(node);visitor.visit(node)
        invoked=set()
        for call in visitor.calls:
            if isinstance(call.func,ast.Name) and call.func.id in params:
                invoked.add(params[call.func.id])
        if invoked:callable_params[symid]=invoked

    edges={(e["source"],e["target"]):dict(e) for e in legacy}
    counters=Counter()
    for symid,node in functions.items():
        sym=symbols[symid]
        clkey=(sym.path,sym.class_name) if sym.class_name else None
        if clkey not in classes:clkey=None
        parents=tuple(sym.qualname.split("."))[:-1]
        env=name_types(node,sym.path,parents)
        visitor=SourceCalls(node);visitor.visit(node)

        def add(target,kind,lineno,via=None):
            if not target or target==symid:return
            key=(symid,target)
            if key in edges:return
            edge={"source":symid,"target":target,"type":kind,
                  "confidence":"high","evidence":sym.path+":"+str(lineno)}
            if via:edge["via_helper"]=via
            edges[key]=edge
            counters[kind]+=1

        for call in visitor.calls:
            target,kind=method_of_attr(call.func,sym.path,clkey,env)
            if target:add(target,kind,call.lineno)
            # resolve passing a same-module or imported function to a helper that invokes it
            helper=None
            fn=call.func
            if isinstance(fn,ast.Name):
                for depth in range(len(parents),-1,-1):
                    name=sym.path+"/"+"/".join(parents[:depth]+(fn.id,))
                    if name in functions:helper=name;break
                if helper is None and fn.id in sym.imports:
                    spec=sym.imports[fn.id]
                    if spec[0]=="symbol":
                        candidates=[s for s in symbols if symbols[s].module==spec[1]
                                    and symbols[s].qualname==spec[2]]
                        if len(candidates)==1:helper=candidates[0]
            if helper not in callable_params:continue
            for pos in callable_params[helper]:
                if pos>=len(call.args):continue
                arg=call.args[pos]
                if not isinstance(arg,ast.Name):continue
                candidate=None
                for depth in range(len(parents),-1,-1):
                    name=sym.path+"/"+"/".join(parents[:depth]+(arg.id,))
                    if name in functions:candidate=name;break
                if candidate is None and arg.id in sym.imports:
                    spec=sym.imports[arg.id]
                    if spec[0]=="symbol":
                        matches=[x for x in symbols if symbols[x].module==spec[1] and
                                 symbols[x].qualname==spec[2]]
                        if len(matches)==1:candidate=matches[0]
                add(candidate,"callback_flow",call.lineno,via=helper)

        for attr in visitor.attrs:
            if not isinstance(attr.ctx,ast.Load):continue
            receiver=class_from_expr(attr.value,sym.path,clkey,env)
            if not receiver:continue
            target=find_method(receiver,attr.attr)
            if not target:continue
            declaring=(symbols[target].path,symbols[target].class_name)
            if target not in classes.get(declaring,{}).get("properties",{}).values():continue
            add(target,"property_getter",attr.lineno)

    # External handlers are not ordinary in-repo caller edges.
    external={}
    for symid,node in functions.items():
        decorators=_decorators(node)
        sym=symbols[symid]
        if any("route" in d or "get"==d or "post"==d or "put"==d for d in decorators):
            external[symid]="route_registration"
        elif node.name.startswith("pytest_") and len(node.name)>7:
            external[symid]="pytest_hook"
        elif sym.path.endswith(("socket.py","_socket.py")) and node.name in (
                "getsockname","getpeername","recv","send","close","connect","accept"):
            external[symid]="protocol_external_dispatch_candidate"
    stats=dict(stats)
    stats.update(v2_new_edge_counts=dict(counters),v2_new_edges=sum(counters.values()),
                 v2_external_entrypoints=external,
                 v2_resolution="conservative_explicit_type_or_import_only")
    return symbols, sorted(edges.values(),key=lambda e:(e["source"],e["target"])), stats
