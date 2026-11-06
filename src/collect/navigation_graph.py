"""Offline Python function graph and issue-to-code retrieval.

Edges are conservative static proxies, not observed runtime calls or causal proofs.
Only syntax is parsed; repository code is never imported or executed.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
import math
from pathlib import Path
import re

SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', 'site-packages', 'build', 'dist', '__pycache__'}
TOKENS = re.compile(r'[A-Za-z][A-Za-z_0-9]*')
STOP = {'the', 'this', 'that', 'into', 'with', 'from', 'when', 'then', 'and', 'but', 'for', 'are', 'was', 'not', 'can', 'has', 'have', 'does', 'should', 'would', 'could', 'will', 'your', 'you', 'issue', 'bug', 'fix', 'test', 'python', 'error'}


def tokens(value):
    value = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', value).replace('_', ' ')
    return [t.lower() for t in TOKENS.findall(value) if len(t) > 2 and t.lower() not in STOP]


def module_name(path):
    parts = list(Path(path).with_suffix('').parts)
    if 'src' in parts:
        parts = parts[parts.index('src') + 1:]
    if parts and parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def relative_module(current, is_package, level, name):
    if not level:
        return name or ''
    pkg = current.split('.') if is_package else current.split('.')[:-1]
    if level - 1 > len(pkg):
        return ''
    pkg = pkg[:len(pkg) - (level - 1)]
    return '.'.join([*pkg, *([name] if name else [])])


@dataclass
class Symbol:
    id: str
    path: str
    module: str
    qualname: str
    class_name: str
    start_line: int
    end_line: int
    source: str
    imports: dict
    is_package: bool

    def public(self):
        result = asdict(self)
        result.pop('imports')
        return result


class Calls(ast.NodeVisitor):
    """Read one function without attributing calls within nested definitions to it."""
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.refs = []

    def visit_FunctionDef(self, node):
        if node is self.root:
            for stmt in node.body:
                self.visit(stmt)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        pass

    def visit_ClassDef(self, node):
        pass

    def visit_Call(self, node):
        self.calls.append(node.func)
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)


def collect_repo(root, max_source_chars=3000):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('Repository checkout does not exist: ' + str(root))
    symbols, contexts, stats = {}, {}, {'files': 0, 'parse_errors': 0, 'unresolved_calls': 0}
    for file in sorted(root.rglob('*.py')):
        rel = file.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        path = rel.as_posix()
        try:
            data = file.read_text(encoding='utf-8')
            tree = ast.parse(data, filename=path)
        except (SyntaxError, ValueError, UnicodeError, OSError):
            stats['parse_errors'] += 1
            continue
        stats['files'] += 1
        mod = module_name(path)
        is_pkg = file.name == '__init__.py'
        imports = {}
        for entry in tree.body:
            if isinstance(entry, ast.Import):
                for alias in entry.names:
                    imports[alias.asname or alias.name.split('.')[0]] = ('module', alias.name if alias.asname else alias.name.split('.')[0])
            elif isinstance(entry, ast.ImportFrom):
                resolved = relative_module(mod, is_pkg, entry.level, entry.module)
                for alias in entry.names:
                    if alias.name != '*':
                        imports[alias.asname or alias.name] = ('symbol', resolved, alias.name)

        def walk(body, parents=(), class_parts=()):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    walk(node.body, parents + (node.name,), class_parts + (node.name,))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual_parts = parents + (node.name,)
                    symid = path + '/' + '/'.join(qual_parts)
                    snippet = ast.get_source_segment(data, node) or ''
                    symbols[symid] = Symbol(symid, path, mod, '.'.join(qual_parts),
                        '.'.join(class_parts), node.lineno, node.end_lineno or node.lineno,
                        snippet[:max_source_chars], imports, is_pkg)
                    contexts[symid] = (node, parents, class_parts)
                    walk(node.body, qual_parts, class_parts)
        walk(tree.body)

    by_module_qual = defaultdict(list)
    by_file_qual = {}
    for sym in symbols.values():
        by_module_qual[(sym.module, sym.qualname)].append(sym.id)
        by_file_qual[(sym.path, sym.qualname)] = sym.id

    def unique(mod, qual):
        candidates = by_module_qual.get((mod, qual), [])
        return candidates[0] if len(candidates) == 1 else None

    edges = {}
    for symid, (node, parents, class_parts) in contexts.items():
        sym = symbols[symid]
        visitor = Calls(node)
        visitor.visit(node)
        for call in visitor.calls:
            target, kind = None, None
            if isinstance(call, ast.Name):
                name = call.id
                # Do not resolve self-recursion or nested scope across unrelated classes.
                for depth in range(len(parents), -1, -1):
                    target = by_file_qual.get((sym.path, '.'.join(parents[:depth] + (name,))))
                    if target:
                        kind = 'lexical_call'
                        break
                if target is None and name in sym.imports:
                    spec = sym.imports[name]
                    if spec[0] == 'symbol':
                        target = unique(spec[1], spec[2]); kind = 'imported_call'
            elif isinstance(call, ast.Attribute) and isinstance(call.value, ast.Name):
                obj, attr = call.value.id, call.attr
                if obj in ('self', 'cls') and class_parts:
                    target = by_file_qual.get((sym.path, '.'.join(class_parts + (attr,))))
                    kind = 'method_call'
                elif obj in sym.imports:
                    spec = sym.imports[obj]
                    if spec[0] == 'module':
                        target = unique(spec[1], attr); kind = 'module_call'
                    elif spec[0] == 'symbol':
                        # `from pkg import module` -- safe only if it maps to a known module.
                        target = unique(spec[1] + '.' + spec[2], attr); kind = 'module_call'
            if target and target != symid:
                key = (symid, target)
                edges[key] = {'source': symid, 'target': target, 'type': kind}
            elif target is None:
                stats['unresolved_calls'] += 1
    return symbols, sorted(edges.values(), key=lambda e: (e['source'], e['target'])), stats


def bm25_rank(query, symbols):
    """Deterministic lexical baseline. For production, select SweRank embeddings."""
    ids = sorted(symbols)
    docs = []
    for key in ids:
        symbol = symbols[key]
        identity = tokens(symbol.qualname + ' ' + symbol.path)
        docs.append(Counter(identity * 3 + tokens(symbol.source[:2400])))
    df = Counter(word for doc in docs for word in doc)
    avg = max(1.0, sum(sum(doc.values()) for doc in docs) / max(1, len(docs)))
    q = Counter(tokens(query[:5000]))
    scored = []
    for key, doc in zip(ids, docs):
        n = sum(doc.values())
        score = sum((1 + math.log1p(freq)) * math.log(1 + (len(ids) - df[word] + .5)/(df[word] + .5))
                    * (count * 2.2)/(count + 1.2 * (.25 + .75 * n / avg))
                    for word, freq in q.items() if (count := doc.get(word, 0)))
        scored.append((key, round(score, 8)))
    return sorted(scored, key=lambda x: (-x[1], x[0]))


# Reuse the encoder and the full-repository code embeddings across issues at the same base commit.
# Cache is process-local; avoid model load + thousands of repeated document encodes per issue.
_SWERANK_MODELS = {}
_SWERANK_DOCUMENTS = {}


def swerank_rank(query, symbols, model_name='Salesforce/SweRankEmbed-Small'):
    """Rank full repository code without recomputing cached embeddings for each issue."""
    import numpy as np
    if model_name not in _SWERANK_MODELS:
        from sentence_transformers import SentenceTransformer
        _SWERANK_MODELS[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    model = _SWERANK_MODELS[model_name]
    ids = tuple(sorted(symbols))
    cache_key = (id(symbols), model_name)
    cached = _SWERANK_DOCUMENTS.get(cache_key)
    if cached is None or cached[0] is not symbols or cached[1] != ids:
        descriptions = [symbols[k].path + '\n' + symbols[k].qualname + '\n' + symbols[k].source for k in ids]
        d = model.encode(descriptions, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        if len(_SWERANK_DOCUMENTS) >= 12:
            _SWERANK_DOCUMENTS.clear()
        _SWERANK_DOCUMENTS[cache_key] = (symbols, ids, np.asarray(d))
    else:
        d = cached[2]
    q = model.encode([query[:3000]], prompt_name='query', normalize_embeddings=True, show_progress_bar=False)[0]
    sims = np.asarray(d) @ np.asarray(q)
    return sorted([(key, float(score)) for key, score in zip(ids, sims)], key=lambda x: (-x[1], x[0]))


def adjacency(edges):
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for e in edges:
        outgoing[e['source']].add(e['target'])
        incoming[e['target']].add(e['source'])
    return outgoing, incoming


def shortest_paths(start, goal, outgoing, max_hops=12, max_paths=32):
    """All minimum-hop variants, truncated deterministically to max_paths."""
    if start == goal:
        return [[start]]
    distances = {start: 0}
    parents = defaultdict(set)
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if distances[node] >= max_hops:
            continue
        for nxt in sorted(outgoing.get(node, ())):
            d = distances[node] + 1
            if nxt not in distances:
                distances[nxt] = d; queue.append(nxt)
            if distances[nxt] == d:
                parents[nxt].add(node)
    if goal not in distances:
        return []
    result = []
    def unwind(n, path):
        if len(result) >= max_paths:
            return
        if n == start:
            result.append([start] + path)
        else:
            for p in sorted(parents[n]):
                unwind(p, [n] + path)
    unwind(goal, [])
    return result


def distance_to_targets(targets, incoming, max_hops=12):
    dist = {key: 0 for key in targets}
    queue = deque(sorted(targets))
    while queue:
        node = queue.popleft()
        if dist[node] == max_hops:
            continue
        for prior in sorted(incoming.get(node, ())):
            if prior not in dist:
                dist[prior] = dist[node] + 1
                queue.append(prior)
    return dist