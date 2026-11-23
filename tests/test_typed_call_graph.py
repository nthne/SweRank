import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"/"collect"))
from typed_call_graph import collect_repo_v2


class TypedGraphTests(unittest.TestCase):
    def graph(self,files):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for path,source in files.items():
                f=root/path
                f.parent.mkdir(parents=True,exist_ok=True)
                f.write_text(source,encoding="utf8")
            symbols,edges,stats=collect_repo_v2(root)
            return symbols,edges,stats

    def test_nested_callee_constructor(self):
        src='''class A:
    def fix(self):
        return 1

def entry():
    return A().fix()
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        e=[e for e in edges if e["target"]=="pkg/mod.py/A/fix"]
        self.assertTrue(any(x["source"]=="pkg/mod.py/entry" and
                            x["type"]=="constructed_object_method" for x in e))

    def test_explicit_typed_local(self):
        src='''class SchedulerLoop:
    def run(self):
        return 1

def entry():
    scheduler_loop = SchedulerLoop()
    scheduler_loop.run()
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        self.assertTrue(any(x["source"]=="pkg/mod.py/entry" and
                            x["target"]=="pkg/mod.py/SchedulerLoop/run" for x in edges))

    def test_property_access(self):
        src='''class Auth:
    @property
    def configured_repositories(self) -> dict:
        return {}

    def get(self):
        return self.configured_repositories["x"]
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        self.assertTrue(any(x["source"]=="pkg/mod.py/Auth/get" and
                            x["target"]=="pkg/mod.py/Auth/configured_repositories" and
                            x["type"]=="property_getter" for x in edges))

    def test_inherited_self_and_super(self):
        src='''class Base:
    def run(self):
        return 1

class Sub(Base):
    def helper(self):
        self.run()
        super().run()
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        self.assertTrue(any(x["source"]=="pkg/mod.py/Sub/helper" and
                            x["target"]=="pkg/mod.py/Base/run" for x in edges))

    def test_callback_flow(self):
        src='''def check_block_like(func,node,errors):
    return func(node,errors)

def check_stmts(node,errors):
    return None

def entry(node,errors):
    return check_block_like(check_stmts,node,errors)
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        self.assertTrue(any(x["source"]=="pkg/mod.py/entry" and
                            x["target"]=="pkg/mod.py/check_stmts" and
                            x["type"]=="callback_flow" for x in edges))

    def test_unknown_receiver_never_matched_by_method_name(self):
        src='''class A:
    def run(self): pass

class B:
    def run(self): pass

def entry(obj):
    obj.run()
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        self.assertFalse(any(e["source"]=="pkg/mod.py/entry" and
                             e["target"] in ("pkg/mod.py/A/run","pkg/mod.py/B/run") for e in edges))

    def test_distinct_methods_same_name_type_correct(self):
        src='''class A:
    def run(self): pass

class B:
    def run(self): pass

def entry():
    obj=B()
    obj.run()
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        target=[e["target"] for e in edges if e["source"]=="pkg/mod.py/entry"]
        self.assertIn("pkg/mod.py/B/run",target)
        self.assertNotIn("pkg/mod.py/A/run",target)

    def test_no_nested_function_attribution(self):
        src='''def target(): pass

def parent():
    def nested():
        target()
    return 0
'''
        _,edges,_=self.graph({"pkg/mod.py":src})
        self.assertFalse(any(e["source"]=="pkg/mod.py/parent" and
                             e["target"]=="pkg/mod.py/target" for e in edges))

if __name__=="__main__":
    unittest.main()
