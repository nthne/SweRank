import importlib.util
import pathlib
import unittest

source = pathlib.Path(__file__).resolve().parents[1] / "src/collect/sample_sweloc.py"
spec = importlib.util.spec_from_file_location("sample_sweloc", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class AuditTests(unittest.TestCase):
    def test_links(self):
        self.assertEqual(m.linked_issues("Fixes #44, resolves issue #50", "a/b"), [44, 50])
        self.assertEqual(m.linked_issues("Fixes c/d#55", "a/b"), [])
        self.assertEqual(m.linked_issues("https://github.com/a/b/issues/7", "a/b"), [7])
    def test_diff(self):
        old = "def f():\n    x = 1\n    return x\n\ndef g():\n    return 2\n"
        patch = "@@ -1,3 +1,3 @@\n def f():\n-    x = 1\n+    x = 3\n     return x"
        self.assertEqual(m.modified_functions(old, "m.py", patch), ["m.py/f"])
    def test_method(self):
        old = "class A:\n    def f(self):\n        return 1\n"
        patch = "@@ -1,3 +1,3 @@\n class A:\n     def f(self):\n-        return 1\n+        return 2"
        self.assertEqual(m.modified_functions(old, "m.py", patch), ["m.py/A/f"])

if __name__ == "__main__":
    unittest.main()
