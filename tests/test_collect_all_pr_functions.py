import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"/"collect"))
from collect_all_pr_functions import spans, diff_lines, modified_symbols, manifest_repos
from build_benchmark_exclusions import canonical, repository_from_row


class TestAllModifiedFunctions(unittest.TestCase):
    def test_new_function_only(self):
        patch_text="@@ -1,1 +1,3 @@\n x = 1\n+def newly_added():\n+    return 2\n"
        symbols, before, after=modified_symbols("pkg/a.py",patch_text,"x = 1\n","x = 1\ndef newly_added():\n    return 2\n")
        self.assertEqual(symbols,{"pkg/a.py/newly_added"})
        self.assertFalse(before)

    def test_removed_function(self):
        patch_text="@@ -1,3 +1,1 @@\n-def gone():\n-    return 1\n x=2\n"
        symbols, before, after=modified_symbols("pkg/a.py",patch_text,"def gone():\n    return 1\nx=2\n","x=2\n")
        self.assertEqual(symbols,{"pkg/a.py/gone"})
        self.assertFalse(after)

    def test_nested_and_method(self):
        code="class X:\n    def method(self):\n        def nested():\n            return 1\n        return nested()\n"
        self.assertIn("pkg/a.py/X/method/nested",spans(code,"pkg/a.py"))
        symbols,_,_=modified_symbols("pkg/a.py","@@ -3,3 +3,3 @@\n         def nested():\n-            return 1\n+            return 2\n         return nested()\n",code,code.replace("return 1","return 2"))
        self.assertIn("pkg/a.py/X/method/nested",symbols)

    def test_no_similarity_filter(self):
        a="def a():\n    return 1\n"
        b="def a():\n    return 2\n"
        selected,_,_=modified_symbols("lib/a.py","@@ -1,2 +1,2 @@\n def a():\n-    return 1\n+    return 2\n",a,b)
        self.assertEqual(selected,{"lib/a.py/a"})

    def test_canonical_repo_names(self):
        self.assertEqual(canonical("https://github.com/Pallets/Flask/issues/123"),"pallets/flask")
        self.assertEqual(repository_from_row({"instance_id":"django__django-121"}),"django/django")
        self.assertEqual(repository_from_row({"repo":"pypa/packaging"}),"pypa/packaging")

    def test_manifest_fail_closed(self):
        with self.assertRaises(FileNotFoundError):
            manifest_repos("/not/a/real/exclusions.json")


if __name__ == "__main__":
    unittest.main()
