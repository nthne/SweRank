import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"/"collect"))
from et1_single_root_paths import single_root, is_test, is_source, fixing_issues

class TestSingleRoot(unittest.TestCase):
    def test_single_edited_function(self):
        before="def first():\n    return 1\n\ndef target():\n    return 2\n"
        after=before.replace("return 2","return 3")
        patch="@@ -4,2 +4,2 @@\n def target():\n-    return 2\n+    return 3\n"
        files=[dict(filename="pkg/main.py",status="modified",patch=patch)]
        target,status=single_root(files,lambda _:before,lambda _:after)
        self.assertEqual(target,"pkg/main.py/target")
        self.assertEqual(status,"accepted")

    def test_test_change_allowed(self):
        before="def target():\n    return 1\n"
        after="def target():\n    return 2\n"
        files=[
            dict(filename="pkg/a.py",status="modified",
                 patch="@@ -1,2 +1,2 @@\n def target():\n-    return 1\n+    return 2\n"),
            dict(filename="tests/test_a.py",status="modified",patch="any")]
        self.assertEqual(single_root(files,lambda _:before,lambda _:after)[1],"accepted")

    def test_global_edit_rejected(self):
        before="CONFIG=1\ndef target():\n    return CONFIG\n"
        after=before.replace("CONFIG=1","CONFIG=2")
        files=[dict(filename="pkg/a.py",status="modified",
                    patch="@@ -1,3 +1,3 @@\n-CONFIG=1\n+CONFIG=2\n def target():\n     return CONFIG\n")]
        self.assertEqual(single_root(files,lambda _:before,lambda _:after)[1],"edit_outside_function")

    def test_config_edit_rejected(self):
        files=[dict(filename="pyproject.toml",status="modified",patch="dummy")]
        self.assertEqual(single_root(files,lambda _:"",lambda _:"")[1],"extra_production_file")

    def test_only_tests_rejected(self):
        files=[dict(filename="tests/test_one.py",status="modified",patch="x")]
        self.assertEqual(single_root(files,lambda _:"",lambda _:"")[1],"function_count_0")

    def test_classify_and_fix_link(self):
        self.assertTrue(is_test("src/tests/test_file.py"))
        self.assertFalse(is_source("src/tests/test_file.py"))
        self.assertTrue(is_source("src/testing_utils.py"))
        pr={"body":"Fixes #123\nCloses other/repo#99\nResolves #456"}
        self.assertEqual(fixing_issues(pr,"owner/repo"),[123,456])

if __name__=="__main__":
    unittest.main()
