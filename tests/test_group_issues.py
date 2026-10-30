import importlib.util
from pathlib import Path
import unittest
source = Path(__file__).resolve().parents[1] / "src" / "collect" / "group_issues.py"
spec = importlib.util.spec_from_file_location("group_issues", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class GroupingTests(unittest.TestCase):
    def test_multi_target(self):
        x = dict(instance_id="repo-1", base_commit="abc", repo="u/r",
                 issue_url="i", query="bug", pr_url="p", review_flags=["multi_function_pr"])
        out = m.group([{**x, "positive_id": "src/a.py/F"},
                       {**x, "positive_id": "tests/test_a.py/test_F"}])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["weak_source_targets"], ["src/a.py/F"])
        self.assertEqual(out[0]["modified_test_functions"], ["tests/test_a.py/test_F"])
    def test_snapshot_versions(self):
        x = dict(instance_id="repo-1", repo="u/r", issue_url="i",
                 query="bug", pr_url="p", positive_id="a.py/F")
        self.assertEqual(len(m.group([{**x, "base_commit":"a"}, {**x, "base_commit":"b"}])), 2)
