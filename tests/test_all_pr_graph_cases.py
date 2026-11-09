"""Offline regression tests for all modified function graph navigation (no LLM)."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"/"collect"))
from evaluate_all_pr_graph_cases import analyze_issue,generate_training

class TestCases(unittest.TestCase):
    def setUp(self):
        self.nodes={x:object() for x in ("A","B","C","D","T","U","Z")}
        self.edges=[{"source":"A","target":"B","type":"lexical_call"},
                    {"source":"B","target":"C","type":"lexical_call"},
                    {"source":"C","target":"D","type":"lexical_call"},
                    {"source":"D","target":"T","type":"lexical_call"},
                    {"source":"Z","target":"U","type":"lexical_call"}]
        self.row={"instance_id":"demo__repo-1-pr2","issue_id":"demo__repo-1",
                  "repo":"demo/repo","issue_url":"https://github.com/demo/repo/issues/1",
                  "pr_url":"https://github.com/demo/repo/pull/2","base_commit":"a"*40,
                  "query":"description","pr_number":2,"complete_extraction":True,
                  "modified_source_functions":["T"],"modified_test_functions":[],"all_modified_functions":["T"]}
    def ranking(self,order):
        return {"model":"Salesforce/SweRankEmbed-Small","ranked_functions":order}
    def test_case1_zero_hops_no_oracle(self):
        c=analyze_issue(self.row,self.ranking(["T","A","B","C","D","U","Z"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["case"],"case1_direct")
        self.assertEqual(c["issue_localization_hops"],0)
        self.assertEqual([x["hops"] for x in c["localization_paths"]],[0])
        self.assertEqual(c["case3_teacher_paths"],[])
    def test_case2_four_hops(self):
        c=analyze_issue(self.row,self.ranking(["A","Z","U","B","C","D","T"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["case"],"case2_forward")
        self.assertEqual(c["issue_localization_hops"],4)
        self.assertEqual(c["localization_paths"][0]["nodes"],["A","B","C","D","T"])
    def test_case3_reverse_and_best_rank(self):
        c=analyze_issue(self.row,self.ranking(["Z","B","A","U","C","D","T"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["case"],"case3_no_topk_path")
        self.assertIsNone(c["issue_localization_hops"])
        self.assertEqual(c["localization_paths"],[])
        self.assertEqual(c["target_details"][0]["best_ancestor_rank"],2)
        self.assertEqual(c["target_details"][0]["best_ancestor_hops"],3)
        self.assertTrue(c["case3_teacher_paths"])
        self.assertTrue(all(p["oracle_only"] and not p["localization_success"] for p in c["case3_teacher_paths"]))
        steps,stops,seeds=generate_training(c,self.ranking(["Z","B","A","U","C","D","T"])["ranked_functions"],self.edges)
        self.assertTrue(steps and stops and seeds)
        self.assertTrue(all(step["oracle_only_teacher"] for step in steps))
    def test_all_modified_multitarget_with_missing_base(self):
        row={**self.row,"modified_test_functions":["missing_added_test"],
             "all_modified_functions":["T","U","missing_added_test"]}
        c=analyze_issue(row,self.ranking(["T","A","B","C","D","U","Z"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["case"],"case1_direct")
        self.assertEqual(len(c["target_details"]),3)
        self.assertEqual(c["targets_not_in_base"],["missing_added_test"])
        self.assertEqual(c["issue_localization_hops"],0)
        self.assertEqual({p["target"] for p in c["localization_paths"]},{"T"})
    def test_no_incoming_edge(self):
        row={**self.row,"all_modified_functions":["Z"],"modified_source_functions":["Z"]}
        c=analyze_issue(row,self.ranking(["U","A","B","C","D","T","Z"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["case"],"case3_no_topk_path")
        self.assertEqual(c["target_details"][0]["in_degree"],0)
        self.assertEqual(c["target_details"][0]["rank_bucket"],"no_incoming_edge")
if __name__=="__main__":unittest.main()
