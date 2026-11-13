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
        self.assertEqual(c["target_details"][0]["case"],"case1_direct")
        self.assertIsNone(c["issue_localization_hops"])
        self.assertEqual([x["hops"] for x in c["localization_paths"]],[0])
        self.assertEqual(c["case3_teacher_paths"],[])
    def test_case2_four_hops(self):
        c=analyze_issue(self.row,self.ranking(["A","Z","U","B","C","D","T"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["target_details"][0]["case"],"case2_forward")
        self.assertEqual(c["target_details"][0]["best_forward_hops"],4)
        self.assertEqual(c["localization_paths"][0]["nodes"],["A","B","C","D","T"])
    def test_case3_reverse_and_best_rank(self):
        c=analyze_issue(self.row,self.ranking(["Z","B","A","U","C","D","T"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["target_details"][0]["case"],"case3_no_topk_path")
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
        self.assertEqual(c["target_details"][0]["case"],"case1_direct")
        self.assertEqual(len(c["target_details"]),3)
        self.assertEqual(c["targets_not_in_base"],["missing_added_test"])
        self.assertIsNone(c["issue_localization_hops"])
        self.assertEqual({p["target"] for p in c["localization_paths"]},{"T"})
    def test_multi_target_three_cases_in_one_issue(self):
        # Top1=T, U has incoming Z, B has no incoming in this graph.
        # Case1 T, Case2 U via Z if Top2 includes Z, Case3 A if no path from seeds.
        row={**self.row,"all_modified_functions":["T","U","A"],"modified_source_functions":["T","U","A"]}
        c=analyze_issue(row,self.ranking(["T","Z","B","C","D","U","A"]),self.nodes,self.edges,2,8,5)
        labels={t["target"]:t["case"] for t in c["target_details"]}
        self.assertEqual(labels["T"],"case1_direct")
        self.assertEqual(labels["U"],"case2_forward")
        self.assertEqual(labels["A"],"case3_no_topk_path")
        self.assertEqual(c["target_case_counts"],{"case1_direct":1,"case2_forward":1,"case3_no_topk_path":1})
        self.assertEqual({(p["target"],p["hops"]) for p in c["localization_paths"]},{("T",0),("U",1)})
        steps,stops,seeds=generate_training(c,self.ranking(["T","Z","B","C","D","U","A"])["ranked_functions"],self.edges)
        self.assertTrue(all("pair_id" in x and "target" in x for x in steps+stops+seeds))

    def test_multiple_equally_short_paths_for_one_target(self):
        edges=[{"source":"A","target":"B","type":"lexical_call"},
               {"source":"A","target":"C","type":"lexical_call"},
               {"source":"B","target":"T","type":"lexical_call"},
               {"source":"C","target":"T","type":"lexical_call"}]
        c=analyze_issue(self.row,self.ranking(["A","Z","B","C","D","T","U"]),self.nodes,edges,1,8,5)
        self.assertEqual(c["target_details"][0]["case"],"case2_forward")
        self.assertEqual(c["target_details"][0]["best_forward_hops"],2)
        self.assertEqual(c["target_details"][0]["num_shortest_paths"],2)
        self.assertEqual({tuple(p["nodes"]) for p in c["localization_paths"]},
                         {("A","B","T"),("A","C","T")})
        steps,_,_=generate_training(c,self.ranking(["A","Z","B","C","D","T","U"])["ranked_functions"],edges)
        first_hops=[s for s in steps if s["current"]=="A"]
        self.assertEqual({x["candidate"] for x in first_hops if x["label"]==1},{"B","C"})

    def test_no_incoming_edge(self):
        row={**self.row,"all_modified_functions":["Z"],"modified_source_functions":["Z"]}
        c=analyze_issue(row,self.ranking(["U","A","B","C","D","T","Z"]),self.nodes,self.edges,1,8,5)
        self.assertEqual(c["target_details"][0]["case"],"case3_no_topk_path")
        self.assertEqual(c["target_details"][0]["in_degree"],0)
        self.assertEqual(c["target_details"][0]["rank_bucket"],"no_incoming_edge")
if __name__=="__main__":unittest.main()
