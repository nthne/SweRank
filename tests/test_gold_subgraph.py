import importlib.util
from pathlib import Path
import unittest
source = Path(__file__).resolve().parents[1] / "src" / "collect" / "build_gold_subgraph.py"
spec = importlib.util.spec_from_file_location("build_gold_subgraph", source)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class ShortestTests(unittest.TestCase):
    def test_branches(self):
        case = {"seeds":["A"], "targets":["T1", "T2"],
                "edges":[{"source":a, "target":b} for a,b in
                         [("A","B"),("B","T1"),("A","C"),("C","T2"),("T1","D")]]}
        x = m.shortest_path_subgraph(case)
        self.assertEqual(x["target_hops"], {"T1":2,"T2":2})
        self.assertEqual(x["gold_nodes"], ["A","B","C","T1","T2"])
    def test_equal_shortest(self):
        case = {"seeds":["A"],"targets":["T"],
                "edges":[{"source":a,"target":b} for a,b in
                         [("A","B"),("A","C"),("B","T"),("C","T"),("A","X"),("X","Y"),("Y","T")]]}
        x = m.shortest_path_subgraph(case)
        self.assertEqual(len(x["gold_edges"]), 4)
        self.assertNotIn("X", x["gold_nodes"])
    def test_unreachable(self):
        case={"seeds":["A"],"targets":["T","Z"],"edges":[{"source":"A","target":"T"}]}
        self.assertEqual(m.shortest_path_subgraph(case)["unreachable_targets"], ["Z"])
    def test_reverse_edges(self):
        case={"seeds":["B"],"targets":["A"],"edges":[{"source":"A","target":"B"}]}
        self.assertEqual(m.shortest_path_subgraph(case)["unreachable_targets"], ["A"])
        self.assertEqual(m.shortest_path_subgraph(case, reverse_edges=True)["unreachable_targets"], [])
    def test_seeds(self):
        case={"seeds":["A","D"],"targets":["T"],
              "edges":[{"source":a,"target":b} for a,b in [("A","B"),("B","T"),("D","T")]]}
        self.assertEqual(m.shortest_path_subgraph(case)["gold_nodes"], ["D","T"])
