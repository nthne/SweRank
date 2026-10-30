import importlib.util,unittest,sys
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'src'/'collect'/'enrich_sweloc.py'
sys.path.insert(0,str(p.parent))
spec=importlib.util.spec_from_file_location('enrich',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class EnrichmentTests(unittest.TestCase):
 def test_cosine(self):
  self.assertAlmostEqual(m.cosine([1,0],[2,0]),1)
  self.assertAlmostEqual(m.cosine([1,0],[0,3]),0)
 def test_ast(self):
  self.assertIn('return 1',m.extract_function('class A:\n def foo(self):\n  return 1\n',['A','foo']))
 def test_judge_validation(self):
  rows=[{'positive_id':'x.py/f'}]
  j={'issue_pr_match':True,'functions':[{'positive_id':'x.py/f','label':'test_change','confidence':'low','evidence':'test'}]}
  self.assertEqual(m.validate_judgement(j,rows)['x.py/f']['label'],'test_change')
 def test_no_fake_score(self):
  row=dict(instance_id='x',repo='org/repo',base_commit='abc',pr_number=1,issue_url='x',
           pr_url='p',positive_id='a.py/f',query='issue',semantic_overlap=0)
  result=m.enrich([row],embedding=False,judge=False)
  self.assertIsNone(result[0]['semantic_similarity'])
  self.assertEqual(result[0]['semantic_status'],'not_requested')
if __name__=='__main__':unittest.main()
