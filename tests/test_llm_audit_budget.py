import importlib.util,sys,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'src'/'collect'
sys.path.insert(0,str(p))
import llm_audit_budget as m
class BudgetTests(unittest.TestCase):
 def test_group_one_request_cache(self):
  rows=[dict(instance_id="i",repo="a/b",pr_number=2,issue_url="issue",base_commit="abc",query="Error",positive_id="src/a.py/f"+str(i),semantic_overlap=0,llm_judge_status="pending") for i in range(2)]
  code={0:"def f0(): pass",1:"def f1(): pass"}
  calls=[]
  def fetch(path):
   return [] if '/files?' in path else {'title':'fix','body':'fix'}
  def req(data):
   calls.append(data)
   return {'output':[{'content':[{'type':'output_text','text':'{"match":true,"labels":[{"i":0,"l":"fix_target","c":"medium","why":"patch"},{"i":1,"l":"uncertain","c":"low","why":"unknown"}]}'}]}],'usage':{'input_tokens':80,'output_tokens':40}}
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   cache=str(Path(d)/'cache.json')
   stats=m.run_budgeted_judge(rows,code,None,'fake','test',1,8,cache,fetch,req)
   self.assertEqual(stats['api_calls'],1)
   self.assertEqual(stats['reviewed'],2)
   self.assertEqual(len(calls),1)
   newer=[dict(r,llm_judge_status='pending') for r in rows]
   stat2=m.run_budgeted_judge(newer,code,None,'fake','test',1,8,cache,fetch,req)
   self.assertEqual(stat2['api_calls'],0)
   self.assertEqual(stat2['cache_hits'],1)
   self.assertEqual(len(calls),1)
 def test_large_group_all_batches(self):
  rows=[dict(instance_id="i",repo="a/b",pr_number=2,issue_url="issue",base_commit="abc",query="Error",positive_id="src/a.py/f"+str(i),semantic_overlap=0,llm_judge_status="pending") for i in range(17)]
  calls=[]
  def fetch(path):
   return [] if '/files?' in path else {'title':'fix','body':'fix'}
  def request(payload):
   calls.append(payload)
   import json
   count=len(json.loads(payload['input'].split('\\n',1)[1])['targets'])
   labels=[{'i':i,'l':'fix_target','c':'low','why':'related'} for i in range(count)]
   return {'output':[{'content':[{'type':'output_text','text':json.dumps({'match':True,'labels':labels})}]}],'usage':{'input_tokens':100,'output_tokens':100}}
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   cache=str(Path(d)/'cache.json')
   stats=m.run_budgeted_judge(rows,{},None,'fake','test',3,8,cache,fetch,request)
   self.assertEqual(stats['api_calls'],3)
   self.assertEqual(stats['reviewed'],17)
   self.assertEqual(len(calls),3)
   self.assertTrue(all(r['llm_judge_status']=='completed_unverified' for r in rows))
   stats2=m.run_budgeted_judge([dict(x,llm_judge_status='pending') for x in rows],{},None,'fake','test',3,8,cache,fetch,request)
   self.assertEqual(stats2['api_calls'],0)
   self.assertEqual(stats2['reviewed'],17)
 def test_parse_labels(self):
  self.assertRaises(ValueError,m.parse_labels,{'match':True,'labels':[]},2)
 def test_zero_calls(self):
  row=dict(instance_id='i',repo='a/b',pr_number=2,issue_url='issue',base_commit='abc',query='Error',positive_id='a.py/f',semantic_overlap=0,llm_judge_status='pending')
  stats=m.run_budgeted_judge([row],{},None,'fake','test',0,8)
  self.assertEqual(stats['api_calls'],0)
  self.assertEqual(row['llm_judge_status'],'skipped_budget')
if __name__=='__main__':unittest.main()
