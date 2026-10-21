import copy, json, unittest
from pathlib import Path
from tools.api_smoke_test import validate_request, validate_response
ROOT=Path(__file__).resolve().parents[1]
class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.req=json.loads((ROOT/'examples/sample_request_252.json').read_text())
        cls.resp=json.loads((ROOT/'examples/sample_response.json').read_text())
    def test_example_request(self): self.assertTrue(validate_request(self.req))
    def test_example_response(self): self.assertTrue(validate_response(self.resp))
    def test_reject_short_request(self):
        r=copy.deepcopy(self.req);r['history'].pop()
        with self.assertRaises(ValueError): validate_request(r)
    def test_reject_bad_sum(self):
        r=copy.deepcopy(self.resp); r['weights']['CASH']=0.19
        with self.assertRaises(ValueError): validate_response(r)
    def test_reject_nan(self):
        r=copy.deepcopy(self.resp);r['weights']['QQQ']=float('nan')
        with self.assertRaises(ValueError): validate_response(r)
    def test_reject_future_row(self):
        r=copy.deepcopy(self.req);r['history'][-1]['date']='2024-01-02'
        with self.assertRaises(ValueError): validate_request(r)
if __name__=='__main__': unittest.main()
