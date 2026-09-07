import os
os.environ.setdefault('OMP_NUM_THREADS','4')
import unittest,json,sqlite3,tempfile
from pathlib import Path
from unittest.mock import patch
import pandas as pd,numpy as np
from src.data.preprocessor import features,risk_band
from src.ml.predict import load_model,predict,explain
from src.talk_to_data.query_runner import execute_query,validate_sql
from src.talk_to_data.prompt_templates import PATTERNS
from src.talk_to_data.nl_to_sql import ask
ROOT=Path(__file__).resolve().parents[1]

class SQLSafety(unittest.TestCase):
    def test_five_real_patterns(self):
        db=ROOT/'runtime/analytics.db'
        if not db.exists():self.skipTest('Run bootstrap.py with the dataset first')
        for name,sql in PATTERNS:
            with self.subTest(name=name):self.assertGreater(execute_query(db,sql)['row_count'],0)
        r=execute_query(db,PATTERNS[0][1])['rows'][0]
        eda=json.loads((ROOT/'reports/eda.json').read_text())
        self.assertEqual(r['applicants'],eda['development_rows'])
        self.assertAlmostEqual(r['difficulty_rate'],eda['target_rate'])
    def test_unsafe_sql_rejected(self):
        bad=['DROP TABLE applicants','SELECT income FROM applicants; DELETE FROM applicants',
             'SELECT * FROM applicants','SELECT name FROM sqlite_master',
             "SELECT load_extension('/tmp/x') FROM applicants",'PRAGMA table_info(applicants)',
             'SELECT readfile("/etc/passwd") FROM applicants',
             'SELECT a.income FROM applicants a JOIN applicants b ON 1=1',
             'SELECT (SELECT income FROM applicants) FROM applicants',
             'SELECT income FROM other.applicants','SELECT income FROM applicants UNION SELECT income FROM applicants',
             'SELECT nonexistent FROM applicants','SELECT randomblob(100000000) FROM applicants',
             'SELECT income FROM applicants LIMIT -1']
        for sql in bad:
            with self.subTest(sql=sql),self.assertRaises(ValueError):validate_sql(sql)
    def test_limit_preserved_and_capped(self):
        self.assertTrue(validate_sql('SELECT income FROM applicants LIMIT 10').endswith('LIMIT 10'))
        self.assertTrue(validate_sql('SELECT income FROM applicants LIMIT 100000').endswith('LIMIT 100'))
    def test_missing_llm_configuration_fails_explicitly(self):
        with patch.dict(os.environ,{'LLM_API_KEY':'','LLM_MODEL':''}):
            with self.assertRaises(ValueError):ask('What is the default rate?',[],ROOT/'runtime/analytics.db')
    def test_followup_memory_contract_mock(self):
        with patch('src.talk_to_data.nl_to_sql.complete',return_value=({'sql':None,'clarification':'Which income type?'},{})) as call:
            ask('And for employed applicants?',[{'question':'Overall rate?','sql':PATTERNS[0][1]}],ROOT/'runtime/analytics.db')
            messages=call.call_args[0][0]
            self.assertEqual(messages[1]['content'],'Overall rate?')
            self.assertIn('SELECT',messages[2]['content'])
            self.assertEqual(messages[-1]['content'],'And for employed applicants?')
    def test_database_authorizer_independent(self):
        db=ROOT/'runtime/analytics.db'
        if not db.exists():self.skipTest('Run bootstrap.py first')
        with patch('src.talk_to_data.query_runner.validate_sql',return_value='DELETE FROM applicants'):
            with self.assertRaises(ValueError):execute_query(db,'ignored')

class ModelBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=load_model(ROOT/'models/risk_model.joblib')
        cls.raw=pd.DataFrame([{'AMT_INCOME_TOTAL':180000,'AMT_CREDIT':600000,'AMT_ANNUITY':30000,'DAYS_BIRTH':-13000,'EXT_SOURCE_2':.5,'EXT_SOURCE_3':.5,'NAME_INCOME_TYPE':'Working'}])
    def test_model_probability_and_unknown_category(self):
        r=self.raw.copy();r['NAME_INCOME_TYPE']='Previously unseen category'
        p=predict(self.b,r).probability.iloc[0];self.assertTrue(0<p<1)
    def test_target_and_id_cannot_influence_prediction(self):
        r=self.raw.copy();r['TARGET']=1;r['SK_ID_CURR']=987654
        np.testing.assert_allclose(predict(self.b,r).probability,predict(self.b,self.raw).probability)
    def test_invalid_credit(self):
        r=self.raw.copy();r['AMT_CREDIT']=-1
        with self.assertRaises(ValueError):predict(self.b,r)
    def test_employment_sentinel_and_zero_denominator(self):
        r=self.raw.copy();r['DAYS_EMPLOYED']=365243;r['AMT_INCOME_TOTAL']=0
        f=features(r);self.assertTrue(pd.isna(f.DAYS_EMPLOYED.iloc[0]));self.assertEqual(f.EMPLOYMENT_UNKNOWN.iloc[0],1)
        self.assertTrue(pd.isna(f.CREDIT_INCOME_RATIO.iloc[0]))
    def test_band_boundaries(self):
        self.assertEqual(list(risk_band([.01,.05,.15],[.05,.15])),['Low','Medium','High'])
    def test_shap_reconstructs_calibrated_prediction(self):
        e=explain(self.b,self.raw)
        np.testing.assert_allclose(e['base_probability']+e['values'].sum(axis=1),predict(self.b,self.raw).probability,atol=1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
