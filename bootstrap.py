"""Build runtime database and held-out profiles from a private local dataset mount."""
import os,json,sqlite3,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from src.data.preprocessor import RAW
ROOT=Path(__file__).resolve().parent
def prepare():
    source=Path(os.getenv('DATA_PATH') or str(ROOT/'data/application_train.csv'))
    runtime=ROOT/'runtime';runtime.mkdir(exist_ok=True)
    if not source.is_file():
        print('Dataset not mounted. Dashboard/manual prediction available; data chat requires DATA_PATH.',flush=True)
        return
    h=hashlib.sha256()
    with source.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    digest=h.hexdigest()
    expected=json.loads((ROOT/'reports/metrics.json').read_text())['dataset_sha256']
    if digest!=expected:raise ValueError('Dataset differs from training data. Retrain explicitly before serving it.')
    marker=runtime/'source.sha256'
    if marker.exists() and marker.read_text()==digest and (runtime/'analytics.db').exists() and (runtime/'profiles.json').exists():return
    d=pd.read_csv(source,usecols=list(dict.fromkeys(RAW+['SK_ID_CURR','TARGET'])))
    dev,test=train_test_split(np.arange(len(d)),test_size=.15,stratify=d.TARGET,random_state=42)
    fit,other=train_test_split(dev,test_size=30/85,stratify=d.TARGET.iloc[dev],random_state=43)
    cal,val=train_test_split(other,test_size=.5,stratify=d.TARGET.iloc[other],random_state=44)
    split=np.full(len(d),'train',dtype=object)
    for name,idx in [('calibration',cal),('validation',val),('test',test)]:split[idx]=name
    a=pd.DataFrame({'applicant_id':d.SK_ID_CURR,'payment_difficulty':d.TARGET,'income':d.AMT_INCOME_TOTAL,
      'credit':d.AMT_CREDIT,'annuity':d.AMT_ANNUITY,'age':-d.DAYS_BIRTH/365.25,'income_type':d.NAME_INCOME_TYPE,
      'education':d.NAME_EDUCATION_TYPE,'housing':d.NAME_HOUSING_TYPE,'contract_type':d.NAME_CONTRACT_TYPE,
      'occupation':d.OCCUPATION_TYPE.fillna('Unknown'),'credit_income_ratio':d.AMT_CREDIT/d.AMT_INCOME_TOTAL,'split':split})
    with sqlite3.connect(runtime/'analytics.db') as con:
        a.to_sql('applicants',con,if_exists='replace',index=False)
        con.execute('CREATE UNIQUE INDEX applicant_idx ON applicants(applicant_id)')
        con.execute('CREATE INDEX split_idx ON applicants(split)')
    profiles=d.iloc[np.sort(test[:100])].drop(columns='TARGET')
    (runtime/'profiles.json').write_text(profiles.to_json(orient='records'))
    marker.write_text(digest)
    print('Runtime ready:',len(a),'applicants and',len(profiles),'held-out profiles',flush=True)
if __name__=='__main__':prepare()
