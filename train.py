"""Reproduce EDA, splits, training, calibration, rules and serving database."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, hashlib, json, sqlite3, time, platform
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
 recall_score, f1_score, fbeta_score, confusion_matrix, brier_score_loss, roc_curve,
 precision_recall_curve)
from sklearn.calibration import calibration_curve
from src.data.preprocessor import features, CAT, RAW, risk_band
from src.ml.rules import derive_rules

ROOT = Path(__file__).resolve().parent

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda v: v.item() if hasattr(v,'item') else str(v), allow_nan=False))

def metrics(y,p,t):
    h=p>=t
    return dict(roc_auc=roc_auc_score(y,p), pr_auc_average_precision=average_precision_score(y,p),
      precision=precision_score(y,h,zero_division=0),recall=recall_score(y,h),
      f1=f1_score(y,h),f2=fbeta_score(y,h,beta=2),brier=brier_score_loss(y,p),
      threshold=float(t),confusion_matrix=confusion_matrix(y,h).tolist(),n=len(y))

def calibrated(model, prep, calibration, x):
    p=model.predict_proba(prep.transform(x))[:,1]
    z=np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7))).reshape(-1,1)
    return calibration.predict_proba(z)[:,1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(ROOT/'data/application_train.csv')); args=ap.parse_args()
    for p in ['models','reports/figures','documents','runtime','notebooks','sql']:
        (ROOT/p).mkdir(parents=True,exist_ok=True)
    start=time.time(); d=pd.read_csv(args.data)
    assert d.SK_ID_CURR.is_unique and set(d.TARGET.unique())=={0,1}
    print('Loaded',d.shape,flush=True)
    y=d.TARGET.astype(int); x=features(d); indices=np.arange(len(d))
    dev,test=train_test_split(indices,test_size=.15,stratify=y,random_state=42)
    fit,other=train_test_split(dev,test_size=30/85,stratify=y.iloc[dev],random_state=43)
    cal,val=train_test_split(other,test_size=.5,stratify=y.iloc[other],random_state=44)
    # EDA is restricted to development rows. Final holdout is never used for feature selection.
    e=d.iloc[dev].copy(); e['AGE_GROUP']=pd.cut(-e.DAYS_BIRTH/365.25,[0,30,40,50,60,100],right=False).astype(str)
    e['CREDIT_INCOME_GROUP']=pd.cut(e.AMT_CREDIT/e.AMT_INCOME_TOTAL,[0,1,2,3,5,1000],right=False).astype(str)
    eda=dict(total_rows=len(d),columns=len(d.columns),development_rows=len(e),
      target_count=int(e.TARGET.sum()),target_rate=float(e.TARGET.mean()),
      duplicate_ids=int(d.SK_ID_CURR.duplicated().sum()),
      numeric_columns=int(len(d.select_dtypes(include='number').columns)),
      categorical_columns=int(len(d.select_dtypes(exclude='number').columns)),
      employment_sentinel_count=int((e.DAYS_EMPLOYED==365243).sum()),
      missing=e.isna().mean().sort_values(ascending=False).head(20).to_dict(),
      target_definition='Payment difficulties: late payment beyond an unspecified X days in the first Y installments. This is not a universal legal default definition.',
      source='https://www.kaggle.com/competitions/home-credit-default-risk/data',groups={},insights=[])
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':11,'figure.dpi':150})
    for col in ['NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE','NAME_CONTRACT_TYPE','AGE_GROUP','CREDIT_INCOME_GROUP']:
        g=e.groupby(col,observed=True).TARGET.agg(['size','sum','mean']).sort_values('mean',ascending=False)
        g=g[g['size']>=500]
        eda['groups'][col]=[dict(group=str(k),count=int(r['size']),positives=int(r['sum']),rate=float(r['mean'])) for k,r in g.iterrows()]
        hi,lo=g.iloc[0],g.iloc[-1]
        eda['insights'].append(f"{col}: {g.index[0]} has a {hi['mean']:.1%} payment-difficulty rate (n={int(hi['size']):,}), compared with {lo['mean']:.1%} for {g.index[-1]} (n={int(lo['size']):,}). Association, not causation.")
        fig,ax=plt.subplots(figsize=(8,4.5)); ax.barh([str(v) for v in g.index],g['mean']*100,color='#147d92'); ax.invert_yaxis(); ax.set_xlabel('Observed payment-difficulty rate (%)'); ax.set_title(col.replace('_',' ').title()); fig.tight_layout(); fig.savefig(ROOT/f'reports/figures/{col}.png'); plt.close(fig)
    save_json(ROOT/'reports/eda.json',eda)
    save_json(ROOT/'reports/feature_catalog.json',{c:dict(dtype=str(d[c].dtype),missing_fraction=float(e[c].isna().mean()),role='target' if c=='TARGET' else ('identifier' if c=='SK_ID_CURR' else ('model input' if c in RAW else 'not used in this model'))) for c in d})
    num=[c for c in x if c not in CAT]
    prep=ColumnTransformer([('numeric',SimpleImputer(strategy='median',add_indicator=True),num),('category',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),CAT)],verbose_feature_names_out=False)
    zfit=prep.fit_transform(x.iloc[fit]); zcal=prep.transform(x.iloc[cal]); zval=prep.transform(x.iloc[val])
    cat_mask=[i>=zfit.shape[1]-len(CAT) for i in range(zfit.shape[1])]
    baseline_prep=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('category',OneHotEncoder(handle_unknown='ignore'),CAT)])
    print('Training logistic baseline',flush=True)
    baseline=make_pipeline(baseline_prep,LogisticRegression(max_iter=500,class_weight='balanced',C=.1,solver='lbfgs'))
    baseline.fit(x.iloc[fit],y.iloc[fit]); bp=baseline.predict_proba(x.iloc[val])[:,1]
    print('Training two histogram boosting candidates',flush=True)
    candidates=[]
    for weight in [None,'balanced']:
        model=HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=23,learning_rate=.07,l2_regularization=10,min_samples_leaf=60,categorical_features=cat_mask,class_weight=weight,random_state=42,early_stopping=True)
        model.fit(zfit,y.iloc[fit]); pcal=model.predict_proba(zcal)[:,1]
        logit=lambda p:np.log(np.clip(p,1e-7,1-1e-7)/(1-np.clip(p,1e-7,1-1e-7))).reshape(-1,1)
        calibrator=LogisticRegression(C=1000,max_iter=200).fit(logit(pcal),y.iloc[cal])
        p=calibrator.predict_proba(logit(model.predict_proba(zval)[:,1]))[:,1]
        scores=metrics(y.iloc[val],p,.1)
        candidates.append((scores['pr_auc_average_precision'],model,calibrator,p,scores,weight))
        print('Candidate',weight,scores,flush=True)
    _,model,calibrator,pval,_,weight=max(candidates,key=lambda a:a[0])
    thresholds=np.linspace(.02,.40,191)
    threshold=float(max(thresholds,key=lambda t:fbeta_score(y.iloc[val],pval>=t,beta=2)))
    boundaries=[float(np.quantile(pval,.5)),float(np.quantile(pval,.85))]
    # Train a compact explanatory surrogate on training predictions, report agreement on validation.
    pfit=calibrated(model,prep,calibrator,x.iloc[fit])
    rule_model,rule_prep,rule_features,rules=derive_rules(
        x.iloc[fit],pfit,x.iloc[val],pval,boundaries)
    save_json(ROOT/'reports/rules.json',rules)
    ptest=calibrated(model,prep,calibrator,x.iloc[test]); rawtest=model.predict_proba(prep.transform(x.iloc[test]))[:,1]
    result=dict(split_counts=dict(train=len(fit),calibration=len(cal),validation=len(val),test=len(test)),
      seeds=[42,43,44],baseline_validation=metrics(y.iloc[val],bp,.5),
      candidates_validation=[dict(class_weight=a[5],metrics=a[4]) for a in candidates],
      selected_class_weight=weight,validation=metrics(y.iloc[val],pval,threshold),test=metrics(y.iloc[test],ptest,threshold),
      test_uncalibrated_brier=float(brier_score_loss(y.iloc[test],rawtest)),
      test_prevalence=float(y.iloc[test].mean()),risk_boundaries=boundaries,
      threshold_rationale='Maximize F2 on validation; recall weighted more than precision. No business cost model supplied.',
      bands_rationale='Validation probability quantiles: bottom 50% Low, next 35% Medium, top 15% High. Relative risk segments, not policy.',
      training_seconds=time.time()-start,python=platform.python_version(),sklearn=sklearn.__version__,
      dataset_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest())
    save_json(ROOT/'reports/metrics.json',result)
    for name,plotter in [('ROC',roc_curve),('Precision_Recall',precision_recall_curve)]:
        fig,ax=plt.subplots(figsize=(7,4)); a,b,_=plotter(y.iloc[test],ptest)
        if name=='ROC': ax.plot(a,b,color='#147d92'); ax.plot([0,1],[0,1],'--',color='gray'); ax.set(xlabel='False positive rate',ylabel='True positive rate')
        else: ax.plot(b,a,color='#147d92'); ax.axhline(y.iloc[test].mean(),ls='--',color='gray'); ax.set(xlabel='Recall',ylabel='Precision')
        ax.set_title(name.replace('_',' ')+' | untouched test set'); fig.tight_layout();fig.savefig(ROOT/f'reports/figures/{name}.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); obs,pred=calibration_curve(y.iloc[test],ptest,n_bins=10,strategy='quantile');ax.plot(pred,obs,'o-',color='#147d92');ax.plot([0,.5],[0,.5],'--',color='gray');ax.set(xlabel='Mean predicted risk',ylabel='Observed difficulty rate',title='Calibration | test set');fig.tight_layout();fig.savefig(ROOT/'reports/figures/Calibration.png');plt.close(fig)
    # Store only training-distribution statistics for local explanations, not raw training records.
    background=x.iloc[fit].sample(1000,random_state=45)
    stats={c:dict(median=float(background[c].median()),std=float(background[c].std() or 1),low=float(background[c].min()),high=float(background[c].max())) for c in num}
    category_values={c:background[c].value_counts(normalize=True).to_dict() for c in CAT}
    bundle=dict(model=model,prep=prep,calibration=calibrator,threshold=threshold,boundaries=boundaries,
      features=list(x.columns),numeric=num,category=CAT,stats=stats,categories=category_values,
      rule_model=rule_model,rule_prep=rule_prep,rule_features=rule_features)
    joblib.dump(bundle,ROOT/'models/risk_model.joblib',compress=3)
    joblib.dump(baseline,ROOT/'models/baseline.joblib',compress=3)
    assignments=pd.DataFrame({'SK_ID_CURR':d.SK_ID_CURR,'split':'train'})
    for group,idx in [('calibration',cal),('validation',val),('test',test)]:assignments.loc[idx,'split']=group
    assignments.to_csv(ROOT/'runtime/split_assignments.csv',index=False)
    analytics=pd.DataFrame({'applicant_id':d.SK_ID_CURR,'payment_difficulty':y,'income':d.AMT_INCOME_TOTAL,
      'credit':d.AMT_CREDIT,'annuity':d.AMT_ANNUITY,'age':-d.DAYS_BIRTH/365.25,
      'income_type':d.NAME_INCOME_TYPE,'education':d.NAME_EDUCATION_TYPE,'housing':d.NAME_HOUSING_TYPE,
      'contract_type':d.NAME_CONTRACT_TYPE,'occupation':d.OCCUPATION_TYPE.fillna('Unknown'),
      'credit_income_ratio':d.AMT_CREDIT/d.AMT_INCOME_TOTAL,'split':assignments['split']})
    with sqlite3.connect(ROOT/'runtime/analytics.db') as con:
        analytics.to_sql('applicants',con,if_exists='replace',index=False)
        con.execute('CREATE UNIQUE INDEX IF NOT EXISTS applicant_idx ON applicants(applicant_id)')
        con.execute('CREATE INDEX IF NOT EXISTS split_idx ON applicants(split)')
    print('COMPLETE',json.dumps(result['test']), 'seconds',time.time()-start,flush=True)

if __name__=='__main__':main()
