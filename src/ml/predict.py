from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.special import expit
from src.data.preprocessor import features,risk_band

def load_model(path):return joblib.load(Path(path))

def probabilities(bundle,x):
    p=bundle['model'].predict_proba(bundle['prep'].transform(x))[:,1]
    p=np.clip(p,1e-7,1-1e-7)
    return bundle['calibration'].predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]

def predict(bundle,raw):
    x=features(raw)
    for c in ['AMT_CREDIT','AMT_INCOME_TOTAL','AMT_ANNUITY']:
        if (x[c].dropna()<=0).any():raise ValueError(c+' must be positive when provided.')
    if ((x.AGE_YEARS.dropna()<18)|(x.AGE_YEARS.dropna()>100)).any():raise ValueError('Age must be between 18 and 100 years.')
    for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']:
        if ((x[c].dropna()<0)|(x[c].dropna()>1)).any():raise ValueError(c+' must be between zero and one.')
    p=probabilities(bundle,x)
    return pd.DataFrame({'probability':p,'risk_score':p*100,'risk_band':risk_band(p,bundle['boundaries']),
                         'above_screening_threshold':p>=bundle['threshold']})

def explain(bundle,raw):
    import shap
    x=features(raw); z=bundle['prep'].transform(x)
    # Model-agnostic SHAP avoids unsupported native categorical tree split conversions.
    # Reference uses training-distribution summaries, not a stored applicant record.
    reference={c:bundle['stats'][c]['median'] for c in bundle['numeric']}
    reference.update({c:max(bundle['categories'][c],key=bundle['categories'][c].get) for c in bundle['category']})
    background=bundle['prep'].transform(pd.DataFrame([reference])[bundle['features']])
    def score(z):
        p=np.clip(bundle['model'].predict_proba(z)[:,1],1e-7,1-1e-7)
        return bundle['calibration'].predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
    names=bundle['prep'].get_feature_names_out().tolist()
    explainer=shap.Explainer(score,background,algorithm='permutation',feature_names=names,seed=42)
    explanation=explainer(z,max_evals=(2*z.shape[1]+1)*3,silent=True)
    contributions=explanation.values
    base=float(explanation.base_values[0])
    reconstructed=explanation.base_values+contributions.sum(axis=1)
    np.testing.assert_allclose(reconstructed,probabilities(bundle,x),atol=1e-5)
    return {'base_probability':base,'values':contributions,'feature_names':names,'reconstructed_probability':reconstructed}
