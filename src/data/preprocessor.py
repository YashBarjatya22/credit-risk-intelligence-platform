"""Deterministic features. Learned transforms are fitted on training rows only."""
import numpy as np
import pandas as pd

NUMERIC = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE',
 'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH',
 'CNT_CHILDREN','CNT_FAM_MEMBERS','REGION_POPULATION_RELATIVE',
 'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_LAST_PHONE_CHANGE',
 'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
 'REG_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY','LIVE_CITY_NOT_WORK_CITY',
 'FLAG_OWN_CAR','FLAG_OWN_REALTY','OBS_30_CNT_SOCIAL_CIRCLE',
 'DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE',
 'AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_QRT']
CAT = ['NAME_CONTRACT_TYPE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE']
RAW = NUMERIC + CAT

def features(raw):
    x = raw.reindex(columns=RAW).copy()
    for c in ['FLAG_OWN_CAR','FLAG_OWN_REALTY']:
        x[c] = x[c].map(lambda v: 1 if v=='Y' else (0 if v=='N' else v))
    for c in NUMERIC:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    x['EMPLOYMENT_UNKNOWN'] = (x.DAYS_EMPLOYED == 365243).astype(float)
    x['DAYS_EMPLOYED'] = x.DAYS_EMPLOYED.mask(x.DAYS_EMPLOYED == 365243)
    x['AGE_YEARS'] = -x.DAYS_BIRTH / 365.25
    x['EMPLOYED_YEARS'] = -x.DAYS_EMPLOYED / 365.25
    for name, a, b in [('CREDIT_INCOME_RATIO','AMT_CREDIT','AMT_INCOME_TOTAL'),
                       ('ANNUITY_INCOME_RATIO','AMT_ANNUITY','AMT_INCOME_TOTAL'),
                       ('CREDIT_ANNUITY_RATIO','AMT_CREDIT','AMT_ANNUITY'),
                       ('CREDIT_GOODS_RATIO','AMT_CREDIT','AMT_GOODS_PRICE')]:
        x[name] = x[a] / x[b].where(x[b] > 0)
    x['EXT_SOURCE_MEAN'] = x[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1)
    for c in CAT:
        x[c] = x[c].fillna('Unknown').astype(str)
    return x.replace([np.inf,-np.inf],np.nan)

def risk_band(probability, boundaries):
    p = np.asarray(probability)
    return np.where(p < boundaries[0], 'Low', np.where(p < boundaries[1], 'Medium', 'High'))
