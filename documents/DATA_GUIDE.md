# Data coverage and business meaning

Source: user-supplied Home Credit `application_train.csv` and `HomeCredit_columns_description.csv`; dictionary read with Latin-1 encoding.

| Category | Example columns | Interpretation / quality considerations |
|---|---|---|
| Demographics | DAYS_BIRTH, CNT_CHILDREN, NAME_FAMILY_STATUS, NAME_EDUCATION_TYPE | Age, household and education; potential fairness implications |
| Financials | AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE | Income, requested credit, annuity and goods; unspecified dataset currency |
| Employment | DAYS_EMPLOYED, OCCUPATION_TYPE, ORGANIZATION_TYPE | Employment duration/type; 365243 is a sentinel |
| External credit indicators | EXT_SOURCE_1/2/3, AMT_REQ_CREDIT_BUREAU_* | Normalized external scores and recent bureau inquiry counts; not full histories |
| Social-circle indicators | OBS_*_CNT_SOCIAL_CIRCLE, DEF_*_CNT_SOCIAL_CIRCLE | Observed contacts and their delinquency indicators; not the applicant's installment ledger |
| Housing and geography | NAME_HOUSING_TYPE, REGION_*, APARTMENTS_*, FLOORSMAX_* | Living situation and neighborhood; many building fields are mostly missing |
| Contact and documents | FLAG_PHONE, FLAG_EMAIL, FLAG_DOCUMENT_* | Application metadata; most are excluded from the small initial model |
| Target and identifier | TARGET, SK_ID_CURR | Outcome for learning and unique ID; neither enters model features |

`reports/feature_catalog.json` records all source columns, observed dtypes, development missingness and whether the model uses each field. The app shows six quantified associations and group sizes; they must not be interpreted as causal or as approved policy.

## Missing tables and a sound extension

Additional `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `installments_payments.csv`, `POS_CASH_balance.csv` and `credit_card_balance.csv` would enable deeper historical analysis. Aggregate each one to one row per current applicant before joining, verify one-to-one joins, define what information is available at application time, and avoid aggregating future outcomes. Examples include prior delinquency frequency, payment lateness, active debt and utilization. Retain the exact applicant split and compare improvements only on development/validation data before a fresh final evaluation.

The current solution does not invent these unavailable signals. It uses the application-level proxies explicitly labelled above.
