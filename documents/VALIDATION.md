# Validation record

## Data and modelling

- Loaded the 307,511 × 122 application table and verified unique applicant identifiers and a binary target.
- Reproduced fixed, stratified training, calibration, validation, and test splits.
- Trained a class-balanced logistic-regression baseline and two histogram-gradient-boosting configurations.
- Selected the final configuration using validation average precision, then evaluated it once on 46,127 untouched test rows.
- Verified that the SHAP baseline plus feature contributions reconstructs the calibrated probability within `1e-5`.
- Kept the target and applicant identifier out of the feature set and handled the `DAYS_EMPLOYED=365243` sentinel explicitly.

## Analytics and application

- Reconciled all five reference SQL patterns with the development-population counts in the EDA report.
- Tested rejection of mutations, joins, subqueries, CTEs, unknown columns, wildcard output, unsupported functions, and excessive limits.
- Tested the independent SQLite authorizer, LLM history contract, input validation, risk-band boundaries, and unseen categories.
- Rendered all six Streamlit sections with Streamlit's application-testing interface and submitted a manual assessment.
- Built and ran the application with Docker Compose; the container reported healthy.
- Ran a live Gemini-to-SQL query locally and confirmed the proposed SQL, returned rows, and provider token usage.
- Opened the public Render deployment successfully.
- Added screenshots of the working application to the presentation.

**Automated test result: 13 tests passed.**

## Reproducibility

From the project root:

```sh
docker compose up --build
docker compose exec credit-risk python -m unittest discover -s tests -v
```

The first command starts the same image used by the web service. The second command runs the model, SQL-safety, LLM-contract, and UI test suite inside the container.

## Scope

The public free deployment omits the raw CSV because applicant data should not be committed to a public repository. Saved dashboards and manual scoring remain available. Held-out examples and database-backed questions require the CSV to be mounted privately; live questions also require provider credentials.

This validation covers an application-data model and a random holdout. It is not temporal or external validation, and it does not replace a formal fairness, policy, or production-security review.
