# Validation record

## Completed in the build environment

- Loaded the supplied 307,511 × 122 application table and verified unique applicant IDs and binary TARGET.
- Reproduced fixed training, calibration, validation and test splits.
- Trained the logistic baseline and two gradient-boosting candidates; selected by validation average precision.
- Evaluated the final calibrated model on 46,127 untouched test rows.
- Verified SHAP baseline plus contributions reconstructs the served calibrated probability within 1e-5.
- Executed all five reference SQL patterns against the full local analytics database and reconciled row counts.
- Tested rejection of mutations, joins, subqueries, CTEs, unknown columns, wildcard output and excess limits.
- Tested the independent SQLite authorizer, LLM request/history contract, input validation, sentinel handling and band boundaries.
- Rendered all Streamlit sections using Streamlit's application test interface and submitted a real manual prediction form.
- Rendered and visually inspected the ten-page presentation PDF.

Final suite: 13 tests passed. The textual run log is not shipped because local absolute paths and verbose framework logs do not help evaluators; the tests remain reproducible in `tests/`.

## Not completed in this environment

- Docker image build and Compose startup: Docker daemon unavailable here.
- Live LLM call: no user/provider API credential was supplied.
- Public deployment: no user hosting account or deployed URL was connected.
- Deployed UI screenshots: depend on the two checks above.

These are tracked as mandatory pre-submission gates rather than represented as successful results.
