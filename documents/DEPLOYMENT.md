# Deploying the actual project

The employer requires a deployed URL. This package has no public deployment yet. Docker files are present; Docker startup must first be verified on the laptop.

## Host requirements

Choose a host that supports a long-running Docker web service, outbound HTTPS for the model API, WebSockets for Streamlit, and writable runtime storage. Start with around 2 GB RAM as a practical trial allocation and watch actual usage; bootstrapping and SHAP may require tuning. A static website host will not execute this Python app.

Build from the included Dockerfile. Route the host's public HTTPS service to port 8501, or set `PORT` to the port required by the host. Health endpoint: `/_stcore/health`. Startup may take longer than a normal static app because CSV conversion occurs first. The container runs as uid 10001; its runtime directory must remain writable.

## Data and secrets

- Transfer `application_train.csv` through the host's private file or volume mechanism and mount it read-only at `/app/data/application_train.csv`, or set `DATA_PATH` to its private absolute path. Never commit it to Git.
- Set `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL` in the host's secret settings. Do not bake them into an image or commit `.env`.
- Optionally set `APP_ACCESS_PASSWORD` and give the evaluator access privately through the submission process. This lightweight gate is for a demonstration, not full production authentication.
- Do not publish raw data or a runtime SQLite database in Git. Only the code, reports and trusted model artifacts belong in the submitted source archive.

## Acceptance checks

1. Public HTTPS URL opens from a fresh browser session and remains reachable after restart.
2. Every UI section opens. Assess a held-out profile and confirm probability, band and explanation appear.
3. Five SQL examples execute. Five real natural-language questions pass the live evaluation script and are manually inspected for semantic correctness.
4. A follow-up question preserves the intended context. Requests for unavailable installment histories get a clarification rather than invented results.
5. A destructive SQL request is refused, API failures are readable, and no secret is displayed in logs or screenshots.
6. Capture the deployed app outputs in the presentation and record the URL in the final form.

Host-specific account creation, billing and secret entry must happen in the user's own account. A permission message alone does not give this chat access to that account or to Antigravity on the laptop.
