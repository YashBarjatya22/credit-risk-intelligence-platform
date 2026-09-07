# Deployment

**Live application:** https://credit-risk-intelligence-platform-1.onrender.com

CreditScope runs as a Docker web service on Render. The service listens on the host-provided `PORT` value, with port 8501 used locally. Streamlit's health endpoint is `/_stcore/health`, and the container runs as the non-root user `appuser` (uid 10001).

## Run locally with Docker Compose

1. Place `application_train.csv` at `data/application_train.csv`.
2. Copy `.env.example` to `.env` and add the optional LLM settings if Talk to Data is required.
3. From the project root, run:

```sh
docker compose up --build
```

4. Open `http://localhost:8501`.
5. Check container health with `docker compose ps`. Stop the service with `docker compose down`.

The startup command runs `bootstrap.py` before Streamlit. When the CSV is available, the script creates a local read-only analytics database and held-out profiles. Saved model and report views remain available when the CSV is not mounted.

## Data and secrets

- The raw Home Credit CSV is intentionally excluded from Git. Mount it read-only at `/app/data/application_train.csv`, or set `DATA_PATH` to its private location.
- Store `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL` in the hosting provider's environment settings.
- `APP_ACCESS_PASSWORD` can be set when a lightweight access gate is useful.
- Do not commit `.env`, the runtime SQLite database, raw applicant data, or API credentials.

The current public free service does not include the private CSV. It therefore serves the saved dashboards and manual risk-assessment workflow, while held-out profiles and database-backed questions require a deployment with the dataset mounted.

## Deployment checks

- The public HTTPS URL opens successfully after a cold start.
- All six application sections render.
- A manual scenario returns a probability, relative risk band, and SHAP explanation.
- With the private CSV mounted, the five reference SQL examples execute against the read-only database.
- With LLM settings configured, `evaluate_live_chat.py` checks the five reference questions.
- Unsafe SQL is rejected, provider errors are readable, and no secret appears in logs or screenshots.
