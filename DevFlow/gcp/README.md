# Google Cloud (Cloud Run) quick notes

- **Image naming (Artifact Registry)**: `REGION-docker.pkg.dev/PROJECT/REPO/IMAGE:TAG`  
  Example: `us-central1-docker.pkg.dev/myproj/devflow/devflow-api:2026-01-01`

- `cloudbuild.yaml` runs `docker build` on `backend/Dockerfile`, pushes the image, and deploys to Cloud Run.

- Point `DATABASE_URL` at a Cloud SQL (PostgreSQL) instance over the Cloud SQL socket or IP (see `terraform/` for example wiring).

- Store `OPENAI_API_KEY` in **Secret Manager** and mount to Cloud Run with `--set-secrets`.

See the main `README.md` for the end-to-end DevFlow runbook.
