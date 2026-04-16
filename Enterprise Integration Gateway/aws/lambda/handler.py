"""
AWS Lambda handler for the Enterprise Integration Gateway.

Uses Mangum to wrap the FastAPI application for execution behind
AWS API Gateway (REST or HTTP API) or an Application Load Balancer.

This enables serverless deployment for:
  - Lightweight / burst traffic patterns
  - Scheduled sync jobs via EventBridge rules
  - Cost-optimized environments with low request volume

Usage:
  - Deploy via AWS SAM (see template.yaml)
  - Or package as a Lambda layer / container image

Note: APScheduler is disabled in Lambda (SCHEDULER_ENABLED=false).
      Use EventBridge scheduled rules to trigger sync endpoints instead.
"""
import os

# Force Lambda-appropriate defaults before importing the app
os.environ.setdefault("SCHEDULER_ENABLED", "false")
os.environ.setdefault("LOG_FORMAT", "json")
os.environ.setdefault("LOG_LEVEL", "INFO")

from mangum import Mangum  # noqa: E402

from app.main import app  # noqa: E402

# Create the Lambda handler
handler = Mangum(app, lifespan="auto")
