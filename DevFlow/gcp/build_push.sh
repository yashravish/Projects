#!/usr/bin/env bash
# Example: build & push to Artifact Registry
#  PROJECT_ID=your-gcp-id REGION=us-central1 REPO=devflow bash gcp/build_push.sh
set -euo pipefail
: "${PROJECT_ID:?}"
: "${REGION:=us-central1}"
: "${REPO:=devflow}"
: "${REGISTRY:=${REGION}-docker.pkg.dev}"
IMAGE="${REGISTRY}/${PROJECT_ID}/${REPO}/devflow-api:${TAG:-latest}"
echo "Building ${IMAGE} ..."
cd "$(dirname "$0")/.."
docker build -f backend/Dockerfile -t "${IMAGE}" backend/
docker push "${IMAGE}"
echo "Pushed: ${IMAGE}"
