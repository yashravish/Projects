#!/usr/bin/env bash
set -eu

TOK=$(curl -sS -X POST -H "Content-Type: application/json" \
  -d '{"email":"seed-admin@example.gov","password":"ChangeMe!2026"}' \
  http://localhost:8000/api/v1/auth/login \
  | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

echo "=== submit training ==="
JOB=$(curl -sS -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
  -d '{"name":"psdi-cross-encoder-reranker","auto_promote":true}' \
  http://localhost:8000/api/v1/training/jobs)
echo "$JOB" | python -m json.tool | head -40
MID=$(echo "$JOB" | python -c 'import sys,json;print(json.load(sys.stdin)["registered_model_id"])')

echo "=== list models ==="
curl -sS -H "Authorization: Bearer $TOK" http://localhost:8000/api/v1/models | python -m json.tool | head -40

echo "=== predict (model_id=$MID) ==="
curl -sS -X POST -H "Authorization: Bearer $TOK" -H "Content-Type: application/json" \
  -d '{"query":"When is the FY26 grant deadline?","passages":["Applications must be submitted by February 28, 2026.","The vendor portal supports SSO."]}' \
  "http://localhost:8000/api/v1/models/$MID/predict" | python -m json.tool

echo "=== list training jobs ==="
curl -sS -H "Authorization: Bearer $TOK" http://localhost:8000/api/v1/training/jobs | python -m json.tool | head -30
