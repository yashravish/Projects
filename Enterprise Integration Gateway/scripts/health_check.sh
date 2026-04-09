#!/usr/bin/env bash
# Quick health check script for all services
set -euo pipefail

APP_URL="${APP_URL:-http://localhost:8000}"
MOCK_URL="${MOCK_URL:-http://localhost:8001}"

check() {
    local name="$1"
    local url="$2"
    local response
    response=$(curl -sf --max-time 5 "$url" 2>/dev/null) && \
        echo "✓ $name — OK" || \
        echo "✗ $name — FAILED ($url)"
}

echo "=== Enterprise Integration Gateway Health Check ==="
check "Main App Health" "$APP_URL/api/v1/health"
check "Mock Providers"  "$MOCK_URL/health"
check "Metrics"         "$APP_URL/api/v1/metrics"
echo "==================================================="
