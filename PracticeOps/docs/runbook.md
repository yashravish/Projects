# PracticeOps Operations Runbook

This document provides operational guidance for debugging common issues, handling failures, and maintaining the PracticeOps API.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Debugging Authentication Issues](#debugging-authentication-issues)
3. [Debugging Email Delivery](#debugging-email-delivery)
4. [Database Connectivity](#database-connectivity)
5. [Common Operational Errors](#common-operational-errors)
6. [Health Monitoring](#health-monitoring)
7. [Log Analysis](#log-analysis)

---

## Quick Reference

### Service URLs

| Environment | API URL | Health Check |
|-------------|---------|--------------|
| Development | `http://localhost:8000` | `GET /health` |
| Production | `https://api.practiceops.app` | `GET /health` |

### Key Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `JWT_SECRET_KEY` | Secret for JWT signing | Yes |
| `CORS_ORIGINS` | Allowed frontend origins | Yes |
| `SMTP_HOST` | SMTP server host | No (uses console in dev) |
| `SMTP_USER` | SMTP username | No |
| `SMTP_PASS` | SMTP password | No |

### Health Check Response

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",      // or "degraded"
  "db": "ok"           // or "error"
}
```

---

## Debugging Authentication Issues

### Symptom: User can't log in

**Check 1: Verify credentials are correct**
```sql
-- Check if user exists
SELECT id, email, display_name, created_at
FROM users
WHERE email = 'user@example.com';
```

**Check 2: Look for rate limiting**
```bash
# Search logs for rate limit events
grep "rate_limit_exceeded" /var/log/practiceops.log | tail -20
```

**Check 3: Verify JWT configuration**
```bash
# Ensure JWT_SECRET_KEY is set
echo $JWT_SECRET_KEY | wc -c  # Should be > 32 characters
```

### Symptom: Token expired immediately

**Cause**: System clock skew between servers

**Fix**: Synchronize NTP
```bash
sudo ntpdate -s time.nist.gov
```

### Symptom: 401 on valid token

**Check 1: Token type confusion**
- Access tokens work for API calls
- Refresh tokens only work for `/auth/refresh`

**Check 2: Token decode**
```python
import jwt
token = "eyJ..."
decoded = jwt.decode(token, options={"verify_signature": False})
print(decoded)  # Check 'type', 'exp', 'sub'
```

**Check 3: User deleted**
```sql
SELECT id, email FROM users WHERE id = '<user_id_from_token>';
```

### Symptom: FORBIDDEN on authorized user

**Check 1: Team membership**
```sql
SELECT tm.*, u.email
FROM team_memberships tm
JOIN users u ON u.id = tm.user_id
WHERE tm.team_id = '<team_id>'
  AND tm.user_id = '<user_id>';
```

**Check 2: Role verification**
```sql
SELECT role, section
FROM team_memberships
WHERE team_id = '<team_id>'
  AND user_id = '<user_id>';
```

---

## Debugging Email Delivery

### Symptom: Emails not sending

**Check 1: SMTP configuration**
```bash
# Verify SMTP is configured
echo "SMTP_HOST: $SMTP_HOST"
echo "SMTP_USER: $SMTP_USER"
echo "SMTP_FROM: $SMTP_FROM"
```

**Check 2: Console provider active**

If SMTP is not configured, emails are logged to console instead of sent.
```bash
grep "Simulating email send" /var/log/practiceops.log
```

**Check 3: Scheduler job status**
```bash
curl -X GET http://localhost:8000/admin/jobs \
  -H "Authorization: Bearer <admin_token>"
```

### Symptom: Job failures

**Check 1: View job logs**
```bash
grep "job_failed" /var/log/practiceops.log | tail -20
```

**Check 2: Manually trigger job**
```bash
curl -X POST http://localhost:8000/admin/jobs/no_log_reminder/run \
  -H "Authorization: Bearer <admin_token>"
```

**Check 3: Verify notification preferences**
```sql
SELECT np.*, u.email
FROM notification_preferences np
JOIN users u ON u.id = np.user_id
WHERE np.email_enabled = true
  AND np.team_id = '<team_id>';
```

### Symptom: Wrong recipients

**Check 1: User email addresses**
```sql
SELECT id, email, display_name
FROM users
WHERE email IS NULL OR email = '';
```

**Check 2: Notification preferences**
```sql
SELECT user_id, email_enabled, no_log_days, weekly_digest_enabled
FROM notification_preferences
WHERE team_id = '<team_id>';
```

---

## Database Connectivity

### Symptom: Health check shows `"db": "error"`

**Check 1: Database is running**
```bash
docker ps | grep postgres
# or
systemctl status postgresql
```

**Check 2: Connection string**
```bash
# Verify DATABASE_URL format
# postgresql+asyncpg://user:pass@host:port/dbname
echo $DATABASE_URL
```

**Check 3: Network connectivity**
```bash
pg_isready -h localhost -p 5433 -d practiceops
```

**Check 4: Credentials**
```bash
psql "$DATABASE_URL" -c "SELECT 1"
```

### Symptom: Connection pool exhausted

**Cause**: Too many concurrent connections

**Check**:
```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'practiceops';
```

**Fix**: Restart the API service to reset connections

### Symptom: Slow queries

**Check 1: Missing indexes**
```sql
-- List indexes on tickets table
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'tickets';
```

**Check 2: Long-running queries**
```sql
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE state = 'active'
  AND datname = 'practiceops'
ORDER BY duration DESC;
```

### Running Migrations

```bash
cd apps/api
alembic upgrade head
```

**Rollback one migration**:
```bash
alembic downgrade -1
```

---

## Common Operational Errors

### Error: `CONFLICT - Email already registered`

**Cause**: User trying to register with existing email

**Resolution**: User should use `/auth/login` instead

### Error: `VALIDATION_ERROR - Invalid transition`

**Cause**: Attempting invalid ticket status transition

**Valid transitions**:
```
OPEN -> IN_PROGRESS
IN_PROGRESS -> OPEN, BLOCKED, RESOLVED
BLOCKED -> IN_PROGRESS, RESOLVED
RESOLVED -> (only VERIFIED via /verify)
VERIFIED -> (terminal, no transitions)
```

### Error: `CONFLICT - Ticket already claimed`

**Cause**: Race condition when multiple users claim same ticket

**Resolution**: Expected behavior, user should refresh and try another

### Error: `RATE_LIMITED`

**Cause**: Too many authentication attempts

**Resolution**: Wait 60 seconds, or investigate potential attack

```bash
# Find the IP making requests
grep "rate_limit_exceeded" /var/log/practiceops.log | \
  grep -oE 'client_ip":"[^"]+' | sort | uniq -c | sort -rn
```

---

## Health Monitoring

### Basic Health Check

```bash
# Simple availability check
curl -s http://localhost:8000/health | jq .
```

### Monitoring Script

```bash
#!/bin/bash
HEALTH=$(curl -s http://localhost:8000/health)
STATUS=$(echo $HEALTH | jq -r '.status')

if [ "$STATUS" != "ok" ]; then
    echo "ALERT: PracticeOps unhealthy"
    echo $HEALTH
    # Send alert to PagerDuty/Slack/etc.
fi
```

### Key Metrics to Monitor

1. **Response time** - `/health` should respond in < 100ms
2. **Error rate** - Monitor 4xx and 5xx responses
3. **Database connections** - Watch for pool exhaustion
4. **Rate limit hits** - May indicate attacks

---

## Log Analysis

### Log Format

Logs are structured JSON for easy parsing:

```json
{
  "timestamp": "2026-01-02T10:30:00.000Z",
  "level": "info",
  "event": "request_completed",
  "request_id": "abc-123-def",
  "method": "POST",
  "path": "/auth/login",
  "status_code": 200
}
```

### Useful Queries

**Find all errors**:
```bash
grep '"level":"error"' /var/log/practiceops.log
```

**Find requests by ID**:
```bash
grep '"request_id":"abc-123"' /var/log/practiceops.log
```

**Find slow requests** (if timing is logged):
```bash
jq 'select(.duration_ms > 1000)' /var/log/practiceops.log
```

**Find auth failures**:
```bash
grep 'UNAUTHORIZED\|FORBIDDEN' /var/log/practiceops.log | tail -50
```

**Find rate limit events**:
```bash
grep 'rate_limit_exceeded' /var/log/practiceops.log
```

### Request Tracing

Every request gets a unique `request_id`:

1. Find the failing request's ID from client or response header
2. Search logs for all entries with that ID
3. Follow the complete request lifecycle

```bash
REQUEST_ID="abc-123-def"
grep "$REQUEST_ID" /var/log/practiceops.log | jq .
```

---

## Emergency Procedures

### API Not Responding

1. Check if process is running:
   ```bash
   ps aux | grep uvicorn
   ```

2. Check logs for crash:
   ```bash
   tail -100 /var/log/practiceops.log
   ```

3. Restart service:
   ```bash
   docker-compose restart api
   # or
   systemctl restart practiceops
   ```

### Database Emergency

1. If corrupted, restore from backup:
   ```bash
   pg_restore -d practiceops /backups/latest.dump
   ```

2. If slow, kill long queries:
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'practiceops'
     AND state = 'active'
     AND query_start < now() - interval '5 minutes';
   ```

### Rate Limit Bypass (Emergency)

If legitimate users are being blocked, temporarily disable rate limiting:

```python
# In app/core/middleware.py
AUTH_RATE_LIMIT = "1000/minute"  # Effectively disabled
```

**Important**: Re-enable normal limits after investigation.

---

## Contact Information

For escalation:
- **On-call Engineer**: Check PagerDuty rotation
- **Database Issues**: Contact DBA team
- **Security Incidents**: Contact security@practiceops.app

