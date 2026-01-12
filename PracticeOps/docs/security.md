# PracticeOps Security Documentation

This document describes the security architecture, authentication flow, RBAC model, rate limiting, and CORS configuration for the PracticeOps API.

## Table of Contents

1. [Authentication Flow](#authentication-flow)
2. [RBAC Model](#rbac-model)
3. [Rate Limiting](#rate-limiting)
4. [CORS Configuration](#cors-configuration)
5. [Error Handling](#error-handling)
6. [Security Best Practices](#security-best-practices)

---

## Authentication Flow

### Overview

PracticeOps uses JWT (JSON Web Tokens) for authentication with a two-token system:

- **Access Token**: Short-lived (15 minutes), used for API authorization
- **Refresh Token**: Long-lived (30 days), used to obtain new access tokens

### Token Flow

```
1. User registers/logs in
   POST /auth/register or POST /auth/login
   └── Returns: { access_token, refresh_token, user }

2. Client stores tokens (localStorage)

3. API requests include access token
   Authorization: Bearer <access_token>

4. On 401 (token expired), client uses refresh token
   POST /auth/refresh { refresh_token }
   └── Returns: { access_token }

5. If refresh fails, user must re-authenticate
```

### Endpoints

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------|
| `/auth/register` | POST | No | Create new user account |
| `/auth/login` | POST | No | Authenticate and get tokens |
| `/auth/refresh` | POST | No | Get new access token |
| `/me` | GET | Yes | Get current user info |

### Password Security

- Passwords are hashed using **bcrypt** via passlib
- Minimum complexity is enforced at the client level
- Password hashes are never exposed in responses

### Token Security

- Tokens are signed using **HS256** algorithm
- Secret key is loaded from environment variable `JWT_SECRET_KEY`
- Tokens include `type` claim to prevent access/refresh token confusion
- Tokens include `exp` (expiration) and `sub` (user ID) claims

---

## RBAC Model

### Roles

PracticeOps implements three roles with increasing privileges:

| Role | Description |
|------|-------------|
| `MEMBER` | Regular team member, can log practice, create tickets |
| `SECTION_LEADER` | Leads a specific section, can manage section members |
| `ADMIN` | Full team administration privileges |

### Role Hierarchy

```
ADMIN
  └── Full access to all team resources
  └── Can manage all members
  └── Can create/modify all assignments
  └── Can verify any ticket
  └── Access to leader dashboard

SECTION_LEADER
  └── Access limited to their section
  └── Can verify tickets in their section
  └── Can create section-scoped assignments
  └── Access to leader dashboard (section-scoped)

MEMBER
  └── Personal resources only
  └── Can log practice
  └── Can create/manage own tickets
  └── Access to member dashboard
```

### RBAC Dependencies

All RBAC checks are implemented via FastAPI dependencies:

```python
# Require authentication
CurrentUser = Annotated[User, Depends(get_current_user)]

# Require team membership
require_membership(team_id)

# Require specific role(s)
require_role(team_id, [Role.ADMIN, Role.SECTION_LEADER])

# Require section leader of specific section
require_section_leader_of_section(team_id, section)
```

### Visibility Rules

Ticket and resource visibility is enforced at three levels:

| Visibility | Who Can See |
|------------|-------------|
| `PRIVATE` | Owner + Admin + Section Leader (if in owner's section) |
| `SECTION` | Members in that section + Section Leader + Admin |
| `TEAM` | All team members |

### Privacy Guarantee

Leader dashboards aggregate private tickets without exposing identifiers:
- `private_ticket_aggregates` contains only: section, category, status, priority, song_ref, due_bucket, count
- No ticket IDs, owner IDs, or user names are exposed
- Aggregates with count < 3 are suppressed to prevent identification

---

## Rate Limiting

### Implementation

Rate limiting is implemented using **slowapi** with in-memory storage.

### Protected Endpoints

| Endpoint | Limit | Window |
|----------|-------|--------|
| `/auth/login` | 10 requests | per minute per IP |
| `/auth/register` | 10 requests | per minute per IP |

### Response on Limit Exceeded

```json
HTTP/1.1 429 Too Many Requests
Retry-After: 60

{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Too many requests. Please try again later.",
    "field": null
  }
}
```

### IP Detection

The rate limiter uses the following logic to detect client IP:
1. Check `X-Forwarded-For` header (for reverse proxy scenarios)
2. Fall back to direct client IP

### Configuration

Rate limits can be adjusted in `app/core/middleware.py`:

```python
AUTH_RATE_LIMIT = "10/minute"  # Format: "count/period"
```

---

## CORS Configuration

### Allowed Origins

CORS is configured via environment variable `CORS_ORIGINS`:

```bash
# Development (default)
CORS_ORIGINS=["http://localhost:5173", "http://127.0.0.1:5173"]

# Production
CORS_ORIGINS=["https://practiceops.app"]
```

### Policy

- **Origins**: Explicit list from environment (no wildcards in production)
- **Credentials**: Allowed (`allow_credentials=True`)
- **Methods**: All methods allowed (`["*"]`)
- **Headers**: All headers allowed (`["*"]`)

### Important

**Never use wildcard (`*`) origins in production** as it allows any website to make authenticated requests to the API.

---

## Error Handling

### Standard Error Format

All API errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "field": "optional_field_name"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Authentication required or failed |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Invalid input data |
| `CONFLICT` | 409 | Resource conflict (e.g., duplicate) |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL` | 500 | Internal server error |

### Security Considerations

- **No stack traces** are exposed in error responses
- **No internal class names** or SQL errors are revealed
- Error messages are user-safe and consistent
- Request IDs are included in logs for correlation

---

## Security Best Practices

### For Operators

1. **Set a strong JWT secret**
   ```bash
   JWT_SECRET_KEY=$(openssl rand -base64 32)
   ```

2. **Use HTTPS in production**
   - All API traffic should be encrypted
   - Set `Secure` flag on cookies if used

3. **Restrict CORS origins**
   - Only include your frontend domain(s)
   - Never use wildcards in production

4. **Monitor rate limit violations**
   - Rate limit exceeded events are logged
   - Watch for brute force patterns

5. **Rotate secrets periodically**
   - JWT secret rotation requires user re-authentication
   - Plan for token invalidation

### For Developers

1. **Always use RBAC dependencies**
   - Never implement inline role checks
   - Use `require_role()`, `require_membership()`, etc.

2. **Validate all inputs**
   - Use Pydantic models with strict validation
   - Enums are validated at the Pydantic layer

3. **Never expose sensitive data**
   - No password hashes in responses
   - No internal IDs in aggregates
   - No stack traces to clients

4. **Log security events**
   - Authentication failures
   - Authorization failures
   - Rate limit violations

---

## Request Tracing

### Request ID

Every request is assigned a unique `request_id` (UUID v4):

- Attached to request context
- Included in all log entries
- Returned in `X-Request-ID` response header

### Correlation

Use `request_id` to trace issues:

```bash
# Find all logs for a specific request
grep "request_id.*abc123" /var/log/practiceops.log
```

---

## Audit Trail

### Ticket Activity

All ticket changes are logged in `ticket_activity`:
- Status transitions
- Claims
- Verifications
- Comments

### What's Logged

- User ID performing action
- Timestamp
- Old/new status for transitions
- Content/notes

This provides a complete audit trail for compliance and debugging.

