# Railway Deployment Setup Guide

## Required Environment Variables

Your Railway project needs these environment variables configured for the API to work properly.

### 1. Database (REQUIRED)
```
DATABASE_URL=${{Postgres.DATABASE_URL}}
```
**Important**: Use Railway's variable reference syntax `${{Postgres.DATABASE_URL}}` to automatically link to your Postgres database.

### 2. Application Settings (REQUIRED)
```
ENVIRONMENT=production
```

### 3. Security (REQUIRED)
```
JWT_SECRET_KEY=<your-secret-key-here>
```
Generate a strong secret key:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
**NEVER** use the same key from .env - generate a unique production key!

### 4. Frontend URL (REQUIRED)
```
FRONTEND_URL=https://your-frontend.railway.app
```
Replace with your actual frontend Railway URL for invite links to work correctly.

### 5. CORS Origins (REQUIRED)
```
CORS_ORIGINS=https://your-frontend.railway.app,https://your-frontend-pr-*.railway.app
```
Comma-separated list of allowed origins. Include your production domain and preview domains.

### 6. Email Configuration (OPTIONAL - for notifications)

#### Option A: Resend API (Recommended)
```
RESEND_API_KEY=re_...
SMTP_FROM=noreply@yourdomain.com
```
Get your API key from https://resend.com

#### Option B: Traditional SMTP
```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
SMTP_FROM=noreply@yourdomain.com
```

### 7. OpenAI Integration (OPTIONAL - for AI summaries)
```
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o-mini
```

## How to Set Environment Variables in Railway

1. Go to your Railway project dashboard
2. Click on your API service
3. Go to the "Variables" tab
4. Click "New Variable"
5. Add each variable from the list above
6. For `DATABASE_URL`, use the reference syntax: `${{Postgres.DATABASE_URL}}`
7. Click "Deploy" to restart with new variables

## Troubleshooting

### Health Check Failing
If your deployment shows "service unavailable" errors:

1. **Check DATABASE_URL** - Ensure it's properly linked: `${{Postgres.DATABASE_URL}}`
2. **Verify Postgres is running** - Your Postgres service should be healthy
3. **Check logs** - View Railway logs for startup errors
4. **Verify all REQUIRED variables are set** - Missing variables will cause startup failures

### Database Connection Errors
- Railway's internal DNS `postgres.railway.internal` only works between Railway services
- Make sure your Postgres service is in the same project
- Use the variable reference syntax, not a hardcoded URL

### CORS Errors
- Update `CORS_ORIGINS` to include your actual frontend domains
- Include both production and preview URLs
- Format: comma-separated, no spaces, full URLs with https://

## Port Configuration

Railway automatically provides a `PORT` environment variable. The Dockerfile has been updated to use this automatically - you don't need to set it manually.

## Verification

After setting variables and deploying:

1. Check deployment logs for startup errors
2. Visit `https://your-api.railway.app/health` - should return:
   ```json
   {"status": "ok", "db": "ok"}
   ```
3. Visit `https://your-api.railway.app/docs` - should show API documentation
4. Test a login from your frontend

## Next Steps

1. Set all REQUIRED environment variables above
2. Deploy your application
3. Test the health endpoint
4. Configure optional features (email, OpenAI) as needed
