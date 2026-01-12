# Railway Deployment Guide

## IMPORTANT: Fix for Failed Build

This is a **pnpm monorepo** and Railway must deploy from the **monorepo root**, not from `apps/api`.

**You MUST configure the Root Directory in Railway:**

1. Go to your Railway dashboard: `railway open`
2. Click on your service
3. Go to **Settings** tab
4. Find **Root Directory** setting
5. Set it to:
   - If your GitHub repo root is `Projects/`: `Signal Desk Lite`
   - If your GitHub repo root is `Signal Desk Lite/`: leave it **empty** or set to `.`
6. **IMPORTANT**: Ensure **Builder** is set to **NIXPACKS** (not Railpack)
7. Click **Save**
8. Redeploy: Click **Deploy** or run `railway up`

## Why This Matters

- The app uses `workspace:*` dependencies that only work with pnpm
- Railway must use **Nixpacks** (not Railpack) to detect pnpm correctly
- The build must run from the monorepo root to access all workspace packages

## Quick Start

1. Open your terminal where you're logged in to Railway
2. Navigate to the **monorepo root** (not apps/api):
   ```bash
   cd "c:\Users\yashr\Documents\GitHub\Projects\Signal Desk Lite"
   ```
3. Deploy from here:
   ```bash
   railway up
   ```

## Manual Deployment Steps

If you prefer to run commands manually:

### 1. Initialize Railway Project

```bash
cd "c:\Users\yashr\Documents\GitHub\Projects\Signal Desk Lite"
railway init
```

This will:
- Create a new Railway project
- Link your local directory to the project

### 2. Set Environment Variables

You can set variables via CLI or Railway dashboard:

#### Option A: Via CLI
```bash
railway variables set DATABASE_URL="your-neon-connection-string"
railway variables set DIRECT_URL="your-neon-direct-connection-string"
railway variables set REDIS_URL="your-upstash-redis-url"
railway variables set UPSTASH_REDIS_REST_URL="your-upstash-rest-url"
railway variables set UPSTASH_REDIS_REST_TOKEN="your-upstash-rest-token"
railway variables set JWT_SECRET="your-jwt-secret"
railway variables set OPENAI_API_KEY="your-openai-key"
railway variables set PORT="3001"
railway variables set API_HOST="0.0.0.0"
railway variables set NODE_ENV="production"
```

#### Option B: Via Dashboard
```bash
railway open
```
Then add the variables in the Variables tab.

### 3. Deploy

```bash
railway up
```

This will:
- Upload your code
- Build using the nixpacks.toml configuration
- Deploy the API

### 4. Add a Domain

```bash
railway domain
```

This generates a Railway public domain for your API.

### 5. Check Status

```bash
railway status
railway logs
```

## Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | Neon PostgreSQL connection (pooled) | `postgresql://user:pass@host/db?sslmode=require` |
| `DIRECT_URL` | Neon PostgreSQL direct connection | `postgresql://user:pass@host/db?sslmode=require` |
| `REDIS_URL` | Upstash Redis connection | `rediss://default:xxx@xxx.upstash.io:6379` |
| `UPSTASH_REDIS_REST_URL` | Upstash REST API URL | `https://xxx.upstash.io` |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash REST API token | `your-token` |
| `JWT_SECRET` | Secret for JWT signing | `your-256-bit-secret` |
| `OPENAI_API_KEY` | OpenAI API key (optional) | `sk-...` |
| `PORT` | Server port | `3001` |
| `API_HOST` | Server host | `0.0.0.0` |
| `NODE_ENV` | Environment | `production` |

## Database Migrations

After first deployment, run migrations:

```bash
railway run npx prisma migrate deploy
```

Or connect to the service and run:
```bash
railway shell
npx prisma migrate deploy
exit
```

## Troubleshooting

### Build Fails
- Check Railway logs: `railway logs`
- Verify nixpacks.toml is correct
- Ensure all dependencies are in package.json

### Database Connection Issues
- Verify DATABASE_URL and DIRECT_URL are correct
- Check that pgvector extension is enabled in Neon
- Run migrations: `railway run npx prisma migrate deploy`

### Redis Connection Issues
- Verify all Upstash credentials are set
- Check REDIS_URL format includes `rediss://` (with double 's')

### Port Issues
- Railway automatically sets PORT - ensure your app uses `process.env.PORT`
- Verify API_HOST is set to `0.0.0.0` (not `localhost`)

## Useful Commands

```bash
# View service URL
railway domain

# View environment variables
railway variables

# View logs
railway logs

# Open Railway dashboard
railway open

# SSH into service
railway shell

# Redeploy
railway up

# Check deployment status
railway status
```

## Next Steps

1. Get your Railway API URL: `railway domain`
2. Update Vercel with the API URL as `NEXT_PUBLIC_API_URL`
3. Test the API endpoints
4. Monitor logs for any issues

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
