# PracticeOps Railway Deployment Guide

Railway is the easiest way to deploy PracticeOps. Everything (frontend, backend, database) runs on one platform with automatic HTTPS and GitHub integration.

## Why Railway?

- Deploy in under 15 minutes
- Built-in PostgreSQL database
- Automatic GitHub deployments
- Free tier: $5 credit/month (enough for small projects)
- No credit card required for trial
- Automatic HTTPS/SSL
- Simple environment variable management

---

## Step 1: Prerequisites

1. GitHub account with your PracticeOps repository
2. Railway account (sign up at [railway.app](https://railway.app))
   - Sign up with GitHub for easiest setup

---

## Step 2: Create Railway Project

1. Go to [railway.app](https://railway.app) and log in
2. Click **"New Project"**
3. Select **"Deploy from GitHub repo"**
4. Choose your **PracticeOps repository**
5. Railway will detect your app but we'll configure it manually for better control

---

## Step 3: Add PostgreSQL Database

1. In your Railway project, click **"+ New"**
2. Select **"Database"** → **"PostgreSQL"**
3. Railway automatically creates the database and generates credentials
4. Note: Database URL will be available as `DATABASE_URL` environment variable

---

## Step 4: Configure Backend (API) Service

### 4.1 Create API Service

1. Click **"+ New"** → **"GitHub Repo"** → Select your repo again
2. Rename this service to **"api"** (click the service name to rename)

### 4.2 Configure API Settings

Click on the **api** service, then go to **Settings**:

1. **Root Directory**: `apps/api`
2. **Build Command**: `pip install -e ".[dev]"`
3. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### 4.3 Add API Environment Variables

Go to **Variables** tab for the API service and add:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | Use "Reference" → Select PostgreSQL service → `DATABASE_URL` |
| `ENVIRONMENT` | `production` |
| `CORS_ORIGINS` | Leave blank for now (will add after frontend is deployed) |
| `JWT_SECRET_KEY` | Generate with: `openssl rand -base64 32` |
| `POSTGRES_PASSWORD` | Use "Reference" → Select PostgreSQL service → `POSTGRES_PASSWORD` |

### 4.4 Generate JWT Secret (if needed)

On your local machine:
```bash
# Linux/Mac/Windows (Git Bash or PowerShell with openssl)
openssl rand -base64 32

# Or use Python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4.5 Enable Public API Domain

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy the generated URL (e.g., `api-production-xxxx.up.railway.app`)
4. **Save this URL** - you'll need it for the frontend

---

## Step 5: Configure Frontend (Web) Service

### 5.1 Create Web Service

1. Click **"+ New"** → **"GitHub Repo"** → Select your repo again
2. Rename this service to **"web"**

### 5.2 Configure Web Settings

Click on the **web** service, then go to **Settings**:

1. **Root Directory**: `apps/web`
2. **Build Command**: `npm install && npm run build`
3. **Start Command**: `npx vite preview --host 0.0.0.0 --port $PORT`

### 5.3 Add Web Environment Variables

Go to **Variables** tab for the WEB service:

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | Your API URL from Step 4.5 (e.g., `https://api-production-xxxx.up.railway.app`) |

### 5.4 Enable Public Web Domain

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy the generated URL (e.g., `web-production-yyyy.up.railway.app`)

---

## Step 6: Update CORS Configuration

Now that you have the frontend URL, update the API's CORS settings:

1. Go to your **API service**
2. Go to **Variables** tab
3. Update or add `CORS_ORIGINS` with your frontend URL:
   ```
   https://web-production-yyyy.up.railway.app
   ```
4. Click **"Deploy"** to restart the API with new settings

---

## Step 7: Run Database Migrations

Railway doesn't automatically run migrations, so we need to do this once:

### Option A: Using Railway CLI (Recommended)

1. Install Railway CLI:
   ```bash
   # Windows (PowerShell)
   iwr https://railway.app/install.ps1 | iex

   # Mac/Linux
   curl -fsSL https://railway.app/install.sh | sh
   ```

2. Login and link to your project:
   ```bash
   railway login
   railway link
   ```

3. Run migrations:
   ```bash
   railway run -s api alembic upgrade head
   ```

### Option B: One-Click Migration Setup

Add a one-time deployment command:

1. Go to **API service** → **Settings**
2. Temporarily change **Start Command** to:
   ```
   alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
3. Wait for deployment to complete
4. Change it back to just:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

---

## Step 8: Seed Demo Data (Optional)

If you want to add demo data:

```bash
# Using Railway CLI
railway run -s api python -m scripts.seed_demo
```

Default login credentials (if seed script creates them):
- Email: `director@example.com`
- Password: `password123`

---

## Step 9: Verify Deployment

### Test API
Visit your API URL in browser:
```
https://api-production-xxxx.up.railway.app/health
https://api-production-xxxx.up.railway.app/docs
```

### Test Frontend
Visit your web URL:
```
https://web-production-yyyy.up.railway.app
```

You should see the PracticeOps login page!

---

## Step 10: Configure Custom Domain (Optional)

### Add Custom Domain to Frontend

1. Go to **Web service** → **Settings** → **Networking**
2. Click **"Custom Domain"**
3. Enter your domain (e.g., `app.yourdomain.com`)
4. Follow DNS instructions to add CNAME record
5. Railway automatically provisions SSL certificate

### Add Custom Domain to API

1. Go to **API service** → **Settings** → **Networking**
2. Add custom domain (e.g., `api.yourdomain.com`)
3. Update frontend `VITE_API_URL` to use new domain
4. Update API `CORS_ORIGINS` to allow your custom frontend domain

---

## Ongoing Operations

### View Logs

In Railway dashboard:
1. Click on any service (api, web, or database)
2. Go to **"Deployments"** tab
3. Click on latest deployment to see logs

### Redeploy Application

Railway automatically deploys when you push to GitHub:
```bash
git add .
git commit -m "Your changes"
git push origin main
```

### Manual Redeploy

In Railway dashboard:
1. Click on the service
2. Go to **"Deployments"** tab
3. Click **"Deploy"** or click ⋮ on a deployment → **"Redeploy"**

### Restart Services

1. Click on the service
2. Click **"⋮"** (three dots) → **"Restart"**

### Database Backups

Railway Pro includes automatic backups. For free tier:

1. Go to **PostgreSQL service**
2. Click **"⋮"** → **"Backups"**
3. You can manually create backups

Or use Railway CLI:
```bash
railway run -s postgres pg_dump > backup.sql
```

### Environment Variables

To update:
1. Click on the service
2. Go to **"Variables"** tab
3. Add/Edit/Delete variables
4. Service automatically redeploys with new variables

---

## Cost Management

### Free Tier
- **$5 in credits per month**
- Enough for small projects with light usage
- No credit card required for trial

### Usage Tips
- Each service uses resources (CPU, RAM, Network)
- Monitor usage in **"Project"** → **"Usage"** tab
- Set up billing alerts
- Consider upgrading to Pro ($20/month) for better limits

### Typical Monthly Cost
- **Free tier**: $0-$5 (usually enough for development/small production)
- **Light production**: $5-$10/month
- **With custom domains/more traffic**: $10-$20/month

---

## Troubleshooting

### Issue: Build Fails

**Check build logs:**
1. Go to service → **"Deployments"**
2. Click on failed deployment
3. Review build logs for errors

**Common solutions:**
- Verify Root Directory is correct (`apps/api` or `apps/web`)
- Check build command syntax
- Ensure all dependencies are in package.json/pyproject.toml

### Issue: Database Connection Fails

**Solutions:**
1. Verify `DATABASE_URL` reference is correct
2. Check if PostgreSQL service is running
3. Ensure migrations have been run
4. Check API logs for specific error messages

### Issue: Frontend Shows Blank Page

**Solutions:**
1. Check browser console for errors (F12)
2. Verify `VITE_API_URL` is set correctly
3. Check if API is accessible
4. Verify CORS settings in API allow frontend domain

### Issue: CORS Errors

**Solutions:**
1. Make sure `CORS_ORIGINS` in API includes full frontend URL with `https://`
2. Don't include trailing slash in CORS_ORIGINS
3. Redeploy API after changing CORS settings

### Issue: 502 Bad Gateway

**Solutions:**
1. Check if service is running in dashboard
2. Review deployment logs for crashes
3. Verify start command is correct
4. Check for runtime errors in logs

---

## Alternative Deployment Options

If Railway doesn't work for you:

### 1. Render
- Similar to Railway
- Free tier available
- [render.com](https://render.com)

### 2. Fly.io
- Great Docker support
- More complex setup
- [fly.io](https://fly.io)

### 3. Vercel + Railway
- Frontend on Vercel (better for static sites)
- Backend + DB on Railway
- More complex but very scalable

### 4. DigitalOcean App Platform
- $5/month minimum
- Good documentation
- More traditional hosting

---

## Next Steps (Optional Enhancements)

1. **Set up monitoring**: Add Sentry for error tracking
2. **Custom domains**: Use your own domain names
3. **CI/CD improvements**: Add GitHub Actions for testing before deploy
4. **Database backups**: Set up automated backup strategy
5. **Scaling**: Upgrade Railway plan as traffic grows

---

## Support Resources

- **Railway Docs**: [docs.railway.app](https://docs.railway.app)
- **Railway Discord**: [discord.gg/railway](https://discord.gg/railway)
- **Railway Status**: [status.railway.app](https://status.railway.app)

---

## Quick Reference

**Your Services:**
- **API**: FastAPI (Python 3.11)
- **Web**: React + Vite
- **Database**: PostgreSQL 16

**Important URLs:**
- Railway Dashboard: [railway.app/dashboard](https://railway.app/dashboard)
- API Docs (after deploy): `https://your-api-url/docs`
- Frontend (after deploy): `https://your-web-url`

**Key Commands:**
```bash
# Install Railway CLI
curl -fsSL https://railway.app/install.sh | sh  # Mac/Linux
iwr https://railway.app/install.ps1 | iex       # Windows

# Link to project
railway link

# Run migrations
railway run -s api alembic upgrade head

# View logs
railway logs -s api
railway logs -s web

# Open service in browser
railway open
```

---

**Estimated Setup Time**: 15-30 minutes
**Monthly Cost**: $0-$5 (free tier)
**Difficulty**: Beginner-friendly

Good luck with your deployment!
