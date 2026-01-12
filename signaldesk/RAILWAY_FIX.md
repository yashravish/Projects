# Railway Deployment Fix

## Problems Fixed

### 1. Invalid Nix Package Name
**Error:**
```
error: undefined variable 'nodejs-20_x'
```
**Fix:** Changed `nodejs-20_x` to `nodejs_20` in nixpacks.toml

### 2. Wrong Package Manager
**Error:**
```
npm error Unsupported URL Type "workspace:": workspace:*
```
**Root Cause:**
- Railway was using Railpack + npm instead of Nixpacks + pnpm
- This is a pnpm monorepo with `workspace:*` dependencies
- Railway must deploy from the monorepo root, not from `apps/api`
- Railway must use Nixpacks (not Railpack) to properly detect and use pnpm

## Solution Steps

### 1. Configure Railway Service
Open your Railway dashboard and update these settings:

```bash
railway open
```

Then in your service settings:

**Root Directory:**
- If your GitHub repo root is `Projects/`: set to `Signal Desk Lite`
- If your GitHub repo root is `Signal Desk Lite/`: leave **empty** or set to `.`

**Builder:**
- Ensure it's set to **NIXPACKS** (not Railpack)

**Save** and **Redeploy**

### 2. Configuration Files
The following files have been updated at the monorepo root:

- [nixpacks.toml](nixpacks.toml) - Nixpacks build configuration
- [railway.json](railway.json) - Railway deployment configuration

Both files are configured to:
1. Install all dependencies using `pnpm install --frozen-lockfile`
2. Build the `@signaldesk/shared` package first
3. Build the `@signaldesk/api` package
4. Generate Prisma client
5. Start the API from `apps/api/dist/index.js`

### 3. Deploy
From the monorepo root:

```bash
cd "c:\Users\yashr\Documents\GitHub\Projects\Signal Desk Lite"
railway up
```

Or push to GitHub if you have automatic deployments enabled.

## Verification
After deployment, check the build logs. You should see:
- ✓ **Nixpacks** being used (not Railpack)
- ✓ **pnpm** being used for installation
- ✓ No "workspace:" protocol errors
- ✓ Both packages building successfully

## Need More Help?
See the full deployment guide: [apps/api/RAILWAY_DEPLOYMENT.md](apps/api/RAILWAY_DEPLOYMENT.md)
