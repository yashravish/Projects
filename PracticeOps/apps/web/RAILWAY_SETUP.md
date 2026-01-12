# Railway Frontend Deployment Setup Guide

## Required Build Arguments

The frontend needs the API URL configured at build time (not runtime) because Vite embeds environment variables into the built JavaScript.

### Set in Railway Service Variables

Go to your Railway Web service → Variables tab and add:

```
VITE_API_URL=https://your-api-service.railway.app
```

**Important**:
- Replace `your-api-service.railway.app` with your actual API service URL
- This URL will be embedded in the built frontend files
- If your API URL changes, you must rebuild the frontend

## How Railway Handles Vite Build Args

Railway automatically passes service variables as both environment variables AND Docker build arguments. The Dockerfile is configured to accept `VITE_API_URL` as a build arg:

```dockerfile
ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
```

## Deployment Steps

### 1. Get Your API URL

First, deploy your API service and get its URL:
- Go to your API service in Railway
- Copy the public domain (e.g., `https://practiceops-api-production.up.railway.app`)

### 2. Configure Frontend Variables

In your Web service → Variables tab, add:

```
VITE_API_URL=https://your-api-service.railway.app
```

### 3. Deploy

Railway will automatically:
1. Clone your repository
2. Run `npm ci` (now that package-lock.json is committed)
3. Build with `npm run build` (which runs `tsc -b && vite build`)
4. Copy built files to nginx
5. Serve your application on port 80

### 4. Verify Deployment

Once deployed:
- Visit your frontend URL (Railway provides this)
- Open browser DevTools → Network tab
- You should see API requests going to your API domain
- Try logging in to verify the connection works

## Troubleshooting

### Build Fails: "npm ci" Error
- **Fixed**: We committed `package-lock.json` to git
- If you see this error again, verify the file is in your repository

### Build Fails: "VITE_API_URL is not set"
- Make sure you added `VITE_API_URL` in Railway Variables
- Redeploy after adding the variable

### Frontend Loads But Can't Connect to API
- Check browser console for API errors
- Verify `VITE_API_URL` points to the correct API domain
- Ensure API service is healthy (check `/health` endpoint)
- Check for CORS errors - API must include frontend URL in `CORS_ORIGINS`

### 404 Errors on Page Refresh
- **Fixed**: nginx.conf includes SPA routing configuration
- All routes correctly fall back to `index.html`

### CORS Errors
The API needs to know about your frontend URL. In your API service variables, set:

```
CORS_ORIGINS=https://your-frontend.railway.app
```

If you have preview deployments, include those too:
```
CORS_ORIGINS=https://your-frontend.railway.app,https://your-frontend-pr-*.railway.app
```

## Environment Variables Summary

### Required (Build Time)
```
VITE_API_URL=https://your-api-service.railway.app
```

### Optional
None - the frontend only needs to know where the API is located.

## Local Development vs Production

- **Local**: Uses `.env` file with `VITE_API_URL=http://localhost:8000`
- **Production**: Uses Railway variable `VITE_API_URL=https://your-api.railway.app`
- The `.env` file is gitignored and never deployed

## Updating API URL

If your API URL changes:
1. Update `VITE_API_URL` in Railway Variables
2. Trigger a new deployment (Railway will rebuild with new URL)
3. The new frontend build will use the updated API URL

## Health Check

Railway is configured to check `/` for frontend health:
- nginx serves the React app on port 80
- Any request to `/` returns the index.html
- Health check should pass immediately after build completes
