# Multi-stage frontend Dockerfile.
#
#   --target dev       → Node + Vite dev server (5173) for `docker compose up`
#   default / prod     → static build served by nginx with /api proxy
#
# The dev target keeps node_modules in a named volume (see compose.yaml) so the
# host's Windows filesystem isn't paged through the bind mount.

FROM node:20-alpine AS base
WORKDIR /app
RUN apk add --no-cache libc6-compat
COPY package.json ./
RUN npm install --no-audit --no-fund

# ---------- dev -----------------------------------------------------------------
FROM base AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev"]

# ---------- build ---------------------------------------------------------------
FROM base AS build
COPY . .
RUN npm run build

# ---------- prod ----------------------------------------------------------------
FROM nginx:1.27-alpine AS prod
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --retries=10 \
    CMD wget -qO- http://localhost/ >/dev/null 2>&1 || exit 1
