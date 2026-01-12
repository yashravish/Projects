# AWS Deployment Prompt for PracticeOps

## Objective
Deploy the PracticeOps full-stack application (FastAPI backend + React frontend + PostgreSQL) to AWS with a **$0-$5/month budget** using Terraform infrastructure-as-code and GitHub Actions CI/CD automation.

---

## Current Application State

### Technology Stack
- **Backend:** FastAPI (Python 3.11) with SQLAlchemy + AsyncPG
- **Frontend:** React 18 + TypeScript + Vite
- **Database:** PostgreSQL 16
- **Authentication:** JWT-based with bcrypt password hashing
- **Containerization:** Docker + Docker Compose
- **Migrations:** Alembic (Python database migrations)
- **Background Jobs:** APScheduler (practice reminders, weekly digests)

### Current Directory Structure
```
PracticeOps/
├── apps/
│   ├── api/                    # FastAPI backend
│   │   ├── app/
│   │   │   ├── routes/         # API endpoints
│   │   │   ├── models/         # SQLAlchemy models
│   │   │   ├── core/           # Auth, security, middleware
│   │   │   ├── services/       # Business logic, scheduler
│   │   │   └── main.py         # FastAPI app entry point
│   │   ├── alembic/            # Database migrations
│   │   ├── scripts/
│   │   │   ├── seed.py         # Basic seed data
│   │   │   └── seed_demo.py    # Comprehensive demo data
│   │   ├── Dockerfile          # Production API image
│   │   └── pyproject.toml      # Python dependencies
│   └── web/                    # React frontend
│       ├── src/
│       │   ├── pages/          # React pages/routes
│       │   ├── components/     # Reusable components
│       │   └── lib/            # Utilities, API client
│       ├── Dockerfile          # Production web image
│       └── package.json        # Node dependencies
├── infra/
│   └── docker-compose.yml      # Local development only
├── docs/
│   └── deployment.md           # AWS deployment plan
└── .gitignore
```

### Key Configuration Files

**Backend Environment Variables (apps/api/app/config.py):**
- `DATABASE_URL` - PostgreSQL connection string
- `ENVIRONMENT` - "development" | "production"
- `JWT_SECRET_KEY` - Secret for JWT token signing
- `CORS_ORIGINS` - Comma-separated allowed origins
- `FRONTEND_URL` - Frontend URL for redirects/emails

**Frontend Environment Variables (apps/web/.env):**
- `VITE_API_URL` - Backend API URL

**Database Schema:**
- Tables: users, teams, team_memberships, rehearsal_cycles, assignments, tickets, practice_logs, notification_preferences, invites
- All tables use UUID primary keys
- Created via Alembic migrations in apps/api/alembic/versions/

---

## Deployment Architecture

### Cost-Optimized AWS Setup ($0-$5/month)

```
┌─────────────────────────────────────────────────────────┐
│                    Internet Users                        │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
             │ (HTML/CSS/JS)             │ (API Requests)
             ▼                            ▼
    ┌─────────────────┐         ┌──────────────────────┐
    │  S3 Bucket      │         │  EC2 t3.micro        │
    │  (Frontend)     │         │  (Ubuntu 22.04)      │
    │                 │         │                      │
    │  - index.html   │         │  Docker Compose:     │
    │  - assets/      │         │  ┌────────────────┐  │
    │  - Static SPA   │         │  │ API Container  │  │
    └─────────────────┘         │  │ (FastAPI)      │  │
                                │  └────────────────┘  │
                                │  ┌────────────────┐  │
                                │  │ DB Container   │  │
                                │  │ (PostgreSQL)   │  │
                                │  └────────────────┘  │
                                └──────────────────────┘
```

**Why This Architecture?**
- S3 static hosting: ~$0.01-0.50/month (pennies)
- EC2 t3.micro: $0 (free tier) or t4g.nano $3-5/month
- No RDS (using containerized Postgres on EC2)
- No ECS/Fargate (using Docker Compose on EC2)
- GitHub Container Registry (GHCR): Free

---

## Implementation Tasks

### PHASE 1: Create Infrastructure Files

#### Task 1.1: Create Terraform Configuration

**File: `infra/main.tf`**
```hcl
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# S3 Bucket for Frontend
resource "aws_s3_bucket" "web" {
  bucket = var.web_bucket_name
}

resource "aws_s3_bucket_public_access_block" "web" {
  bucket                  = aws_s3_bucket.web.id
  block_public_acls       = true
  block_public_policy     = false
  ignore_public_acls      = true
  restrict_public_buckets = false
}

resource "aws_s3_bucket_website_configuration" "web" {
  bucket = aws_s3_bucket.web.id
  index_document { suffix = "index.html" }
  error_document { key = "index.html" }  # SPA fallback
}

data "aws_iam_policy_document" "web_public" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.web.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
  }
}

resource "aws_s3_bucket_policy" "web" {
  bucket = aws_s3_bucket.web.id
  policy = data.aws_iam_policy_document.web_public.json
}

# EC2 Instance for API + Database
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "api_sg" {
  name        = "${var.project_name}-api-sg"
  description = "Allow SSH and API access"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_cidr]
  }

  ingress {
    description = "API"
    from_port   = var.api_port
    to_port     = var.api_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "api" {
  ami                    = data.aws_ami.ubuntu.id
  instance_type          = var.instance_type
  key_name               = var.ssh_key_name
  vpc_security_group_ids = [aws_security_group.api_sg.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh", {
    api_port = var.api_port
  })

  tags = {
    Name = "${var.project_name}-api"
  }
}
```

**File: `infra/variables.tf`**
```hcl
variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "practiceops"
}

variable "web_bucket_name" {
  type        = string
  description = "Unique S3 bucket name for frontend"
}

variable "instance_type" {
  type    = string
  default = "t3.micro"  # Free tier eligible
}

variable "api_port" {
  type    = number
  default = 8000
}

variable "ssh_key_name" {
  type        = string
  description = "Name of AWS EC2 key pair"
}

variable "ssh_cidr" {
  type    = string
  default = "0.0.0.0/0"
  description = "CIDR block for SSH access (tighten to your IP)"
}
```

**File: `infra/outputs.tf`**
```hcl
output "api_public_ip" {
  value       = aws_instance.api.public_ip
  description = "Public IP of API server"
}

output "web_bucket" {
  value       = aws_s3_bucket.web.bucket
  description = "S3 bucket name"
}

output "web_website_url" {
  value       = aws_s3_bucket_website_configuration.web.website_endpoint
  description = "S3 website URL"
}
```

**File: `infra/user_data.sh`**
```bash
#!/bin/bash
set -e

# Update system
apt-get update -y
apt-get install -y ca-certificates curl gnupg git

# Install Docker
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu jammy stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Configure Docker for ubuntu user
usermod -aG docker ubuntu

# Prepare deployment directory
mkdir -p /opt/practiceops
chown -R ubuntu:ubuntu /opt/practiceops

# Signal readiness
echo "Server ready for PracticeOps deploy" > /opt/practiceops/READY.txt
```

#### Task 1.2: Create Production Docker Compose

**File: `docker-compose.prod.yml` (project root)**
```yaml
services:
  db:
    image: postgres:16
    container_name: practiceops-db-prod
    environment:
      POSTGRES_DB: practiceops
      POSTGRES_USER: practiceops
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U practiceops -d practiceops"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  api:
    image: ${API_IMAGE}
    container_name: practiceops-api-prod
    environment:
      DATABASE_URL: postgresql+asyncpg://practiceops:${POSTGRES_PASSWORD}@db:5432/practiceops
      ENVIRONMENT: production
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
      CORS_ORIGINS: ${CORS_ORIGINS}
      FRONTEND_URL: ${FRONTEND_URL}
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

volumes:
  db_data:
```

#### Task 1.3: Update API Dockerfile for Production

**Verify/Update: `apps/api/Dockerfile`**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run migrations and start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Task 1.4: Update Web Dockerfile for Production Build

**Verify/Update: `apps/web/Dockerfile`**
```dockerfile
FROM node:20-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build for production
RUN npm run build

# Production stage - serve with nginx
FROM nginx:alpine

# Copy build output
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx config for SPA routing
RUN echo 'server { \
  listen 80; \
  location / { \
    root /usr/share/nginx/html; \
    index index.html; \
    try_files $uri $uri/ /index.html; \
  } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

---

### PHASE 2: GitHub Actions CI/CD Pipeline

#### Task 2.1: Create Deployment Workflow

**File: `.github/workflows/deploy.yml`**
```yaml
name: Deploy PracticeOps to AWS

on:
  push:
    branches: ["main"]
  workflow_dispatch:  # Allow manual trigger

permissions:
  contents: read
  packages: write

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      # ============== AWS Setup ==============
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      # ============== Terraform ==============
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.5.0

      - name: Terraform Init
        working-directory: infra
        run: terraform init

      - name: Terraform Plan
        working-directory: infra
        env:
          TF_VAR_web_bucket_name: ${{ secrets.TF_VAR_web_bucket_name }}
          TF_VAR_ssh_key_name: ${{ secrets.TF_VAR_ssh_key_name }}
        run: terraform plan

      - name: Terraform Apply
        working-directory: infra
        env:
          TF_VAR_web_bucket_name: ${{ secrets.TF_VAR_web_bucket_name }}
          TF_VAR_ssh_key_name: ${{ secrets.TF_VAR_ssh_key_name }}
        run: terraform apply -auto-approve

      # ============== Build & Push API Docker Image ==============
      - name: Login to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and Push API Image
        run: |
          API_IMAGE="ghcr.io/${{ github.repository_owner }}/practiceops-api:latest"
          docker build -t "$API_IMAGE" ./apps/api
          docker push "$API_IMAGE"
          echo "API_IMAGE=$API_IMAGE" >> $GITHUB_ENV

      # ============== Build & Deploy Frontend ==============
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: apps/web/package-lock.json

      - name: Build Frontend
        working-directory: apps/web
        env:
          VITE_API_URL: http://${{ secrets.EC2_HOST }}:8000
        run: |
          npm ci
          npm run build

      - name: Deploy Frontend to S3
        run: |
          aws s3 sync apps/web/dist "s3://${{ secrets.TF_VAR_web_bucket_name }}" --delete

      # ============== Deploy Backend to EC2 ==============
      - name: Deploy API to EC2 via SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_PRIVATE_KEY }}
          script: |
            set -e

            # Navigate to deployment directory
            mkdir -p /opt/practiceops
            cd /opt/practiceops

            # Create environment file
            cat > .env << 'EOF'
            API_IMAGE=${{ env.API_IMAGE }}
            POSTGRES_PASSWORD=${{ secrets.POSTGRES_PASSWORD }}
            JWT_SECRET_KEY=${{ secrets.JWT_SECRET_KEY }}
            CORS_ORIGINS=${{ secrets.CORS_ORIGINS }}
            FRONTEND_URL=${{ secrets.FRONTEND_URL }}
            EOF

            # Download production docker-compose
            curl -fsSL https://raw.githubusercontent.com/${{ github.repository }}/main/docker-compose.prod.yml -o docker-compose.prod.yml

            # Pull latest images
            docker compose -f docker-compose.prod.yml pull

            # Start services
            docker compose -f docker-compose.prod.yml up -d

            # Wait for database to be ready
            sleep 10

            # Run database migrations
            docker compose -f docker-compose.prod.yml exec -T api alembic upgrade head

            # Optionally seed demo data (only on first deploy)
            # docker compose -f docker-compose.prod.yml exec -T api python -m scripts.seed_demo

            echo "Deployment complete!"

      - name: Health Check
        run: |
          sleep 15
          curl -f http://${{ secrets.EC2_HOST }}:8000/health || exit 1
          echo "Health check passed!"
```

---

### PHASE 3: Configuration & Secrets Setup

#### Task 3.1: Generate Production Secrets

**Run locally to generate secure secrets:**

```bash
# Generate JWT secret (32-byte random string)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate PostgreSQL password (32-byte random string)
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

#### Task 3.2: Create AWS EC2 Key Pair

**In AWS Console:**
1. Navigate to EC2 → Key Pairs
2. Create new key pair named `practiceops-key`
3. Download the `.pem` file
4. Store private key securely (needed for GitHub Secret)

#### Task 3.3: Choose Unique S3 Bucket Name

**Format:** `practiceops-web-[your-unique-identifier]`
**Example:** `practiceops-web-prod-abc123`

Must be globally unique across all AWS accounts.

#### Task 3.4: Configure GitHub Secrets

**Repository → Settings → Secrets and variables → Actions → New repository secret**

Add the following secrets:

| Secret Name | Value | Description |
|------------|-------|-------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | IAM user with EC2, S3, Terraform permissions |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | Corresponding secret |
| `AWS_REGION` | `us-east-1` | AWS region |
| `TF_VAR_web_bucket_name` | `practiceops-web-[unique]` | S3 bucket name |
| `TF_VAR_ssh_key_name` | `practiceops-key` | EC2 key pair name |
| `EC2_HOST` | (blank initially) | Will be filled after first Terraform run |
| `EC2_SSH_PRIVATE_KEY` | Contents of `.pem` file | Full private key including headers |
| `POSTGRES_PASSWORD` | Generated password | From Task 3.1 |
| `JWT_SECRET_KEY` | Generated secret | From Task 3.1 |
| `CORS_ORIGINS` | `http://[bucket-name].s3-website-[region].amazonaws.com` | Will update after deployment |
| `FRONTEND_URL` | Same as CORS_ORIGINS | Frontend URL |

---

### PHASE 4: Deployment Execution

#### Task 4.1: Initialize Terraform Backend (Optional but Recommended)

**Create S3 backend for Terraform state:**

```bash
# Create S3 bucket for Terraform state
aws s3 mb s3://practiceops-terraform-state-[unique] --region us-east-1

# Update infra/main.tf to add backend config
terraform {
  backend "s3" {
    bucket = "practiceops-terraform-state-[unique]"
    key    = "practiceops/terraform.tfstate"
    region = "us-east-1"
  }
}
```

#### Task 4.2: Local Terraform Initialization

```bash
cd infra/

# Initialize Terraform
terraform init

# Create terraform.tfvars
cat > terraform.tfvars << EOF
aws_region       = "us-east-1"
project_name     = "practiceops"
web_bucket_name  = "practiceops-web-[your-unique-id]"
instance_type    = "t3.micro"
ssh_key_name     = "practiceops-key"
ssh_cidr         = "0.0.0.0/0"  # Tighten to your IP for security
EOF

# Plan deployment
terraform plan

# Apply infrastructure
terraform apply
```

**Capture outputs:**
```bash
terraform output api_public_ip
# Update GitHub Secret EC2_HOST with this IP
```

#### Task 4.3: Update GitHub Secrets with IP

1. Copy `api_public_ip` from Terraform output
2. Update GitHub Secret `EC2_HOST` with this IP
3. Update `CORS_ORIGINS` to `http://[S3-website-endpoint]`
4. Update `FRONTEND_URL` to match `CORS_ORIGINS`

#### Task 4.4: Trigger Deployment

```bash
# Commit and push to trigger GitHub Actions
git add .
git commit -m "Add AWS deployment infrastructure"
git push origin main
```

**Monitor deployment:**
- GitHub → Actions tab
- Watch workflow execution
- Check for successful completion

#### Task 4.5: Verify Deployment

```bash
# Test API health
curl http://[EC2_PUBLIC_IP]:8000/health

# Test API docs
open http://[EC2_PUBLIC_IP]:8000/docs

# Test frontend
open http://[S3-BUCKET-NAME].s3-website-[REGION].amazonaws.com
```

---

### PHASE 5: Post-Deployment Configuration

#### Task 5.1: Seed Demo Data (Optional)

```bash
# SSH into EC2
ssh -i practiceops-key.pem ubuntu@[EC2_PUBLIC_IP]

# Navigate to deployment directory
cd /opt/practiceops

# Run demo data seeding
docker compose -f docker-compose.prod.yml exec api python -m scripts.seed_demo

# Verify data
docker compose -f docker-compose.prod.yml exec api python -c "
from app.database import async_session_maker
from app.models import User
from sqlalchemy import select
import asyncio

async def check():
    async with async_session_maker() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()
        print(f'Users in database: {len(users)}')

asyncio.run(check())
"
```

#### Task 5.2: Configure Custom Domain (Optional)

**Using Route 53:**

1. Purchase/configure domain in Route 53
2. Create A record pointing to EC2 IP
3. Create CNAME for www pointing to S3 website endpoint
4. Update CORS_ORIGINS and FRONTEND_URL secrets
5. Re-deploy

#### Task 5.3: Enable HTTPS (Future Enhancement)

**Options:**
- **For S3:** Use CloudFront with ACM certificate
- **For EC2:** Use Let's Encrypt with reverse proxy (nginx/Caddy)
- **Cost Impact:** CloudFront may exceed $5/month budget

---

## Security Hardening Checklist

- [ ] Tighten SSH access (`ssh_cidr`) to your IP only
- [ ] Use AWS Secrets Manager for production secrets (adds cost)
- [ ] Enable S3 bucket versioning
- [ ] Set up CloudWatch alerts for EC2 instance health
- [ ] Configure automatic security updates on EC2
- [ ] Implement rate limiting in FastAPI (already present)
- [ ] Enable database backups (pg_dump cron job)
- [ ] Review CORS origins to prevent unauthorized access

---

## Monitoring & Maintenance

### Cost Monitoring

```bash
# Check current AWS costs
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost
```

### Application Logs

```bash
# SSH into EC2
ssh -i practiceops-key.pem ubuntu@[EC2_PUBLIC_IP]

# View API logs
docker compose -f /opt/practiceops/docker-compose.prod.yml logs -f api

# View database logs
docker compose -f /opt/practiceops/docker-compose.prod.yml logs -f db
```

### Database Backups

```bash
# Create backup script
cat > /opt/practiceops/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/practiceops/backups"
mkdir -p $BACKUP_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

docker compose -f /opt/practiceops/docker-compose.prod.yml exec -T db \
  pg_dump -U practiceops practiceops | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz

# Keep only last 7 days of backups
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
EOF

chmod +x /opt/practiceops/backup.sh

# Add to crontab (daily at 2 AM)
(crontab -l 2>/dev/null; echo "0 2 * * * /opt/practiceops/backup.sh") | crontab -
```

---

## Troubleshooting Guide

### Issue: Terraform apply fails

**Solution:**
```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify S3 bucket name is unique
aws s3 ls | grep practiceops

# Check Terraform state
terraform state list
```

### Issue: Docker image pull fails on EC2

**Solution:**
```bash
# SSH into EC2
ssh -i practiceops-key.pem ubuntu@[EC2_PUBLIC_IP]

# Login to GHCR manually
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull image manually
docker pull ghcr.io/[owner]/practiceops-api:latest
```

### Issue: Database migrations fail

**Solution:**
```bash
# Connect to API container
docker exec -it practiceops-api-prod bash

# Check database connectivity
python -c "from app.database import engine; import asyncio; asyncio.run(engine.connect())"

# Run migrations manually
alembic upgrade head

# Check migration status
alembic current
```

### Issue: Frontend shows 404 on routes

**Solution:**
- Verify S3 error document is set to `index.html` in Terraform
- Check that Vite built correctly: `apps/web/dist/index.html` exists
- Verify S3 bucket policy allows public read access

### Issue: CORS errors in browser

**Solution:**
```bash
# Verify CORS_ORIGINS includes S3 website URL
echo $CORS_ORIGINS

# Update in GitHub Secrets if needed
# Re-deploy via GitHub Actions
```

---

## Rollback Procedure

### Rollback Application

```bash
# SSH into EC2
ssh -i practiceops-key.pem ubuntu@[EC2_PUBLIC_IP]

cd /opt/practiceops

# Pull previous image version
docker pull ghcr.io/[owner]/practiceops-api:[previous-tag]

# Update .env with previous image
# Restart containers
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d
```

### Rollback Infrastructure

```bash
cd infra/

# Destroy resources
terraform destroy

# Or revert to previous state
terraform state pull > backup.tfstate
```

---

## Success Criteria

✅ **Infrastructure Deployed:**
- [ ] S3 bucket created and configured
- [ ] EC2 instance running and accessible
- [ ] Security groups configured correctly
- [ ] Terraform outputs captured

✅ **Application Running:**
- [ ] API responds to `/health` endpoint
- [ ] Frontend loads in browser
- [ ] Can login with demo credentials
- [ ] Database contains seed data

✅ **CI/CD Working:**
- [ ] GitHub Actions workflow succeeds
- [ ] Automated deployments on push to main
- [ ] Docker images building and pushing

✅ **Cost Target Met:**
- [ ] Monthly AWS bill projected at $0-5
- [ ] No unexpected charges
- [ ] Free tier or minimal instance used

---

## Expected Deployment Timeline

- **Phase 1 (Infra Files):** 30-45 minutes
- **Phase 2 (GitHub Actions):** 15-20 minutes
- **Phase 3 (Secrets Setup):** 20-30 minutes
- **Phase 4 (Execution):** 15-20 minutes
- **Phase 5 (Post-Deploy):** 10-15 minutes

**Total:** ~2 hours for first-time deployment

---

## Cost Breakdown Estimate

| Resource | Free Tier | Post-Free Tier |
|----------|-----------|----------------|
| EC2 t3.micro | $0 (750 hrs/mo) | ~$8.50/month |
| EC2 t4g.nano | N/A | ~$3-5/month |
| EBS 20GB gp3 | $0 (30GB free) | ~$1.60/month |
| S3 Storage (1GB) | $0 | ~$0.02/month |
| S3 Requests | $0 | ~$0.10/month |
| Data Transfer Out (1GB) | $0 (1GB free) | ~$0.09/GB |
| **TOTAL (Free Tier)** | **~$0/month** | - |
| **TOTAL (t4g.nano)** | - | **~$3-7/month** |

**Recommended:** Start with t3.micro (free tier), switch to t4g.nano after 12 months.

---

## Contact & Support

- **GitHub Issues:** Use for deployment questions
- **AWS Support:** Free tier includes basic support
- **Terraform Docs:** terraform.io/docs
- **Docker Docs:** docs.docker.com

---

*This deployment prompt is tailored specifically for the PracticeOps application based on the current codebase state as of January 2026.*
