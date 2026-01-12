Below is a **practical “$0–$5 MVP” AWS plan** with:

1. **Terraform infra** (minimal + cheap)
2. **Cost-optimized architecture** (realistic)
3. **One-click GitHub Actions pipeline** (build + deploy)

I’m optimizing for **lowest ongoing cost** while still being reliable.

---

## Architecture for $0–$5 MVP (AWS)

### Cheapest realistic setup

* **Frontend:** S3 static hosting (+ optional CloudFront)
* **Backend + Postgres:** **single EC2 instance** running Docker Compose

  * This is the key to hitting **$0–$5**.
  * ECS + RDS will almost always exceed $5/mo.

### Cost notes

* If you’re in **AWS Free Tier (first 12 months)**:

  * **EC2 t3.micro** can be ~$0
  * S3 costs are pennies
* If not free tier:

  * **t4g.nano** (ARM) is often around the $3–$5 range depending on region (plus small EBS)
  * Still within your target.

---

## Terraform (Minimal MVP)

### Repo structure (suggested)

```
infra/
  main.tf
  variables.tf
  outputs.tf
  user_data.sh
apps/
  api/
  web/
docker-compose.prod.yml
```

### `infra/main.tf`

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

# ---------
# S3: Frontend
# ---------
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
  error_document { key    = "index.html" } # SPA fallback
}

# Public-read policy ONLY for MVP simplicity.
# For production, swap to CloudFront + OAC and block public access.
data "aws_iam_policy_document" "web_public" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.web.arn}/*"]
    principals { type = "*", identifiers = ["*"] }
  }
}

resource "aws_s3_bucket_policy" "web" {
  bucket = aws_s3_bucket.web.id
  policy = data.aws_iam_policy_document.web_public.json
}

# ---------
# EC2: API + Postgres via Docker Compose
# ---------
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

resource "aws_security_group" "api_sg" {
  name        = "${var.project_name}-api-sg"
  description = "Allow SSH + API"

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

### `infra/user_data.sh`

```bash
#!/bin/bash
set -e

# Basic setup
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

usermod -aG docker ubuntu

mkdir -p /opt/practiceops
chown -R ubuntu:ubuntu /opt/practiceops

# We will deploy docker-compose + env via GitHub Actions over SSH
echo "Server ready for PracticeOps deploy" > /opt/practiceops/READY.txt
```

### `infra/variables.tf`

```hcl
variable "aws_region"       { type = string, default = "us-east-1" }
variable "project_name"     { type = string, default = "practiceops" }
variable "web_bucket_name"  { type = string }
variable "instance_type"    { type = string, default = "t3.micro" } # free-tier friendly
variable "api_port"         { type = number, default = 8000 }
variable "ssh_key_name"     { type = string }
variable "ssh_cidr"         { type = string, default = "0.0.0.0/0" } # tighten to your IP
```

### `infra/outputs.tf`

```hcl
output "api_public_ip" {
  value = aws_instance.api.public_ip
}

output "web_bucket" {
  value = aws_s3_bucket.web.bucket
}

output "web_website_url" {
  value = aws_s3_bucket_website_configuration.web.website_endpoint
}
```

---

## Docker Compose (API + Postgres on EC2)

### `docker-compose.prod.yml` (repo root)

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: practiceops
      POSTGRES_USER: practiceops
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - db_data:/var/lib/postgresql/data
    restart: unless-stopped

  api:
    image: ${API_IMAGE}
    environment:
      DATABASE_URL: postgresql+asyncpg://practiceops:${POSTGRES_PASSWORD}@db:5432/practiceops
      ENVIRONMENT: production
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
      CORS_ORIGINS: ${CORS_ORIGINS}
      FRONTEND_URL: ${FRONTEND_URL}
    ports:
      - "8000:8000"
    depends_on:
      - db
    restart: unless-stopped

volumes:
  db_data:
```

> This avoids RDS cost entirely and keeps you in the $0–$5 zone.

---

## One-Click GitHub Actions Pipeline (Build + Deploy)

### What it will do on every push to `main`

1. `terraform init/plan/apply` (optional: gated)
2. Build API Docker image and push to **GHCR** (free)
3. Build frontend and deploy to **S3**
4. SSH into EC2, pull latest image, run migrations, restart compose

### Required GitHub Secrets

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_REGION`
* `TF_VAR_web_bucket_name`
* `TF_VAR_ssh_key_name`
* `EC2_HOST` (from terraform output `api_public_ip`)
* `EC2_SSH_PRIVATE_KEY` (private key matching `ssh_key_name`)
* `POSTGRES_PASSWORD`
* `JWT_SECRET_KEY`
* `CORS_ORIGINS`
* `FRONTEND_URL`

### `.github/workflows/deploy.yml`

```yaml
name: Deploy PracticeOps

on:
  push:
    branches: [ "main" ]

permissions:
  contents: read
  packages: write

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      # ---------- AWS ----------
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      # ---------- Terraform (optional but “one-click”) ----------
      - name: Setup Terraform
        uses: hashicorp/setup-terraform@v3

      - name: Terraform Init
        working-directory: infra
        run: terraform init

      - name: Terraform Apply
        working-directory: infra
        env:
          TF_VAR_web_bucket_name: ${{ secrets.TF_VAR_web_bucket_name }}
          TF_VAR_ssh_key_name: ${{ secrets.TF_VAR_ssh_key_name }}
        run: terraform apply -auto-approve

      # ---------- Build API image -> GHCR ----------
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and Push API
        run: |
          API_IMAGE="ghcr.io/${{ github.repository_owner }}/practiceops-api:latest"
          docker build -t "$API_IMAGE" ./apps/api
          docker push "$API_IMAGE"
          echo "API_IMAGE=$API_IMAGE" >> $GITHUB_ENV

      # ---------- Build + Deploy Frontend to S3 ----------
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"

      - name: Build Web
        working-directory: apps/web
        env:
          VITE_API_URL: http://${{ secrets.EC2_HOST }}:8000
        run: |
          npm ci
          npm run build

      - name: Deploy Web to S3
        run: |
          aws s3 sync apps/web/dist "s3://${{ secrets.TF_VAR_web_bucket_name }}" --delete

      # ---------- SSH Deploy on EC2 ----------
      - name: Deploy API on EC2 via SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_PRIVATE_KEY }}
          script: |
            set -e
            mkdir -p /opt/practiceops
            cd /opt/practiceops

            # Write env
            cat > .env << 'EOF'
            API_IMAGE=${{ env.API_IMAGE }}
            POSTGRES_PASSWORD=${{ secrets.POSTGRES_PASSWORD }}
            JWT_SECRET_KEY=${{ secrets.JWT_SECRET_KEY }}
            CORS_ORIGINS=${{ secrets.CORS_ORIGINS }}
            FRONTEND_URL=${{ secrets.FRONTEND_URL }}
            EOF

            # Copy compose from repo (simple approach: curl raw file)
            curl -fsSL https://raw.githubusercontent.com/${{ github.repository }}/main/docker-compose.prod.yml -o docker-compose.prod.yml

            # Pull latest + restart
            docker compose -f docker-compose.prod.yml pull

            # Start DB + API
            docker compose -f docker-compose.prod.yml up -d

            # Run migrations inside API container (if alembic exists)
            docker compose -f docker-compose.prod.yml exec -T api alembic upgrade head || true
```

> If you don’t want Terraform on every push, split into two workflows: `infra.yml` (manual) + `deploy.yml` (automatic). But the above is truly “one-click.”

---

## How to keep this $0–$5

* Use **t3.micro (free tier)** or **t4g.nano** (cheap)
* Use **S3 website hosting** (skip CloudFront until needed)
* Keep DB on the instance (skip RDS)
* Use GHCR for images (free)

