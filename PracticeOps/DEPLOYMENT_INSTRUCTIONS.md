# PracticeOps AWS Deployment Instructions

This guide will walk you through deploying PracticeOps to AWS with a $0-$5/month budget.

## Prerequisites

- AWS Account with billing alerts configured
- GitHub account with this repository
- Terraform installed locally (v1.5.0+)
- AWS CLI configured with your credentials
- Git installed

---

## Step 1: AWS Setup

### 1.1 Create IAM User for Deployment

1. Log in to AWS Console
2. Navigate to IAM → Users → Create User
3. Create user named `practiceops-deploy`
4. Attach policies:
   - `AmazonEC2FullAccess`
   - `AmazonS3FullAccess`
   - `IAMFullAccess` (for creating service roles)
5. Create access key for this user
6. **Save the Access Key ID and Secret Access Key** - you'll need these for GitHub Secrets

### 1.2 Create EC2 Key Pair

1. Navigate to EC2 → Key Pairs (in the left sidebar under Network & Security)
2. Click "Create key pair"
3. Name: `practiceops-key`
4. Key pair type: RSA
5. Private key file format: `.pem`
6. Click "Create key pair"
7. **Download and save the `.pem` file securely** - you cannot download it again
8. If on Linux/Mac, set permissions: `chmod 400 practiceops-key.pem`

### 1.3 Choose a Unique S3 Bucket Name

S3 bucket names must be globally unique across ALL AWS accounts.

**Recommended format:** `practiceops-web-[your-name]-[random]`

**Example:** `practiceops-web-johnsmith-7x9k`

**Save this name** - you'll use it in multiple places.

---

## Step 2: Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each of the following:

| Secret Name | Value | Where to Get It |
|-------------|-------|-----------------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | From Step 1.1 |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | From Step 1.1 |
| `AWS_REGION` | `us-east-1` | Or your preferred region |
| `TF_VAR_web_bucket_name` | Your unique S3 bucket name | From Step 1.3 |
| `TF_VAR_ssh_key_name` | `practiceops-key` | From Step 1.2 |
| `EC2_HOST` | Leave blank for now | Will be filled after first Terraform run |
| `EC2_SSH_PRIVATE_KEY` | Full contents of `.pem` file | From Step 1.2 (including BEGIN/END lines) |
| `POSTGRES_PASSWORD` | `Lq4sC2mr7_gbnOmzE5pGB4EKeTyFq883vFWdsB7WqoI` | Generated secret (or generate your own) |
| `JWT_SECRET_KEY` | `IWH5Vb0C79Y3n3R7MGtaj009CA20MZQEUlDLavbLRx8` | Generated secret (or generate your own) |
| `CORS_ORIGINS` | Leave blank for now | Will be filled after infrastructure is created |
| `FRONTEND_URL` | Leave blank for now | Will be filled after infrastructure is created |

### How to Copy EC2_SSH_PRIVATE_KEY

On Windows (PowerShell):
```powershell
Get-Content practiceops-key.pem | Set-Clipboard
```

On Linux/Mac:
```bash
cat practiceops-key.pem | pbcopy  # macOS
cat practiceops-key.pem | xclip -selection clipboard  # Linux
```

The key should look like this (paste the entire thing):
```
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
...
-----END RSA PRIVATE KEY-----
```

---

## Step 3: Initial Infrastructure Setup with Terraform

Before using GitHub Actions, we need to create the infrastructure once to get the EC2 IP address.

### 3.1 Configure Terraform Variables

Create a file `infra/terraform.tfvars`:

```bash
cd infra
cat > terraform.tfvars << 'EOF'
aws_region       = "us-east-1"
project_name     = "practiceops"
web_bucket_name  = "practiceops-web-YOUR-UNIQUE-NAME"  # Replace with your bucket name from Step 1.3
instance_type    = "t3.micro"  # Free tier eligible
ssh_key_name     = "practiceops-key"
ssh_cidr         = "0.0.0.0/0"  # For better security, replace with your IP: "YOUR.IP.ADDRESS.HERE/32"
EOF
```

**Important:** Replace `practiceops-web-YOUR-UNIQUE-NAME` with your actual bucket name.

### 3.2 Initialize and Apply Terraform

```bash
# Initialize Terraform
terraform init

# Preview the infrastructure that will be created
terraform plan

# Create the infrastructure
terraform apply
```

Type `yes` when prompted.

### 3.3 Capture Terraform Outputs

After Terraform completes, run:

```bash
terraform output api_public_ip
terraform output web_website_url
```

**Save these values!**

Example outputs:
- `api_public_ip = "18.234.56.78"`
- `web_website_url = "practiceops-web-johnsmith-7x9k.s3-website-us-east-1.amazonaws.com"`

### 3.4 Update GitHub Secrets with Infrastructure Info

Go back to your GitHub repository secrets and update:

| Secret Name | Value |
|-------------|-------|
| `EC2_HOST` | The IP from `api_public_ip` (e.g., `18.234.56.78`) |
| `CORS_ORIGINS` | The full S3 website URL with `http://` prefix (e.g., `http://practiceops-web-johnsmith-7x9k.s3-website-us-east-1.amazonaws.com`) |
| `FRONTEND_URL` | Same as `CORS_ORIGINS` |

---

## Step 4: Deploy Application via GitHub Actions

### 4.1 Commit and Push

```bash
# Add all new files
git add .

# Commit the infrastructure code
git commit -m "Add AWS deployment infrastructure and workflows"

# Push to main branch to trigger deployment
git push origin main
```

### 4.2 Monitor Deployment

1. Go to GitHub → **Actions** tab
2. You should see a workflow run titled "Deploy PracticeOps to AWS"
3. Click on the run to see detailed logs
4. Wait for all steps to complete (usually 10-15 minutes)

### 4.3 Check for Success

Look for:
- ✅ All steps showing green checkmarks
- ✅ "Health check passed!" message at the end

---

## Step 5: Verify Deployment

### 5.1 Test API

Open your browser or use curl:

```bash
# Health check
curl http://YOUR_EC2_IP:8000/health

# API documentation
# Visit in browser: http://YOUR_EC2_IP:8000/docs
```

### 5.2 Test Frontend

Open in your browser:
```
http://YOUR_S3_BUCKET_NAME.s3-website-YOUR_REGION.amazonaws.com
```

You should see the PracticeOps login page.

### 5.3 Seed Demo Data (Optional)

SSH into your EC2 instance:

```bash
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP
```

Then run:

```bash
cd /opt/practiceops
docker compose -f docker-compose.prod.yml exec api python -m scripts.seed_demo
```

This will create demo users, teams, and practice data.

Default login credentials:
- Email: `director@example.com`
- Password: `password123`

---

## Step 6: Post-Deployment Configuration

### 6.1 Enable Database Backups

SSH into EC2 and create a backup script:

```bash
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP

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

### 6.2 Tighten SSH Access (Recommended)

For better security, update the security group to only allow SSH from your IP:

1. Get your current IP: `curl ifconfig.me`
2. Update `infra/terraform.tfvars`:
   ```hcl
   ssh_cidr = "YOUR.IP.ADDRESS/32"
   ```
3. Run `terraform apply` again

---

## Ongoing Operations

### View Application Logs

```bash
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP

# API logs
docker compose -f /opt/practiceops/docker-compose.prod.yml logs -f api

# Database logs
docker compose -f /opt/practiceops/docker-compose.prod.yml logs -f db
```

### Redeploy Application

Just push to the main branch:

```bash
git add .
git commit -m "Your changes"
git push origin main
```

GitHub Actions will automatically rebuild and redeploy.

### Manual Restart

```bash
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP
cd /opt/practiceops
docker compose -f docker-compose.prod.yml restart
```

### Check AWS Costs

```bash
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost
```

Or check in AWS Console → Billing Dashboard

---

## Troubleshooting

### Issue: GitHub Actions workflow fails at Terraform step

**Solution:** Verify your AWS credentials in GitHub Secrets are correct:
```bash
aws sts get-caller-identity
```

### Issue: Cannot SSH into EC2

**Solutions:**
1. Verify security group allows SSH from your IP
2. Check that you're using the correct key file
3. Ensure key file has correct permissions (Linux/Mac): `chmod 400 practiceops-key.pem`

### Issue: Health check fails

**Solution:** SSH into EC2 and check container status:
```bash
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP
docker ps
docker logs practiceops-api-prod
```

### Issue: Frontend shows blank page

**Solutions:**
1. Check browser console for errors
2. Verify `VITE_API_URL` environment variable was set correctly during build
3. Check CORS settings in API

### Issue: Database migrations fail

**Solution:**
```bash
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP
cd /opt/practiceops
docker compose -f docker-compose.prod.yml exec api alembic upgrade head
```

---

## Destroying Infrastructure

If you need to tear down everything:

```bash
cd infra
terraform destroy
```

**Warning:** This will delete:
- EC2 instance and all data
- S3 bucket and all files
- Security groups
- All database data (unless you've backed it up)

---

## Cost Monitoring Checklist

- [ ] Set up AWS billing alerts for $5/month
- [ ] Monitor EC2 instance usage
- [ ] Check S3 storage size monthly
- [ ] Review data transfer costs
- [ ] After 12 months free tier, consider switching to t4g.nano

---

## Security Hardening Checklist

- [ ] Tighten SSH access to your IP only
- [ ] Enable S3 bucket versioning
- [ ] Set up CloudWatch alerts for EC2 health
- [ ] Configure automatic security updates on EC2
- [ ] Review and restrict CORS origins
- [ ] Implement database backup strategy
- [ ] Consider AWS Secrets Manager for production secrets

---

## Next Steps (Optional Enhancements)

1. **Custom Domain:** Configure Route 53 with your domain
2. **HTTPS:** Add CloudFront + ACM certificate for S3, Let's Encrypt for EC2
3. **Monitoring:** Set up CloudWatch dashboards
4. **CI/CD Improvements:** Add staging environment, run tests before deploy
5. **Scaling:** Move to RDS and ECS when traffic increases

---

## Support

- **GitHub Issues:** Report problems or ask questions
- **AWS Documentation:** https://docs.aws.amazon.com/
- **Terraform Documentation:** https://terraform.io/docs

---

**Deployment Date:** January 2026
**Estimated Setup Time:** 2 hours
**Estimated Monthly Cost:** $0 (free tier) to $5/month

---

## Quick Reference

**Your Configuration:**
- AWS Region: `us-east-1` (or your chosen region)
- S3 Bucket: `practiceops-web-[YOUR-UNIQUE-NAME]`
- EC2 Key Pair: `practiceops-key`
- EC2 Instance Type: `t3.micro`
- API Port: `8000`
- Database: PostgreSQL 16 (containerized)

**Important Files:**
- Infrastructure: `infra/`
- Production Compose: `docker-compose.prod.yml`
- GitHub Workflow: `.github/workflows/deploy.yml`
- API Dockerfile: `apps/api/Dockerfile`
- Web Dockerfile: `apps/web/Dockerfile`

**Useful Commands:**
```bash
# Check infrastructure status
cd infra && terraform show

# Get current IP
curl ifconfig.me

# View EC2 system status
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP 'docker ps && df -h'

# Create manual backup
ssh -i practiceops-key.pem ubuntu@YOUR_EC2_IP '/opt/practiceops/backup.sh'
```

---

Good luck with your deployment! 🚀
