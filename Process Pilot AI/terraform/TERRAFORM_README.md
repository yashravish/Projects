# ProcessPilot AI — Terraform Infrastructure Scaffold

This directory contains Infrastructure as Code (IaC) definitions for deploying ProcessPilot AI on AWS using Terraform. The configuration provisions a production-grade, cloud-native architecture with ECS Fargate, RDS PostgreSQL, and an Application Load Balancer.

> **Portfolio Note:** This Terraform scaffold demonstrates cloud architecture thinking and IaC proficiency. It is syntactically valid and structurally complete, but has not been deployed to a live AWS account. Secrets are placeholder values and certain production-hardening steps (private subnets, NAT gateways, WAF, HTTPS) are noted as future work.

---

## What Gets Provisioned

| Resource | Purpose | Maps to App Component |
|---|---|---|
| **VPC + Subnets** | Isolated network with 2 public subnets across AZs | Network foundation |
| **Internet Gateway + Route Table** | Public internet access for ALB and Fargate tasks | Connectivity |
| **Application Load Balancer** | HTTP traffic entry point with path-based routing | Reverse proxy |
| **ECS Cluster** | Container orchestration platform | Runtime environment |
| **ECS Service — Backend** | Runs FastAPI containers on Fargate | `backend/` |
| **ECS Service — Frontend** | Runs nginx + React build on Fargate | `frontend/` |
| **ECS Task Definitions** | Container specs (CPU, memory, env vars, logging) | Container config |
| **RDS PostgreSQL** | Managed relational database (postgres:15) | Data persistence |
| **Security Groups** | Network-level access control (ALB → ECS → RDS) | Security boundary |
| **IAM Execution Role** | Permissions for ECS to pull images and write logs | Permissions |
| **CloudWatch Log Groups** | Centralized log aggregation for both services | Observability |

## Architecture Mapping

```
Internet
   │
   ▼
┌─────────────────────────────┐
│  Application Load Balancer  │
│  (Port 80)                  │
│                             │
│  /api/*  → Backend TG       │
│  /*      → Frontend TG      │
└──────┬──────────┬───────────┘
       │          │
       ▼          ▼
┌────────────┐ ┌────────────┐
│ ECS Backend│ │ECS Frontend│
│ (Fargate)  │ │ (Fargate)  │
│ Port 8000  │ │  Port 80   │
└──────┬─────┘ └────────────┘
       │
       ▼
┌────────────┐
│    RDS     │
│ PostgreSQL │
│ Port 5432  │
└────────────┘
```

## How to Deploy

### Prerequisites

1. **AWS CLI** configured with credentials (`aws configure`)
2. **Terraform >= 1.0** installed ([download](https://developer.hashicorp.com/terraform/downloads))
3. **Docker images** pushed to Amazon ECR (or another registry)
4. A `terraform.tfvars` file with your secrets (see below)

### Deployment Steps

```bash
# 1. Initialize Terraform and download provider plugins
cd terraform
terraform init

# 2. Preview the infrastructure changes
terraform plan

# 3. Apply the changes (type 'yes' to confirm)
terraform apply

# 4. Retrieve the ALB URL
terraform output alb_dns_name
```

### Tear Down

```bash
terraform destroy
```

## Environment Variables & Secrets

The task definitions reference several environment variables. In production you should **never** hard-code secrets in Terraform files. Instead:

| Variable | How to Manage |
|---|---|
| `DATABASE_URL` | Constructed from RDS outputs; store password in AWS Secrets Manager |
| `JWT_SECRET` | Generate a strong random value; store in Secrets Manager |
| `OPENAI_API_KEY` | Store in Secrets Manager; reference via `secrets` block in task definition |
| `APP_ENV` | Set via `terraform.tfvars` or CI/CD pipeline variable |

**Recommended approach:** Use the `aws_secretsmanager_secret` resource and reference secrets in the ECS task definition's `secrets` block rather than `environment`.

## Cost Estimation

Approximate monthly costs for the default configuration (us-east-1):

| Resource | Configuration | Est. Monthly Cost |
|---|---|---|
| ECS Fargate (backend) | 2 tasks × 0.25 vCPU / 512 MiB | ~$18 |
| ECS Fargate (frontend) | 2 tasks × 0.25 vCPU / 512 MiB | ~$18 |
| RDS PostgreSQL | db.t3.micro, 20 GB | ~$15 |
| Application Load Balancer | 1 ALB + LCUs | ~$18 |
| CloudWatch Logs | Minimal volume | ~$1 |
| **Total** | | **~$70/month** |

> Costs vary with traffic volume and region. Use the [AWS Pricing Calculator](https://calculator.aws/) for precise estimates.

## Security Considerations

**Implemented in this scaffold:**
- Security groups enforce least-privilege network access (ALB → ECS → RDS chain)
- RDS is not publicly accessible
- IAM execution role follows the managed policy for ECS tasks
- Container Insights enabled for observability

**Recommended for production hardening:**
- Move ECS tasks and RDS into **private subnets** with NAT gateways
- Add **HTTPS** via ACM certificate on the ALB listener (port 443)
- Enable **AWS WAF** on the ALB for DDoS and injection protection
- Use **Secrets Manager** for all sensitive values (DB password, JWT secret, API keys)
- Enable **RDS encryption at rest** and automated backups with longer retention
- Add **auto-scaling policies** for ECS services based on CPU/memory thresholds
- Implement **VPC Flow Logs** for network audit trail
- Restrict IAM roles with custom policies instead of managed policies

## File Structure

```
terraform/
├── provider.tf          # AWS provider and Terraform version constraints
├── variables.tf         # Input variables with defaults and descriptions
├── main.tf              # All infrastructure resources
├── outputs.tf           # Exported values (ALB DNS, RDS endpoint, etc.)
└── TERRAFORM_README.md  # This file
```
