# AWS Deployment Guide

This document covers deploying the Enterprise Integration Gateway to AWS using two approaches: **ECS Fargate** (recommended for production) and **Lambda** (serverless, cost-optimized for low traffic).

---

## Architecture on AWS

```
                    Internet
                       │
                ┌──────▼──────┐
                │   Route 53  │
                │  (DNS)      │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │     ALB     │ ◄── Public subnets
                │  (port 80/  │
                │   443)      │
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │       Private subnets
  ┌─────▼─────┐  ┌─────▼─────┐  ┌────▼────┐
  │ ECS Task  │  │ ECS Task  │  │ Lambda  │
  │ (Fargate) │  │ (Fargate) │  │ (SAM)   │
  └─────┬─────┘  └─────┬─────┘  └────┬────┘
        │              │              │
   ┌────┴──────────────┴──────────────┴────┐
   │                                        │
   │  ┌──────────┐  ┌────────┐  ┌───────┐  │
   │  │ RDS      │  │ Redis  │  │ MSK   │  │
   │  │ Postgres │  │ Elasti │  │ Kafka │  │
   │  │          │  │ Cache  │  │       │  │
   │  └──────────┘  └────────┘  └───────┘  │
   │                                        │
   └────────────────────────────────────────┘
           Private subnets (data tier)
```

---

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Docker** installed locally for building images
3. **AWS SAM CLI** (for Lambda deployment only)
4. An **ECR repository** created: `aws ecr create-repository --repository-name eig-app`

---

## Option A: ECS Fargate Deployment

### Step 1 — Deploy Infrastructure

```bash
aws cloudformation deploy \
  --template-file aws/cloudformation/infrastructure.yaml \
  --stack-name eig-production-infra \
  --parameter-overrides \
    EnvironmentName=production \
    DbMasterPassword=<SECURE_PASSWORD> \
  --capabilities CAPABILITY_IAM \
  --tags Project=enterprise-integration-gateway
```

This creates: VPC, subnets, RDS PostgreSQL, ElastiCache Redis, MSK Kafka, ECS cluster, ALB, security groups, and CloudWatch log groups.

### Step 2 — Build and Push Docker Image

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t eig-app .
docker tag eig-app:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/eig-app:latest

# Push
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/eig-app:latest
```

### Step 3 — Register Task Definition

```bash
# Replace placeholders in task-definition.json with actual values
aws ecs register-task-definition \
  --cli-input-json file://aws/ecs/task-definition.json
```

### Step 4 — Create ECS Service

```bash
aws ecs create-service \
  --cli-input-json file://aws/ecs/service-definition.json
```

### Step 5 — Verify

```bash
# Get ALB DNS name from CloudFormation outputs
ALB_DNS=$(aws cloudformation describe-stacks \
  --stack-name eig-production-infra \
  --query "Stacks[0].Outputs[?OutputKey=='ALBEndpoint'].OutputValue" \
  --output text)

curl http://$ALB_DNS/api/v1/health
```

---

## Option B: Lambda Deployment (Serverless)

### Step 1 — Deploy with SAM

```bash
cd aws/lambda

# Build
sam build

# Deploy (guided first time)
sam deploy --guided \
  --stack-name eig-lambda \
  --capabilities CAPABILITY_IAM
```

### Step 2 — Verify

```bash
# Get API endpoint from stack outputs
API_URL=$(aws cloudformation describe-stacks \
  --stack-name eig-lambda \
  --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" \
  --output text)

curl $API_URL/api/v1/health
```

### Lambda Considerations

| Feature | ECS Fargate | Lambda |
|---------|-------------|--------|
| APScheduler | ✅ Runs natively | ❌ Use EventBridge rules |
| Kafka consumer | ✅ Background thread | ❌ Use MSK Lambda trigger |
| Cold starts | None | ~2-5 seconds |
| Cost at low traffic | Higher (always-on) | Lower (pay-per-request) |
| Cost at high traffic | Lower (predictable) | Higher (per-invocation) |
| Max request duration | Unlimited | 30 seconds |

---

## Environment Variables — Local vs AWS

| Local (`.env`) | AWS (SSM Parameter Store) |
|----------------|---------------------------|
| `DATABASE_URL=postgresql://...@localhost:5432/eig_db` | `/eig/production/DATABASE_URL` → RDS endpoint |
| `REDIS_URL=redis://localhost:6379/0` | `/eig/production/REDIS_URL` → ElastiCache endpoint |
| `KAFKA_BOOTSTRAP_SERVERS=localhost:9092` | `/eig/production/KAFKA_BOOTSTRAP_SERVERS` → MSK brokers |
| `CRM_BASE_URL=http://localhost:8001` | `/eig/production/CRM_BASE_URL` → actual CRM URL |
| `VENDOR_BASE_URL=http://localhost:8001` | `/eig/production/VENDOR_BASE_URL` → actual vendor URL |

Sensitive values are stored in SSM Parameter Store and injected at runtime via the ECS task definition `secrets` block.

---

## CI/CD Pipeline with CodeBuild

The `aws/buildspec.yml` defines a three-phase pipeline:

1. **pre_build** — Run test suite, authenticate with ECR
2. **build** — Build and tag Docker image
3. **post_build** — Push to ECR, generate `imagedefinitions.json`

Set up a CodePipeline with:
- **Source**: GitHub (via CodeStar connection)
- **Build**: CodeBuild (uses `aws/buildspec.yml`)
- **Deploy**: ECS (uses `imagedefinitions.json` for rolling deploy)

---

## Auto-Scaling

### ECS Service Auto-Scaling

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/eig-production-cluster/eig-app-service \
  --min-capacity 2 \
  --max-capacity 10

# CPU-based scaling policy
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/eig-production-cluster/eig-app-service \
  --policy-name eig-cpu-scaling \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'
```

---

## Monitoring & Alerting

### CloudWatch Alarms (recommended)

| Alarm | Metric | Threshold |
|-------|--------|-----------|
| High CPU | `ECSServiceAverageCPUUtilization` | > 80% for 5 min |
| High Memory | `ECSServiceAverageMemoryUtilization` | > 85% for 5 min |
| 5xx Errors | `ALB HTTPCode_Target_5XX_Count` | > 10 in 5 min |
| Unhealthy Hosts | `ALB UnHealthyHostCount` | > 0 for 5 min |
| DB Connections | `RDS DatabaseConnections` | > 80% of max |
| Cache Hit Rate | `ElastiCache CacheHitRate` | < 50% for 15 min |

### Structured Logging

All application logs are JSON-formatted and sent to CloudWatch via the `awslogs` driver. Use CloudWatch Insights for queries:

```
fields @timestamp, @message
| filter event_type = "sync_job_finished"
| stats count(*) by status
| sort @timestamp desc
```

---

## Cost Estimation (us-east-1)

| Resource | Config | Est. Monthly Cost |
|----------|--------|-------------------|
| ECS Fargate (2 tasks) | 0.5 vCPU, 1 GB | ~$30 |
| RDS PostgreSQL | db.t3.micro | ~$15 |
| ElastiCache Redis | cache.t3.micro | ~$13 |
| MSK Kafka (2 brokers) | kafka.t3.small | ~$60 |
| ALB | Standard | ~$16 |
| NAT Gateway | Single AZ | ~$32 |
| CloudWatch | Logs + metrics | ~$5 |
| **Total** | | **~$171/month** |

### Cost Optimization Tips

1. Use **Reserved Instances** for RDS and ElastiCache (up to 40% savings)
2. Use **Savings Plans** for Fargate (up to 50% savings)
3. Consider **Lambda** for low-traffic environments ($5-10/month)
4. Set CloudWatch log retention to 30 days (already configured)
5. Use **single-AZ** NAT Gateway for staging environments
