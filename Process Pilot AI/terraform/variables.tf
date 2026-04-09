variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Application name used for resource naming and tagging"
  type        = string
  default     = "processpilot"
}

variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
  default     = "production"
}

variable "db_instance_class" {
  description = "RDS instance class for PostgreSQL"
  type        = string
  default     = "db.t3.micro"
}

variable "db_name" {
  description = "Name of the PostgreSQL database"
  type        = string
  default     = "processpilot"
}

variable "db_username" {
  description = "Master username for the RDS instance"
  type        = string
  default     = "processpilot_admin"
}

variable "container_cpu" {
  description = "CPU units for ECS Fargate tasks (1 vCPU = 1024)"
  type        = number
  default     = 256
}

variable "container_memory" {
  description = "Memory in MiB for ECS Fargate tasks"
  type        = number
  default     = 512
}

variable "desired_count" {
  description = "Desired number of ECS task instances per service"
  type        = number
  default     = 2
}
