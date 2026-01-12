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
