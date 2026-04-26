# Sample Terraform: Artifact Registry, Cloud SQL (Postgres), Secret Manager, Cloud Run.
# - Does not set production IAM / VPC details (tune for your org).
# - After apply: build/push an image, then set Cloud Run image + DATABASE_URL/Secret wiring (console or gcloud).

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
  }
}

variable "project_id" {
  type        = string
  description = "GCP project id"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "db_password" {
  type        = string
  description = "Cloud SQL app user password; pass via TF_VAR_db_password"
  sensitive   = true
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_artifact_registry_repository" "devflow" {
  location      = var.region
  repository_id = "devflow"
  format        = "DOCKER"
  description   = "DevFlow API images (example: REGION-docker.pkg.dev/PROJECT/devflow/devflow-api:TAG)"
}

resource "google_sql_database_instance" "main" {
  name             = "devflow-pg"
  region           = var.region
  database_version = "POSTGRES_16"
  settings {
    tier = "db-f1-micro" # increase for production
    ip_configuration {
      ipv4_enabled = true
    }
  }
  deletion_protection = false
}

resource "google_sql_database" "app" {
  name     = "devflow"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "app" {
  name     = "devflow"
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

resource "google_secret_manager_secret" "openai" {
  secret_id = "devflow_openai_api_key"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "openai" {
  secret      = google_secret_manager_secret.openai.id
  secret_data = "replace-me-in-ui"
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/devflow"
}

output "cloud_sql_connection_name" {
  value = google_sql_database_instance.main.connection_name
}

output "openai_secret_id" {
  value = google_secret_manager_secret.openai.name
}
