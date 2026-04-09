output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer — use this to access the app"
  value       = aws_lb.main.dns_name
}

output "rds_endpoint" {
  description = "Connection endpoint for the PostgreSQL RDS instance"
  value       = aws_db_instance.main.endpoint
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster running the application services"
  value       = aws_ecs_cluster.main.name
}

output "backend_service_name" {
  description = "Name of the ECS service running the FastAPI backend"
  value       = aws_ecs_service.backend.name
}

output "frontend_service_name" {
  description = "Name of the ECS service running the React frontend"
  value       = aws_ecs_service.frontend.name
}
