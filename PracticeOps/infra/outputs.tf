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
