# infrastructure/aws/ecs.tf
resource "aws_ecs_cluster" "cyber_cluster" {
  name = "cyber-response-cluster"
}

resource "aws_security_group" "elk_sg" {
  ingress {
    from_port = 5044
    to_port = 5044
    protocol = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
}