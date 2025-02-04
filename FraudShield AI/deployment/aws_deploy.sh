#!/bin/bash
# deployment/aws_deploy.sh
# This script deploys FraudShield AI components to AWS using CloudFormation and the AWS CLI.

set -e

echo "Deploying AWS Infrastructure using CloudFormation..."

# Replace STACK_NAME and TEMPLATE_FILE as needed.
STACK_NAME="FraudShieldAIStack"
TEMPLATE_FILE="aws_infrastructure.yaml"

aws cloudformation deploy --stack-name $STACK_NAME --template-file $TEMPLATE_FILE --capabilities CAPABILITY_NAMED_IAM

echo "AWS Infrastructure deployed successfully."

# Additional deployment steps for Lambda functions and other services can be added here.
