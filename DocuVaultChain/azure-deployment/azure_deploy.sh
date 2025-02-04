#!/bin/bash
# azure-deployment/azure_deploy.sh
set -e

RESOURCE_GROUP="DocuVaultChainRG"
TEMPLATE_FILE="azure_infrastructure.json"
DEPLOYMENT_NAME="DocuVaultChainDeployment"

echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location "East US"

echo "Deploying infrastructure..."
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --name $DEPLOYMENT_NAME \
  --template-file $TEMPLATE_FILE

echo "Deployment complete."
