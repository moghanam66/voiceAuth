#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
RESOURCE_GROUP="your-resource-group-name"
APP_NAME="your-app-service-name"
LOCATION="your-location" # e.g., "East US"
PLAN_NAME="your-app-service-plan-name"

# Create a resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create an App Service plan
az appservice plan create --name $PLAN_NAME --resource-group $RESOURCE_GROUP --sku B1 --is-linux

# Create the web app
az webapp create --name $APP_NAME --resource-group $RESOURCE_GROUP --plan $PLAN_NAME --runtime "PYTHON|3.8"

# Configure the web app to use the startup script
az webapp config set --resource-group $RESOURCE_GROUP --name $APP_NAME --startup-file startup.sh

# Deploy the application
az webapp up --name $APP_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --plan $PLAN_NAME --runtime "PYTHON|3.8"

# Output the URL of the deployed app
echo "Your app is deployed at: https://$APP_NAME.azurewebsites.net"