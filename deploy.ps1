# Quick deployment script for Azure
Write-Host "========================================"
Write-Host "Azure Deployment Script"
Write-Host "========================================"
Write-Host ""

# Variables (UPDATE THESE)
$RESOURCE_GROUP = "wakeword-rg"
$APP_NAME = "wakeword-api-$(Get-Random -Maximum 9999)"
$LOCATION = "eastus"
$SKU = "P1v2"

Write-Host "Configuration:"
Write-Host "  Resource Group: $RESOURCE_GROUP"
Write-Host "  App Name: $APP_NAME"
Write-Host "  Location: $LOCATION"
Write-Host "  SKU: $SKU"
Write-Host ""

# Login check
Write-Host "Checking Azure login..."
$account = az account show 2>$null | ConvertFrom-Json
if (!$account) {
    Write-Host "Not logged in. Running az login..."
    az login
}
Write-Host "✓ Logged in as: $($account.user.name)"
Write-Host ""

# Create resource group
Write-Host "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION
Write-Host ""

# Create App Service plan
Write-Host "Creating App Service plan..."
az appservice plan create `
  --name "$APP_NAME-plan" `
  --resource-group $RESOURCE_GROUP `
  --sku $SKU `
  --is-linux
Write-Host ""

# Create web app
Write-Host "Creating web app..."
az webapp create `
  --resource-group $RESOURCE_GROUP `
  --plan "$APP_NAME-plan" `
  --name $APP_NAME `
  --runtime "PYTHON:3.11"
Write-Host ""

# Configure app settings
Write-Host "Configuring app settings..."
az webapp config set `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --startup-file "startup.sh"

az webapp config appsettings set `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --settings `
    WEBSITES_PORT=5000 `
    SCM_DO_BUILD_DURING_DEPLOYMENT=true `
    ENABLE_ORYX_BUILD=true
Write-Host ""

# Deploy code
Write-Host "Deploying code..."
az webapp up `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --runtime "PYTHON:3.11" `
  --sku $SKU
Write-Host ""

# Get URL
$url = az webapp show `
  --resource-group $RESOURCE_GROUP `
  --name $APP_NAME `
  --query defaultHostName `
  --output tsv

Write-Host "========================================"
Write-Host "✓ Deployment Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "Your API is available at:"
Write-Host "  https://$url"
Write-Host ""
Write-Host "Test health endpoint:"
Write-Host "  curl https://$url/health"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Upload CEO voice embedding (embeddings/ceo_voice.pkl)"
Write-Host "  2. Wait for Vosk model to download (~5-10 minutes)"
Write-Host "  3. Check logs: az webapp log tail -g $RESOURCE_GROUP -n $APP_NAME"
Write-Host ""
Write-Host "Kudu console:"
Write-Host "  https://$APP_NAME.scm.azurewebsites.net"
Write-Host ""
