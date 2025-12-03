# 🚀 Azure App Service Deployment Guide

## Overview
Deploy the Wake Word Detection API to Azure App Service for production use.

---

## 📋 Prerequisites

1. **Azure Account** with active subscription
2. **Azure CLI** installed ([Download](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
3. **Git** installed
4. **Python 3.9+** locally for testing

---

## 🎯 Deployment Steps

### Step 1: Prepare Your Project

```powershell
# Navigate to deployment folder
cd c:\wakeword__\wakeword-azure-deployment

# Verify files are present
ls
```

Required files:
- `api_server_deployed.py`
- `requirements.txt`
- `startup.sh`
- `.deployment`
- `config.py`
- `voice_authenticator.py`
- `embeddings/ceo_voice.pkl`

---

### Step 2: Login to Azure

```powershell
# Login to Azure
az login

# Set your subscription (if you have multiple)
az account set --subscription "YOUR_SUBSCRIPTION_NAME"

# Verify
az account show
```

---

### Step 3: Create Resource Group

```powershell
# Create resource group
az group create --name wakeword-rg --location eastus

# Verify
az group list --output table
```

**Available locations:**
- `eastus`, `westus`, `westeurope`, `eastasia`, `australiaeast`

---

### Step 4: Create App Service Plan

```powershell
# Create Linux App Service Plan (P1v2 or higher recommended)
az appservice plan create `
  --name wakeword-plan `
  --resource-group wakeword-rg `
  --sku P1v2 `
  --is-linux

# For development/testing, use B1 (cheaper)
# az appservice plan create `
#   --name wakeword-plan `
#   --resource-group wakeword-rg `
#   --sku B1 `
#   --is-linux
```

**Recommended SKUs:**
- **Development:** B1 ($13/month)
- **Production:** P1v2 ($80/month) - 3.5 GB RAM
- **High Performance:** P2v2 ($160/month) - 7 GB RAM

---

### Step 5: Create Web App

```powershell
# Create Python web app
az webapp create `
  --resource-group wakeword-rg `
  --plan wakeword-plan `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --runtime "PYTHON:3.9" `
  --deployment-local-git

# Save the Git URL from output
```

**Note:** Replace `YOUR_UNIQUE_NAME` with something unique (e.g., your company name or random string).

---

### Step 6: Configure App Settings

```powershell
# Set startup command
az webapp config set `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --startup-file "startup.sh"

# Increase timeout (important for model loading)
az webapp config set `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --web-sockets-enabled true

# Configure environment variables
az webapp config appsettings set `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --settings `
    WEBSITES_PORT=5000 `
    SCM_DO_BUILD_DURING_DEPLOYMENT=true `
    ENABLE_ORYX_BUILD=true `
    POST_BUILD_COMMAND="python download_vosk_model.py"
```

---

### Step 7: Deploy Code

#### Option A: Deploy via Git

```powershell
# Initialize git (if not already done)
git init

# Add Azure remote
git remote add azure <GIT_URL_FROM_STEP_5>

# Add and commit files
git add .
git commit -m "Initial deployment"

# Deploy to Azure
git push azure main

# Or if your branch is master:
# git push azure master:main
```

#### Option B: Deploy via ZIP

```powershell
# Create deployment package
Compress-Archive -Path * -DestinationPath deploy.zip

# Deploy
az webapp deployment source config-zip `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --src deploy.zip
```

#### Option C: Deploy via Azure CLI (Recommended)

```powershell
# Deploy directly
az webapp up `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --runtime "PYTHON:3.9" `
  --sku P1v2 `
  --location eastus
```

---

### Step 8: Upload Large Files (Vosk Model)

The Vosk model (1.8 GB) is too large for git. Upload separately:

#### Option A: Kudu Console

1. Go to: `https://wakeword-api-YOUR_UNIQUE_NAME.scm.azurewebsites.net`
2. Navigate to **Debug Console** → **CMD**
3. Navigate to: `/home/site/wwwroot`
4. Upload `vosk-model` folder via drag-and-drop

#### Option B: FTP

```powershell
# Get FTP credentials
az webapp deployment list-publishing-credentials `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --query "{username:publishingUserName, password:publishingPassword}"

# Use FileZilla or WinSCP to upload:
# Host: ftp://wakeword-api-YOUR_UNIQUE_NAME.scm.azurewebsites.net
# Upload vosk-model folder to /site/wwwroot/
```

#### Option C: Azure Storage (Recommended for large files)

```powershell
# Create storage account
az storage account create `
  --name wakewordstore `
  --resource-group wakeword-rg `
  --location eastus `
  --sku Standard_LRS

# Upload model to blob storage
az storage container create `
  --name models `
  --account-name wakewordstore

az storage blob upload-batch `
  --destination models `
  --source c:\wakeword__\vosk-model `
  --account-name wakewordstore

# Download in startup script
# Add to startup.sh:
# az storage blob download-batch --destination vosk-model --source models --account-name wakewordstore
```

---

### Step 9: Verify Deployment

```powershell
# Check app status
az webapp show `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --query state

# Get URL
az webapp show `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --query defaultHostName

# Test health endpoint
curl https://wakeword-api-YOUR_UNIQUE_NAME.azurewebsites.net/health
```

---

### Step 10: Monitor Logs

```powershell
# Stream logs
az webapp log tail `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME

# Or view in portal:
# https://portal.azure.com → Your App → Monitoring → Log stream
```

---

## 🔧 Configuration Files

### `requirements.txt`

```txt
flask==3.0.0
flask-cors==4.0.0
vosk==0.3.45
numpy==1.24.3
torch==2.0.1
torchaudio==2.0.2
speechbrain==0.5.16
sounddevice==0.4.6
gunicorn==21.2.0
```

### `startup.sh`

```bash
#!/bin/bash

echo "Starting Wake Word Detection API..."

# Download Vosk model if not present
if [ ! -d "vosk-model" ]; then
    echo "Downloading Vosk model..."
    python download_vosk_model.py
fi

# Check if CEO embedding exists
if [ ! -f "embeddings/ceo_voice.pkl" ]; then
    echo "WARNING: CEO voice not enrolled!"
fi

# Start Gunicorn
echo "Starting Gunicorn server..."
gunicorn --bind 0.0.0.0:5000 --timeout 300 --workers 2 api_server_deployed:app
```

### `.deployment`

```ini
[config]
command = bash startup.sh
```

### `download_vosk_model.py` (for auto-download)

```python
import os
import urllib.request
import zipfile

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip"
MODEL_ZIP = "vosk-model.zip"
MODEL_DIR = "vosk-model"

if not os.path.exists(MODEL_DIR):
    print(f"Downloading model from {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_ZIP)
    
    print("Extracting model...")
    with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
        zip_ref.extractall()
    
    # Rename extracted folder
    extracted = "vosk-model-en-us-0.22"
    if os.path.exists(extracted):
        os.rename(extracted, MODEL_DIR)
    
    # Cleanup
    os.remove(MODEL_ZIP)
    print(f"Model ready at {MODEL_DIR}")
else:
    print(f"Model already exists at {MODEL_DIR}")
```

---

## 🔐 Security Configuration

### Enable HTTPS Only

```powershell
az webapp update `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --https-only true
```

### Configure CORS

```powershell
az webapp cors add `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --allowed-origins "https://yourdomain.com"
```

### Add Custom Domain

```powershell
# Add custom domain
az webapp config hostname add `
  --resource-group wakeword-rg `
  --webapp-name wakeword-api-YOUR_UNIQUE_NAME `
  --hostname api.yourdomain.com

# Enable SSL
az webapp config ssl bind `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --certificate-thumbprint <THUMBPRINT> `
  --ssl-type SNI
```

---

## 📊 Scaling Configuration

### Auto-scaling

```powershell
# Enable auto-scale (requires Standard tier or higher)
az monitor autoscale create `
  --resource-group wakeword-rg `
  --resource wakeword-plan `
  --resource-type Microsoft.Web/serverfarms `
  --name wakeword-autoscale `
  --min-count 1 `
  --max-count 5 `
  --count 2

# Scale based on CPU
az monitor autoscale rule create `
  --resource-group wakeword-rg `
  --autoscale-name wakeword-autoscale `
  --condition "Percentage CPU > 70 avg 5m" `
  --scale out 1
```

### Manual Scaling

```powershell
# Scale to 3 instances
az appservice plan update `
  --name wakeword-plan `
  --resource-group wakeword-rg `
  --number-of-workers 3
```

---

## 🐛 Troubleshooting

### Issue: App won't start

```powershell
# Check logs
az webapp log tail --resource-group wakeword-rg --name wakeword-api-YOUR_UNIQUE_NAME

# Check app status
az webapp show --resource-group wakeword-rg --name wakeword-api-YOUR_UNIQUE_NAME

# Restart app
az webapp restart --resource-group wakeword-rg --name wakeword-api-YOUR_UNIQUE_NAME
```

### Issue: Model loading timeout

Increase timeout in `startup.sh`:
```bash
gunicorn --bind 0.0.0.0:5000 --timeout 600 --workers 2 api_server_deployed:app
```

### Issue: Out of memory

Upgrade to larger SKU:
```powershell
az appservice plan update `
  --name wakeword-plan `
  --resource-group wakeword-rg `
  --sku P2v2
```

### Issue: Slow response times

- Reduce workers in startup.sh
- Increase instance size
- Enable caching
- Optimize model loading

---

## 💰 Cost Optimization

### Development Environment

```powershell
# Use B1 tier (~$13/month)
az appservice plan update `
  --name wakeword-plan `
  --resource-group wakeword-rg `
  --sku B1
```

### Production Environment

```powershell
# Use P1v2 tier (~$80/month)
az appservice plan update `
  --name wakeword-plan `
  --resource-group wakeword-rg `
  --sku P1v2
```

### Stop when not in use

```powershell
# Stop app
az webapp stop --resource-group wakeword-rg --name wakeword-api-YOUR_UNIQUE_NAME

# Start app
az webapp start --resource-group wakeword-rg --name wakeword-api-YOUR_UNIQUE_NAME
```

---

## 📱 Testing Deployed API

### Test from PowerShell

```powershell
# Health check
Invoke-RestMethod -Uri "https://wakeword-api-YOUR_UNIQUE_NAME.azurewebsites.net/health"

# Process audio
$audioFile = "test.wav"
$uri = "https://wakeword-api-YOUR_UNIQUE_NAME.azurewebsites.net/process_audio"
$form = @{
    audio = Get-Item -Path $audioFile
}
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

### Test from Python

```python
import requests

url = "https://wakeword-api-YOUR_UNIQUE_NAME.azurewebsites.net"

# Health check
response = requests.get(f"{url}/health")
print(response.json())

# Process audio
with open('test.wav', 'rb') as f:
    files = {'audio': ('test.wav', f, 'audio/wav')}
    response = requests.post(f"{url}/process_audio", files=files)
    print(response.json())
```

---

## 🔄 CI/CD with GitHub Actions

Create `.github/workflows/azure-deploy.yml`:

```yaml
name: Deploy to Azure App Service

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Deploy to Azure Web App
      uses: azure/webapps-deploy@v2
      with:
        app-name: 'wakeword-api-YOUR_UNIQUE_NAME'
        publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
```

Get publish profile:
```powershell
az webapp deployment list-publishing-profiles `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME `
  --xml
```

Add to GitHub Secrets as `AZURE_WEBAPP_PUBLISH_PROFILE`.

---

## 🗑️ Cleanup (Delete Resources)

```powershell
# Delete entire resource group
az group delete --name wakeword-rg --yes --no-wait

# Or delete just the web app
az webapp delete `
  --resource-group wakeword-rg `
  --name wakeword-api-YOUR_UNIQUE_NAME
```

---

## ✅ Deployment Checklist

- [ ] Azure CLI installed and logged in
- [ ] Resource group created
- [ ] App Service Plan created (P1v2 or higher)
- [ ] Web App created
- [ ] Startup command configured
- [ ] Environment variables set
- [ ] Code deployed (via Git/ZIP/CLI)
- [ ] Vosk model uploaded (1.8 GB)
- [ ] CEO embedding uploaded (`embeddings/ceo_voice.pkl`)
- [ ] HTTPS enabled
- [ ] Health endpoint responds
- [ ] Test audio processing works
- [ ] Logs reviewed for errors
- [ ] CORS configured (if needed)
- [ ] Monitoring enabled
- [ ] Backup strategy defined

---

## 📞 Support Resources

- **Azure Documentation:** https://docs.microsoft.com/en-us/azure/app-service/
- **Azure CLI Reference:** https://docs.microsoft.com/en-us/cli/azure/
- **Pricing Calculator:** https://azure.microsoft.com/en-us/pricing/calculator/

---

**Your API will be available at:**
`https://wakeword-api-YOUR_UNIQUE_NAME.azurewebsites.net`

🎉 **Deployment Complete!**
