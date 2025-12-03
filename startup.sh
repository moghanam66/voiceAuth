#!/bin/bash

echo "Starting Voice Authentication API..."

# Download Vosk model if not present
if [ ! -d "vosk-model" ]; then
    echo "Downloading Vosk model..."
    python download_vosk_model.py
fi

# Check if CEO embedding exists
if [ ! -f "embeddings/ceo_voice.pkl" ]; then
    echo "WARNING: CEO voice not enrolled!"
    echo "Use POST /enroll endpoint to enroll CEO voice"
fi

# Start Gunicorn with cloud-ready API server
echo "Starting Gunicorn server..."
exec gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 2 api_server_cloud:app