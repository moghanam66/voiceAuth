#!/bin/bash

echo "Starting Voice Authentication API (Authentication Only)..."

# Check if CEO embedding exists
if [ ! -f "embeddings/ceo_voice.pkl" ]; then
    echo "WARNING: CEO voice not enrolled!"
    echo "Use POST /enroll endpoint to enroll CEO voice"
fi

# Start Gunicorn with authentication-only API server
echo "Starting Gunicorn server..."
exec gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 2 api_server_auth:app