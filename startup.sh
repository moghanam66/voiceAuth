#!/bin/bash
echo "Starting Voice Authentication API..."
mkdir -p embeddings
gunicorn --bind=0.0.0.0:$PORT --timeout=300 --workers=2 api_server_auth:app