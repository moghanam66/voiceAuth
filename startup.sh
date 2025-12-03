#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Start the Flask application
exec python api_server.py