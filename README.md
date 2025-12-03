# Wake Word Detection and Voice Authentication API

This project implements a REST API server for wake word detection and voice authentication using Flask. It provides real-time status updates via HTTP endpoints and is designed to be deployed on Azure App Service.

## Project Structure

```
wakeword-azure-deployment
├── api_server.py            # REST API server implementation
├── main_realtime.py         # Continuous voice detection and authentication logic
├── config.py                # Configuration settings (logging levels, formats, etc.)
├── requirements.txt         # Python dependencies
├── startup.sh               # Shell script to start the application on Azure
├── .deployment              # Deployment configuration files
├── deploy.sh                # Automation script for deployment to Azure
├── azure                    # Azure-specific configuration and templates
│   ├── app-service-config.json  # Azure App Service configuration settings
│   └── deployment-template.json  # ARM template for resource deployment
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
└── README.md                # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd wakeword-azure-deployment
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables by copying `.env.example` to `.env` and modifying as needed.

## Usage

To run the API server locally, execute:
```
python api_server.py
```

The server will start on `http://localhost:5000`, and you can access the following endpoints:

- `GET /status` - Full status (wake_word, voice_authenticated, listening, score)
- `GET /wake_word` - Wake word detection status only
- `GET /voice_auth` - Voice authentication status only
- `GET /health` - Health check

## Deployment on Azure

To deploy the application on Azure App Service, follow these steps:

1. Ensure you have the Azure CLI installed and are logged in:
   ```
   az login
   ```

2. Run the deployment script:
   ```
   ./deploy.sh
   ```

This will set up the necessary resources and deploy the application to Azure.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.