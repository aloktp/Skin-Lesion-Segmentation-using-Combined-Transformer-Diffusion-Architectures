import os
import sys
from flask import Flask
from dotenv import load_dotenv
from app.routes import main as main_blueprint
from app.utils import setup_logging, load_environment, check_essential_paths

# Conditionally load .env file only if running outside Docker
load_environment()

# Initialize Flask app
app = Flask(__name__)
app.register_blueprint(main_blueprint)

# Set up logging
logger = setup_logging()
logger.info("Logger initialized.")

# Load configurations from environment variables
app.config['APP_DATA'] = os.getenv('APP_DATA_PATH', '/shared/data')
app.config['APP_RESULT'] = os.getenv('APP_RESULT_PATH', '/shared/result')
app.config['APP_LOG'] = os.path.join(os.getenv('APP_LOG_PATH', '/shared/logs'), 'app.log')
app.config['APP_MODEL'] = os.getenv('APP_MODEL_PATH', '/shared/models')

# Check essential paths and environment variables
check_essential_paths(logger)

# Start the application
if __name__ == "__main__":
    try:
        web_host = os.getenv("WEB_HOST", "0.0.0.0")
        web_port = int(os.getenv("WEB_PORT", 8000))
        app.run(
            host=web_host,
            port=web_port,
            debug=(os.getenv("ENV") == "development")
        )
    except ValueError as e:
        logger.error("Invalid port configuration.")
        print(f"Error: {e}")
        sys.exit(1)
