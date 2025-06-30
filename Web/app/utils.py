import os
import sys
import logging
import json
from dotenv import load_dotenv
import csv
import re

load_dotenv()

def load_environment():
    if not os.getenv("DOCKER_ENV"):  
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        if not load_dotenv(env_path):
            print("Error: .env file could not be loaded.")
            sys.exit(1)


def verify_images_in_mapping(task_type, dataset_name, image_files):
    mapping_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, "mapping.csv")
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")

    # Read the mapping.csv and gather all images listed in the "image_name" column
    mapped_images = set()
    with open(mapping_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            mapped_images.add(row["image_name"])

    # Check for missing images in the mapping
    missing_images = [img for img in image_files if img not in mapped_images]
    return missing_images

def setup_logging():
    log_dir = os.getenv("APP_LOG_PATH", "/shared/logs")
    os.makedirs(log_dir, exist_ok=True)  
    log_file = os.path.join(log_dir, "app.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging has been configured.")
    return logger

def check_essential_paths(logger):
    required_paths = {
        "APP_DATA_PATH": os.getenv("APP_DATA_PATH"),
        "APP_LOG_PATH": os.getenv("APP_LOG_PATH"),
        "APP_RESULT_PATH": os.getenv("APP_RESULT_PATH")
    }
    
    for path_name, path_value in required_paths.items():
        if not path_value:
            logger.error(f"{path_name} is not set.")
            print(f"Error: {path_name} is not set in the environment.")
            sys.exit(1)
        elif not os.path.exists(path_value):
            logger.error(f"Path specified in {path_name} does not exist: {path_value}")
            print(f"Error: Path specified in {path_name} does not exist: {path_value}")
            sys.exit(1)

def get_model_list():
    model_folder = os.getenv("APP_MODEL_PATH", "/shared/models")
    if not os.path.isdir(model_folder):
        raise FileNotFoundError(f"Model folder not found at {model_folder}")
    return [f for f in os.listdir(model_folder) if f.endswith((".pt", ".pth", ".pkl"))]

def get_dataset_list():
    ground_truth_folder = os.path.join(os.getenv("APP_DATA_PATH"), "ground_truth")
    if not os.path.isdir(ground_truth_folder):
        raise FileNotFoundError(f"Ground truth folder not found at {ground_truth_folder}")
    return [folder for folder in os.listdir(ground_truth_folder) if os.path.isdir(os.path.join(ground_truth_folder, folder))]

def get_task_types_from_data():
    data_record_path = os.path.join(os.getenv("APP_DATA_PATH"), "data_record.json")
    if not os.path.exists(data_record_path):
        raise FileNotFoundError(f"Data record file not found at {data_record_path}")
    
    with open(data_record_path, 'r') as file:
        records = json.load(file)
    return {record["Task Type"] for record in records if record["Active"]}

def get_datasets_by_task(task_type):
    data_record_path = os.path.join(os.getenv("APP_DATA_PATH"), "data_record.json")
    if not os.path.exists(data_record_path):
        raise FileNotFoundError(f"Data record file not found at {data_record_path}")

    with open(data_record_path, 'r') as file:
        records = json.load(file)
    return {record["Dataset Name"] for record in records if record["Task Type"] == task_type and record["Active"]}


def extract_id(filename, pattern):
    match = re.search(pattern, filename, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        return None