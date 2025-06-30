import os
import json
import logging
import shutil
from flask import Flask, request, jsonify
import random
from flask_cors import CORS
import csv
import requests
import subprocess
app = Flask(__name__)
CORS(app) 

# Configure logger
log_dir = os.getenv("APP_LOG_PATH", "/shared/logs")
os.makedirs(log_dir, exist_ok=True)  
log_file_path = os.path.join(log_dir, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

ai_host = os.getenv("AI_HOST", "localhost")
ai_port = os.getenv("AI_PORT", "3000")
base_url = f"http://{ai_host}:{ai_port}"

# Helper function to generate a unique 10-digit ID
def generate_unique_id():
    return str(random.randint(1000000000, 9999999999))

def create_mapping_csv(result_dir, mapped_images):
    mapping_file_path = os.path.join(result_dir, "mapping.csv")
    with open(mapping_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "test_image_path", "ground_truth_path"])
        for image in mapped_images:
            writer.writerow([image["image_id"], image["test_image_path"], image["ground_truth_path"]])
    logger.info(f"Mapping file created at {mapping_file_path}")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    logger.info("Received request at /predict")

    # Determine if it's a folder or single image upload
    if 'testFolder' in request.files:
        upload_type = "Folder"
        test_files = request.files.getlist("testFolder")
    elif 'testImage' in request.files:
        upload_type = "Image"
        test_files = [request.files.get("testImage")]
    else:
        logger.error("Missing files or form data in the request.")
        return jsonify({"error": "Missing files or form data in the request."}), 400

    # Validate form data
    task_type = request.form.get("taskType")
    dataset_name = request.form.get("dataset")
    model_name = request.form.get("modelName")
    model_file = request.form.get("modelFile")
    if not task_type or not dataset_name or not model_name or not model_file:
        logger.error("Missing form data in the request.")
        return jsonify({"error": "Missing form data in the request."}), 400

    # Generate unique ID and create result directory
    unique_id = generate_unique_id()
    result_dir = os.path.join(os.getenv("APP_RESULT_PATH", "/shared/result"), unique_id)
    test_folder_dir = os.path.join(result_dir, "test_folder")
    ground_truth_dir = os.path.join(result_dir, "ground_truth")
    sampled_dir = os.path.join(result_dir, "sampled")
                               
    logger.info(f"Generated unique_id: {unique_id}, result_dir: {result_dir}")
    modelname_dir = os.path.join(os.getenv("APP_MODEL_PATH", "/shared/models"), model_name)
    modelfile_dir = os.path.join(os.getenv("APP_MODEL_PATH", "/shared/models"), model_name, "model", model_file)

    # Create necessary directories
    os.makedirs(test_folder_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    os.makedirs(sampled_dir, exist_ok=True)

    # Save uploaded file(s)
    for file in test_files:
        file_path = os.path.join(test_folder_dir, os.path.basename(file.filename))
        file.save(file_path)
        logger.info(f"Saved file: {file_path}")

    # Load and process mapping file if it exists
    mapping_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, "mapping.csv")
    mapped_images = []
    
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as mapping_file:
            reader = csv.reader(mapping_file)
            next(reader) 

            uploaded_filenames = {os.path.basename(file.filename) for file in test_files}
            sampled_path = os.path.join(modelname_dir, "sampled")
            all_samples_found = True

            # Copy ground truth images based on the mapping file
            for row in reader:
                image_id, image_name, test_image_path, ground_truth_path = row

                # Check if the image_name is in the set of uploaded filenames
                if image_name in uploaded_filenames:
                    # Copy the ground truth image to the destination directory
                    ground_truth_image_name = os.path.basename(ground_truth_path)
                    source_path = os.path.join(os.getenv("APP_DATA_PATH", "/shared/data"), task_type, dataset_name, ground_truth_path)
                    dest_path = os.path.join(ground_truth_dir, ground_truth_image_name)
                    shutil.copy(source_path, dest_path)
                    logger.info(f"Copied ground truth image: {source_path} to {dest_path}")

                    # Add to mapped_images list for mapping.csv
                    mapped_images.append({
                        "image_id": image_name,
                        "test_image_path": os.path.join("test_folder", image_name),
                        "ground_truth_path": os.path.join("ground_truth", ground_truth_image_name)
                    })

            # Create mapping.csv before evaluation
            create_mapping_csv(result_dir, mapped_images)

            for image_name in uploaded_filenames:   
                    # Check if sample exists
                    base_name = image_name.split("_", 1)[-1].split(".")[0]
                    sample_filename = f"{base_name}_output_ens.jpg"
                    sample_path = os.path.join(sampled_path, sample_filename)
                    if os.path.exists(sample_path):
                        shutil.copy(sample_path, sampled_dir)
                        logger.info(f"Found existing sample: {sample_path}")

                    else:
                        all_samples_found = False
                        logger.warning(f"Sample not found for image: {image_name}")
                        break
            if all_samples_found:
                # If all samples were found, proceed to /evaluation
                eval_response = requests.post(f"{base_url}/evaluation", json={
                    "unique_id": unique_id,
                    "model_path": modelfile_dir,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "ground_truth_dir": ground_truth_dir,
                    "test_folder_dir": test_folder_dir
                })
                if eval_response.status_code != 200:
                    logger.error("valuation failed after finding all samples.")
                    shutil.rmtree(result_dir, ignore_errors=True)
                    logger.info(f"Deleted folder for unique_id {unique_id} due to evaluation error.")
                    return jsonify({"error": "Evaluation failed."}), 500
                metrics = eval_response.json().get("metrics", {})
                logger.info(f"Evaluation metrics after finding all samples: {metrics}")
            else:
                # If any sample is missing, call /sample and then /evaluation
                sample_response = requests.post(f"{base_url}/sample", json={
                    "unique_id": unique_id,
                    "model_path": modelfile_dir,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "ground_truth_dir": ground_truth_dir,
                    "test_folder_dir": test_folder_dir
                })
                
                if sample_response.status_code != 200:
                    logger.error("Sampling failed.")
                    shutil.rmtree(result_dir, ignore_errors=True)
                    logger.info(f"Deleted folder for unique_id {unique_id} due to sampling error.")
                    return jsonify({"error": "Sampling failed."}), 500
                
                eval_response = requests.post(f"{base_url}/evaluation", json={
                    "unique_id": unique_id,
                    "model_path": modelfile_dir,
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "ground_truth_dir": ground_truth_dir,
                    "test_folder_dir": test_folder_dir
                })
                if eval_response.status_code != 200:
                    logger.error("Evaluation failed after sampling.")
                    shutil.rmtree(result_dir, ignore_errors=True)
                    logger.info(f"Deleted folder for unique_id {unique_id} due to evaluation error.")
                    return jsonify({"error": "Evaluation failed."}), 500
                metrics = eval_response.json().get("metrics", {})
                logger.info(f"Evaluation metrics after sampling: {metrics}")
    else:
        logger.error(f"Mapping file not found at {mapping_path}")
        shutil.rmtree(result_dir, ignore_errors=True)
        return jsonify({"error": "Mapping file not found"}), 404

    # Prepare JSON data for the result record
    json_data = {
        "Unique ID": unique_id,
        "Type": upload_type,
        "Test_image": test_folder_dir,
        "ground_truth": ground_truth_dir,
        "Dataset Name": dataset_name,
        "Model": modelfile_dir,
        "sampled_path": sampled_dir,
        "metrics": metrics
    }

    # Append to result_record.json or create it if it doesn't exist
    result_record_path = os.path.join(os.getenv("APP_RESULT_PATH", "/shared/result"), "result_record.json")
    if not os.path.exists(result_record_path):
        with open(result_record_path, "w") as f:
            json.dump([json_data], f, indent=4)
    else:
        with open(result_record_path, "r+") as f:
            data = json.load(f)
            data.append(json_data)
            f.seek(0)
            json.dump(data, f, indent=4)
    logger.info(f"Data successfully saved for Unique ID: {unique_id}")

    return jsonify({"success": True, "unique_id": unique_id}), 200


@app.route("/sample", methods=["POST"])
def sample():
    unique_id = request.json.get("unique_id")
    result_dir = os.path.join(os.getenv("APP_RESULT_PATH", "/shared/result"), unique_id)
    sampled_dir = os.path.join(result_dir, "sampled")
    model_path = request.json.get("model_path")
    dataset_name = request.json.get("dataset_name")
    model_name = request.json.get("model_name")
    test_folder_dir = request.json.get("test_folder_dir")
    ground_truth_dir = request.json.get("ground_truth_dir")

    logger.info(f"Running sample with unique_id: {unique_id}, dataset_name: {dataset_name}, model_name: {model_name}, model_path: {model_path}")
    
    if model_name == "MedSegDiffv2":
        sample_process = subprocess.run([
            "python", "/shared/models/MedSegDiffv2/segmentation_sample.py",
            "--data_name", dataset_name,
            "--data_dir", result_dir,
            "--out_dir", sampled_dir,
            "--model_path", model_path,
            "--image_size", "256",
            "--num_channels", "128",
            "--class_cond", "False",
            "--num_res_blocks", "2",
            "--num_heads", "1",
            "--learn_sigma", "True",
            "--use_scale_shift_norm", "False",
            "--attention_resolutions", "16",
            "--diffusion_steps", "1000",
            "--noise_schedule", "linear",
            "--rescale_learned_sigmas", "False",
            "--rescale_timesteps", "False",
            "--num_ensemble", "5"
        ])
    elif model_name == "TBConvl-Net":
        sample_process = subprocess.run([
            "python", "/shared/models/TBConvl-Net/segmentation.py",
            "--model_pth", model_path,
            "--test_images_dir", test_folder_dir,
            "--test_masks_dir", ground_truth_dir,
            "--save_dir_pred", sampled_dir
        ])
    else:
        logger.error(f"Unsupported model_name: {model_name}")
        return jsonify({"error": f"Unsupported model_name: {model_name}"}), 400


    if sample_process.returncode == 0:
        logger.info("segmentation_sample.py executed successfully.")
        return jsonify({"success": True}), 200
    else:
        logger.error("segmentation_sample.py failed.")
        return jsonify({"error": "segmentation_sample.py failed"}), 500

@app.route("/evaluation", methods=["POST"])
def evaluation():
    unique_id = request.json.get("unique_id")
    result_dir = os.path.join(os.getenv("APP_RESULT_PATH", "/shared/result"), unique_id)
    sampled_dir = os.path.join(result_dir, "sampled")
    ground_truth_dir = os.path.join(result_dir, "ground_truth")
    model_path = request.json.get("model_path")
    model_name = request.json.get("model_name") 
    test_folder_dir = request.json.get("test_folder_dir")
    ground_truth_dir = request.json.get("ground_truth_dir")

    logger.info(f"Running evaluation with unique_id: {unique_id}, model_name: {model_path}")

    # Choose the evaluation script based on the model_name
    if model_name == "MedSegDiffv2":
        eval_output = subprocess.check_output([
            "python", "/shared/models/MedSegDiffv2/segmentation_eval.py",
            "--inp_pth", sampled_dir,
            "--out_pth", ground_truth_dir
        ], text=True)
    elif model_name == "TBConvl-Net":
        eval_output = subprocess.check_output([
            "python", "/shared/models/TBConvl-Net/evaluate.py",
            "--test_images_dir", test_folder_dir,
            "--test_masks_dir", ground_truth_dir,
            "--model_pth", model_path
        ], text=True)
    else:
        logger.error(f"Unsupported model_name: {model_name}")
        return jsonify({"error": f"Unsupported model_name: {model_name}"}), 400

    metrics = {}
    for line in eval_output.strip().splitlines():
        if line.startswith("IoU:"):
            metrics["IoU"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Dice Coefficient:"):
            metrics["DiceScore"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Accuracy:"):
            metrics["Accuracy"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Sensitivity:"):
            metrics["Sensitivity"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("Specificity:"):
            metrics["Specificity"] = round(float(line.split(":")[1].strip()), 5)
        elif line.startswith("F1 Score:"):
            metrics["F1Score"] = round(float(line.split(":")[1].strip()), 5)

    logger.info(f"Evaluation metrics: {metrics}")
    return jsonify({"success": True, "metrics": metrics}), 200


# Health Check Endpoint
@app.route('/health', methods=['GET'])
def health_check():
    logger.info("Health check endpoint accessed")
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host=os.getenv('AI_HOST', '0.0.0.0'), port=int(os.getenv('AI_PORT', 3000)), debug=(os.getenv("ENV") == "development"))