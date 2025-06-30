import os
import json
import pandas as pd
import requests
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, flash, send_file, send_from_directory
from app.utils import (
    setup_logging,
    load_environment,
    check_essential_paths,
    get_task_types_from_data,
    get_datasets_by_task,
    verify_images_in_mapping,
    extract_id,
)
from app.data_setup import process_dataset_form 
from datetime import datetime
import shutil
from fpdf import FPDF


main = Blueprint('main', __name__)
logger = setup_logging()
load_environment()
check_essential_paths(logger)

@main.route('/')
def index():
    records = []
    no_data = False
    data_file = os.path.join(os.getenv('APP_DATA_PATH'), 'data_record.json')
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
            records = [record for record in data if record.get("Active")]
        if not records:
            no_data = True
    except FileNotFoundError:
        no_data = True 
    
    return render_template('index.html', records=records, no_data=no_data)

@main.route('/get_task_types')
def get_task_types():
    task_types = get_task_types_from_data()
    return jsonify(taskTypes=list(task_types))

@main.route('/get_datasets')
def get_datasets():
    task_type = request.args.get('taskType')
    datasets = get_datasets_by_task(task_type)
    return jsonify(datasets=list(datasets))

@main.route('/get_model_names', methods=['GET'])
def get_model_names():
    model_base_path = os.getenv('APP_MODEL_PATH')
    if not os.path.isdir(model_base_path):
        return jsonify({'models': []})
    
    model_names = [folder for folder in os.listdir(model_base_path) if os.path.isdir(os.path.join(model_base_path, folder))]
    return jsonify(models=model_names)


@main.route('/get_model_files', methods=['GET'])
def get_model_files():
    model_name = request.args.get('modelName')
    model_folder = os.path.join(os.getenv('APP_MODEL_PATH'), model_name, "model")
    
    if not os.path.isdir(model_folder):
        return jsonify({'modelFiles': []})
    
    model_files = [file for file in os.listdir(model_folder) if file.endswith((".pt", ".pkl", ".pth"))]
    return jsonify(modelFiles=model_files)


@main.route('/verify_images_in_mapping', methods=['POST'])
def verify_images_in_mapping():
    data = request.get_json()
    image_names = data['imageNames']
    task_type = data['taskType']
    dataset = data['dataset']
    mapping_file_path = os.path.join(os.getenv("APP_DATA_PATH"), task_type, dataset, "mapping.csv")

    try:
        df = pd.read_csv(mapping_file_path)
        missing_images = [img for img in image_names if img not in df['image_name'].values]
        if len(missing_images) > 10:
            # Return a message with only the count of missing images if there are more than 10
            return jsonify({"missingImages": f"{len(missing_images)} images are missing from mapping.csv"})
        else:
            # Return the list of missing images if there are 10 or fewer
            return jsonify({"missingImages": missing_images})
    except FileNotFoundError:
        return jsonify({"error": "Mapping file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@main.route('/datasetup', methods=['GET', 'POST'])
def datasetup():
    logger.info("Received request at /datasetup")
    message = ""
    replace_existing = request.form.get('replace_existing', 'false').lower() == 'true'
    app_data_path = os.getenv('APP_DATA_PATH')
    data_record_path = os.path.join(app_data_path, 'data_record.json')

    # Ensure data_record.json exists
    if not os.path.exists(data_record_path):
        os.makedirs(os.path.dirname(data_record_path), exist_ok=True)
        with open(data_record_path, 'w') as f:
            json.dump([], f)
        logger.info(f"Initialized data_record.json at {data_record_path}")

    if request.method == 'POST':
        # Extract form data
        task_type = request.form.get('taskType')
        dataset_name = request.form.get('datasetName')
        test_images = request.files.getlist('testImages')
        ground_truth_images = request.files.getlist('groundTruthImages')
        

        # Check if taskType is 'Other' and retrieve specific task type
        if task_type == "Other":
            specific_task_type = request.form.get('custom_task_type')
            if specific_task_type:
                task_type = specific_task_type
            else:
                # Handle case where specific task type isn't provided
                return jsonify({'success': False, 'error': 'Please specify a task type when selecting "Other".'})


        logger.info(f"Processing dataset setup for task type: {task_type}, dataset name: {dataset_name}")
        
        base_dir = os.path.join(app_data_path, task_type, dataset_name)


         # Check if base directory already exists
        if os.path.exists(base_dir):
            # If replace_existing is not set, prompt the user for replacement confirmation
            if not replace_existing:
                logger.info("Prompting user for dataset replacement confirmation.")
                return jsonify({'replacePrompt': True, 'message': "Dataset already exists. Would you like to replace it?"})

            # If replace_existing is true, delete the existing base directory and deactivate the old record
            logger.info(f"Deleting existing dataset at {base_dir} as replace_existing is set to true.")
            shutil.rmtree(base_dir)

            # Load existing records from data_record.json
            with open(data_record_path, 'r+') as f:
                records = json.load(f)
                # Deactivate the record with matching taskType and datasetName
                for record in records:
                    if record['Task Type'] == task_type and record['Dataset Name'] == dataset_name:
                        record['Active'] = False
                # Write the updated records back to data_record.json
                f.seek(0)
                f.truncate()
                json.dump(records, f, indent=4)

        # Proceed to process the dataset form
        success, message, unique_id = process_dataset_form(task_type, dataset_name, test_images, ground_truth_images)
        
        if success:
            logger.info("Dataset processed successfully.")
            return jsonify({'success': True, 'message': message, 'unique_id': unique_id})
        else:
            logger.error(f"Error processing dataset: {message}")
            # If unique_id is provided, clean up the created directory and JSON record
            if unique_id:
                if os.path.exists(base_dir):
                    shutil.rmtree(base_dir)
                    logger.info(f"Deleted directory at {base_dir} due to error.")
                
                with open(data_record_path, 'r+') as f:
                    records = json.load(f)
                    records = [record for record in records if record.get("Unique ID") != unique_id]
                    f.seek(0)
                    f.truncate()
                    json.dump(records, f, indent=4)
                logger.info(f"Removed JSON record with Unique ID: {unique_id} due to error.")

            return jsonify({'success': False, 'error': message})

    # Render the template with the message
    logger.info("Rendering data_setup.html with message.")
    return render_template('data_setup.html', message=message)


@main.route("/folder_upload", methods=["GET", "POST"])
def folder_upload():
    logger.info("Received request at /folder_upload")
    task_types = get_task_types_from_data() 
    ai_host = os.getenv("AI_HOST", "localhost")
    ai_port = os.getenv("AI_PORT", "3000")
    models, datasets, model_files = [], [], []

    if request.method == "POST":
        selected_task = request.form.get("taskType")
        selected_dataset = request.form.get("dataset")
        selected_model_name = request.form.get("modelName")
        selected_model_file = request.form.get("modelFile")
        test_folder = request.files.getlist("testFolder")

        # Filter and validate image files
        image_files = [f for f in test_folder if f.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

        # Validation checks
        if not selected_task or not selected_dataset or not selected_model_name or not selected_model_file:
            flash("Please select task type, dataset, model name, and model file.", "error")
            return render_template(
                "folder_upload.html", task_types=task_types, datasets=datasets, models=models, model_files=model_files
            )

        if len(image_files) == 0:
            flash("Please upload a folder containing at least one image file.", "error")
            return render_template(
                "folder_upload.html", task_types=task_types, datasets=datasets, models=models, model_files=model_files
            )

        # Check if images are listed in the mapping file
        mapping_file_path = os.path.join(os.getenv("APP_DATA_PATH"), selected_task, selected_dataset, "mapping.csv")
        missing_images = verify_images_in_mapping([f.filename for f in image_files], mapping_file_path)

        if missing_images:
            missing_count = len(missing_images)
            if missing_count > 10:
                flash(f"{missing_count} images are missing from mapping.csv.", "error")
            else:
                flash(f"Some images are not listed in the mapping file: {', '.join(missing_images)}", "error")
            return render_template(
                "folder_upload.html", task_types=task_types, datasets=datasets, models=models, model_files=model_files
            )

        # Make request to AI API /predict endpoint
        api_url = f"http://{ai_host}:{ai_port}/predict"
        files = [('testFolder', (file.filename, file.stream, file.mimetype)) for file in image_files]
        data = {
            'taskType': selected_task,
            'dataset': selected_dataset,
            'modelName': selected_model_name,
            'modelFile': selected_model_file
        }

        try:
            response = requests.post(api_url, files=files, data=data)
            
            # Log the raw response and status code
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Raw response content: {response.content.decode('utf-8')}")
            
            # Attempt to parse JSON response
            try:
                response_data = response.json()
                logger.info(f"Parsed JSON response_data: {response_data}")
                
                if response.status_code == 200 and response_data.get('success'):
                    unique_id = response_data['unique_id']
                    return redirect(url_for('main.result', unique_id=unique_id))
                else:
                    flash("An error occurred while processing your request.", "error")
            except ValueError as json_error:
                # Log JSON parsing error and raw content
                logger.error("Failed to parse JSON response")
                logger.error(f"JSON error: {json_error}")
                logger.error(f"Raw response content: {response.text}")
                flash("An error occurred while processing the response from the AI service.", "error")

        except requests.RequestException as e:
            flash("Unable to connect to the AI service.", "error")
            logger.error(f"Error contacting AI API: {e}")

    return render_template("folder_upload.html", task_types=task_types, datasets=datasets, models=models, AI_HOST=ai_host, AI_PORT=ai_port)


@main.route("/img_upload", methods=["GET", "POST"])
def img_upload():
    logger.info("Received request at /img_upload")
    task_types = get_task_types_from_data() 
    ai_host = os.getenv("AI_HOST", "localhost")
    ai_port = os.getenv("AI_PORT", "3000")
    models, datasets, model_files = [], [], []

    if request.method == "POST":
        # Retrieve form data and the uploaded image
        selected_task = request.form.get("taskType")
        selected_dataset = request.form.get("dataset")
        selected_model_name = request.form.get("modelName")
        selected_model_file = request.form.get("modelFile")
        test_image = request.files.get("testImage")

        # Validation checks
        if not selected_task or not selected_dataset or not selected_model_name or not selected_model_file:
            flash("Please select task type, dataset, model name, and model file.", "error")
            return render_template(
                "img_upload.html", task_types=task_types, datasets=datasets, 
                models=models, model_files=model_files
            )

        # Check if a valid image file is uploaded
        if not test_image or not test_image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            flash("Please upload a valid image file in .png, .jpg, .jpeg, or .bmp format.", "error")
            return render_template(
                "img_upload.html", task_types=task_types, datasets=datasets, 
                models=models, model_files=model_files
            )
        
        # Check if image is listed in the mapping file
        mapping_file_path = os.path.join(os.getenv("APP_DATA_PATH"), selected_task, selected_dataset, "mapping.csv")
        missing_images = verify_images_in_mapping([test_image.filename], mapping_file_path)

        if missing_images:
            missing_count = len(missing_images)
            if missing_count > 10:
                flash(f"{missing_count} images are missing from mapping.csv.", "error")
            else:
                flash(f"Image is not listed in the mapping file: {', '.join(missing_images)}", "error")
            return render_template("img_upload.html", task_types=task_types, datasets=datasets, models=models, model_files=model_files)

        # Make request to AI API /predict endpoint
        api_url = f"http://{os.getenv('AI_HOST')}:{os.getenv('AI_PORT')}/predict"
        files = {'testImage': (test_image.filename, test_image.stream, test_image.mimetype)}
        data = {
            'taskType': selected_task,
            'dataset': selected_dataset,
            'modelName': selected_model_name,
            'modelFile': selected_model_file
        }

        try:
            response = requests.post(api_url, files=files, data=data)
            
            # Log the raw response and status code
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Raw response content: {response.content.decode('utf-8')}")
            
            # Attempt to parse JSON response
            try:
                response_data = response.json()
                logger.info(f"Parsed JSON response_data: {response_data}")
                
                if response.status_code == 200 and response_data.get('success'):
                    unique_id = response_data['unique_id']
                    return redirect(url_for('main.result', unique_id=unique_id))
                else:
                    flash("An error occurred while processing your request.", "error")
            except ValueError as json_error:
                # Log JSON parsing error and raw content
                logger.error("Failed to parse JSON response")
                logger.error(f"JSON error: {json_error}")
                logger.error(f"Raw response content: {response.text}")
                flash("An error occurred while processing the response from the AI service.", "error")

        except requests.RequestException as e:
            flash("Unable to connect to the AI service.", "error")
            logger.error(f"Error contacting AI API: {e}")

    return render_template("img_upload.html", task_types=task_types, datasets=datasets, models=models, AI_HOST=ai_host, AI_PORT=ai_port)

@main.route('/result/<result_id>')
def result(result_id):
    RESULT_PATH = os.getenv("APP_RESULT_PATH", "/shared/result")
    RESULT_JSON_PATH = os.path.join(RESULT_PATH, "result_record.json")

    # Load result JSON file
    try:
        with open(RESULT_JSON_PATH, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        logger.error(f"Result file not found at {RESULT_JSON_PATH}")
        return "Result file not found", 404

    # Find the result entry with the specified result_id
    result = next((r for r in results if r["Unique ID"] == result_id), None)
    if not result:
        logger.warning(f"No result entry found for Unique ID: {result_id}")
        return "Result not found", 404

    # Log the found result entry
    logger.info(f"Found result entry for Unique ID {result_id}: {result}")

    # Initialize dictionaries to store image paths
    try:
        test_images = {
            extract_id(file, r"ISIC_(\d+)\.jpg"): f"{result_id}/test_folder/{file}"
            for file in os.listdir(f"{RESULT_PATH}/{result_id}/test_folder")
        }
        logger.debug(f"Extracted test image IDs: {list(test_images.keys())}")

        ground_truth_images = {
            extract_id(file, r"ISIC_(\d+)_Segmentation\.png"): f"{result_id}/ground_truth/{file}"
            for file in os.listdir(f"{RESULT_PATH}/{result_id}/ground_truth")
        }
        logger.debug(f"Extracted ground truth image IDs: {list(ground_truth_images.keys())}")

        sampled_images = {
            extract_id(file, r"(\d+)_output_ens\.jpg"): f"{result_id}/sampled/{file}"
            for file in os.listdir(f"{RESULT_PATH}/{result_id}/sampled")
        }
        logger.debug(f"Extracted sampled image IDs: {list(sampled_images.keys())}")
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        return "Directory not found", 404

    # Log image file dictionaries for debugging
    logger.debug(f"Test Images: {test_images}")
    logger.debug(f"Ground Truth Images: {ground_truth_images}")
    logger.debug(f"Sampled Images: {sampled_images}")

    # Align images by matching IDs
    matched_images = []
    for img_id in test_images:
        if img_id in ground_truth_images and img_id in sampled_images:
            matched_image = {
                "test_image": url_for('main.serve_result_file', subpath=test_images[img_id]),
                "ground_truth_image": url_for('main.serve_result_file', subpath=ground_truth_images[img_id]),
                "sampled_image": url_for('main.serve_result_file', subpath=sampled_images[img_id])
            }
            matched_images.append(matched_image)
            logger.info(f"Matched image set for ID {img_id}: {matched_image}")
        else:
            logger.warning(f"Image ID {img_id} missing in one or more categories (test, ground truth, sampled)")

    # Log the matched images
    logger.debug(f"Matched Images: {matched_images}")

    # Pass matched images and scores to the template
    return render_template(
        "result.html",
        matched_images=matched_images,
        scores={
            "F1Score": result.get("metrics", {}).get("F1Score", "N/A"),
            "Specificity": result.get("metrics", {}).get("Specificity", "N/A"),
            "Sensitivity": result.get("metrics", {}).get("Sensitivity", "N/A"),
            "Accuracy": result.get("metrics", {}).get("Accuracy", "N/A"),
            "DiceScore": result.get("metrics", {}).get("DiceScore", "N/A"),
            "IoU": result.get("metrics", {}).get("IoU", "N/A")
        },
        result_id=result_id
    )



@main.route('/download_pdf/<result_id>')
def download_pdf(result_id):
    RESULT_PATH = os.getenv("APP_RESULT_PATH", "/shared/result")
    RESULT_JSON_PATH = os.path.join(RESULT_PATH, "result_record.json")
    
    # Read result_record.json within the download_pdf route
    with open(RESULT_JSON_PATH, "r") as f:
        results = json.load(f)

    result = next((r for r in results if r["Unique ID"] == result_id), None)
    if not result:
        return "Result not found", 404

    # Paths to local image files
    test_images = {
        extract_id(file, r"ISIC_(\d+)\.jpg"): os.path.join(RESULT_PATH, result_id, "test_folder", file)
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/test_folder")
    }
    ground_truth_images = {
        extract_id(file, r"ISIC_(\d+)_Segmentation\.png"): os.path.join(RESULT_PATH, result_id, "ground_truth", file)
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/ground_truth")
    }
    sampled_images = {
        extract_id(file, r"(\d+)_output_ens\.jpg"): os.path.join(RESULT_PATH, result_id, "sampled", file)
        for file in os.listdir(f"{RESULT_PATH}/{result_id}/sampled")
    }

    # Align images by matching IDs
    matched_images = [
        {
            "test_image": test_images[img_id],
            "ground_truth_image": ground_truth_images[img_id],
            "sampled_image": sampled_images[img_id]
        }
        for img_id in test_images
        if img_id in ground_truth_images and img_id in sampled_images
    ]

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_font("Arial", "B", 14)

    # Title for Scores Section
    pdf.cell(0, 10, "Segmentation Scores", ln=True, align="C")
    pdf.ln(5)
    
    # Center the table on the page
    table_x = (210 - 80) / 2  
    pdf.set_xy(table_x, pdf.get_y())

    # Create table headers
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Metric", border=1, align="C")
    pdf.cell(40, 10, "Score", border=1, align="C")
    pdf.ln()

    # Table content
    pdf.set_font("Arial", "", 12)
    score_keys = ["F1Score", "Specificity", "Sensitivity", "Accuracy", "DiceScore", "IoU"]
    for key in score_keys:
        pdf.set_x(table_x)
        pdf.cell(40, 10, key, border=1, align="C")
        pdf.cell(40, 10, str(result["metrics"].get(key, 'N/A')), border=1, align="C")
        pdf.ln()  

    pdf.ln(10) 

    # Image Table Section Title
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Detailed Comparison", ln=True, align="C")
    pdf.set_font("Arial", "B", 12)
    
    # Adjusted Table Headers for Images
    headers = ["Test Image", "Ground Truth", "Sampled Image"]
    cell_width = 60 
    for header in headers:
        pdf.cell(cell_width, 10, header, border=1, align="C")
    pdf.ln() 

    # Image Rows
    pdf.set_font("Arial", "", 10)
    padding = 5 
    image_height = 60  
    for images in matched_images:
        if pdf.get_y() + image_height + 2 * padding > pdf.page_break_trigger:
            pdf.add_page()
            for header in headers:
                pdf.cell(cell_width, 10, header, border=1, align="C")
            pdf.ln() 

        y_start = pdf.get_y()  

        pdf.cell(cell_width, image_height + 2 * padding, "", border=1) 
        pdf.cell(cell_width, image_height + 2 * padding, "", border=1)  
        pdf.cell(cell_width, image_height + 2 * padding, "", border=1)  
        pdf.ln() 

        x_start = 15 
        pdf.image(images["test_image"], x=x_start + padding, y=y_start + padding, w=cell_width - 2 * padding, h=image_height)
        pdf.image(images["ground_truth_image"], x=x_start + cell_width + padding, y=y_start + padding, w=cell_width - 2 * padding, h=image_height)
        pdf.image(images["sampled_image"], x=x_start + 2 * cell_width + padding, y=y_start + padding, w=cell_width - 2 * padding, h=image_height)

    pdf_file_path = f"/tmp/Segmentation_Result_{result_id}.pdf"
    pdf.output(pdf_file_path)
    return send_file(pdf_file_path, as_attachment=True, download_name=f"Segmentation_Result_{result_id}.pdf")

@main.route('/shared/result/<path:subpath>')
def serve_result_file(subpath):
    result_path = os.getenv("APP_RESULT_PATH", "/shared/result")
    return send_from_directory(result_path, subpath)