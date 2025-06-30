import os
import json
import pandas as pd
from app.utils import setup_logging
import time


logger = setup_logging()


def generate_unique_id(json_file):
    unique_id = int(time.time() * 1000) % 10000000000
    with open(json_file, 'r') as f:
        records = json.load(f)
        while any(record["Unique ID"] == unique_id for record in records):
            unique_id = (unique_id + 1) % 10000000000
    return unique_id

def get_directory_size(directory):
    total_size = sum(
        os.path.getsize(os.path.join(directory, f)) 
        for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    )
    if total_size < 1e9: 
        return f"{total_size / 1e6:.2f} MB"
    else:
        return f"{total_size / 1e9:.2f} GB"

def update_json_record(json_file, unique_id, task_type, dataset_name, test_count, test_size, gt_count, gt_size, active_status):
    with open(json_file, "r+") as f:
        records = json.load(f)

        # Create new record entry
        new_record = {
            "Unique ID": unique_id,
            "Task Type": task_type,
            "Dataset Name": dataset_name,
            "Path": f"/shared/data/{task_type}/{dataset_name}",
            "Number of Test Images": test_count,
            "Size of Test Images": test_size,
            "Number of Ground Truth Images": gt_count,
            "Size of Ground Truth Images": gt_size,
            "Active": active_status
        }
        records.append(new_record)
        f.seek(0)
        json.dump(records, f, indent=4)
    logger.info(f"Dataset record updated in {json_file}")

def process_dataset_form(task_type, dataset_name, test_images, ground_truth_images):
    try:
        # Define base directory for storing datasets
        app_data_path = os.getenv('APP_DATA_PATH')
        base_dir = os.path.join(app_data_path, task_type, dataset_name)
        test_dir = os.path.join(base_dir, "Test_images")
        ground_truth_dir = os.path.join(base_dir, "Ground_truth")
        mapping_file = os.path.join(base_dir, "mapping.csv")
        json_file = os.path.join(app_data_path, "data_record.json")


        # Create necessary directories
        os.makedirs(test_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)

        # Save test and ground truth images and map them
        data = []
        unmapped_images = []

        logger.info("Mapping test images to ground truth images.")
        for test_image in test_images:
            test_id = os.path.splitext(test_image.filename)[0].split('_')[-1]
            test_image_filename = os.path.basename(test_image.filename) 
            gt_image_filename = f"{dataset_name}_{test_id}_Segmentation.png"
            # Generate possible ground truth filenames
            possible_gt_filenames = [
                f"{dataset_name}_{test_id}_Segmentation.png",
                f"{dataset_name}_{test_id}_Segmentation.jpg",
                f"{dataset_name}_{test_id}_segmentation.png",
                f"{dataset_name}_{test_id}_segmentation.jpg"
            ]
            # Check if the corresponding ground truth image exists
            matching_gt_image = next(
                (img for img in ground_truth_images if os.path.basename(img.filename) in possible_gt_filenames), None
            )

            if matching_gt_image:
                # Save test and ground truth images
                test_image.save(os.path.join(test_dir, test_image_filename))
                matching_gt_image.save(os.path.join(ground_truth_dir, os.path.basename(matching_gt_image.filename)))

                # Add paths to the mapping list
                data.append({
                    "#": test_id,
                    "image_name": test_image_filename,
                    "test_images": f"Test_images/{test_image_filename}",
                    "ground_truth": f"Ground_truth/{os.path.basename(matching_gt_image.filename)}"
                })
                logger.info(f"Mapping created for test image {test_image.filename}")
            else:
                unmapped_images.append(test_image.filename)
                logger.warning(f"No matching ground truth found for test image {test_image.filename}")

        # Save mapping to CSV
        df = pd.DataFrame(data, columns=["#", "image_name", "test_images", "ground_truth"])
        df.to_csv(mapping_file, index=False)
        logger.info(f"Mapping file saved at {mapping_file}")

        # Calculate directory sizes
        test_size = get_directory_size(test_dir)
        gt_size = get_directory_size(ground_truth_dir)
        unique_id = generate_unique_id(json_file)

        # Update JSON with dataset record
        update_json_record(
            json_file=json_file,
            unique_id=unique_id,
            task_type=task_type,
            dataset_name=dataset_name,
            test_count=len(test_images),
            test_size=test_size,
            gt_count=len(data),
            gt_size=gt_size,
            active_status=True
        )

        # Return success message with details about unmapped images, if any
        if unmapped_images:
            return True, f"Dataset processed successfully, but the following images were unmapped: {', '.join(unmapped_images)}"
        return True, "Dataset processed successfully.", unique_id

    except Exception as e:
        logger.error("An error occurred while processing the dataset form", exc_info=True)
        return False, f"An error occurred: {str(e)}", unique_id
