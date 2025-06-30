import os
import shutil
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import time

class DatasetFormApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Upload Form")

        # JSON File Path
        self.json_file = "../shared/data/data_record.json"
        self.initialize_json()

        # Task Type
        tk.Label(root, text="Task Type:").grid(row=0, column=0, sticky="w")
        self.task_type = tk.StringVar()
        self.task_type.trace("w", self.show_other_task_entry)
        self.task_type_menu = tk.OptionMenu(root, self.task_type, "Skin_Lesion", "Multi-Organ_Segmentation", "Other")
        self.task_type_menu.grid(row=0, column=1, sticky="w")

        # Other Task Type Entry (hidden by default)
        self.other_task_entry = tk.Entry(root)
        self.other_task_label = tk.Label(root, text="Specify Task Type:")
        self.other_task_entry.bind("<FocusOut>", self.format_other_task_type)

        # Dataset Name
        tk.Label(root, text="Dataset Name:").grid(row=2, column=0, sticky="w")

        # Bind an event to convert the dataset name to uppercase and strip leading spaces
        def format_dataset_name(event):
            dataset_name = self.dataset_name_entry.get().strip().upper()
            self.dataset_name_entry.delete(0, tk.END)
            self.dataset_name_entry.insert(0, dataset_name)
        
        self.dataset_name_entry = tk.Entry(root)
        self.dataset_name_entry.grid(row=2, column=1, sticky="w")
        self.dataset_name_entry.bind("<FocusOut>", format_dataset_name)

        # Test Images Folder
        tk.Label(root, text="Test Images Folder:").grid(row=4, column=0, sticky="w")
        self.test_images_path = tk.StringVar()
        tk.Button(root, text="Upload", command=self.upload_test_images).grid(row=4, column=1, sticky="w")
        self.test_images_label = tk.Label(root, text="", fg="red")
        self.test_images_label.grid(row=4, column=2, sticky="w")
        self.test_hint_label = tk.Label(root, text="Format: {dataset_name}_{image_id}.jpg or .png", fg="gray")
        self.test_hint_label.grid(row=5, column=1, sticky="w")

        # Ground Truth Images Folder
        tk.Label(root, text="Ground Truth Images Folder:").grid(row=6, column=0, sticky="w")
        self.ground_truth_path = tk.StringVar()
        tk.Button(root, text="Upload", command=self.upload_ground_truth_images).grid(row=6, column=1, sticky="w")
        self.ground_truth_label = tk.Label(root, text="", fg="red")
        self.ground_truth_label.grid(row=6, column=2, sticky="w")
        self.gt_hint_label = tk.Label(root, text="Format: {dataset_name}_{image_id}_Segmentation.jpg or .png", fg="gray")
        self.gt_hint_label.grid(row=7, column=1, sticky="w")

        # Submit Button (disabled initially)
        self.submit_button = tk.Button(root, text="Submit", command=self.submit_form, state=tk.DISABLED)
        self.submit_button.grid(row=8, column=0, pady=10)

        # Reset Button
        self.reset_button = tk.Button(root, text="Reset", command=self.reset_form)
        self.reset_button.grid(row=8, column=1, pady=10)

    def initialize_json(self):
        if not os.path.exists(self.json_file):
            os.makedirs(os.path.dirname(self.json_file), exist_ok=True)
            with open(self.json_file, "w") as f:
                json.dump([], f)  

    def show_other_task_entry(self, *args):
        if self.task_type.get() == "Other":
            self.other_task_label.grid(row=1, column=0, sticky="w")
            self.other_task_entry.grid(row=1, column=1, sticky="w")
        else:
            self.other_task_label.grid_forget()
            self.other_task_entry.grid_forget()

    def format_other_task_type(self, event):
        other_task_type = self.other_task_entry.get().strip()
        self.other_task_entry.delete(0, tk.END)
        self.other_task_entry.insert(0, other_task_type)

    def upload_test_images(self):
        dataset_name = self.dataset_name_entry.get().strip().lower()  # Convert dataset name to lowercase for comparison
        path = filedialog.askdirectory()
        if path:
            test_images = [f for f in os.listdir(path) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
            invalid_files = [
                f for f in test_images
                if not (f.lower().startswith(f"{dataset_name}_") and 
                        (f.lower().endswith('.jpg') or f.lower().endswith('.png')) and 
                        f[len(dataset_name) + 1:].split('.')[0].isdigit())  # Check if the part after dataset_name_ is numeric
            ]
            if invalid_files:
                messagebox.showwarning(
                    "Invalid File Format",
                    f"The following files do not match the required format {dataset_name}_{{image_id}}.jpg or .png:\n"
                    + "\n".join(invalid_files)
                )
                return
            self.test_images_path.set(path)
            self.test_images_count = len(test_images)
            self.test_images_label.config(text=f"{path} ({self.test_images_count} images)", fg="green")
            self.check_ready_to_submit()

    def upload_ground_truth_images(self):
        dataset_name = self.dataset_name_entry.get().strip().lower()  # Convert dataset name to lowercase for comparison
        path = filedialog.askdirectory()
        if path:
            ground_truth_images = [f for f in os.listdir(path) if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
            invalid_files = [
                f for f in ground_truth_images
                if not (f.lower().startswith(f"{dataset_name}_") and 
                        (f.lower().endswith("_segmentation.jpg") or f.lower().endswith("_segmentation.png")) and 
                        f[len(dataset_name) + 1:].split('_')[0].isdigit())  # Check if the part after dataset_name_ is numeric
            ]
            if invalid_files:
                messagebox.showwarning(
                    "Invalid File Format",
                    f"The following files do not match the required format {dataset_name}_{{image_id}}_Segmentation.jpg or .png:\n"
                    + "\n".join(invalid_files)
                )
                return
            self.ground_truth_path.set(path)
            self.ground_truth_count = len(ground_truth_images)
            self.ground_truth_label.config(text=f"{path} ({self.ground_truth_count} images)", fg="green")
            self.check_ready_to_submit()



    def check_ready_to_submit(self):
        test_count = getattr(self, 'test_images_count', 0)
        gt_count = getattr(self, 'ground_truth_count', 0)

        if test_count == gt_count and gt_count > 0:
            self.submit_button.config(state=tk.NORMAL)
        else:
            self.submit_button.config(state=tk.DISABLED)

    def reset_form(self):
        self.task_type.set("")
        self.dataset_name_entry.delete(0, tk.END)
        self.test_images_path.set("")
        self.ground_truth_path.set("")
        self.test_images_label.config(text="", fg="red")
        self.ground_truth_label.config(text="", fg="red")
        self.submit_button.config(state=tk.DISABLED)
        self.test_images_count = 0
        self.ground_truth_count = 0

    def generate_unique_id(self):
        return int(time.time() * 1000) % 10000000000

    def get_directory_size(self, directory):
        total_size = sum(
            os.path.getsize(os.path.join(directory, f)) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
        )
        if total_size < 1e9:  
            return f"{total_size / 1e6:.2f} MB"
        else:
            return f"{total_size / 1e9:.2f} GB"

    def update_json_record(self, task_type, dataset_name, test_count, test_size, gt_count, gt_size, active_status):
        with open(self.json_file, "r") as f:
            records = json.load(f)

        for record in records:
            if record["Task Type"] == task_type and record["Dataset Name"] == dataset_name:
                record["Active"] = False

        # Create new record entry
        new_record = {
            "Unique ID": self.generate_unique_id(),
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

        # Save back to JSON
        with open(self.json_file, "w") as f:
            json.dump(records, f, indent=4)

    def submit_form(self):
        task_type = self.other_task_entry.get().strip() if self.task_type.get() == "Other" else self.task_type.get()
        dataset_name = self.dataset_name_entry.get().strip()
        test_images_path = self.test_images_path.get()
        ground_truth_path = self.ground_truth_path.get()

        if not (task_type and dataset_name and test_images_path and ground_truth_path):
            messagebox.showwarning("Incomplete Form", "Please fill in all fields.")
            return

        base_dir = f"../shared/data/{task_type}/{dataset_name}"
        test_dir = os.path.join(base_dir, "Test_images")
        ground_truth_dir = os.path.join(base_dir, "Ground_truth")
        mapping_file = os.path.join(base_dir, "mapping.csv")

        try:
            if os.path.exists(base_dir):
                replace = messagebox.askyesno("Replace Dataset?", f"{dataset_name} already exists. Do you want to replace it?")
                if replace:
                    shutil.rmtree(base_dir)
                else:
                    return
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(ground_truth_dir, exist_ok=True)

            # Map images and save CSV
            test_images = [f for f in os.listdir(test_images_path) if f.endswith('.jpg') or f.endswith('.png')]
            data = []
            unmapped_images = []

            for test_image in test_images:
                test_id = os.path.splitext(test_image)[0].split('_')[-1]
                test_path = os.path.join(test_images_path, test_image)
                gt_image = f"{dataset_name}_{test_id}_Segmentation.png"
                gt_path = os.path.join(ground_truth_path, gt_image)

                if os.path.exists(gt_path):
                    shutil.copy(test_path, os.path.join(test_dir, test_image))
                    shutil.copy(gt_path, os.path.join(ground_truth_dir, gt_image))

                    # Use forward slashes for paths in the CSV data
                    data.append({
                        "#": test_id,
                        "image_name": test_image,
                        "test_images": f"Test_images/{test_image}",
                        "ground_truth": f"Ground_truth/{gt_image}"
                    })
                else:
                    unmapped_images.append(test_image)

            # Save mapping to CSV with forward slashes
            df = pd.DataFrame(data, columns=["#", "image_name", "test_images", "ground_truth"])
            df.to_csv(mapping_file, index=False)

            # Update JSON and reset form if successful
            if not unmapped_images:
                test_size = self.get_directory_size(test_dir)
                gt_size = self.get_directory_size(ground_truth_dir)
                self.update_json_record(task_type, dataset_name, len(test_images), test_size, len(data), gt_size, active_status=True)
                messagebox.showinfo("Success", "Dataset processed and mapping file created successfully.")
                self.reset_form()
            else:
                error_message = "The following test images could not be mapped to ground truth images:\n" + "\n".join(unmapped_images)
                messagebox.showwarning("Unmapped Images", error_message)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetFormApp(root)
    root.mainloop()
