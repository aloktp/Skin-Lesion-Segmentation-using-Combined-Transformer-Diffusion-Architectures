# python evaluate.py --test_images_dir '/srv/scratch/z5450230/dataset/ISIC2018_Task1-2_Validation_Input' --test_masks_dir '/srv/scratch/z5450230/dataset/ISIC2018_Task1_Validation_GroundTruth' --model_pth '/srv/scratch/z5450230/tbconv/model/1modelWeights_Swin_Trans_Weights_Swin_Trans_Leather.pth'


import numpy as np
import torch
import argparse
from sklearn.metrics import confusion_matrix, jaccard_score, f1_score, precision_score, recall_score
from SwinUnet import SwinUNet
from SegmentationDataset import SegmentationDataset
from torchvision import transforms
from torch.utils.data import DataLoader

# Set up argument parser
parser = argparse.ArgumentParser(description="Evaluation of Segmentation Model")
parser.add_argument('--model_pth', type=str, required=True, help='Path to the saved model directory')
parser.add_argument('--test_images_dir', type=str, required=True, help='Path to the test images directory')
parser.add_argument('--test_masks_dir', type=str, required=True, help='Path to the test masks directory')

args = parser.parse_args()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = SwinUNet(input_channels=3, output_channels=1, embed_dim=32, num_heads=[4, 8], window_size=4, mlp_ratio=4., depth=2)

# Load state dict, removing 'module.' if necessary
state_dict = torch.load(args.model_pth, weights_only=True, map_location='cpu')
if any(key.startswith("module.") for key in state_dict.keys()):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model = torch.nn.DataParallel(model)  # This will split batches across both GPUs
model = model.to(device)  # Move to GPU if available
model.eval()

# Data loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
test_dataset = SegmentationDataset(args.test_images_dir, args.test_masks_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def evaluate_metrics_pytorch(model, dataloader, device):
    model.eval()
    all_metrics = {
        'Accuracy': [],
        'Dice Coefficient': [],
        'IoU': [],
        'Sensitivity': [],
        'Specificity': [],
        'F1 Score': []
    }

    with torch.no_grad():
        for images, masks, img_name in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = outputs > 0.5 

            preds = preds.cpu().numpy().astype(np.uint8)
            masks = masks.cpu().numpy().astype(np.uint8)
            masks = (masks > 0).astype(np.uint8)  
            for pred, mask in zip(preds, masks):
                pred_flat = pred.flatten()
                mask_flat = mask.flatten()

                # Calculate metrics
                tn, fp, fn, tp = confusion_matrix(mask_flat, pred_flat, labels=[0,1]).ravel()

                accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
                iou = jaccard_score(mask_flat, pred_flat, zero_division=0)
                dice = f1_score(mask_flat, pred_flat, zero_division=0)
                specificity = tn / (tn + fp + 1e-8)
                sensitivity = recall_score(mask_flat, pred_flat, zero_division=0)
                f1 = dice 

                all_metrics['IoU'].append(iou)
                all_metrics['Dice Coefficient'].append(dice)
                all_metrics['Accuracy'].append(accuracy)
                all_metrics['Sensitivity'].append(sensitivity)
                all_metrics['Specificity'].append(specificity)
                all_metrics['F1 Score'].append(f1)

    # Compute average metrics
    avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}

    print("Evaluation Metrics:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")

    return avg_metrics

# Run evaluation
evaluate_metrics_pytorch(model, test_loader, device)
