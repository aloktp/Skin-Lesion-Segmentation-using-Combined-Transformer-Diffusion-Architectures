# train.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from SwinUnet import SwinUNet, DiceLoss, initialize_weights  # Import from SwinUNet.py
from SegmentationDataset import SegmentationDataset  # Assuming you have dataset.py for SegmentationDataset

import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Training script for SwinUNet segmentation model")
parser.add_argument('--train_images_dir', type=str, required=True, help='Path to the training images directory')
parser.add_argument('--train_masks_dir', type=str, required=True, help='Path to the training masks directory')
parser.add_argument('--save_model_path', type=str, required=True, help='Path to save the trained model')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
args = parser.parse_args()

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load training data
train_dataset = SegmentationDataset(args.train_images_dir, args.train_masks_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Initialize model, loss, optimizer, and scheduler
model = SwinUNet(input_channels=3, output_channels=1).to(device)
model = torch.nn.DataParallel(model)  # This will split batches across both GPUs
model.apply(initialize_weights)
criterion = DiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=5, min_lr=1e-9)

# Training loop with early stopping
best_val_loss = float('inf')
for epoch in range(args.num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    # Update learning rate
    scheduler.step(running_loss / len(train_loader.dataset))

    # Save the best model
    if running_loss < best_val_loss:
        best_val_loss = running_loss
        torch.save(model.state_dict(), args.save_model_path)
    print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss:.4f}")

print("Training Complete!")


# python train.py --train_images_dir '/path/to/train_images' \
#                 --train_masks_dir '/path/to/train_masks' \
#                 --save_model_path '/path/to/save/model.pth' \
#                 --batch_size 32 \
#                 --learning_rate 0.0001 \
#                 --num_epochs 10


# python train.py --train_images_dir '/srv/scratch/z5450230/dataset/ISIC2018_Task1-2_Training_Input' --train_masks_dir '/srv/scratch/z5450230/dataset/ISIC2018_Task1_Training_GroundTruth' --save_model_path '/srv/scratch/z5450230/tbconv/model/model.pth' --batch_size 24  --learning_rate 0.0001  --num_epochs 10