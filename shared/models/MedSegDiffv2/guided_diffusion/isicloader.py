import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class ISICDataset(Dataset):
    def __init__(self, args, data_path, transform=None):

        # Load mapping.csv to get image and ground truth paths
        mapping_file = os.path.join(data_path, 'mapping.csv')
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Mapping file not found at: {mapping_file}")
        
        df = pd.read_csv(mapping_file)
        
        # Use the correct column names from mapping.csv
        self.image_paths = df['test_image_path'].apply(lambda x: os.path.join(data_path, x) if not os.path.isabs(x) else x).tolist()
        self.mask_paths = df['ground_truth_path'].apply(lambda x: os.path.join(data_path, x) if not os.path.isabs(x) else x).tolist()

        # Store transform and data path
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        img_path = self.image_paths[index]
        msk_path = self.mask_paths[index]

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # Apply transformations, if specified
        if self.transform:
            # Apply the same random transformations to both the image and mask
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        # Extract image name for reference
        name = os.path.basename(img_path)

        return img, mask, name
