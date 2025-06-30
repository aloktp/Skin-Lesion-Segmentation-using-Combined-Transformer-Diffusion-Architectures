from PIL import Image
import os
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, images_dir, masks_dir, transform=None):
        super(SegmentationDataset, self).__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_filenames = os.listdir(images_dir)

        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure RGB
        # Load mask
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        mask = Image.open(mask_path).convert('L')    # Grayscale
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, img_name
