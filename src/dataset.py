import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define color mapping
COLOR_NAMES = [
    "red", "green", "blue", "yellow", "cyan", 
    "magenta", "black", "white", "orange", "purple"
]

COLOR_TO_IDX = {color: idx for idx, color in enumerate(COLOR_NAMES)}
IDX_TO_COLOR = {idx: color for color, idx in COLOR_TO_IDX.items()}

def create_color_onehot(color_name, n_colors=10):
    """Create one-hot encoding for color name"""
    onehot = np.zeros(n_colors, dtype=np.float32)
    if color_name in COLOR_TO_IDX:
        onehot[COLOR_TO_IDX[color_name]] = 1.0
    return onehot

class PolygonDataset(Dataset):
    """
    Dataset class for polygon-color pairs
    """
    def __init__(self, data_root, split="training", transform=None, augment=True):
        """
        Args:
            data_root: Root directory containing the dataset
            split: Either "training" or "validation"
            transform: Transform to apply to images
            augment: Whether to apply data augmentation
        """
        self.data_root = data_root
        self.split = split
        self.augment = augment and (split == "training")

        # Paths
        self.input_dir = os.path.join(data_root, split, "inputs")
        self.output_dir = os.path.join(data_root, split, "outputs")
        self.json_path = os.path.join(data_root, split, "data.json")

        # Load metadata
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)

        # Setup transforms
        self.setup_transforms()

        print(f"Loaded {len(self.data)} samples for {split}")

    def setup_transforms(self):
      TARGET = 128             # ← Pick the resolution you want (128 or 256)

      resize = A.Resize(height=TARGET, width=TARGET,        # 🔹 ADD THIS
                        interpolation=cv2.INTER_NEAREST)

      if self.augment:                                   # training split
          self.aug_transform = A.Compose([
              A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT,
                      value=0, mask_value=0, p=0.5),
              A.RandomScale(scale_limit=0.2, p=0.5),
              A.HorizontalFlip(p=0.5),
              A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                                rotate_limit=15,
                                border_mode=cv2.BORDER_CONSTANT,
                                value=0, mask_value=0, p=0.5),
              resize                                           # 🔹 HERE
          ])
      else:                                             # validation split
          self.aug_transform = resize                    # 🔹 HERE

      # Normalisation definitions stay exactly the same
      self.norm_transform = A.Compose([
          A.Normalize(mean=[0.0], std=[1.0]),
          ToTensorV2()
      ])
      self.norm_transform_rgb = A.Compose([
          A.Normalize(mean=[0.0, 0.0, 0.0],
                      std=[1.0, 1.0, 1.0]),
          ToTensorV2()
      ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.data[idx]

        # Load images
        input_path = os.path.join(self.input_dir, sample["input_polygon"])
        output_path = os.path.join(self.output_dir, sample["output_image"])

        # Read input (grayscale polygon)
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if input_img is None:
            raise FileNotFoundError(f"Could not load input image: {input_path}")

        # Read output (colored polygon)
        output_img = cv2.imread(output_path, cv2.IMREAD_COLOR)
        if output_img is None:
            raise FileNotFoundError(f"Could not load output image: {output_path}")

        # Convert BGR to RGB
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        # Apply augmentation if enabled (both input and output together)
        if self.aug_transform is not None:
            # Combine input and output for consistent augmentation
            combined = np.concatenate([
                input_img[..., np.newaxis],  # Add channel dimension
                output_img
            ], axis=2)  # Shape: (H, W, 4)

            augmented = self.aug_transform(image=combined)["image"]
            input_img = augmented[..., 0]  # First channel
            output_img = augmented[..., 1:4]  # Next 3 channels

        # Normalize and convert to tensors
        input_tensor = self.norm_transform(image=input_img)["image"]  # (1, H, W)
        output_tensor = self.norm_transform_rgb(image=output_img)["image"]  # (3, H, W)

        # Create color one-hot encoding
        colour_name   = sample["colour"].lower()    
        color_onehot = create_color_onehot(colour_name)
        color_tensor = torch.from_numpy(color_onehot)

        return {
            "input_polygon": input_tensor,
            "colour_onehot": color_tensor,
            "output_image": output_tensor,
            "color_name": colour_name,
            "input_fname": sample["input_polygon"],
            "output_fname": sample["output_image"]
        }

def create_dataloaders(data_root, batch_size=32, num_workers=4):
    """Create training and validation dataloaders"""

    train_dataset = PolygonDataset(
        data_root=data_root,
        split="training",
        augment=True
    )

    val_dataset = PolygonDataset(
        data_root=data_root,
        split="validation",
        augment=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader

def visualize_sample(dataset, idx):
    """Visualize a sample from the dataset"""
    import matplotlib.pyplot as plt

    sample = dataset[idx]

    input_img = sample["input"].squeeze().numpy()  # Remove channel dimension
    output_img = sample["output"].permute(1, 2, 0).numpy()  # CHW -> HWC
    color_name = sample["color_name"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title(f"Input Polygon")
    axes[0].axis('off')

    axes[1].imshow(output_img)
    axes[1].set_title(f"Output ({color_name})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset creation...")

    # This would work if you have the actual dataset
    # dataset = PolygonDataset("data/dataset", "training")
    # print(f"Dataset size: {len(dataset)}")
    # sample = dataset[0]
    # print(f"Sample keys: {sample.keys()}")
    # print(f"Input shape: {sample['input'].shape}")
    # print(f"Output shape: {sample['output'].shape}")
    # print(f"Color shape: {sample['color'].shape}")
    # print(f"Color name: {sample['color_name']}")
