import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List

def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) for batch of images

    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
        threshold: Threshold for binarization

    Returns:
        Mean IoU across the batch
    """
    # Binarize predictions and targets
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    # Calculate intersection and union
    intersection = (pred_binary * target_binary).sum(dim=[1, 2, 3])
    union = pred_binary.sum(dim=[1, 2, 3]) + target_binary.sum(dim=[1, 2, 3]) - intersection

    # Calculate IoU (add small epsilon to avoid division by zero)
    iou = (intersection + 1e-8) / (union + 1e-8)

    return iou.mean()

def calculate_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Calculate pixel-wise accuracy"""
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()

    correct = (pred_binary == target_binary).float()
    accuracy = correct.mean()

    return accuracy

def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, val_ious, best_iou, filepath):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_ious': val_ious,
        'best_iou': best_iou
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint

def visualize_results(input_img, target_img, pred_img, color_name, save_path=None):
    """
    Visualize input, target, and prediction

    Args:
        input_img: Input polygon image (H, W) or (1, H, W)
        target_img: Target colored image (3, H, W) or (H, W, 3)
        pred_img: Predicted colored image (3, H, W) or (H, W, 3)
        color_name: Name of the color
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy and handle dimensions
    if torch.is_tensor(input_img):
        input_img = input_img.cpu().numpy()
    if torch.is_tensor(target_img):
        target_img = target_img.cpu().numpy()
    if torch.is_tensor(pred_img):
        pred_img = pred_img.cpu().numpy()

    # Handle different input formats
    if input_img.ndim == 3 and input_img.shape[0] == 1:
        input_img = input_img.squeeze(0)

    if target_img.ndim == 3 and target_img.shape[0] == 3:
        target_img = target_img.transpose(1, 2, 0)

    if pred_img.ndim == 3 and pred_img.shape[0] == 3:
        pred_img = pred_img.transpose(1, 2, 0)

    # Ensure values are in [0, 1] range
    target_img = np.clip(target_img, 0, 1)
    pred_img = np.clip(pred_img, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_img, cmap='gray')
    axes[0].set_title('Input Polygon')
    axes[0].axis('off')

    axes[1].imshow(target_img)
    axes[1].set_title(f'Target ({color_name})')
    axes[1].axis('off')

    axes[2].imshow(pred_img)
    axes[2].set_title(f'Prediction ({color_name})')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def create_color_legend():
    """Create a legend showing all available colors"""
    from dataset import COLOR_NAMES, COLOR_TO_IDX

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    # Define RGB values for colors (approximate)
    color_rgb = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'yellow': [1, 1, 0],
        'cyan': [0, 1, 1],
        'magenta': [1, 0, 1],
        'black': [0, 0, 0],
        'white': [1, 1, 1],
        'orange': [1, 0.5, 0],
        'purple': [0.5, 0, 1]
    }

    for i, color_name in enumerate(COLOR_NAMES):
        rgb = color_rgb.get(color_name, [0.5, 0.5, 0.5])
        axes[i].imshow([[rgb]], aspect='auto')
        axes[i].set_title(f'{color_name} ({COLOR_TO_IDX[color_name]})')
        axes[i].axis('off')

    plt.suptitle('Available Colors')
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, val_ious, save_path=None):
    """Plot training curves"""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # IoU curve
    ax2.plot(epochs, val_ious, 'g-', label='Validation IoU')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('IoU')
    ax2.set_title('Validation IoU')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

def preprocess_image_for_inference(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Preprocess an image for inference

    Args:
        image_path: Path to the input image
        target_size: Target size (height, width)

    Returns:
        Preprocessed image tensor (1, 1, H, W)
    """
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")

    # Resize image
    img = cv2.resize(img, (target_size[1], target_size[0]))

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert to tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    return img_tensor

def postprocess_output(output_tensor: torch.Tensor) -> np.ndarray:
    """
    Postprocess model output for visualization

    Args:
        output_tensor: Model output (1, 3, H, W) or (3, H, W)

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    if output_tensor.dim() == 4:
        output_tensor = output_tensor.squeeze(0)  # Remove batch dimension

    # Convert to numpy and transpose from CHW to HWC
    output_np = output_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    # Clip values to [0, 1]
    output_np = np.clip(output_np, 0, 1)

    return output_np

def create_sample_polygon(shape: str, size: Tuple[int, int] = (256, 256), 
                         thickness: int = 2) -> np.ndarray:
    """
    Create a sample polygon for testing

    Args:
        shape: Shape type ('triangle', 'square', 'pentagon', 'hexagon', 'octagon')
        size: Image size (height, width)
        thickness: Line thickness for drawing

    Returns:
        Grayscale image with polygon (H, W)
    """
    img = np.zeros(size, dtype=np.uint8)
    center = (size[1] // 2, size[0] // 2)
    radius = min(size) // 3

    if shape == 'triangle':
        pts = np.array([
            [center[0], center[1] - radius],
            [center[0] - int(radius * 0.866), center[1] + radius // 2],
            [center[0] + int(radius * 0.866), center[1] + radius // 2]
        ], np.int32)
    elif shape == 'square':
        half_side = radius // 2
        pts = np.array([
            [center[0] - half_side, center[1] - half_side],
            [center[0] + half_side, center[1] - half_side],
            [center[0] + half_side, center[1] + half_side],
            [center[0] - half_side, center[1] + half_side]
        ], np.int32)
    elif shape == 'pentagon':
        angles = np.linspace(0, 2 * np.pi, 6)[:-1] - np.pi/2
        pts = np.array([[center[0] + int(radius * np.cos(a)), 
                        center[1] + int(radius * np.sin(a))] for a in angles], np.int32)
    elif shape == 'hexagon':
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        pts = np.array([[center[0] + int(radius * np.cos(a)), 
                        center[1] + int(radius * np.sin(a))] for a in angles], np.int32)
    elif shape == 'octagon':
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        pts = np.array([[center[0] + int(radius * np.cos(a)), 
                        center[1] + int(radius * np.sin(a))] for a in angles], np.int32)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    cv2.polylines(img, [pts], True, 255, thickness)

    return img

def generate_synthetic_dataset(output_dir: str, num_samples: int = 100):
    """
    Generate synthetic dataset for testing

    Args:
        output_dir: Directory to save the generated dataset
        num_samples: Number of samples to generate
    """
    import json
    import random
    from dataset import COLOR_NAMES

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'outputs'), exist_ok=True)

    shapes = ['triangle', 'square', 'pentagon', 'hexagon', 'octagon']
    data = []

    color_rgb = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        'yellow': [255, 255, 0],
        'cyan': [0, 255, 255],
        'magenta': [255, 0, 255],
        'black': [0, 0, 0],
        'white': [255, 255, 255],
        'orange': [255, 128, 0],
        'purple': [128, 0, 255]
    }

    for i in range(num_samples):
        # Random shape and color
        shape = random.choice(shapes)
        color = random.choice(COLOR_NAMES)

        # Generate polygon
        polygon = create_sample_polygon(shape)

        # Create colored version
        colored = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = polygon > 0
        colored[mask] = color_rgb[color]

        # Save images
        input_filename = f"{shape}_{i:04d}.png"
        output_filename = f"{shape}_{color}_{i:04d}.png"

        cv2.imwrite(os.path.join(output_dir, 'inputs', input_filename), polygon)
        cv2.imwrite(os.path.join(output_dir, 'outputs', output_filename), colored)

        data.append({
            "input": input_filename,
            "output": output_filename,
            "color": color
        })

    # Save metadata
    with open(os.path.join(output_dir, 'data.json'), 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {num_samples} synthetic samples in {output_dir}")

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test color legend
    create_color_legend()

    # Test sample polygon creation
    shapes = ['triangle', 'square', 'pentagon', 'hexagon', 'octagon']
    fig, axes = plt.subplots(1, len(shapes), figsize=(15, 3))

    for i, shape in enumerate(shapes):
        polygon = create_sample_polygon(shape)
        axes[i].imshow(polygon, cmap='gray')
        axes[i].set_title(shape.capitalize())
        axes[i].axis('off')

    plt.suptitle('Sample Polygons')
    plt.tight_layout()
    plt.show()
