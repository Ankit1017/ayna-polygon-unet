#!/usr/bin/env python3
"""
Google Colab Setup Script for Ayna Polygon UNet
Run this cell first in your Colab notebook to set up everything
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "tqdm>=4.64.0",
        "wandb>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.6.0",
        "matplotlib>=3.5.0",
        "albumentations>=1.3.0",
        "Pillow>=9.0.0"
    ]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

    print("✅ All packages installed successfully!")

def setup_gpu():
    """Check GPU availability"""
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU available. Please enable GPU in Runtime > Change runtime type")

def download_dataset():
    """Download the dataset from Google Drive"""
    try:
        subprocess.check_call(["pip", "install", "-q", "gdown"])
        subprocess.check_call([
            "gdown", "--id", "1QXLgo3ZfQPorGwhYVmZUEWO_sU3i1pHM", "-O", "dataset.zip"
        ])

        # Create data directory and extract
        os.makedirs("data", exist_ok=True)
        subprocess.check_call(["unzip", "-q", "dataset.zip", "-d", "data/"])

        print("✅ Dataset downloaded and extracted successfully!")

        # List dataset structure
        dataset_path = "data/dataset"
        if os.path.exists(dataset_path):
            for split in ["training", "validation"]:
                split_path = os.path.join(dataset_path, split)
                if os.path.exists(split_path):
                    inputs = len(os.listdir(os.path.join(split_path, "inputs")))
                    outputs = len(os.listdir(os.path.join(split_path, "outputs")))
                    print(f"  {split}: {inputs} input images, {outputs} output images")

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")

def setup_wandb():
    """Setup Weights & Biases"""
    try:
        import wandb
        print("🔗 Weights & Biases is ready!")
        print("   Run 'wandb.login()' in the next cell to authenticate")
        print("   Your API key can be found at: https://wandb.ai/authorize")
    except ImportError:
        print("❌ Wandb not installed properly")

def main():
    """Main setup function"""
    print("🚀 Setting up Ayna Polygon UNet on Google Colab...")
    print("=" * 50)

    print("\n📦 Installing packages...")
    install_requirements()

    print("\n🖥️ Checking GPU...")
    setup_gpu()

    print("\n📊 Downloading dataset...")
    download_dataset()

    print("\n📈 Setting up Wandb...")
    setup_wandb()

    print("\n" + "=" * 50)
    print("✅ Setup complete! You're ready to train the model.")
    print("\nNext steps:")
    print("1. Run 'wandb.login()' if you want to track experiments")
    print("2. Start training with: !python src/train.py --data_root data/dataset --use_wandb")

if __name__ == "__main__":
    main()
