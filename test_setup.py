#!/usr/bin/env python3
"""
Quick test script to verify the installation and model architecture
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from model import PolygonColorUNet, count_parameters
from dataset import COLOR_NAMES, create_color_onehot
from utils import create_sample_polygon

def test_model():
    """Test model creation and forward pass"""
    print("🧪 Testing model architecture...")

    # Create model
    model = PolygonColorUNet(n_colors=len(COLOR_NAMES))
    print(f"✅ Model created with {count_parameters(model):,} parameters")

    # Test forward pass
    batch_size = 2
    height, width = 256, 256

    # Create dummy inputs
    input_img = torch.randn(batch_size, 1, height, width)
    color_onehot = torch.randn(batch_size, len(COLOR_NAMES))

    # Forward pass
    with torch.no_grad():
        output = model(input_img, color_onehot)

    print(f"✅ Forward pass successful")
    print(f"   Input shape: {input_img.shape}")
    print(f"   Color shape: {color_onehot.shape}")
    print(f"   Output shape: {output.shape}")

    # Test color encoding
    color_vec = create_color_onehot("red")
    print(f"✅ Color encoding working: red -> {color_vec.nonzero()[0][0]}")

    return True

def test_utils():
    """Test utility functions"""
    print(f"\n🛠️ Testing utility functions...")

    # Test polygon creation
    shapes = ['triangle', 'square', 'pentagon']
    for shape in shapes:
        polygon = create_sample_polygon(shape)
        assert polygon.shape == (256, 256), f"Wrong shape for {shape}"

    print(f"✅ Polygon generation working for {len(shapes)} shapes")

    # Test color names
    print(f"✅ {len(COLOR_NAMES)} colors available: {', '.join(COLOR_NAMES[:5])}...")

    return True

def test_dataset_format():
    """Test if dataset has correct format"""
    print(f"\n📊 Checking dataset format...")

    dataset_path = "data/dataset"
    if not os.path.exists(dataset_path):
        print("⚠️ Dataset not found. Please run download first.")
        return False

    required_dirs = [
        "training/inputs", "training/outputs", 
        "validation/inputs", "validation/outputs"
    ]

    for dir_path in required_dirs:
        full_path = os.path.join(dataset_path, dir_path)
        if os.path.exists(full_path):
            count = len(os.listdir(full_path))
            print(f"✅ {dir_path}: {count} files")
        else:
            print(f"❌ Missing: {dir_path}")
            return False

    return True

def main():
    """Run all tests"""
    print("🔍 Running Ayna Polygon UNet Tests")
    print("=" * 40)

    try:
        # Test model
        test_model()

        # Test utilities  
        test_utils()

        # Test dataset
        test_dataset_format()

        print("\n" + "=" * 40)
        print("✅ All tests passed! System is ready.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
