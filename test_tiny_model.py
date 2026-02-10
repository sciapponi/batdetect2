#!/usr/bin/env python3
"""Quick test to verify the XiConv U-Net model works."""

import torch
from pathlib import Path

from batdetect2.models import build_model
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
import yaml

# Load config
config_path = Path("example_data/config_xiconv_unet.yaml")
with open(config_path) as f:
    config_dict = yaml.safe_load(f)

config = validate_config(config_dict)

# Build model
targets = build_targets(config=config.targets)
model = build_model(
    config=config.model,
    targets=targets,
)
detector = model.detector
print(f"\nModel built successfully!")
print(f"Total parameters: {sum(p.numel() for p in detector.parameters()):,}")

# Test forward pass
batch_size = 2
channels = 1
height = 128
width = 400  # arbitrary time dimension

dummy_input = torch.randn(batch_size, channels, height, width)
print(f"\nInput shape: {dummy_input.shape}")

try:
    with torch.no_grad():
        output = detector(dummy_input)
    print(f"✓ Forward pass successful!")
    if isinstance(output, dict):
        print(f"Detection output shape: {output['detection'].shape}")
        print(f"Class output shape: {output['class'].shape}")
        print(f"Size output shape: {output['size'].shape}")
    else:
        print(f"Output type: {type(output)}")
        print(f"Output: {output}")
except Exception as e:
    print(f"✗ Forward pass failed!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
