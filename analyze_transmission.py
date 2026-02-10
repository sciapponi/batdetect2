#!/usr/bin/env python3
"""Calculate transmission size for encoder output."""

import torch
from pathlib import Path
from batdetect2.models import build_model
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
import yaml

# Load original config
config_path = Path("example_data/config.yaml")
with open(config_path) as f:
    config_dict = yaml.safe_load(f)

config = validate_config(config_dict)
targets = build_targets(config=config.targets)
model = build_model(config=config.model, targets=targets)

# Test with realistic input size
# 0.256s clip at 256kHz = 65,536 samples
# After STFT: 128 freq bins × 400 time steps
batch_size = 1
height = 128
width = 400  # ~0.256s clip

dummy_input = torch.randn(batch_size, 1, height, width)

# Get encoder output
with torch.no_grad():
    encoder_out = model.detector.backbone.encoder(dummy_input)
    if isinstance(encoder_out, list):
        bottleneck_input = encoder_out[-1]
    else:
        bottleneck_input = encoder_out

print(f"Input spectrogram: {dummy_input.shape}")
print(f"Encoder output: {bottleneck_input.shape}")
print(f"\nBottleneck transmission:")
print(f"  Shape: {bottleneck_input.shape[1]} channels × {bottleneck_input.shape[2]} height × {bottleneck_input.shape[3]} width")
print(f"  Elements: {bottleneck_input.numel():,}")
print(f"  Float32: {bottleneck_input.numel() * 4:,} bytes ({bottleneck_input.numel() * 4 / 1024:.1f} KB)")
print(f"  Float16: {bottleneck_input.numel() * 2:,} bytes ({bottleneck_input.numel() * 2 / 1024:.1f} KB)")
print(f"  Int8: {bottleneck_input.numel():,} bytes ({bottleneck_input.numel() / 1024:.1f} KB)")

# Calculate encoder parameters
encoder_params = sum(p.numel() for p in model.detector.backbone.encoder.parameters())
decoder_params = sum(p.numel() for p in model.detector.backbone.decoder.parameters())
bottleneck_params = sum(p.numel() for p in model.detector.backbone.bottleneck.parameters())
heads_params = sum(p.numel() for p in model.detector.classifier_head.parameters()) + \
               sum(p.numel() for p in model.detector.bbox_head.parameters())

print(f"\nParameter split:")
print(f"  On-device (encoder): {encoder_params:,} params ({encoder_params * 4 / 1024 / 1024:.2f} MB)")
print(f"  Cloud (bottleneck): {bottleneck_params:,} params")
print(f"  Cloud (decoder): {decoder_params:,} params")
print(f"  Cloud (heads): {heads_params:,} params")
print(f"  Cloud total: {bottleneck_params + decoder_params + heads_params:,} params")
