#!/usr/bin/env python
"""Debug config loading."""

from batdetect2.models.config import load_backbone_config
from batdetect2.models import build_classifier

# Load configs
config_full = load_backbone_config("example_data/config_encoder_litemla_classifier.yaml")
config_half = load_backbone_config("example_data/config_half_classifier.yaml")

print("FULL CONFIG:")
print(f"  Encoder layers: {len(config_full.encoder.layers)}")
print(f"  Layer 0 out_channels: {config_full.encoder.layers[0].out_channels}")
print(f"  Layer 1 out_channels: {config_full.encoder.layers[1].out_channels}")
print(f"  Bottleneck channels: {config_full.bottleneck.channels}")
if config_full.bottleneck.layers:
    print(f"  Bottleneck layer 0: {config_full.bottleneck.layers[0]}")

print("\nHALF CONFIG:")
print(f"  Encoder layers: {len(config_half.encoder.layers)}")
print(f"  Layer 0 out_channels: {config_half.encoder.layers[0].out_channels}")
print(f"  Layer 1 out_channels: {config_half.encoder.layers[1].out_channels}")
print(f"  Bottleneck channels: {config_half.bottleneck.channels}")
if config_half.bottleneck.layers:
    print(f"  Bottleneck layer 0: {config_half.bottleneck.layers[0]}")

# Build and check actual models
model_full = build_classifier(num_classes=17, config=config_full)
model_half = build_classifier(num_classes=17, config=config_half)

print("\nACTUAL MODEL ENCODER CHANNELS:")
print(f"  Full encoder out_channels: {model_full.encoder.out_channels}")
print(f"  Half encoder out_channels: {model_half.encoder.out_channels}")
print(f"  Full bottleneck out_channels: {model_full.bottleneck.out_channels}")
print(f"  Half bottleneck out_channels: {model_half.bottleneck.out_channels}")
