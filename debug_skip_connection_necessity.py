#!/usr/bin/env python3
"""Understand why StandardConv fails without skip connections but FreqCoord works."""

import torch
import yaml
from pathlib import Path
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
from batdetect2.models import build_model

print("="*70)
print("DIAGNOSING: StandardConv vs FreqCoordConv without skip connections")
print("="*70)

# Compare two configs
configs = {
    "StandardConv (NO SKIP)": "/home/stefanoc/batdetect2/example_data/config_standard_conv_skips.yaml",
    "FreqCoord (NO SKIP)": "/home/stefanoc/batdetect2/example_data/config_half_the_size.yaml",
}

# Modify configs to disable skip connections
for name, config_path in configs.items():
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Force no skip connections
    config_dict['model']['encoder']['return_skip'] = False
    config_dict['model']['decoder']['use_skip'] = False
    
    config = validate_config(config_dict)
    targets = build_targets(config=config.targets)
    model = build_model(config=config.model, targets=targets)
    
    print(f"\n--- Encoder Architecture ---")
    for i, layer in enumerate(model.detector.backbone.encoder.layers):
        layer_type = type(layer).__name__
        print(f"  Layer {i}: {layer_type}")
        
        # Check if layer has coordinate features
        if hasattr(layer, 'coords'):
            if layer.coords is not None:
                print(f"    ✓ Has FREQUENCY COORDINATES")
            else:
                print(f"    ✗ No coordinates")
        else:
            print(f"    ✗ No coordinate feature")
        
        # Check conv layer input channels
        if hasattr(layer, 'conv'):
            print(f"    Conv: {layer.conv.in_channels} -> {layer.conv.out_channels} channels")
    
    print(f"\n--- Testing Forward Pass ---")
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 400)
    
    with torch.no_grad():
        model.eval()
        
        # Pass through encoder
        encoder_out = model.detector.backbone.encoder(x)
        print(f"Encoder output shape: {encoder_out.shape}")
        
        # Pass through full model
        outputs = model.detector(x)
        
        # Check feature diversity
        features = outputs.features
        print(f"\nFeature map statistics:")
        print(f"  Shape: {features.shape}")
        print(f"  Mean: {features.mean().item():.6f}")
        print(f"  Std: {features.std().item():.6f}")
        
        # Check if features have spatial variation
        # Calculate variance across spatial dimensions
        spatial_var = features.var(dim=(2, 3)).mean().item()
        channel_var = features.var(dim=1).mean().item()
        
        print(f"  Spatial variance (avg across channels): {spatial_var:.6f}")
        print(f"  Channel variance (avg across space): {channel_var:.6f}")
        
        if spatial_var < 0.001:
            print(f"  ⚠️  Very low spatial variance - features are too uniform!")
        
        # Check output diversity
        det_probs = outputs.detection_probs
        print(f"\nDetection probabilities:")
        print(f"  Mean: {det_probs.mean().item():.6f}")
        print(f"  Spatial std: {det_probs.std(dim=(2,3)).mean().item():.6f}")

print(f"\n{'='*70}")
print("THEORY:")
print("="*70)
print("Without skip connections, the model must recover spatial/frequency")
print("information through the decoder. This requires:")
print()
print("1. FreqCoordConv blocks:")
print("   - Add explicit frequency coordinate to each layer")
print("   - Maintain frequency position awareness through encoder/decoder")
print("   - Even if downsampled, coordinates preserve frequency structure")
print()
print("2. StandardConv blocks:")
print("   - NO positional information!")
print("   - After downsampling + bottleneck, all frequency info is implicit")
print("   - Decoder can't recover spatial structure without skip connections")
print()
print("CONCLUSION:")
print("  FreqCoord blocks encode positional inductive bias")
print("  StandardConv blocks NEED skip connections without this bias")
