#!/usr/bin/env python3
"""Debug ESP32 architecture to find the issue."""

import yaml
import torch
from pathlib import Path
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
from batdetect2.models import build_model

def analyze_model(config_path, name):
    """Analyze a model's architecture."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {name}")
    print(f"Config: {config_path}")
    print(f"{'='*70}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = validate_config(config_dict)
    targets = build_targets(config=config.targets)
    model = build_model(config=config.model, targets=targets)
    
    print(f"\n--- Encoder Architecture ---")
    print(f"Input: {config.model.in_channels} channels, {config.model.input_height} height")
    for i, layer in enumerate(model.detector.backbone.encoder.layers):
        layer_type = type(layer).__name__
        print(f"  Layer {i}: {layer_type}")
        if hasattr(layer, 'conv'):
            print(f"    Conv: {layer.conv.in_channels} -> {layer.conv.out_channels}")
            if hasattr(layer, 'coords'):
                if layer.coords is not None:
                    print(f"    ✓ Has frequency coordinates (shape: {layer.coords.shape})")
                else:
                    print(f"    ✗ Coordinates disabled!")
        elif hasattr(layer, 'layers'):  # LayerGroup
            print(f"    LayerGroup with {len(layer.layers)} sublayers:")
            for j, sublayer in enumerate(layer.layers):
                sublayer_type = type(sublayer).__name__
                print(f"      {j}: {sublayer_type}")
                if hasattr(sublayer, 'conv'):
                    print(f"        Conv: {sublayer.conv.in_channels} -> {sublayer.conv.out_channels}")
    
    print(f"\n--- Decoder Architecture ---")
    for i, layer in enumerate(model.detector.backbone.decoder.layers):
        layer_type = type(layer).__name__
        print(f"  Layer {i}: {layer_type}")
        if hasattr(layer, 'conv'):
            print(f"    Conv: {layer.conv.in_channels} -> {layer.conv.out_channels}")
            if hasattr(layer, 'coords'):
                if layer.coords is not None:
                    print(f"    ✓ Has frequency coordinates (shape: {layer.coords.shape})")
                else:
                    print(f"    ✗ Coordinates disabled!")
        elif hasattr(layer, 'layers'):  # LayerGroup
            print(f"    LayerGroup with {len(layer.layers)} sublayers:")
            for j, sublayer in enumerate(layer.layers):
                sublayer_type = type(sublayer).__name__
                print(f"      {j}: {sublayer_type}")
                if hasattr(sublayer, 'conv'):
                    print(f"        Conv: {sublayer.conv.in_channels} -> {sublayer.conv.out_channels}")
    
    # Test forward pass
    print(f"\n--- Testing Forward Pass ---")
    batch_size = 2
    height = config.model.input_height
    width = 400
    
    x = torch.randn(batch_size, config.model.in_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    try:
        with torch.no_grad():
            encoder_out = model.detector.backbone.encoder(x)
            if isinstance(encoder_out, list):
                bottleneck_in = encoder_out[-1]
            else:
                bottleneck_in = encoder_out
            print(f"Encoder output shape: {bottleneck_in.shape}")
            
            bottleneck_out = model.detector.backbone.bottleneck(bottleneck_in)
            print(f"Bottleneck output shape: {bottleneck_out.shape}")
            
            decoder_out = model.detector.backbone.decoder(bottleneck_out)
            print(f"Decoder output shape: {decoder_out.shape}")
            
            final_out = model.detector.backbone(x)
            print(f"Final backbone output shape: {final_out.shape}")
            
            detections = model.detector(x)
            print(f"Detection head output shape: {detections.shape}")
            
            print("✓ Forward pass successful!")
            
    except Exception as e:
        print(f"✗ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Model Size ---")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

# Compare configs
esp32_model = analyze_model("/home/stefanoc/batdetect2/example_data/config_esp32_optimized.yaml", "ESP32 Optimized")
print("\n\n")
half_model = analyze_model("/home/stefanoc/batdetect2/example_data/config_half_the_size.yaml", "Half Size (Working)")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("\nBoth models should have similar structure if they're using FreqCoord blocks.")
print("Check for differences in coordinate handling or layer configuration above.")
