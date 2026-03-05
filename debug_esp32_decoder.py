#!/usr/bin/env python3
"""Compare ESP32 decoder (ConvTranspose) vs standard decoder (bilinear)."""

import torch
import yaml
from pathlib import Path
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
from batdetect2.models import build_model

print("="*70)
print("COMPARING ESP32 DECODER vs STANDARD DECODER")
print("="*70)

configs = {
    "ESP32 (ConvTranspose upsampling)": "/home/stefanoc/batdetect2/example_data/config_esp32_optimized.yaml",
    "Standard (Bilinear upsampling)": "/home/stefanoc/batdetect2/example_data/config_half_the_size.yaml",
}

for name, config_path in configs.items():
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = validate_config(config_dict)
    targets = build_targets(config=config.targets)
    model = build_model(config=config.model, targets=targets)
    
    print(f"\nDecoder layers:")
    for i, layer in enumerate(model.detector.backbone.decoder.layers):
        layer_type = type(layer).__name__
        print(f"  Layer {i}: {layer_type}")
        
        if hasattr(layer, 'coords'):
            if layer.coords is not None:
                print(f"    ✓ Has frequency coordinates")
            else:
                print(f"    ✗ No coordinates")
        
        if hasattr(layer, 'upsample'):
            upsample_type = type(layer.upsample).__name__
            print(f"    Upsampling: {upsample_type}")
        elif hasattr(layer, 'up_mode'):
            print(f"    Upsampling: F.interpolate (mode={layer.up_mode})")
    
    # Test gradient flow
    print(f"\n--- Testing Gradient Flow ---")
    batch_size = 2
    x = torch.randn(batch_size, 1, 128, 400, requires_grad=True)
    
    model.train()
    outputs = model.detector(x)
    
    # Simple loss
    loss = outputs.detection_probs.mean()
    loss.backward()
    
    # Check decoder gradients
    print(f"\nDecoder layer gradients:")
    for i, layer in enumerate(model.detector.backbone.decoder.layers):
        if hasattr(layer, 'conv'):
            if layer.conv.weight.grad is not None:
                grad_norm = layer.conv.weight.grad.norm().item()
                print(f"  Layer {i} conv grad norm: {grad_norm:.6f}")
                
                if grad_norm < 1e-8:
                    print(f"    ⚠️  Vanishing gradient!")
            else:
                print(f"  Layer {i}: NO GRADIENT")
        
        if hasattr(layer, 'upsample'):
            if layer.upsample.weight.grad is not None:
                grad_norm = layer.upsample.weight.grad.norm().item()
                print(f"  Layer {i} upsample grad norm: {grad_norm:.6f}")
                
                if grad_norm < 1e-8:
                    print(f"    ⚠️  Vanishing gradient!")

print(f"\n{'='*70}")
print("DIAGNOSIS:")
print("="*70)
print("If ESP32 ConvTranspose2d layers have vanishing gradients,")
print("that explains why the model doesn't learn!")
print()
print("ConvTranspose2d adds learnable parameters to upsampling,")
print("but may have different initialization/gradient flow properties")
print("compared to bilinear interpolation.")
