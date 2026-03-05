#!/usr/bin/env python3
"""Test if ESP32 ConvTranspose upsampling blocks have issues."""

import torch
import torch.nn as nn
from batdetect2.models.blocks_esp32 import FreqCoordConvTransposeUpBroadcastBlock
from batdetect2.models.blocks import FreqCoordConvUpBlock

def test_upsampling_block(block, name, input_shape):
    """Test upsampling block forward/backward."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Create random input
    x = torch.randn(*input_shape, requires_grad=True)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    y = block(x)
    print(f"Output shape: {y.shape}")
    print(f"Expected output height: {input_shape[2] * 2}")
    print(f"Actual output height: {y.shape[2]}")
    
    if y.shape[2] != input_shape[2] * 2:
        print("⚠️  HEIGHT MISMATCH!")
    else:
        print("✓ Height correct")
    
    # Check for NaNs or Infs
    if torch.isnan(y).any():
        print("✗ OUTPUT HAS NaNs!")
    elif torch.isinf(y).any():
        print("✗ OUTPUT HAS Infs!")
    else:
        print("✓ Output is clean")
    
    # Test gradient flow
    loss = (y ** 2).mean()
    print(f"Loss: {loss.item():.6f}")
    
    loss.backward()
    
    if x.grad is None:
        print("✗ NO INPUT GRADIENT!")
    else:
        print(f"✓ Input gradient mean: {x.grad.mean().item():.6f}")
        print(f"✓ Input gradient std: {x.grad.std().item():.6f}")
        
        if torch.isnan(x.grad).any():
            print("✗ INPUT GRADIENT HAS NaNs!")
        elif torch.isinf(x.grad).any():
            print("✗ INPUT GRADIENT HAS Infs!")
    
    # Check block parameter gradients
    for param_name, param in block.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"✗ {param_name}: NO GRADIENT")
            elif torch.isnan(param.grad).any():
                print(f"✗ {param_name}: Gradient has NaNs")
            elif torch.isinf(param.grad).any():
                print(f"✗ {param_name}: Gradient has Infs")
            else:
                print(f"✓ {param_name}: grad OK (mean={param.grad.mean().item():.6f})")

# Test configuration
batch_size = 4
in_channels = 128
input_height = 16
width = 50

print(f"Test configuration:")
print(f"  Batch size: {batch_size}")
print(f"  Input channels: {in_channels}")
print(f"  Input height: {input_height}")
print(f"  Width: {width}")
print(f"  Expected output height: {input_height * 2}")

# Create ESP32 ConvTranspose block
esp32_block = FreqCoordConvTransposeUpBroadcastBlock(
    in_channels=in_channels,
    out_channels=32,
    input_height=input_height,
)

# Create standard interpolation block for comparison
standard_block = FreqCoordConvUpBlock(
    in_channels=in_channels,
    out_channels=32,
    input_height=input_height,
)

# Test ESP32 block
test_upsampling_block(
    esp32_block,
    "FreqCoordConvTransposeUpBroadcast (ESP32)",
    (batch_size, in_channels, input_height, width)
)

# Test standard block
test_upsampling_block(
    standard_block,
    "FreqCoordConvUpBlock (Standard)",
    (batch_size, in_channels, input_height, width)
)

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")

# Compare outputs with same input
torch.manual_seed(42)
x = torch.randn(batch_size, in_channels, input_height, width)

with torch.no_grad():
    esp32_out = esp32_block(x)
    standard_out = standard_block(x)
    
    print(f"ESP32 output stats:")
    print(f"  Mean: {esp32_out.mean().item():.6f}")
    print(f"  Std: {esp32_out.std().item():.6f}")
    print(f"  Min: {esp32_out.min().item():.6f}")
    print(f"  Max: {esp32_out.max().item():.6f}")
    
    print(f"\nStandard output stats:")
    print(f"  Mean: {standard_out.mean().item():.6f}")
    print(f"  Std: {standard_out.std().item():.6f}")
    print(f"  Min: {standard_out.min().item():.6f}")
    print(f"  Max: {standard_out.max().item():.6f}")
    
    print(f"\nOutput difference:")
    diff = (esp32_out - standard_out).abs()
    print(f"  Mean absolute difference: {diff.mean().item():.6f}")
    print(f"  Max absolute difference: {diff.max().item():.6f}")
