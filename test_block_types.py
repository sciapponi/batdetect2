#!/usr/bin/env python3
"""Test different block types to see if there's a training issue."""

import torch
import torch.nn as nn
from batdetect2.models.blocks import (
    FreqCoordConvDownBroadcastBlock,
    DepthwiseSeparableConvDownBlock,
    StandardConvDownBlock,
    FreqCoordConvTransposeUpBroadcastBlock,
    DepthwiseSeparableConvTransposeUpBlock,
    StandardConvUpBlock,
)

def test_forward_backward(block, name, input_shape):
    """Test forward and backward pass through a block."""
    print(f"\n=== Testing {name} ===")
    print(f"Input shape: {input_shape}")
    
    # Create input and target
    x = torch.randn(*input_shape, requires_grad=True)
    
    # Forward pass
    y = block(x)
    print(f"Output shape: {y.shape}")
    
    # Create a simple loss (mean squared error with zeros)
    loss = (y ** 2).mean()
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check if gradients exist
    if x.grad is not None:
        print(f"Input gradient mean: {x.grad.mean().item():.6f}")
        print(f"Input gradient std: {x.grad.std().item():.6f}")
    
    # Check parameter gradients
    has_grads = False
    for name_param, param in block.named_parameters():
        if param.grad is not None:
            has_grads = True
            print(f"  {name_param} grad mean: {param.grad.mean().item():.6f}")
    
    if not has_grads:
        print("  WARNING: No parameter gradients!")
    
    return y.shape

print("=" * 60)
print("Testing DOWNSAMPLING blocks")
print("=" * 60)

batch_size = 4
input_channels = 16
height = 128
width = 400

# Test FreqCoordConv
freq_down = FreqCoordConvDownBroadcastBlock(
    in_channels=input_channels,
    out_channels=32,
    input_height=height,
)
out_shape = test_forward_backward(
    freq_down, 
    "FreqCoordConvDownBroadcast",
    (batch_size, input_channels, height, width)
)

# Test DepthwiseSeparable
depthwise_down = DepthwiseSeparableConvDownBlock(
    in_channels=input_channels,
    out_channels=32,
)
test_forward_backward(
    depthwise_down,
    "DepthwiseSeparableConvDown",
    (batch_size, input_channels, height, width)
)

# Test Standard
standard_down = StandardConvDownBlock(
    in_channels=input_channels,
    out_channels=32,
)
test_forward_backward(
    standard_down,
    "StandardConvDown",
    (batch_size, input_channels, height, width)
)

print("\n" + "=" * 60)
print("Testing UPSAMPLING blocks") 
print("=" * 60)

up_input_channels = 32
up_height = 64
up_width = 200

# Test FreqCoordConv Up
from batdetect2.models.blocks_esp32 import FreqCoordConvTransposeUpBroadcastBlock
freq_up = FreqCoordConvTransposeUpBroadcastBlock(
    in_channels=up_input_channels,
    out_channels=16,
    input_height=up_height,
)
test_forward_backward(
    freq_up,
    "FreqCoordConvTransposeUpBroadcast", 
    (batch_size, up_input_channels, up_height, up_width)
)

# Test DepthwiseSeparable Up
depthwise_up = DepthwiseSeparableConvTransposeUpBlock(
    in_channels=up_input_channels,
    out_channels=16,
    input_height=up_height,
)
test_forward_backward(
    depthwise_up,
    "DepthwiseSeparableConvTransposeUp",
    (batch_size, up_input_channels, up_height, up_width)
)

# Test Standard Up
standard_up = StandardConvUpBlock(
    in_channels=up_input_channels,
    out_channels=16,
)
test_forward_backward(
    standard_up,
    "StandardConvUp",
    (batch_size, up_input_channels, up_height, up_width)
)

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print("All blocks should show gradients flowing properly.")
print("If any block shows missing gradients, that's a problem!")
