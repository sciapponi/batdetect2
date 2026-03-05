#!/usr/bin/env python3
"""Check if different block types have initialization issues."""

import torch
import torch.nn as nn
from batdetect2.models.blocks import (
    FreqCoordConvDownBroadcastBlock,
    DepthwiseSeparableConvDownBlock,
    StandardConvDownBlock,
)

def check_block_weights(block, name):
    """Check weight statistics for a block."""
    print(f"\n=== {name} ===")
    for param_name, param in block.named_parameters():
        if "weight" in param_name:
            print(f"  {param_name}:")
            print(f"    Shape: {param.shape}")
            print(f"    Mean: {param.mean().item():.6f}")
            print(f"    Std: {param.std().item():.6f}")
            print(f"    Min: {param.min().item():.6f}")
            print(f"    Max: {param.max().item():.6f}")

# Create instances of different block types
print("Initializing blocks with same parameters (in_channels=16, out_channels=32)...")

freq_block = FreqCoordConvDownBroadcastBlock(
    in_channels=16,
    out_channels=32,
    input_height=128,
    kernel_size=3,
    pad_size=1,
)

depthwise_block = DepthwiseSeparableConvDownBlock(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    pad_size=1,
)

standard_block = StandardConvDownBlock(
    in_channels=16,
    out_channels=32,
    kernel_size=3,
    pad_size=1,
)

# Check weights
check_block_weights(freq_block, "FreqCoordConvDownBroadcast")
check_block_weights(depthwise_block, "DepthwiseSeparableConvDown")
check_block_weights(standard_block, "StandardConvDown")

# Key difference: FreqCoordConv takes in_channels+1 due to coord channel
print("\n" + "="*60)
print("CRITICAL DIFFERENCE FOUND:")
print("="*60)
print(f"\nFreqCoordConvBlock conv layer expects: {freq_block.conv.in_channels} channels")
print(f"  (in_channels + 1 for frequency coordinate)")
print(f"\nDepth wiseSeparableConvBlock depthwise expects: {depthwise_block.depthwise.in_channels} channels")
print(f"StandardConvBlock conv expects: {standard_block.conv.in_channels} channels")

print("\n" + "="*60)
print("NOTE: All blocks use PyTorch default initialization")
print("  Conv2d uses kaiming_uniform_ for weights")
print("  This should be similar across block types")
print("="*60)
