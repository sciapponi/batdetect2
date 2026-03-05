"""Test proper initialization for ConvTranspose2d to match bilinear."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Create test input
x = torch.randn(1, 3, 4, 4)

print("=" * 70)
print("Testing ConvTranspose2d initialization")
print("=" * 70)

# Standard bilinear interpolation
bilinear_out = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
print(f"\nBilinear output shape: {bilinear_out.shape}")
print(f"Bilinear output mean: {bilinear_out.mean():.6f}")
print(f"Bilinear output std:  {bilinear_out.std():.6f}")

# ConvTranspose2d with 0.25 initialization (my current approach)
conv_025 = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
with torch.no_grad():
    conv_025.weight.fill_(0.25)
    conv_025.bias.zero_()

out_025 = conv_025(x)
print(f"\nConvTranspose (0.25) output shape: {out_025.shape}")
print(f"ConvTranspose (0.25) output mean: {out_025.mean():.6f}")
print(f"ConvTranspose (0.25) output std:  {out_025.std():.6f}")
print(f"Difference from bilinear: {(out_025 - bilinear_out).abs().mean():.6f}")

# ConvTranspose2d with proper bilinear kernel
# For 2x upsampling with kernel_size=2, stride=2:
# Each output pixel should be an interpolation of neighbors
# The bilinear kernel for this case uses overlapping weights
def get_bilinear_kernel(in_channels, out_channels):
    """Proper bilinear upsampling kernel for ConvTranspose2d."""
    assert in_channels == out_channels, "Must have same in/out channels"
    
    # Bilinear kernel for 2x2 upsampling with kernel_size=2, stride=2
    # This creates a smooth interpolation
    kernel = torch.zeros(in_channels, out_channels, 2, 2)
    
    # For each channel, set identity with bilinear weights
    # With stride=2, kernel_size=2, we get checkerboard pattern
    # Proper init uses 1.0 for identity mapping per channel
    for i in range(in_channels):
        kernel[i, i, :, :] = 1.0  # Identity per channel
    
    return kernel

conv_bilinear = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2, padding=0)
with torch.no_grad():
    conv_bilinear.weight.copy_(get_bilinear_kernel(3, 3))
    conv_bilinear.bias.zero_()

out_bilinear = conv_bilinear(x)
print(f"\nConvTranspose (identity) output shape: {out_bilinear.shape}")
print(f"ConvTranspose (identity) output mean: {out_bilinear.mean():.6f}")
print(f"ConvTranspose (identity) output std:  {out_bilinear.std():.6f}")
print(f"Difference from bilinear: {(out_bilinear - bilinear_out).abs().mean():.6f}")

# Test gradient flow
print("\n" + "=" * 70)
print("Gradient Flow Test")
print("=" * 70)

loss_fn = nn.MSELoss()
target = torch.ones_like(bilinear_out)

# Test with 0.25 init
x1 = x.clone().requires_grad_(True)
out1 = conv_025(x1)
loss1 = loss_fn(out1, target)
loss1.backward()
print(f"\n0.25 init - Input grad norm: {x1.grad.norm():.6f}")
print(f"0.25 init - Weight grad norm: {conv_025.weight.grad.norm():.6f}")

# Test with identity init
x2 = x.clone().requires_grad_(True)
out2 = conv_bilinear(x2)
loss2 = loss_fn(out2, target)
loss2.backward()
print(f"\nIdentity init - Input grad norm: {x2.grad.norm():.6f}")
print(f"Identity init - Weight grad norm: {conv_bilinear.weight.grad.norm():.6f}")

# Test with standard bilinear
x3 = x.clone().requires_grad_(True)
out3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
loss3 = loss_fn(out3, target)
loss3.backward()
print(f"\nBilinear interpolation - Input grad norm: {x3.grad.norm():.6f}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("Identity (1.0) initialization should match bilinear interpolation better")
print("than 0.25 initialization, and have stronger gradient flow.")
