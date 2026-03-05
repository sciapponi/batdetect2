#!/usr/bin/env python3
"""Debug detection head and loss computation."""

import torch
import yaml
from pathlib import Path
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
from batdetect2.models import build_model

print("="*70)
print("INVESTIGATING DETECTION HEAD BEHAVIOR")
print("="*70)

# Load a config
config_path = Path("/home/stefanoc/batdetect2/example_data/config_half_the_size.yaml")
with open(config_path) as f:
    config_dict = yaml.safe_load(f)

config = validate_config(config_dict)
targets = build_targets(config=config.targets)
model = build_model(config=config.model, targets=targets)

print(f"\n--- Detection Head Architecture ---")
det_head = model.detector.detection_head
print(f"Type: {type(det_head)}")
print(f"Layers: {det_head}")

# Check the final layer
if hasattr(det_head, 'conv'):
    print(f"\nFinal conv layer:")
    print(f"  In channels: {det_head.conv.in_channels}")
    print(f"  Out channels: {det_head.conv.out_channels}")
    print(f"  Bias: {det_head.conv.bias is not None}")
    if det_head.conv.bias is not None:
        print(f"  Initial bias values: {det_head.conv.bias.data}")

# Test forward pass
print(f"\n--- Testing Detection Head Output ---")
batch_size = 4
features = torch.randn(batch_size, 16, 128, 400)  # Typical backbone output

with torch.no_grad():
    det_logits = det_head(features)
    print(f"Detection logits shape: {det_logits.shape}")
    print(f"Detection logits stats:")
    print(f"  Mean: {det_logits.mean().item():.6f}")
    print(f"  Std: {det_logits.std().item():.6f}")
    print(f"  Min: {det_logits.min().item():.6f}")
    print(f"  Max: {det_logits.max().item():.6f}")
    
    # Apply sigmoid
    det_probs = torch.sigmoid(det_logits)
    print(f"\nAfter sigmoid:")
    print(f"  Mean: {det_probs.mean().item():.6f}")
    print(f"  Std: {det_probs.std().item():.6f}")
    print(f"  Min: {det_probs.min().item():.6f}")
    print(f"  Max: {det_probs.max().item():.6f}")
    print(f"  % > 0.5: {(det_probs > 0.5).float().mean().item() * 100:.2f}%")
    print(f"  % > 0.9: {(det_probs > 0.9).float().mean().item() * 100:.2f}%")
    
    # This is the problem! If logits have high positive bias, sigmoid will be ~1 everywhere
    if det_logits.mean().item() > 2:
        print(f"\n⚠️  PROBLEM: Logits are too high! Mean logit = {det_logits.mean().item():.2f}")
        print(f"   sigmoid({det_logits.mean().item():.2f}) = {torch.sigmoid(torch.tensor(det_logits.mean().item())).item():.4f}")
        print(f"   This explains the ~94% detection probability everywhere!")

# Check how detector applies sigmoid
print(f"\n--- Checking Full Detector Output ---")
detector = model.detector
x = torch.randn(2, 1, 128, 400)

with torch.no_grad():
    outputs = detector(x)
    print(f"Detection probs from full detector:")
    print(f"  Mean: {outputs.detection_probs.mean().item():.6f}")
    print(f"  All > 0.5? {(outputs.detection_probs > 0.5).all().item()}")

# Now check if the detection head has the right initialization
print(f"\n--- Checking Weight Initialization ---")
print(f"\nDetection head final layer weights:")
if hasattr(det_head, 'conv'):
    w = det_head.conv.weight
    print(f"  Weight shape: {w.shape}")
    print(f"  Weight mean: {w.mean().item():.6f}")
    print(f"  Weight std: {w.std().item():.6f}")
    
    if det_head.conv.bias is not None:
        b = det_head.conv.bias
        print(f"  Bias shape: {b.shape}")
        print(f"  Bias value: {b.item():.6f}")
        
        # For detection, the bias should be initialized to produce low initial probabilities
        # Common practice: bias = -log((1-p)/p) where p is target initial probability
        # For p=0.01, bias should be around -4.6
        expected_prob = torch.sigmoid(b).item()
        print(f"  Initial detection probability: {expected_prob:.4f}")
        
        if expected_prob > 0.5:
            print(f"\n⚠️  BUG FOUND: Detection head bias is initialized too high!")
            print(f"   Current bias = {b.item():.2f} -> prob = {expected_prob:.2f}")
            print(f"   Should be around -4.6 for initial prob ~0.01")
            print(f"\n   FIX: Initialize detection head bias to log(prior/(1-prior))")
            print(f"        For prior=0.01: bias = log(0.01/0.99) = -4.595")

# Check classification head too
print(f"\n--- Classification Head ---")
cls_head = model.detector.classification_head
print(f"Type: {type(cls_head)}")

if hasattr(cls_head, 'conv'):
    print(f"Classification head:")
    print(f"  Out channels: {cls_head.conv.out_channels}")
    if cls_head.conv.bias is not None:
        cls_bias = cls_head.conv.bias
        print(f"  Bias mean: {cls_bias.mean().item():.6f}")
        print(f"  Bias std: {cls_bias.std().item():.6f}")
        print(f"  Bias range: [{cls_bias.min().item():.2f}, {cls_bias.max().item():.2f}]")

print(f"\n{'='*70}")
print("RECOMMENDATION:")
print("="*70)
print("If detection head bias is initialized incorrectly, the model will")
print("start with ~94% detection probability everywhere, making it hard to learn.")
print("The detection head should use proper bias initialization!")
