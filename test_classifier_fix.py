#!/usr/bin/env python3
"""Test if the bias initialization fix works."""

import torch
import yaml
from pathlib import Path
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
from batdetect2.models import build_model

print("="*70)
print("TESTING FIXED CLASSIFIER HEAD INITIALIZATION")
print("="*70)

# Build model with fixed initialization
config_path = Path("/home/stefanoc/batdetect2/example_data/config_half_the_size.yaml")
with open(config_path) as f:
    config_dict = yaml.safe_load(f)

config = validate_config(config_dict)
targets = build_targets(config=config.targets)
model = build_model(config=config.model, targets=targets)

print(f"\n--- Classifier Head Initialization ---")
classifier = model.detector.classifier_head
print(f"Number of classes: {classifier.num_classes}")
print(f"Classifier bias shape: {classifier.classifier.bias.shape}")
print(f"Classifier bias values:")
print(f"  Class biases (first 5): {classifier.classifier.bias[:5].tolist()}")
print(f"  Background bias (last): {classifier.classifier.bias[-1].item()}")

# Test with random input
print(f"\n--- Testing Model Output ---")
batch_size = 2
x = torch.randn(batch_size, 1, 128, 400)

with torch.no_grad():
    model.eval()
    outputs = model.detector(x)
    
    print(f"\nDetection probabilities:")
    print(f"  Mean: {outputs.detection_probs.mean().item():.6f}")
    print(f"  Std: {outputs.detection_probs.std().item():.6f}")
    print(f"  Min: {outputs.detection_probs.min().item():.6f}")
    print(f"  Max: {outputs.detection_probs.max().item():.6f}")
    print(f"  % > 0.5: {(outputs.detection_probs > 0.5).float().mean().item() * 100:.2f}%")
    print(f"  % > 0.1: {(outputs.detection_probs > 0.1).float().mean().item() * 100:.2f}%")
    print(f"  % > 0.01: {(outputs.detection_probs > 0.01).float().mean().item() * 100:.2f}%")
    
    print(f"\nClassification probabilities:")
    print(f"  Mean: {outputs.class_probs.mean().item():.6f}")
    print(f"  Std: {outputs.class_probs.std().item():.6f}")
    print(f"  Max: {outputs.class_probs.max().item():.6f}")
    
    print(f"\nPer-class probabilities (first 5 classes):")
    for i in range(min(5, outputs.class_probs.shape[1])):
        class_prob = outputs.class_probs[:, i, :, :].mean().item()
        print(f"  Class {i}: {class_prob:.6f}")

print(f"\n{'='*70}")
print("EXPECTED RESULTS:")
print("="*70)
print("✓ Classifier bias: -6.0 for classes, +6.0 for background")
print("✓ Detection probability: ~1-2% (not ~94%!)")
print("✓ Each class probability: ~0.1% or less")
print("\nIf detection is still ~94%, the fix didn't work.")
print("If detection is now ~1-2%, the bug is FIXED!")
