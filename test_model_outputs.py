#!/usr/bin/env python3
"""Check if ESP32 model outputs are correct during inference."""

import torch
import yaml
from pathlib import Path
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
from batdetect2.models import build_model

def test_model_outputs(config_path, model_name):
    """Test model outputs for sanity."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}\n")
    
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = validate_config(config_dict)
    targets = build_targets(config=config.targets)
    model = build_model(config=config.model, targets=targets)
    detector = model.detector  # Get just the detector part
    detector.eval()  # Set to eval mode
    
    # Create test input (spectrogram, not audio)
    batch_size = 2
    height = config.model.input_height
    width = 400  # ~0.256s clip
    
    x = torch.randn(batch_size, 1, height, width)
    
    with torch.no_grad():
        outputs = detector(x)
        
        print(f"Model outputs:")
        print(f"  Type: {type(outputs)}")
        print(f"  Fields: {outputs._fields}")
        
        if hasattr(outputs, 'detection_probs'):
            det = outputs.detection_probs
            print(f"\nDetection output:")
            print(f"  Shape: {det.shape}")
            print(f"  Mean: {det.mean().item():.6f}")
            print(f"  Std: {det.std().item():.6f}")
            print(f"  Min: {det.min().item():.6f}")
            print(f"  Max: {det.max().item():.6f}")
            print(f"  % > 0.5: {(det > 0.5).float().mean().item() * 100:.2f}%")
            print(f"  % > 0.1: {(det > 0.1).float().mean().item() * 100:.2f}%")
            print(f"  % > 0.01: {(det > 0.01).float().mean().item() * 100:.2f}%")
            
            # Check for NaN or Inf
            if torch.isnan(det).any():
                print(f"  ⚠️  HAS NaNs!")
            if torch.isinf(det).any():
                print(f"  ⚠️  HAS Infs!")
        
        if hasattr(outputs, 'class_probs'):
            cls = outputs.class_probs
            print(f"\nClassification output:")
            print(f"  Shape: {cls.shape}")
            print(f"  Mean: {cls.mean().item():.6f}")
            print(f"  Std: {cls.std().item():.6f}")
            print(f"  Min: {cls.min().item():.6f}")
            print(f"  Max: {cls.max().item():.6f}")
            
            # Check per-class outputs
            print(f"\nPer-class statistics:")
            for i in range(min(5, cls.shape[1])):  # First 5 classes
                class_scores = cls[:, i, :, :]
                print(f"  Class {i}: mean={class_scores.mean().item():.6f}, "
                      f"max={class_scores.max().item():.6f}")
            
            if torch.isnan(cls).any():
                print(f"  ⚠️  HAS NaNs!")
            if torch.isinf(cls).any():
                print(f"  ⚠️  HAS Infs!")
        
        if hasattr(outputs, 'size_preds'):
            size = outputs.size_preds
            print(f"\nSize output:")
            print(f"  Shape: {size.shape}")
            print(f"  Mean: {size.mean().item():.6f}")
            print(f"  Std: {size.std().item():.6f}")
            print(f"  Min: {size.min().item():.6f}")
            print(f"  Max: {size.max().item():.6f}")
    
    # Test with training mode
    detector.train()
    with torch.no_grad():
        outputs_train = detector(x)
        if hasattr(outputs_train, 'detection_probs'):
            det_train = outputs_train.detection_probs
            det_eval = outputs.detection_probs
            
            print(f"\n\nTrain vs Eval mode comparison:")
            print(f"Detection difference:")
            print(f"  Mean absolute diff: {(det_train - det_eval).abs().mean().item():.6f}")
            print(f"  Max absolute diff: {(det_train - det_eval).abs().max().item():.6f}")
            
            if (det_train - det_eval).abs().max().item() > 0.1:
                print(f"  ⚠️  Large difference between train/eval modes!")
    
    return detector

# Test both configs
esp32_model = test_model_outputs(
    "/home/stefanoc/batdetect2/example_data/config_esp32_optimized.yaml",
    "ESP32 Optimized (BAD)"
)

working_model = test_model_outputs(
    "/home/stefanoc/batdetect2/example_data/config_half_the_size.yaml",
    "Half Size (GOOD)"
)

print(f"\n\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}\n")

# Run same input through both
torch.manual_seed(42)
x_test = torch.randn(1, 1, 128, 400)

with torch.no_grad():
    esp32_model.eval()
    working_model.eval()
    
    esp32_out = esp32_model(x_test)
    working_out = working_model(x_test)
    
    print("Detection outputs for same input:")
    print(f"  ESP32: mean={esp32_out.detection_probs.mean():.6f}, max={esp32_out.detection_probs.max():.6f}")
    print(f"  Working: mean={working_out.detection_probs.mean():.6f}, max={working_out.detection_probs.max():.6f}")
    
    print("\nClassification outputs for same input:")
    print(f"  ESP32: mean={esp32_out.class_probs.mean():.6f}, max={esp32_out.class_probs.max():.6f}")
    print(f"  Working: mean={working_out.class_probs.mean():.6f}, max={working_out.class_probs.max():.6f}")

print("\nIf ESP32 outputs are significantly different (e.g., all near zero), that explains the poor performance!")
