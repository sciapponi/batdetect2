#!/usr/bin/env python3
"""Analyze training metrics to suggest better loss coefficient combinations."""

import pandas as pd
import numpy as np
import yaml

# Read the metrics
metrics = pd.read_csv('outputs/logs/version_5/metrics.csv')

# Read the hparams to see current configuration
with open('outputs/logs/version_5/hparams.yaml', 'r') as f:
    hparams = yaml.safe_load(f)

print("=" * 80)
print("CURRENT LOSS COEFFICIENTS (version_5)")
print("=" * 80)
current_weights = hparams['train']['loss']
print(f"Detection weight: {current_weights['detection']['weight']}")
print(f"Size weight: {current_weights['size']['weight']}")
print(f"Classification weight: {current_weights['classification']['weight']}")
print()

# Get validation metrics only (rows where val columns are not null)
val_metrics = metrics[metrics['total_loss/val'].notna()].copy()

print("=" * 80)
print("TRAINING PROGRESSION ANALYSIS")
print("=" * 80)
print(f"Total validation steps: {len(val_metrics)}")
print()

# Analyze loss components at different stages
print("LOSS COMPONENTS AT DIFFERENT TRAINING STAGES:")
print("-" * 80)

stages = [
    ("Early (epoch ~5)", val_metrics[val_metrics['epoch'] <= 5]),
    ("Mid (epoch ~25)", val_metrics[(val_metrics['epoch'] > 20) & (val_metrics['epoch'] <= 30)]),
    ("Late (epoch ~45-49)", val_metrics[val_metrics['epoch'] >= 45])
]

for stage_name, stage_data in stages:
    if len(stage_data) > 0:
        print(f"\n{stage_name}:")
        print(f"  Detection loss (val):     {stage_data['detection_loss/val'].mean():.2f} ± {stage_data['detection_loss/val'].std():.2f}")
        print(f"  Size loss (val):          {stage_data['size_loss/val'].mean():.2f} ± {stage_data['size_loss/val'].std():.2f}")
        print(f"  Classification loss (val): {stage_data['classification_loss/val'].mean():.2f} ± {stage_data['classification_loss/val'].std():.2f}")
        print(f"  Total loss (val):         {stage_data['total_loss/val'].mean():.2f} ± {stage_data['total_loss/val'].std():.2f}")

# Get final metrics
final_metrics = val_metrics[val_metrics['epoch'] == val_metrics['epoch'].max()].iloc[-1]

print("\n" + "=" * 80)
print("FINAL PERFORMANCE (last validation)")
print("=" * 80)
print(f"Epoch: {final_metrics['epoch']}")
print(f"\nLoss values:")
print(f"  Detection loss (val):      {final_metrics['detection_loss/val']:.2f}")
print(f"  Size loss (val):           {final_metrics['size_loss/val']:.2f}")
print(f"  Classification loss (val): {final_metrics['classification_loss/val']:.2f}")
print(f"  Total loss (val):          {final_metrics['total_loss/val']:.2f}")

# Get classification metrics
class_cols = [c for c in metrics.columns if 'classification/average_precision/' in c]
final_class_aps = []
for col in class_cols:
    if pd.notna(final_metrics[col]):
        final_class_aps.append(final_metrics[col])

if final_class_aps:
    print(f"\nClassification Performance:")
    print(f"  Mean AP (across species): {np.mean(final_class_aps):.3f}")
    print(f"  Detection AP:             {final_metrics['detection/average_precision']:.3f}")
    print(f"  Classification mAP:       {final_metrics['classification/mean_average_precision']:.3f}")

print("\n" + "=" * 80)
print("LOSS CONTRIBUTION ANALYSIS")
print("=" * 80)

# Calculate the actual contribution of each loss component to total loss
# Total = detection_weight * det_loss + size_weight * size_loss + class_weight * class_loss
det_w = current_weights['detection']['weight']
size_w = current_weights['size']['weight']
class_w = current_weights['classification']['weight']

final_det_loss = final_metrics['detection_loss/val']
final_size_loss = final_metrics['size_loss/val']
final_class_loss = final_metrics['classification_loss/val']

det_contribution = det_w * final_det_loss
size_contribution = size_w * final_size_loss
class_contribution = class_w * final_class_loss
total = det_contribution + size_contribution + class_contribution

print(f"\nWeighted contributions to final total loss:")
print(f"  Detection:      {det_w} × {final_det_loss:.2f} = {det_contribution:.2f} ({100*det_contribution/total:.1f}%)")
print(f"  Size:           {size_w} × {final_size_loss:.2f} = {size_contribution:.2f} ({100*size_contribution/total:.1f}%)")
print(f"  Classification: {class_w} × {final_class_loss:.2f} = {class_contribution:.2f} ({100*class_contribution/total:.1f}%)")
print(f"  Total:          {total:.2f}")

print("\n" + "=" * 80)
print("RECOMMENDED LOSS COEFFICIENT ADJUSTMENTS")
print("=" * 80)

# Analysis based on the metrics
print("\nBased on the training progression:")
print()

# Detection loss is very high compared to others
if final_det_loss > 10 * final_class_loss:
    print("1. DETECTION LOSS is very high ({:.1f}), dominating the total loss.".format(final_det_loss))
    print("   Current weight: {}, contributing {:.1f}% to total loss".format(det_w, 100*det_contribution/total))
    print("   → RECOMMENDATION: This is actually too low! The raw detection loss is huge,")
    print("     but its weighted contribution is small. Consider INCREASING to 0.1 or even 1.0")
    print("     to force the model to focus more on detection.")
    print()

# Size loss analysis
if final_size_loss < final_class_loss:
    print("2. SIZE LOSS is relatively low ({:.1f}).".format(final_size_loss))
    print("   Current weight: {}, contributing {:.1f}% to total loss".format(size_w, 100*size_contribution/total))
    print("   → RECOMMENDATION: Size estimation seems to be learned well.")
    print("     Current weight of {} seems reasonable.".format(size_w))
    print()

# Classification is dominant
if class_contribution/total > 0.6:
    print("3. CLASSIFICATION LOSS is dominating ({:.1f}% of total).".format(100*class_contribution/total))
    print("   Current weight: {}, raw loss: {:.1f}".format(class_w, final_class_loss))
    print("   → RECOMMENDATION: Consider reducing to 0.5 to balance better with other tasks.")
    print()

print("\nSUGGESTED COEFFICIENT COMBINATIONS TO TRY:")
print("-" * 80)

suggestions = [
    {
        "name": "Balanced v1",
        "detection": 0.1,
        "size": 0.1,
        "classification": 0.5,
        "rationale": "Increase detection focus while reducing classification dominance"
    },
    {
        "name": "Balanced v2",
        "detection": 1.0,
        "size": 0.1,
        "classification": 1.0,
        "rationale": "Equal emphasis on detection and classification"
    },
    {
        "name": "Detection-focused",
        "detection": 1.0,
        "size": 0.05,
        "classification": 0.5,
        "rationale": "Prioritize detection performance"
    },
    {
        "name": "Conservative",
        "detection": 0.05,
        "size": 0.1,
        "classification": 0.7,
        "rationale": "Modest adjustment from current settings"
    }
]

for i, sug in enumerate(suggestions, 1):
    print(f"\n{i}. {sug['name']}:")
    print(f"   detection:      {sug['detection']}")
    print(f"   size:           {sug['size']}")
    print(f"   classification: {sug['classification']}")
    print(f"   Rationale: {sug['rationale']}")

print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("""
1. Update example_data/config.yaml with new loss coefficients
2. Run training with: uv run batdetect2 train --val-dataset example_data/uk_diff_val.yaml --config example_data/config.yaml example_data/uk_diff_train.yaml
3. Compare new version's metrics with version_5
4. Adjust further based on results
""")
