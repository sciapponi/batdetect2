import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read metrics
df = pd.read_csv('outputs/logs/version_92/metrics.csv')

# Get validation metrics (rows with validation data)
val_metrics = df[df['total_loss/val'].notna()].copy()

# Get per-species classification AP columns
species_ap_cols = [col for col in df.columns if col.startswith('classification/average_precision/')]

# Extract species names 
species_names = [col.replace('classification/average_precision/', '') for col in species_ap_cols]

# Get per-genus classification AP columns (from species predictions)
genus_ap_cols = [col for col in df.columns 
                 if col.startswith('classification/genus/average_precision/')]

# Extract genus names
genus_names = [col.replace('classification/genus/average_precision/', '') 
               for col in genus_ap_cols]

# Get per-genus head AP columns (direct genus head predictions)
genus_head_ap_cols = [col for col in df.columns 
                      if col.startswith('genus_head/average_precision/')]

# Extract genus head names
genus_head_names = [col.replace('genus_head/average_precision/', '') 
                    for col in genus_head_ap_cols]

# Get the latest validation epoch
latest_epoch = val_metrics.iloc[-1]

# Extract per-species AP values for latest epoch (convert to percentages)
species_ap_values = []
species_labels = []
for species_name, col in zip(species_names, species_ap_cols):
    value = latest_epoch[col]
    if not pd.isna(value):
        species_ap_values.append(value * 100)  # Convert to percentage
        species_labels.append(species_name)

# Extract per-genus AP values for latest epoch (convert to percentages)
genus_ap_values = []
genus_labels = []
for genus_name, col in zip(genus_names, genus_ap_cols):
    value = latest_epoch[col]
    if not pd.isna(value):
        genus_ap_values.append(value * 100)  # Convert to percentage
        genus_labels.append(genus_name)

# Extract per-genus head AP values for latest epoch (convert to percentages)
genus_head_ap_values = []
genus_head_labels = []
for genus_name, col in zip(genus_head_names, genus_head_ap_cols):
    value = latest_epoch[col]
    if not pd.isna(value):
        genus_head_ap_values.append(value * 100)  # Convert to percentage
        genus_head_labels.append(genus_name)

# Determine number of subplots
num_plots = 1
if genus_ap_values:
    num_plots += 1
if genus_head_ap_values:
    num_plots += 1

fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6 * num_plots))
if num_plots == 1:
    axes = [axes]

plot_idx = 0

# Plot 1: Species Classification AP
ax = axes[plot_idx]
plot_idx += 1
x_pos = np.arange(len(species_labels))
bars = ax.bar(x_pos, species_ap_values, color='steelblue', alpha=0.8)

# Add value labels on top of bars
for i, (bar, val) in enumerate(zip(bars, species_ap_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Species', fontsize=12)
ax.set_ylabel('Average Precision (%)', fontsize=12)
ax.set_title(f'Per-Species Classification AP (Epoch {int(latest_epoch["epoch"])})', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(species_labels, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(species_ap_values) * 1.15 if species_ap_values else 100)

# Plot 2: Genus Classification AP (if available - derived from species predictions)
if genus_ap_values:
    ax = axes[plot_idx]
    plot_idx += 1
    x_pos = np.arange(len(genus_labels))
    bars = ax.bar(x_pos, genus_ap_values, color='darkgreen', alpha=0.8)
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, genus_ap_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Genus', fontsize=12)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title(f'Per-Genus AP from Species Predictions (Epoch {int(latest_epoch["epoch"])})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genus_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(genus_ap_values) * 1.15 if genus_ap_values else 100)

# Plot 3: Direct Genus Head AP (if available - from genus head output)
if genus_head_ap_values:
    ax = axes[plot_idx]
    plot_idx += 1
    x_pos = np.arange(len(genus_head_labels))
    bars = ax.bar(x_pos, genus_head_ap_values, color='darkorange', alpha=0.8)
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, genus_head_ap_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Genus', fontsize=12)
    ax.set_ylabel('Average Precision (%)', fontsize=12)
    ax.set_title(f'Direct Genus Head AP (Epoch {int(latest_epoch["epoch"])})', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(genus_head_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(genus_head_ap_values) * 1.15 if genus_head_ap_values else 100)

plt.tight_layout()
plt.savefig('outputs/species_classification_ap.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to outputs/species_classification_ap.png")

print("\n" + "="*60)
print(f"VALIDATION METRICS - EPOCH {int(latest_epoch['epoch'])}")
print("="*60)

# Print detection metrics
if 'detection/average_precision' in latest_epoch and not pd.isna(latest_epoch['detection/average_precision']):
    print(f"\nDetection AP: {latest_epoch['detection/average_precision'] * 100:.2f}%")

print(f"\nClassification Mean AP: {latest_epoch['classification/mean_average_precision'] * 100:.2f}%")

print(f"\nPer-Species Classification AP:")
for name, val in zip(species_labels, species_ap_values):
    print(f"  {name}: {val:.2f}%")

if genus_ap_values:
    print(f"\nPer-Genus AP (from Species Predictions):")
    for name, val in zip(genus_labels, genus_ap_values):
        print(f"  {name}: {val:.2f}%")
    if 'classification/genus/mean_average_precision' in latest_epoch and not pd.isna(latest_epoch['classification/genus/mean_average_precision']):
        print(f"  Mean: {latest_epoch['classification/genus/mean_average_precision'] * 100:.2f}%")

if genus_head_ap_values:
    print(f"\nDirect Genus Head AP:")
    for name, val in zip(genus_head_labels, genus_head_ap_values):
        print(f"  {name}: {val:.2f}%")
    if 'genus_head/mean_average_precision' in latest_epoch and not pd.isna(latest_epoch['genus_head/mean_average_precision']):
        print(f"  Mean: {latest_epoch['genus_head/mean_average_precision'] * 100:.2f}%")

print("="*60 + "\n")

plt.show()

