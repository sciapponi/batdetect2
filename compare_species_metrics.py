import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read metrics from both versions
df_75 = pd.read_csv('outputs/logs/version_75/metrics.csv')
df_77 = pd.read_csv('outputs/logs/version_77/metrics.csv')

# Get validation metrics (rows with validation data)
val_metrics_75 = df_75[df_75['total_loss/val'].notna()].copy()
val_metrics_77 = df_77[df_77['total_loss/val'].notna()].copy()

# Get per-species classification AP columns
species_ap_cols = [col for col in df_75.columns 
                   if col.startswith('classification/average_precision/')]

# Extract species names
species_names = [col.replace('classification/average_precision/', '') 
                 for col in species_ap_cols]

# Get the latest validation epoch from each version
latest_epoch_75 = val_metrics_75.iloc[-1]
latest_epoch_77 = val_metrics_77.iloc[-1]

# Extract per-species AP values for both versions (convert to percentages)
species_ap_75 = []
species_ap_77 = []
species_labels = []

for species_name, col in zip(species_names, species_ap_cols):
    value_75 = latest_epoch_75[col]
    value_77 = latest_epoch_77[col]
    
    # Only include if both versions have the value
    if not pd.isna(value_75) and not pd.isna(value_77):
        species_ap_75.append(value_75 * 100)  # Convert to percentage
        species_ap_77.append(value_77 * 100)  # Convert to percentage
        species_labels.append(species_name)

# Create comparison plot
fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(species_labels))
width = 0.35

# Create bars (v75 first as it's the trained reference)
bars1 = ax.bar(x_pos - width/2, species_ap_75, width,
               label=f'Version 75 - Half Size [trained] (Epoch {int(latest_epoch_75["epoch"])})',
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, species_ap_77, width, 
               label=f'Version 77 - Full Size w/ Skip+Attn [training] (Epoch {int(latest_epoch_77["epoch"])})',
               color='coral', alpha=0.8)

# Add value labels on top of bars
for bar, val in zip(bars1, species_ap_75):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=8)

for bar, val in zip(bars2, species_ap_77):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Species', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Precision (%)', fontsize=12, fontweight='bold')
ax.set_title('Species Classification AP: Half-Size (v75) vs Full-Size w/ Skip+Attn (v77)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(species_labels, rotation=45, ha='right')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(max(species_ap_75), max(species_ap_77)) * 1.15 if species_ap_75 else 100)

plt.tight_layout()
plt.savefig('outputs/species_comparison_75_vs_77.png', dpi=150, bbox_inches='tight')
print(f"Plot saved to outputs/species_comparison_75_vs_77.png")

# Print comparison statistics
print("\n" + "="*80)
print("SPECIES CLASSIFICATION AP: HALF-SIZE (v75) vs FULL-SIZE (v77)")
print("="*80)

# Get overall metrics
mean_ap_75 = latest_epoch_75['classification/mean_average_precision'] * 100
mean_ap_77 = latest_epoch_77['classification/mean_average_precision'] * 100

print(f"\nOverall Mean AP:")
print(f"  Version 75 - Half Size [trained] (Epoch {int(latest_epoch_75['epoch'])}): {mean_ap_75:.2f}%")
print(f"  Version 77 - Full Size w/ Skip+Attn [training] (Epoch {int(latest_epoch_77['epoch'])}): {mean_ap_77:.2f}%")
print(f"  Gap (v77 - v75): {mean_ap_77 - mean_ap_75:+.2f}%")

print(f"\nPer-Species Comparison:")
print(f"{'Species':<15} {'V75 Half-Size':>14} {'V77 Full-Size':>14} {'Gap (v77-v75)':>16}")
print("-" * 80)

differences = []
for name, val_75, val_77 in zip(species_labels, species_ap_75, species_ap_77):
    diff = val_77 - val_75  # Gap: how much worse/better v77 (full-size) is compared to v75 (half-size)
    differences.append(diff)
    print(f"{name:<15} {val_75:>13.2f}% {val_77:>13.2f}% {diff:>+15.2f}%")

print("-" * 80)
print(f"{'Mean':<15} {np.mean(species_ap_75):>13.2f}% {np.mean(species_ap_77):>13.2f}% {np.mean(differences):>+15.2f}%")

# Count improvements/degradations (v77 full-size relative to v75 half-size)
better = sum(1 for d in differences if d > 0)
worse = sum(1 for d in differences if d < 0)
same = sum(1 for d in differences if d == 0)

print(f"\nSummary (Full-Size v77 vs Half-Size v75):")
print(f"  Species where full-size is better: {better}/{len(species_labels)}")
print(f"  Species where full-size is worse: {worse}/{len(species_labels)}")
print(f"  Species where full-size is same: {same}/{len(species_labels)}")

print("="*80 + "\n")

plt.show()
