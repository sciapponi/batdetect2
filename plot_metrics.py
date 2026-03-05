import pandas as pd
import matplotlib.pyplot as plt

# Read metrics
df = pd.read_csv('outputs/logs/version_92/metrics.csv')

# Get validation metrics (rows with validation data)
val_metrics = df[df['total_loss/val'].notna()].copy()

# Check if VQ metrics are available
has_vq = any(col.startswith('vq/') and 'perplexity/val' in col and 'perplexity_loss' not in col 
             for col in df.columns)

# Determine subplot layout
n_plots = 4 if has_vq else 3
fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3*n_plots))

# Total validation loss
# Drop NaN values for cleaner plotting
total_loss_data = val_metrics[['epoch', 'total_loss/val']].dropna()
axes[0].plot(total_loss_data['epoch'], total_loss_data['total_loss/val'], 'b-', linewidth=2)
axes[0].set_ylabel('Total Loss (Val)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Detection AP
detection_ap_data = val_metrics[['epoch', 'detection/average_precision']].dropna()
axes[1].plot(detection_ap_data['epoch'], detection_ap_data['detection/average_precision'], 'g-', linewidth=2)
axes[1].set_ylabel('Detection AP', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Classification mAP
classification_map_data = val_metrics[['epoch', 'classification/mean_average_precision']].dropna()
axes[2].plot(classification_map_data['epoch'], classification_map_data['classification/mean_average_precision'], 'r-', linewidth=2)
axes[2].set_ylabel('Classification mAP', fontsize=12)
if not has_vq:
    axes[2].set_xlabel('Epoch', fontsize=12)
axes[2].grid(True, alpha=0.3)

# VQ Perplexity if available
if has_vq:
    # Find perplexity columns (exclude perplexity_loss)
    perplexity_cols = [col for col in df.columns 
                       if 'perplexity/val' in col and 'perplexity_loss' not in col]
    
    for col in perplexity_cols:
        # Extract VQ type from column name
        vq_type = col.split('_perplexity')[0].replace('vq/', '')
        # Drop NaN values for this specific column
        perplexity_data = val_metrics[['epoch', col]].dropna()
        axes[3].plot(perplexity_data['epoch'], perplexity_data[col], 
                    linewidth=2, label=vq_type)
    
    axes[3].set_ylabel('Codebook Perplexity (Val)', fontsize=12)
    axes[3].set_xlabel('Epoch', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    if len(perplexity_cols) > 1:
        axes[3].legend()

plt.tight_layout()
plt.savefig('outputs/validation_metrics.png', dpi=150, bbox_inches='tight')
print("Plot saved to outputs/validation_metrics.png")
plt.show()
