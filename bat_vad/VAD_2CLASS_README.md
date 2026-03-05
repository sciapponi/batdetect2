# 2-Class VAD Training (Bat vs Non-Bat)

This directory contains scripts for training and evaluating a 2-class Voice Activity Detection model that distinguishes between bat calls and non-bat sounds.

## Overview

**Why 2-class instead of single output?**
- The "non-bat" class is very diverse (rodents, insects, environmental sounds)
- Explicit modeling of both classes helps learn better discriminative features
- Allows for class-specific regularization and easier extension to multi-class

## Files

- `train_vad_2class.py` - Training script for 2-class VAD with AUC-ROC and Average Precision metrics
- `eval_vad_on_non_bat.py` - Evaluation script to test on non-bat sounds
- `configs/config_vad_2class.yaml` - Model configuration (2 output classes)
- `configs/non_bat_dataset.yaml` - Non-bat dataset configuration

## Training

### Basic Training Command

```bash
cd bat_vad
uv run python train_vad_2class.py \
  --config configs/config_vad_2class.yaml \
  --bat-train ../example_data/brazil_train.yaml \
  --bat-val ../example_data/brazil_val.yaml \
  --non-bat configs/non_bat_dataset.yaml \
  --epochs 20 \
  --lr 1e-3 \
  --label-smoothing 0.1 \
  --save-path vad_2class_model.pt \
  --csv-log training_metrics.csv
```

```bash
  cd bat_vad
  uv run python train_vad_2class.py \
    --config configs/config_vad_2class.yaml \
    --bat-train ../example_data/uk_diff_train.yaml \
    --bat-val ../example_data/uk_diff_val.yaml \
    --non-bat configs/non_bat_dataset.yaml \
    --clip-duration 0.5 \
    --epochs 20
```
### Parameters

- `--config`: Model architecture config (use `configs/config_vad_2class.yaml` for 2-class)
- `--bat-train/--bat-val`: Dataset configs for bat calls (positive examples)
- `--non-bat`: Dataset config for non-bat sounds (negative examples, optional)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 1e-3)
- `--label-smoothing`: Label smoothing for regularization (default: 0.1)
- `--save-path`: Where to save the trained model
- `--csv-log`: Path to save training metrics as CSV (default: auto-generated from save-path)
- `--clip-duration`: Duration of each training clip in seconds (default: 0.2)
- `--samples-per-epoch`: Number of training samples per epoch (default: 10000)
- `--val-samples`: Number of validation samples per epoch (default: 1000)
- `--positive-ratio`: Ratio of positive (bat) samples, 0.5 = balanced (default: 0.5)

### Training Without Non-Bat Data

If you don't have dedicated non-bat recordings, the script will use regions from bat recordings that don't contain bat calls as negative samples:

```bash
uv run python train_vad_2class.py \
  --config configs/config_vad_2class.yaml \
  --bat-train ../example_data/brazil_train.yaml \
  --bat-val ../example_data/brazil_val.yaml \
  --epochs 20
```

### Changing Clip Duration

To train with longer or shorter clips, use the `--clip-duration` parameter:

```bash
uv run python train_vad_2class.py \
  --config configs/config_vad_2class.yaml \
  --bat-train ../example_data/brazil_train.yaml \
  --bat-val ../example_data/brazil_val.yaml \
  --non-bat configs/non_bat_dataset.yaml \
  --clip-duration 0.5 \
  --epochs 20
```

**Note:** Longer clips (e.g., 0.5s) can capture more context but may include multiple calls or background noise. Shorter clips (e.g., 0.1s) focus on individual calls but may miss context. The default 0.2s works well for most bat species.

## Evaluation on Non-Bat Sounds

After training, test the model on non-bat sounds to measure false positive rate:

```bash
uv run python eval_vad_on_non_bat.py \
  --model vad_2class_model.pt \
  --non-bat-dir /path/to/non_bat \
  --clip-duration 0.2 \
  --stride 0.1
```

This will:
- Process all `.wav` files in the directory using sliding windows
- Report overall false positive rate (windows incorrectly classified as bat)
- Show which files have the most false positives
- Display probability statistics

### Evaluation Parameters

- `--model`: Path to trained model file
- `--non-bat-dir`: Directory containing non-bat audio files
- `--clip-duration`: Window duration in seconds (default: 0.2)
- `--stride`: Stride between windows in seconds (default: 0.1)
- `--device`: cpu or cuda (default: cpu)

## Model Architecture

The 2-class VAD model uses:
- Depthwise-separable convolutions for efficiency
- 5 downsampling layers (8→16→32→32→64 channels)
- Global average pooling
- 2-class softmax output (non-bat, bat)

Input: Spectrogram (64 frequency bins, variable time)
Output: 2-class probabilities [P(non-bat), P(bat)]

## Metrics

During training, the following metrics are logged (both to console and CSV):

**Training metrics:**
- Loss, Accuracy
- Bat class accuracy, Non-bat class accuracy

**Validation metrics:**
- Overall accuracy: (TP + TN) / Total
- **AUC-ROC**: Area Under the ROC Curve (discrimination ability)
- **Average Precision**: Area under precision-recall curve
- Bat class: Precision, Recall, F1, Accuracy
- Non-bat class: Specificity, Accuracy
- Confusion matrix: TP, TN, FP, FN

The CSV file contains all metrics for each epoch, making it easy to analyze training progress and create plots.

## CSV Output Format

The CSV file contains the following columns:
```
epoch, train_loss, train_acc, train_bat_acc, train_non_bat_acc,
val_loss, val_acc, val_precision, val_recall, val_f1,
val_specificity, val_fpr, val_bat_acc, val_non_bat_acc,
val_auc_roc, val_avg_precision,
val_tp, val_tn, val_fp, val_fn
```

## Tips

1. **Balance your data**: Use `--positive-ratio 0.5` to balance bat/non-bat samples
2. **Label smoothing**: Helps prevent overconfidence, recommended value: 0.1
3. **Learning rate**: Start with 1e-3, reduce if training is unstable
4. **Epochs**: 20-50 epochs should be sufficient
5. **False positives**: If FP rate is high on non-bat data, consider:
   - Training longer
   - Adding more diverse non-bat examples
   - Increasing label smoothing
   - Adjusting the classification threshold at inference
6. **Monitor AUC-ROC**: Values > 0.95 indicate excellent discrimination between bat and non-bat

## Example Non-Bat Sounds

Your non-bat dataset should contain:
- Rodent calls (mice, voles, rats, dormice)
- Shrew trills
- Bush-cricket stridulations
- Environmental noises (wind, rain, footsteps, vehicles)
- Electronic interference
- Bird calls

These are common sources of false positives in bat acoustic monitoring.

## Example Results

On a test with 234 non-bat audio files:
- **False positive rate: 4.17%**
- **Clean files: 90.2%** (no false positives)
- **AUC-ROC: 0.95+** (excellent discrimination)
- Main confusers: bush-crickets, fox calls, vehicle noises
