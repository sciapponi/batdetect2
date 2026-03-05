# Voice Activity Detection (VAD) Training

## 2D Spectrogram-based VAD

**Model:** 5,131 parameters, encoder-only with global average pooling  
**Input:** 64-height spectrograms (variable width)  
**Performance:** ~96-98% F1

```bash
uv run python train_vad.py \
  --config example_data/config_vad_tiny_v2.yaml \
  --train-dataset example_data/uk_diff_train.yaml \
  --val-dataset example_data/uk_diff_val.yaml \
  --epochs 50
```

## 1D Raw Audio VAD

**Model:** 46,545 parameters, 1D convolutions on raw waveforms  
**Input:** Raw audio clips at 256kHz  
**Performance:** ~92-96% F1

```bash
# 50ms clips (recommended, fastest)
uv run python train_vad_1d.py \
  --train-dataset example_data/uk_diff_train.yaml \
  --val-dataset example_data/uk_diff_val.yaml \
  --clip-duration 0.05 \
  --channels 32,64,128,256 \
  --batch-size 16 \
  --epochs 5

# 200ms clips (more context)
uv run python train_vad_1d.py \
  --train-dataset example_data/uk_diff_train.yaml \
  --val-dataset example_data/uk_diff_val.yaml \
  --clip-duration 0.2 \
  --channels 32,64,128,256 \
  --batch-size 16 \
  --epochs 5
```

## Key Parameters

**1D Model:**
- `--clip-duration`: Clip length in seconds (0.05, 0.1, 0.2)
- `--channels`: Channel progression (e.g., `32,64,128,256`)
- `--batch-size`: Training batch size
- `--epochs`: Number of training epochs

**2D Model:**
- Uses config file for architecture
- Automatically handles variable-width spectrograms

## Export to ESP32

After training, export models to ESP32-compatible format (int8 quantized):

### 2D Spectrogram VAD

```bash
# 1. Export to ONNX
uv run python export_vad_onnx.py \
  --config example_data/config_vad_tiny_v2.yaml \
  --checkpoint lightning_logs/version_18/checkpoints/epoch=49-step=15650.ckpt \
  --output export/vad_2d.onnx \
  --height 64 \
  --width 100

# 2. Quantize to ESP32 format
uv run python vad_onnx_to_esp.py \
  --model export/vad_2d.onnx \
  --output export/vad_2d.espdl \
  --input-shape 1,1,64,100
```

**Example output:**
- ONNX model: 25.8 KB (float32)
- ESPDL model: 22.7 KB (int8 quantized)
- Compression: 1.1x

### 1D Raw Audio VAD

```bash
# 1. Export to ONNX (50ms clips = 12,800 samples @ 256kHz)
uv run python export_vad_1d_onnx.py \
  --checkpoint lightning_logs/version_13/checkpoints/epoch=4-step=3125.ckpt \
  --output export/vad_1d_50ms.onnx \
  --channels 32,64,128,256 \
  --clip-duration 0.05

# 2. Quantize to ESP32 format
uv run python vad_onnx_to_esp.py \
  --model export/vad_1d_50ms.onnx \
  --output export/vad_1d_50ms.espdl \
  --is-1d \
  --input-shape 1,1,12800
```

**Example output:**
- ONNX model: 182.5 KB (float32)
- ESPDL model: 57.4 KB (int8 quantized)
- Compression: 3.2x

**Note:** Replace checkpoint paths with your actual trained model paths.
