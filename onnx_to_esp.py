import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from esp_ppq.api import espdl_quantize_onnx
from pathlib import Path
import librosa
import onnx
from onnxsim import simplify


def generate_data():
    """Generate calibration data from real audio files."""
    # Use actual audio files for calibration if available
    audio_dir = Path("example_data/audio")
    audio_files = list(audio_dir.glob("*.wav"))[:10]  # Use first 10 files
    
    if not audio_files:
        print("WARNING: No audio files found, using synthetic data")
        return generate_synthetic_data()
    
    print(f"Loading {len(audio_files)} audio files for calibration...")
    spectrograms = []
    target_samples = 100
    
    for audio_file in audio_files:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=256000, mono=True)
        
        # Generate multiple spectrograms from each audio file
        # Split into chunks with smaller overlap to get more samples
        chunk_duration = 0.4  # seconds
        chunk_samples = int(chunk_duration * sr)
        
        # Use smaller steps for more samples
        step_size = chunk_samples // 4  # 75% overlap for more variation
        
        for i in range(0, len(audio) - chunk_samples + 1, step_size):
            chunk = audio[i:i + chunk_samples]
            
            # Compute spectrogram
            n_fft = 512
            hop_length = 128
            spec = librosa.stft(chunk, n_fft=n_fft, hop_length=hop_length)
            spec = np.abs(spec)
            
            # Take 128 frequency bins and exactly 400 time steps
            if spec.shape[0] >= 128 and spec.shape[1] >= 400:
                spec = spec[:128, :400]
                # Normalize
                mean = spec.mean()
                std = spec.std()
                if std > 1e-8:
                    spec = (spec - mean) / std
                spec = np.clip(spec, -3, 3)
                spectrograms.append(spec)
            
            if len(spectrograms) >= target_samples:
                break
        
        if len(spectrograms) >= target_samples:
            break
    
    # If we don't have enough samples, repeat what we have
    if len(spectrograms) < target_samples:
        print(f"Only generated {len(spectrograms)} spectrograms, repeating to reach {target_samples}")
        original_count = len(spectrograms)
        while len(spectrograms) < target_samples:
            spectrograms.append(spectrograms[len(spectrograms) % original_count])
    
    # Convert to tensor
    x = torch.from_numpy(np.array(spectrograms)).float().unsqueeze(1)
    y = torch.zeros(len(spectrograms), dtype=torch.long)
    
    print(f"Generated {len(spectrograms)} spectrograms from real audio")
    return x, y


def generate_synthetic_data():
    """Fallback: Generate synthetic calibration data."""
    num_samples = 100
    torch.manual_seed(42)
    x = torch.randn(num_samples, 1, 128, 400, dtype=torch.float32) * 0.3
    x = torch.clamp(x, -2.0, 2.0)
    y = torch.zeros(num_samples, dtype=torch.long)
    return x, y


def collate_fn(batch):
    """Custom collate function for the dataloader."""
    # batch is already a tuple of (inputs, labels) tensors after default collation
    # We only need to return the inputs
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        return batch[0]
    return batch


if __name__ == '__main__':
    # Use ESP32-optimized ONNX model with depthwise-separable convolutions
    ONNX_MODEL_PATH = "/home/stefanoc/batdetect2/export/config_esp32_depthwise.onnx"
    ESPDL_MODEL_PATH = "/home/stefanoc/batdetect2/export/esp32_depthwise.espdl"
    INPUT_SHAPE = [1, 1, 128, 400]
    TARGET = "esp32s3"
    NUM_OF_BITS = 8
    DEVICE = "cpu"

    # Simplify ONNX model aggressively to remove unsupported ops (Shape, Max, etc.)
    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    model = onnx.load(ONNX_MODEL_PATH)
    
    # Check what ops are in the model
    ops_before = [node.op_type for node in model.graph.node]
    print(f"Operations before simplification: {sorted(set(ops_before))}")
    
    # Enable simplification for graph optimization
    # The graph_optimization=True flag in espdl_quantize_onnx will handle layout conversion
    print("Simplifying ONNX model to optimize graph structure...")
    model_simp, check = simplify(model, skip_fuse_bn=False)
    
    if check:
        print("✓ ONNX model simplified successfully")
        ops_after = [node.op_type for node in model_simp.graph.node]
        print(f"Operations after simplification: {sorted(set(ops_after))}")
        
        # Save simplified model
        simplified_path = ONNX_MODEL_PATH.replace(".onnx", "_simplified.onnx")
        onnx.save(model_simp, simplified_path)
        print(f"Saved simplified model to: {simplified_path}")
        ONNX_MODEL_PATH = simplified_path
    else:
        print("⚠ Simplification check failed, using original model")

    x, y = generate_data()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # Use batch size 1 to avoid numerical issues

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=ONNX_MODEL_PATH,
        espdl_export_file=ESPDL_MODEL_PATH,
        calib_dataloader=dataloader,
        calib_steps=32,  # 校准的步数
        input_shape=INPUT_SHAPE,  # 输入形状，批次为 1
        inputs=None,
        target=TARGET,  # 量化目标类型
        num_of_bits=NUM_OF_BITS,  # 量化位数
        collate_fn=collate_fn,
        dispatching_override=None,
        device=DEVICE,
        error_report=False,  # Disable error analysis to avoid empty tensor issue
        skip_export=False,
        export_test_values=False,  # Disable test values to avoid empty tensor issue
        verbose=1,  # 输出详细日志信息
        # Enable graph optimization for ESP32-S3 PIE accelerator (NHWC layout)
        optimization_level=2,  # Level 2: Maximum optimization (fuses transposes, optimizes layout)
        graph_optimization=True,  # Vital for removing redundant Transposes and NCHW->NHWC conversion
    )