import torch
import onnxruntime as ort
import numpy as np
from pathlib import Path
import librosa

# Generate one real spectrogram
audio_dir = Path("example_data/audio")
audio_files = list(audio_dir.glob("*.wav"))

if audio_files:
    audio, sr = librosa.load(audio_files[0], sr=256000, mono=True)
    chunk_samples = int(0.4 * sr)
    chunk = audio[:chunk_samples]
    
    spec = librosa.stft(chunk, n_fft=512, hop_length=128)
    spec = np.abs(spec)[:128, :400]
    
    mean = spec.mean()
    std = spec.std()
    if std > 1e-8:
        spec = (spec - mean) / std
    spec = np.clip(spec, -3, 3)
    
    input_data = spec.astype(np.float32).reshape(1, 1, 128, 400)
    
    print(f"Input stats: min={input_data.min():.4f}, max={input_data.max():.4f}, mean={input_data.mean():.4f}, std={input_data.std():.4f}")
    print(f"Has NaN: {np.isnan(input_data).any()}")
    print(f"Has Inf: {np.isinf(input_data).any()}")
    
    # Run ONNX inference
    session = ort.InferenceSession("export/config_half_the_size_broadcast.onnx")
    outputs = session.run(None, {"spectrogram": input_data})
    
    print("\nONNX Output stats:")
    for i, output in enumerate(outputs):
        print(f"Output {i}: shape={output.shape}, min={output.min():.4f}, max={output.max():.4f}, has_nan={np.isnan(output).any()}")
