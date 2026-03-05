#!/usr/bin/env python3
"""Convert VAD ONNX models to ESP32 format with quantization."""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from esp_ppq.api import espdl_quantize_onnx
from onnxsim import simplify
from torch.utils.data import DataLoader, TensorDataset


def generate_2d_calibration_data(num_samples=100):
    """Generate calibration data for 2D spectrogram VAD."""
    print("Generating 2D spectrogram calibration data...")
    # Simulate realistic spectrogram statistics
    torch.manual_seed(42)
    # Spectrograms are typically normalized to mean=0, std=1
    x = torch.randn(num_samples, 1, 64, 100, dtype=torch.float32)
    x = torch.clamp(x, -3.0, 3.0)  # Clip outliers
    y = torch.zeros(num_samples, dtype=torch.long)
    
    print(f"Generated {num_samples} spectrogram samples")
    return x, y


def generate_1d_calibration_data(num_samples=100, clip_samples=12800):
    """Generate calibration data for 1D raw audio VAD."""
    print(f"Generating 1D audio calibration data ({clip_samples} samples per clip)...")
    # Simulate realistic audio waveform statistics
    torch.manual_seed(42)
    # Audio waveforms are typically normalized to [-1, 1]
    x = torch.randn(num_samples, 1, clip_samples, dtype=torch.float32) * 0.3
    x = torch.clamp(x, -1.0, 1.0)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    print(f"Generated {num_samples} audio samples")
    return x, y


def collate_fn(batch):
    """Custom collate function for the dataloader."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        return batch[0]
    return batch


def quantize_vad_model(
    onnx_path: Path,
    output_path: Path,
    is_1d: bool = False,
    input_shape: list[int] = None,
    num_samples: int = 100,
    target: str = "esp32s3",
    num_bits: int = 8,
):
    """Quantize VAD ONNX model to ESP32 format.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Path to output .espdl file
        is_1d: Whether this is a 1D raw audio model
        input_shape: Input shape [batch, channels, height/samples, width] or [batch, channels, samples]
        num_samples: Number of calibration samples
        target: Target device (esp32s3)
        num_bits: Quantization bits (8)
    """
    print(f"\n{'='*80}")
    print(f"Quantizing {'1D' if is_1d else '2D'} VAD model for ESP32")
    print(f"{'='*80}\n")
    
    # Load and simplify ONNX model
    print(f"Loading ONNX model from: {onnx_path}")
    model = onnx.load(str(onnx_path))
    
    # Check operations before simplification
    ops_before = [node.op_type for node in model.graph.node]
    print(f"Operations before simplification: {sorted(set(ops_before))}")
    
    # Simplify
    print("Simplifying ONNX model...")
    model_simp, check = simplify(model, skip_fuse_bn=False)
    
    if check:
        print("✓ ONNX model simplified successfully")
        ops_after = [node.op_type for node in model_simp.graph.node]
        print(f"Operations after simplification: {sorted(set(ops_after))}")
        
        # Save simplified model
        simplified_path = str(onnx_path).replace(".onnx", "_simplified.onnx")
        onnx.save(model_simp, simplified_path)
        print(f"Saved simplified model to: {simplified_path}")
        onnx_path = simplified_path
    else:
        print("⚠ Simplification check failed, using original model")
    

    # Optionally load real calibration data from .npz
    real_calib_path = None
    import inspect
    frame = inspect.currentframe()
    if frame is not None and "real_calib_path" in frame.f_back.f_locals:
        real_calib_path = frame.f_back.f_locals["real_calib_path"]
    if real_calib_path is not None:
        print(f"Loading real calibration spectrograms from: {real_calib_path}")
        arr = np.load(real_calib_path)["specs"]
        x = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)  # [N, 1, H, W]
        y = torch.zeros(x.shape[0], dtype=torch.long)
    else:
        if is_1d:
            clip_samples = input_shape[2] if input_shape else 12800
            x, y = generate_1d_calibration_data(num_samples, clip_samples)
        else:
            x, y = generate_2d_calibration_data(num_samples)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nQuantizing model...")
    print(f"  Target: {target}")
    print(f"  Bits: {num_bits}")
    print(f"  Input shape: {input_shape}")
    print(f"  Calibration samples: {num_samples}")
    
    # Quantize
    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=str(onnx_path),
        espdl_export_file=str(output_path),
        calib_dataloader=dataloader,
        calib_steps=min(32, num_samples),
        input_shape=input_shape,
        inputs=None,
        target=target,
        num_of_bits=num_bits,
        collate_fn=collate_fn,
        dispatching_override=None,
        device="cpu",
        error_report=False,
        skip_export=False,
        export_test_values=False,
        verbose=1,
        optimization_level=2,
        graph_optimization=True,
    )
    
    print(f"\n{'='*80}")
    print(f"✓ Quantized model saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Print file sizes
    onnx_size = Path(onnx_path).stat().st_size / 1024
    espdl_size = output_path.stat().st_size / 1024
    print(f"Model sizes:")
    print(f"  ONNX:  {onnx_size:.1f} KB")
    print(f"  ESPDL: {espdl_size:.1f} KB (int8 quantized)")
    print(f"  Compression: {onnx_size/espdl_size:.1f}x")
    
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert VAD ONNX model to ESP32 format with quantization"
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to ONNX model")
    parser.add_argument("--output", type=Path, default=None, help="Output .espdl path (default: auto)")
    parser.add_argument("--is-1d", action="store_true", help="Model is 1D raw audio (not 2D spectrogram)")
    parser.add_argument("--input-shape", type=str, default=None,
                       help="Input shape as comma-separated values (e.g., '1,1,64,100' or '1,1,12800')")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of calibration samples")
    parser.add_argument("--real-calib", type=str, default=None, help="Path to .npz file with real calibration spectrograms")
    parser.add_argument("--target", type=str, default="esp32s3", help="Target device")
    parser.add_argument("--num-bits", type=int, default=8, help="Quantization bits")
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output is None:
        args.output = args.model.parent / args.model.name.replace('.onnx', '.espdl')
    
    # Parse input shape
    if args.input_shape:
        input_shape = [int(x) for x in args.input_shape.split(',')]
    else:
        # Default shapes
        if args.is_1d:
            input_shape = [1, 1, 12800]  # 50ms @ 256kHz
        else:
            input_shape = [1, 1, 64, 100]  # 64 freq bins, 100 time steps
    
    if args.real_calib:
        real_calib_path = args.real_calib
        quantize_vad_model(
            onnx_path=args.model,
            output_path=args.output,
            is_1d=args.is_1d,
            input_shape=input_shape,
            num_samples=args.num_samples,
            target=args.target,
            num_bits=args.num_bits,
            # Pass real_calib_path via frame hack
        )
    else:
        quantize_vad_model(
            onnx_path=args.model,
            output_path=args.output,
            is_1d=args.is_1d,
            input_shape=input_shape,
            num_samples=args.num_samples,
            target=args.target,
            num_bits=args.num_bits,
        )
