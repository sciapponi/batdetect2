"""Test ONNX model inference with dummy input.

Note: The quantized .espdl model can only be tested on ESP32 hardware.
This script validates the FP32 ONNX model that was used to create the .espdl file.
"""

import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Paths
ONNX_PATH = "export/config_half_the_size_broadcast.onnx"
ESPDL_PATH = "export/half_the_size_broadcast.espdl"

def create_dummy_input():
    """Create a dummy spectrogram input."""
    # Shape: (1, 1, 128, 400)
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 1, 128, 400, dtype=torch.float32) * 0.5
    dummy_input = torch.clamp(dummy_input, -3, 3)
    return dummy_input

def test_original_onnx(input_tensor):
    """Test the original float32 ONNX model."""
    print("=" * 60)
    print("Testing original ONNX model (FP32)")
    print("=" * 60)
    
    session = ort.InferenceSession(ONNX_PATH)
    input_name = session.get_inputs()[0].name
    
    input_np = input_tensor.numpy()
    outputs = session.run(None, {input_name: input_np})
    
    print(f"Input shape: {input_np.shape}")
    print(f"Number of outputs: {len(outputs)}")
    
    output_names = ["detection_probs", "size_preds", "class_probs", "features"]
    for i, (name, output) in enumerate(zip(output_names, outputs)):
        print(f"\n{name}:")
        print(f"  Shape: {output.shape}")
        print(f"  Range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"  Mean: {output.mean():.4f}, Std: {output.std():.4f}")
    
    return outputs

def test_quantized_info():
    """Display info about the quantized model files."""
    print("\n" + "=" * 60)
    print("Quantized Model Files (INT8 - for ESP32)")
    print("=" * 60)
    
    if not Path(ESPDL_PATH).exists():
        print(f"\n✗ ESPDL model not found: {ESPDL_PATH}")
        print("  Run: uv run python3 onnx_to_esp.py")
        return False
    
    espdl_size = Path(ESPDL_PATH).stat().st_size / 1024
    json_path = ESPDL_PATH.replace('.espdl', '.json')
    info_path = ESPDL_PATH.replace('.espdl', '.info')
    
    print(f"\n✓ Quantized model files ready for ESP32:")
    print(f"  - {Path(ESPDL_PATH).name} ({espdl_size:.1f} KB)")
    print(f"    └─ INT8 quantized weights & graph")
    
    if Path(json_path).exists():
        json_size = Path(json_path).stat().st_size / 1024
        print(f"  - {Path(json_path).name} ({json_size:.1f} KB)")
        print(f"    └─ Quantization config (scales, zero-points)")
    
    if Path(info_path).exists():
        info_size = Path(info_path).stat().st_size / 1024
        print(f"  - {Path(info_path).name} ({info_size:.0f} KB)")
        print(f"    └─ Model structure & debug info")
    
    # Calculate compression ratio
    if Path(ONNX_PATH).exists():
        onnx_size = Path(ONNX_PATH).stat().st_size / 1024
        compression = onnx_size / espdl_size
        print(f"\n📊 Compression: {onnx_size:.1f} KB (FP32) → {espdl_size:.1f} KB (INT8)")
        print(f"   Ratio: {compression:.2f}x smaller")
    
    print("\nℹ️  To test the quantized model:")
    print("   • Deploy to ESP32-S3 hardware")
    print("   • Use ESP-DL runtime APIs")
    print("   • Expected ~1-3% accuracy loss vs FP32")
    
    return True

def compare_outputs(original, quantized):
    """This would compare outputs if we could run the quantized model."""
    # Note: Can't run .espdl files in Python - they're for ESP32 hardware only
    pass

def main():
    print("ONNX Model Inference Test (FP32)")
    print("=" * 60)
    
    # Check files exist
    if not Path(ONNX_PATH).exists():
        print(f"ERROR: ONNX model not found at {ONNX_PATH}")
        return
    
    # Create dummy input
    print("\nGenerating dummy input...")
    dummy_input = create_dummy_input()
    print(f"Input shape: {dummy_input.shape}")
    print(f"Input range: [{dummy_input.min():.4f}, {dummy_input.max():.4f}]")
    
    # Test original ONNX
    original_outputs = test_original_onnx(dummy_input)
    
    # Show quantized model info
    has_espdl = test_quantized_info()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print("\n✓ FP32 ONNX model tested successfully")
    if has_espdl:
        print("✓ INT8 ESPDL model ready for ESP32 deployment")
        print("\nNext step: Deploy to ESP32-S3 and test on-device")
    else:
        print("\n⚠ Generate ESPDL model first: uv run python3 onnx_to_esp.py")

if __name__ == "__main__":
    main()
