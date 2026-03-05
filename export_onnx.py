#!/usr/bin/env python3
"""Export BatDetect2 model to ONNX format and test with ONNX Runtime."""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from batdetect2.config import validate_config
from batdetect2.models import build_model
from batdetect2.targets import build_targets


def export_model_to_onnx(
    config_path: Path,
    output_path: Path,
    batch_size: int = 1,
    height: int = 128,
    width: int = 400,
    checkpoint_path: Path | None = None,
):
    """
    Export BatDetect2 model to ONNX format.
    
    Parameters
    ----------
    config_path : Path
        Path to the model configuration YAML file
    output_path : Path
        Path where the ONNX model will be saved
    batch_size : int
        Batch size for the exported model (default: 1)
    height : int
        Input height in pixels (default: 128)
    width : int
        Input width in pixels (default: 400)
    """
    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = validate_config(config_dict)
    
    # Create export directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build targets and model
    print("Building model...")
    targets = build_targets(config=config.targets)
    model = build_model(
        config=config.model,
        targets=targets,
    )
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Lightning checkpoints store model weights under 'state_dict' key
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix from Lightning checkpoint keys
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✓ Checkpoint loaded successfully")
    else:
        print("WARNING: No checkpoint provided - using untrained model!")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get the detector (the core neural network)
    detector = model.detector
    
    # Print model info
    num_params = sum(p.numel() for p in detector.parameters())
    print(f"Model built successfully!")
    print(f"Total parameters: {num_params:,}")
    print(f"Number of classes: {len(targets.class_names)}")
    print(f"Class names: {targets.class_names}")
    
    # Create dummy input with dynamic batch and width dimensions
    dummy_input = torch.randn(batch_size, 1, height, width)
    print(f"\nDummy input shape: {dummy_input.shape}")
    
    # Test forward pass
    print("Testing forward pass with PyTorch...")
    with torch.no_grad():
        output = detector(dummy_input)
    
    print("Forward pass successful!")
    print(f"Output type: {type(output).__name__}")
    print(f"Output fields: {output._fields if hasattr(output, '_fields') else 'N/A'}")
    print(f"  detection_probs: {output.detection_probs.shape}")
    print(f"  size_preds: {output.size_preds.shape}")
    print(f"  class_probs: {output.class_probs.shape}")
    print(f"  features: {output.features.shape}")
    if output.genus_probs is not None:
        print(f"  genus_probs: {output.genus_probs.shape}")
    
    # Wrap the model to drop genus_probs from output
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            out = self.model(x)
            # Only return the 4 main outputs, drop genus_probs
            return out.detection_probs, out.size_preds, out.class_probs, out.features
    
    wrapped_model = OnnxWrapper(detector)
    wrapped_model.eval()
    
    # Export to ONNX
    print(f"\nExporting model to ONNX: {output_path}")
    
    # ONNX export with 4 outputs, genus_probs dropped
    output_names = ['detection_probs', 'size_preds', 'class_probs', 'features']
    
    # Use legacy exporter to avoid external data files
    with torch.onnx.select_model_mode_for_export(wrapped_model, torch.onnx.TrainingMode.EVAL):
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['spectrogram'],
            output_names=output_names,
            dynamic_axes={
                'spectrogram': {0: 'batch_size', 3: 'time'},
                'detection_probs': {0: 'batch_size', 3: 'time'},
                'size_preds': {0: 'batch_size', 3: 'time'},
                'class_probs': {0: 'batch_size', 3: 'time'},
                'features': {0: 'batch_size', 3: 'time'},
            },
            dynamo=False,  # Use legacy exporter
        )
    print(f"✓ Model exported successfully to {output_path}")
    
    return output_path


def test_onnx_inference(onnx_path: Path, batch_size: int = 1, height: int = 128, width: int = 400):
    """
    Test inference with ONNX Runtime.
    
    Parameters
    ----------
    onnx_path : Path
        Path to the ONNX model file
    batch_size : int
        Batch size for inference (default: 1)
    height : int
        Input height in pixels (default: 128)
    width : int
        Input width in pixels (default: 400)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: onnxruntime is not installed!")
        print("Please install it with: pip install onnxruntime")
        print("="*60)
        return
    
    print(f"\nLoading ONNX model from: {onnx_path}")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(str(onnx_path))
    
    # Get model input/output info
    print("\nModel input information:")
    for input_meta in session.get_inputs():
        print(f"  Name: {input_meta.name}")
        print(f"  Shape: {input_meta.shape}")
        print(f"  Type: {input_meta.type}")
    
    print("\nModel output information:")
    for output_meta in session.get_outputs():
        print(f"  Name: {output_meta.name}")
        print(f"  Shape: {output_meta.shape}")
        print(f"  Type: {output_meta.type}")
    
    # Create random input data
    input_data = np.random.randn(batch_size, 1, height, width).astype(np.float32)
    print(f"\nTest input shape: {input_data.shape}")
    
    # Run inference
    print("Running ONNX inference...")
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    
    print("✓ ONNX inference successful!")
    print(f"\nOutput shapes:")
    for i, output in enumerate(outputs):
        output_name = session.get_outputs()[i].name
        print(f"  {output_name}: {output.shape}")
        print(f"    Min: {output.min():.4f}, Max: {output.max():.4f}, Mean: {output.mean():.4f}")
    
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Export BatDetect2 model to ONNX and test with ONNX Runtime"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("example_data/config.yaml"),
        help="Path to model configuration YAML file (default: example_data/config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("export/batdetect2_model.onnx"),
        help="Output path for ONNX model (default: export/batdetect2_model.onnx)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for the model (default: 1)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Input height in pixels (default: 128)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=400,
        help="Input width in time steps (default: 400)"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip ONNX Runtime inference test"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file to load trained weights (default: None - untrained model)"
    )
    
    args = parser.parse_args()
    
    # Export model
    onnx_path = export_model_to_onnx(
        config_path=args.config,
        output_path=args.output,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        checkpoint_path=args.checkpoint,
    )
    
    # Test with ONNX Runtime
    if not args.skip_test:
        test_onnx_inference(
            onnx_path=onnx_path,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
        )
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
