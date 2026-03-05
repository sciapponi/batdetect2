#!/usr/bin/env python3
"""Export BatDetect2 model to ONNX format with STATIC shapes for ESP deployment."""

import argparse
from pathlib import Path

import torch
import yaml

from batdetect2.config import validate_config
from batdetect2.models import build_model
from batdetect2.targets import build_targets


def export_model_static_onnx(
    config_path: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
    batch_size: int = 1,
    height: int = 128,
    width: int = 400,
):
    """
    Export BatDetect2 model to ONNX format with STATIC shapes.
    
    Parameters
    ----------
    config_path : Path
        Path to the model configuration YAML file
    output_path : Path
        Path where the ONNX model will be saved
    checkpoint_path : Path | None
        Path to checkpoint file
    batch_size : int
        Batch size for the exported model (default: 1, FIXED)
    height : int
        Input height in pixels (default: 128, FIXED)
    width : int
        Input width in pixels (default: 400, FIXED)
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
    
    # Convert model to channels_last (NHWC) memory format for ESP32-S3 PIE accelerator
    # This should create a natively NHWC ONNX graph without constant NCHW<->NHWC transposes
    print("Converting model to channels_last (NHWC) memory format...")
    detector = detector.to(memory_format=torch.channels_last)
    print("✓ Model converted to channels_last")
    
    # HACK: Pre-expand frequency coordinate tensors to avoid Expand op in ONNX
    # First do a forward pass to capture actual shapes each layer sees
    print("Running forward pass to determine actual tensor shapes at each layer...")
    actual_widths = {}
    
    def make_shape_hook(layer_name):
        def hook(module, input, output):
            if hasattr(module, 'coords'):
                # For encoder layers: use input width (before maxpool in layer)
                # For decoder layers: use output width (after interpolation in layer)
                if 'decoder' in layer_name:
                    # Decoder upsamples, so use output width
                    actual_widths[layer_name] = output.shape[3]
                else:
                    # Encoder, use input width
                    actual_widths[layer_name] = input[0].shape[3]
        return hook
    
    hooks = []
    for name, module in detector.named_modules():
        if hasattr(module, 'coords'):
            hook = module.register_forward_hook(make_shape_hook(name))
            hooks.append(hook)
    
    # Create dummy input and run forward pass
    dummy_input = torch.randn(batch_size, 1, height, width).to(memory_format=torch.channels_last)
    with torch.no_grad():
        _ = detector(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Now pre-expand coords to the actual shapes
    print("Pre-expanding frequency coordinate parameters to avoid ONNX Expand operation...")
    for name, module in detector.named_modules():
        if hasattr(module, 'coords') and isinstance(module.coords, torch.nn.Parameter):
            coords_shape = module.coords.shape
            actual_width = actual_widths.get(name, width)
            # Pre-expand to batch_size=1 and actual width for this layer
            expanded_coords = module.coords.repeat(batch_size, 1, 1, actual_width)
            # Replace parameter with pre-expanded constant
            module.coords = torch.nn.Parameter(expanded_coords, requires_grad=False)
            print(f"  {name}.coords: {coords_shape} -> {expanded_coords.shape}")
    
    # Print model info
    num_params = sum(p.numel() for p in detector.parameters())
    print(f"Model built successfully!")
    print(f"Total parameters: {num_params:,}")
    print(f"Number of classes: {len(targets.class_names)}")
    
    # Create dummy input with FIXED dimensions in channels_last format
    dummy_input = torch.randn(batch_size, 1, height, width).to(memory_format=torch.channels_last)
    print(f"\nDummy input shape (FIXED, channels_last): {dummy_input.shape}")
    
    # Test forward pass
    print("Testing forward pass with PyTorch...")
    with torch.no_grad():
        output = detector(dummy_input)
    
    print("Forward pass successful!")
    print(f"  detection_probs: {output.detection_probs.shape}")
    print(f"  size_preds: {output.size_preds.shape}")
    print(f"  class_probs: {output.class_probs.shape}")
    print(f"  features: {output.features.shape}")
    
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
    
    # Export to ONNX with STATIC shapes (NO dynamic_axes!)
    print(f"\nExporting model to ONNX with STATIC shapes: {output_path}")
    
    output_names = ['detection_probs', 'size_preds', 'class_probs', 'features']
    
    with torch.onnx.select_model_mode_for_export(wrapped_model, torch.onnx.TrainingMode.EVAL):
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,  # Use opset 13 for better ESP compatibility
            do_constant_folding=True,
            input_names=['spectrogram'],
            output_names=output_names,
            # NO dynamic_axes - all shapes are FIXED!
            dynamo=False,
        )
    print(f"✓ Model exported with STATIC shapes to {output_path}")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export BatDetect2 model to ONNX with static shapes for ESP"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path for output ONNX file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Fixed batch size (default: 1)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="Fixed input height (default: 128)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=400,
        help="Fixed input width (default: 400)",
    )
    
    args = parser.parse_args()
    
    export_model_static_onnx(
        config_path=args.config,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
    )
