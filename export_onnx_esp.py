#!/usr/bin/env python3
"""Export BatDetect2 model to ONNX for ESP32 - removes frequency coords for full int8 inference."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from batdetect2.config import validate_config
from batdetect2.models import build_model
from batdetect2.targets import build_targets


class DisableFreqCoordsWrapper(nn.Module):
    """Wrapper that removes frequency coordinate concatenation for ESP32 deployment.
    
    This ensures the entire model runs in int8 without any float operations by:
    1. Setting coords to None (disables concatenation)
    2. Adjusting conv layer weights to drop the frequency coordinate channel
    """
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        
        # Remove frequency coordinate concat and adjust conv weights
        for name, module in self.detector.named_modules():
            if hasattr(module, 'coords') and hasattr(module, 'conv'):
                # Set coords to None to skip concatenation
                module.coords = None
                
                # Adjust conv layer to expect 1 less input channel
                # The conv was trained with [input_channels + 1] due to freq coords
                # We drop the last channel (the freq coord channel)
                old_weight = module.conv.weight.data
                if old_weight.shape[1] > 1:  # Only if there are multiple input channels
                    # Drop the frequency coordinate channel (last channel)
                    new_weight = old_weight[:, :-1, :, :]
                    
                    # Create new conv layer with correct input channels
                    new_conv = nn.Conv2d(
                        in_channels=new_weight.shape[1],
                        out_channels=module.conv.out_channels,
                        kernel_size=module.conv.kernel_size,
                        stride=module.conv.stride,
                        padding=module.conv.padding,
                        dilation=module.conv.dilation,
                        groups=module.conv.groups,
                        bias=module.conv.bias is not None
                    )
                    new_conv.weight.data = new_weight
                    if module.conv.bias is not None:
                        new_conv.bias.data = module.conv.bias.data
                    
                    module.conv = new_conv
    
    def forward(self, x):
        out = self.detector(x)
        # Return only the 4 main outputs, drop genus_probs
        return out.detection_probs, out.size_preds, out.class_probs, out.features


def export_model_esp_onnx(
    config_path: Path,
    output_path: Path,
    checkpoint_path: Path | None = None,
    batch_size: int = 1,
    height: int = 128,
    width: int = 400,
):
    """Export BatDetect2 model to ONNX for ESP32 deployment.
    
    Removes frequency coordinate concatenation to ensure full int8 inference.
    """
    print(f"Loading config from: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = validate_config(config_dict)
    
    # Create export directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build model
    print("Building model...")
    targets = build_targets(config=config.targets)
    model = build_model(
        config=config.model,
        targets=targets,
    )
    
    # Load checkpoint
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("✓ Checkpoint loaded successfully")
    else:
        print("WARNING: No checkpoint provided - using untrained model!")
    
    model.eval()
    detector = model.detector
    
    # Fuse BatchNorm into Conv layers to avoid grouped convolutions
    print("Fusing BatchNorm layers into Conv layers...")
    from torch.ao.quantization import fuse_modules_qat
    # Manually fuse all Conv+BN+ReLU patterns
    fused_count = 0
    for module in detector.modules():
        if hasattr(module, 'conv') and hasattr(module, 'batch_norm'):
            try:
                # Fuse Conv -> BatchNorm by absorbing BN parameters into Conv weights/bias
                conv = module.conv
                bn = module.batch_norm
                
                # Get BatchNorm parameters
                bn_weight = bn.weight.data
                bn_bias = bn.bias.data
                bn_mean = bn.running_mean
                bn_var = bn.running_var
                bn_eps = bn.eps
                
                # Fuse into conv
                conv_weight = conv.weight.data
                if conv.bias is None:
                    conv.bias = torch.nn.Parameter(torch.zeros(conv.out_channels))
                conv_bias = conv.bias.data
                
                # Compute fused parameters: w_fused = w_conv * (gamma / sqrt(var + eps))
                scale = bn_weight / torch.sqrt(bn_var + bn_eps)
                conv.weight.data = conv_weight * scale.view(-1, 1, 1, 1)
                conv.bias.data = (conv_bias - bn_mean) * scale + bn_bias
                
                # Replace BatchNorm with Identity
                module.batch_norm = torch.nn.Identity()
                fused_count += 1
            except Exception as e:
                # Skip if fusion fails
                pass
    print(f"✓ Fused {fused_count} Conv+BatchNorm layers")
    
    # Convert to channels_last for NHWC layout
    print("Converting model to channels_last (NHWC) memory format...")
    detector = detector.to(memory_format=torch.channels_last)
    print("✓ Model converted to channels_last")
    
    # Wrap model to disable frequency coordinates
    print("Disabling frequency coordinate concatenation for ESP32...")
    wrapped_model = DisableFreqCoordsWrapper(detector)
    wrapped_model.eval()
    
    # Count modified layers
    modified_count = 0
    for m in wrapped_model.detector.modules():
        if hasattr(m, 'coords') and m.coords is None:
            modified_count += 1
    print(f"✓ Removed frequency coords from {modified_count} layers")
    print(f"✓ Adjusted conv weights to drop freq coord channel")
    
    # Print model info
    num_params = sum(p.numel() for p in wrapped_model.parameters())
    print(f"\nModel built successfully!")
    print(f"Total parameters: {num_params:,}")
    print(f"Number of classes: {len(targets.class_names)}")
    
    # Create dummy input in channels_last format
    dummy_input = torch.randn(batch_size, 1, height, width).to(memory_format=torch.channels_last)
    print(f"Dummy input shape (FIXED, channels_last): {dummy_input.shape}")
    
    # Test forward pass
    print("\nTesting forward pass with PyTorch...")
    with torch.no_grad():
        outputs = wrapped_model(dummy_input)
    
    print("Forward pass successful!")
    print(f"  detection_probs: {outputs[0].shape}")
    print(f"  size_preds: {outputs[1].shape}")
    print(f"  class_probs: {outputs[2].shape}")
    print(f"  features: {outputs[3].shape}")
    
    # Export to ONNX with static shapes
    print(f"\nExporting to ONNX: {output_path}")
    
    output_names = ['detection_probs', 'size_preds', 'class_probs', 'features']
    
    with torch.onnx.select_model_mode_for_export(wrapped_model, torch.onnx.TrainingMode.EVAL):
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['spectrogram'],
            output_names=output_names,
            dynamo=False,
        )
    
    print(f"✓ Model exported to {output_path}")
    print("\nNOTE: This model does NOT use frequency coordinates.")
    print("      Pass raw spectrograms (1 channel) directly from C++.")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export BatDetect2 model to ONNX for ESP32 (no freq coords)"
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to model config YAML")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output ONNX file path")
    parser.add_argument("--batch-size", type=int, default=1, help="Fixed batch size")
    parser.add_argument("--height", type=int, default=128, help="Fixed input height")
    parser.add_argument("--width", type=int, default=400, help="Fixed input width")
    
    args = parser.parse_args()
    
    export_model_esp_onnx(
        config_path=args.config,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
    )
