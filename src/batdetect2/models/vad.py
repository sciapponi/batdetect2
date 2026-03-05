from typing import Optional

import torch
import torch.nn as nn
from loguru import logger

from batdetect2.models.encoder import Encoder, build_encoder
from batdetect2.typing.models import DetectionModel, ModelOutput

__all__ = [
    "VADModel",
    "build_vad_model",
    "VAD1DModel",
    "build_vad_1d_model",
]


class VADHead(nn.Module):
    """Simple classification head for Voice Activity Detection."""

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class VADModel(DetectionModel):
    """
    Voice Activity Detection Model (Encoder-only).
    
    This model takes a spectrogram input, processes it through an encoder
    (often depthwise-separable convolutions for efficiency), pools the features,
    and produces a binary classification output (Bat / No Bat).
    """

    encoder: Encoder
    head: VADHead

    def __init__(
        self,
        encoder: Encoder,
        head: VADHead,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, spec: torch.Tensor) -> ModelOutput:
        """
        Forward pass.
        
        Args:
            spec: Input spectrogram tensor (B, 1, F, T)
            
        Returns:
            ModelOutput with `class_probs` populated.
            The structure reuses ModelOutput, but detection-specific fields
            will be None or dummy values.
        """
        # Encode features
        features = self.encoder(spec)
        
        # Get the final feature map (bottleneck)
        # The encoder may return a single tensor or a list depending on return_skip
        if isinstance(features, list):
            bottleneck = features[-1]
        else:
            bottleneck = features
        
        # Classification head
        logits = self.head(bottleneck)
        
        # In this simplified model, we don't have bounding boxes or per-pixel maps.
        # We return the logits as class probabilities (after sigmoid/softmax if needed).
        # For compatibility with training loop, we might need to wrap it.
        
        # If binary classification (num_classes=1), use sigmoid to get prob
        # If 2 classes (recommended for bat/non-bat), use softmax
        if self.head.fc.out_features == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)
            
        # Returning a ModelOutput.
        # Note: ModelOutput fields are for detection (class_probs usually (B, T, C)).
        # Here we have (B, C) or (B, num_classes).
        # We use detection_probs for the binary/multi-class output.
        # Set features to bottleneck if features is a tensor, else to None
        return_features = bottleneck if not isinstance(features, list) else bottleneck
        
        # For 2-class case, extract bat probability for detection_probs
        if self.head.fc.out_features == 2:
            detection_probs = probs[:, 1:2]  # Bat class (assuming index 1)
        else:
            detection_probs = probs
        
        return ModelOutput(
            detection_probs=detection_probs,  # Use detection_probs for the bat probability
            size_preds=torch.zeros(probs.size(0), 2, device=probs.device),  # Dummy size predictions
            class_probs=probs,  # Full class probabilities (all classes)
            features=return_features,
            genus_probs=None,
        )


def build_vad_model(config, input_channels: int = 1) -> VADModel:
    """Builds a VADModel from configuration.
    
    Args:
        config: VADConfig with model configuration
        input_channels: Number of input channels (default: 1)
        
    Returns:
        VADModel instance
        
    Note:
        For binary classification, set config.out_channels to 2 (recommended)
        to explicitly model bat and non-bat classes, or 1 for single sigmoid output.
    """
    
    # Build encoder
    encoder = build_encoder(
        in_channels=input_channels,
        input_height=config.input_height,
        config=config.encoder,
    )
    
    # Infer output channels from encoder config
    # The last layer of encoder determines the channel count
    enc_out_channels = encoder.out_channels
    
    head = VADHead(in_channels=enc_out_channels, num_classes=config.out_channels)
    
    return VADModel(encoder=encoder, head=head)


class DepthwiseSeparableConv1D(nn.Module):
    """1D Depthwise-Separable Convolution for efficient 1D processing."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class VAD1DEncoder(nn.Module):
    """1D Encoder for raw audio waveform processing."""
    
    def __init__(self, in_channels: int = 1, channels_list: list = None):
        super().__init__()
        if channels_list is None:
            # Default: progressively increase channels
            channels_list = [16, 32, 64, 128, 256]
        
        self.in_channels = in_channels
        self.out_channels = channels_list[-1]
        
        layers = []
        current_channels = in_channels
        
        for out_ch in channels_list:
            # Use larger kernel for first layer to capture more context
            kernel_size = 15 if len(layers) == 0 else 7
            padding = kernel_size // 2
            
            layers.append(DepthwiseSeparableConv1D(
                in_channels=current_channels,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=4,  # Aggressive downsampling
                padding=padding
            ))
            current_channels = out_ch
        
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class VAD1DHead(nn.Module):
    """Classification head for 1D VAD."""
    
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class VAD1DModel(nn.Module):
    """Voice Activity Detection Model for raw audio (1D)."""
    
    def __init__(self, encoder: VAD1DEncoder, head: VAD1DHead):
        super().__init__()
        self.encoder = encoder
        self.head = head
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Raw audio waveform (B, C, T) where T is number of samples
            
        Returns:
            Logits for binary classification
        """
        features = self.encoder(x)
        logits = self.head(features)
        return logits


def build_vad_1d_model(channels_list: list = None, num_classes: int = 1) -> VAD1DModel:
    """Builds a 1D VAD model for raw audio.
    
    Args:
        channels_list: List of channel counts for each layer. 
                      Default: [16, 32, 64, 128, 256]
        num_classes: Number of output classes (1 for binary)
    """
    if channels_list is None:
        channels_list = [16, 32, 64, 128, 256]
    
    encoder = VAD1DEncoder(in_channels=1, channels_list=channels_list)
    head = VAD1DHead(in_channels=encoder.out_channels, num_classes=num_classes)
    
    return VAD1DModel(encoder=encoder, head=head)
