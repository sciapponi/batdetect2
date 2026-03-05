#!/usr/bin/env python
"""Simple usage example for Encoder + LiteMLA classifier."""

import torch
from batdetect2.models import build_classifier
from batdetect2.models.config import load_backbone_config

# Option 1: Build with config file
config = load_backbone_config("example_data/config_encoder_litemla_classifier.yaml")
model = build_classifier(
    num_classes=17,
    config=config,
    pooling_mode="avg",
    dropout=0.1,
)

# Option 2: Build programmatically
from batdetect2.models.config import BackboneConfig
from batdetect2.models.encoder import EncoderConfig, FreqCoordConvDownConfig, ConvConfig, LayerGroupConfig
from batdetect2.models.bottleneck import BottleneckConfig, LiteMLAConfig

config_manual = BackboneConfig(
    input_height=128,
    in_channels=1,
    encoder=EncoderConfig(
        return_skip=False,
        layers=[
            FreqCoordConvDownConfig(out_channels=32),
            FreqCoordConvDownConfig(out_channels=64),
            LayerGroupConfig(layers=[
                FreqCoordConvDownConfig(out_channels=128),
                ConvConfig(out_channels=256),
            ]),
        ],
    ),
    bottleneck=BottleneckConfig(
        channels=256,
        layers=[
            LiteMLAConfig(out_channels=256, attention_channels=256),
        ],
    ),
)

model_manual = build_classifier(num_classes=17, config=config_manual, pooling_mode="avg")

# Test
dummy_spec = torch.randn(4, 1, 128, 192)
output = model(dummy_spec)

print("Encoder + LiteMLA Classifier")
print("=" * 70)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Input:        {dummy_spec.shape}")
print(f"Output:       {output.probs.shape}")
print(f"Predictions:  {output.probs.argmax(dim=1)}")
print("\n✓ Ready to train!")
