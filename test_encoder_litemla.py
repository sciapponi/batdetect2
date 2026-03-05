#!/usr/bin/env python
"""Truly minimal classifier: Encoder -> LiteMLA -> Pool -> FC."""

import torch
from batdetect2.models import build_classifier
from batdetect2.models.config import BackboneConfig
from batdetect2.models.encoder import EncoderConfig, FreqCoordConvDownConfig, ConvConfig, LayerGroupConfig
from batdetect2.models.bottleneck import BottleneckConfig, LiteMLAConfig

def build_encoder_only_classifier():
    """Build classifier with ZERO bottleneck (encoder -> pool -> FC)."""
    
    config = BackboneConfig(
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
            layers=[],  # Truly empty
        ),
    )
    
    return build_classifier(num_classes=17, config=config, pooling_mode="avg")


def build_encoder_litemla_classifier():
    """Build: Encoder -> Single LiteMLA -> Pool -> FC."""
    
    config = BackboneConfig(
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
                LiteMLAConfig(
                    out_channels=256,
                    attention_channels=256,
                ),  # Single attention layer
            ],
        ),
    )
    
    return build_classifier(num_classes=17, config=config, pooling_mode="avg")


def compare_all_configs():
    """Compare all classifier configurations."""
    
    print("=" * 80)
    print("COMPLETE CLASSIFIER COMPARISON")
    print("=" * 80)
    
    configs = {
        "1. Encoder only (no attention)": build_encoder_only_classifier(),
        "2. Encoder + LiteMLA (minimal)": build_encoder_litemla_classifier(),
        "3. Full (default with attention)": None,  # Fill below
        "4. Full Detector (reference)": None,
    }
    
    # Add full versions
    from batdetect2.models import build_classifier, build_detector
    configs["3. Full (default with attention)"] = build_classifier(num_classes=17)
    configs["4. Full Detector (reference)"] = build_detector(num_classes=17)
    
    print("\nTOTAL PARAMETERS:")
    print("-" * 80)
    
    results = {}
    for name, model in configs.items():
        total = sum(p.numel() for p in model.parameters())
        results[name] = total
        print(f"{name:40s} {total:>12,} params")
    
    # Detailed breakdown for encoder + LiteMLA
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN: Encoder + LiteMLA (your request)")
    print("=" * 80)
    
    model = configs["2. Encoder + LiteMLA (minimal)"]
    encoder_p = sum(p.numel() for p in model.encoder.parameters())
    bottleneck_p = sum(p.numel() for p in model.bottleneck.parameters())
    fc_p = sum(p.numel() for p in model.fc.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    
    print(f"\nEncoder:       {encoder_p:>10,} params ({encoder_p/total_p*100:>5.1f}%)")
    print(f"  - Layer 0:   {sum(p.numel() for p in model.encoder.layers[0].parameters()):>10,}")
    print(f"  - Layer 1:   {sum(p.numel() for p in model.encoder.layers[1].parameters()):>10,}")
    print(f"  - Layer 2:   {sum(p.numel() for p in model.encoder.layers[2].parameters()):>10,}")
    print(f"\nLiteMLA:       {bottleneck_p:>10,} params ({bottleneck_p/total_p*100:>5.1f}%)")
    print(f"FC Head:       {fc_p:>10,} params ({fc_p/total_p*100:>5.1f}%)")
    print(f"{'─'*50}")
    print(f"TOTAL:         {total_p:>10,} params")
    
    # Show reduction
    print("\n" + "=" * 80)
    print("SIZE REDUCTION")
    print("=" * 80)
    encoder_litemla = results["2. Encoder + LiteMLA (minimal)"]
    full_classifier = results["3. Full (default with attention)"]
    detector = results["4. Full Detector (reference)"]
    
    print(f"\nEncoder + LiteMLA vs Full Classifier: {(1 - encoder_litemla/full_classifier)*100:>5.1f}% smaller")
    print(f"Encoder + LiteMLA vs Detector:        {(1 - encoder_litemla/detector)*100:>5.1f}% smaller")
    
    # Test it works
    print("\n" + "=" * 80)
    print("FUNCTIONALITY TEST")
    print("=" * 80)
    
    dummy = torch.randn(4, 1, 128, 192)
    output = model(dummy)
    
    print(f"\nInput shape:   {dummy.shape}")
    print(f"Output shape:  {output.probs.shape}")
    print(f"Predictions:   {output.probs.argmax(dim=1).tolist()}")
    print(f"\n✓ Model works correctly!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_all_configs()
