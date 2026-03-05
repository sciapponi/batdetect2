#!/usr/bin/env python
"""Show detailed parameter breakdown for Classifier vs Detector."""

import torch
from batdetect2.models import build_classifier, build_detector

def show_detailed_breakdown():
    """Show where all the parameters are."""
    num_classes = 17
    
    print("=" * 80)
    print("PARAMETER BREAKDOWN")
    print("=" * 80)
    
    # Build both models
    classifier = build_classifier(num_classes=num_classes, pooling_mode="avg")
    detector = build_detector(num_classes=num_classes)
    
    # Classifier breakdown
    print("\nCLASSIFIER (Encoder + Bottleneck + FC):")
    print("-" * 80)
    encoder_params = sum(p.numel() for p in classifier.encoder.parameters())
    bottleneck_params = sum(p.numel() for p in classifier.bottleneck.parameters())
    fc_params = sum(p.numel() for p in classifier.fc.parameters())
    dropout_params = sum(p.numel() for p in classifier.dropout.parameters())
    total_classifier = sum(p.numel() for p in classifier.parameters())
    
    print(f"  Encoder:    {encoder_params:>10,} params  ({encoder_params/total_classifier*100:>5.1f}%)")
    print(f"  Bottleneck: {bottleneck_params:>10,} params  ({bottleneck_params/total_classifier*100:>5.1f}%)")
    print(f"  FC layer:   {fc_params:>10,} params  ({fc_params/total_classifier*100:>5.1f}%)")
    print(f"  Dropout:    {dropout_params:>10,} params  ({dropout_params/total_classifier*100:>5.1f}%)")
    print(f"  {'─'*40}")
    print(f"  TOTAL:      {total_classifier:>10,} params")
    
    # Detector breakdown
    print("\nDETECTOR (Encoder + Bottleneck + Decoder + 3 Heads):")
    print("-" * 80)
    backbone = detector.backbone
    encoder_params_d = sum(p.numel() for p in backbone.encoder.parameters())
    bottleneck_params_d = sum(p.numel() for p in backbone.bottleneck.parameters())
    decoder_params = sum(p.numel() for p in backbone.decoder.parameters())
    classifier_head_params = sum(p.numel() for p in detector.classifier_head.parameters())
    bbox_head_params = sum(p.numel() for p in detector.bbox_head.parameters())
    total_detector = sum(p.numel() for p in detector.parameters())
    
    print(f"  Encoder:          {encoder_params_d:>10,} params  ({encoder_params_d/total_detector*100:>5.1f}%)")
    print(f"  Bottleneck:       {bottleneck_params_d:>10,} params  ({bottleneck_params_d/total_detector*100:>5.1f}%)")
    print(f"  Decoder:          {decoder_params:>10,} params  ({decoder_params/total_detector*100:>5.1f}%)")
    print(f"  Classifier Head:  {classifier_head_params:>10,} params  ({classifier_head_params/total_detector*100:>5.1f}%)")
    print(f"  BBox Head:        {bbox_head_params:>10,} params  ({bbox_head_params/total_detector*100:>5.1f}%)")
    print(f"  {'─'*40}")
    print(f"  TOTAL:            {total_detector:>10,} params")
    
    # Comparison
    print("\nCOMPARISON:")
    print("-" * 80)
    print(f"  Detector has decoder:     {decoder_params:>10,} params")
    print(f"  Detector has 2 heads:     {classifier_head_params + bbox_head_params:>10,} params")
    print(f"  Classifier has FC:        {fc_params:>10,} params")
    print(f"  {'─'*40}")
    print(f"  Difference:               {total_detector - total_classifier:>10,} params")
    print(f"  Classifier is {(1 - total_classifier/total_detector)*100:.1f}% smaller")
    
    # Show that encoder is the same
    print("\nKEY INSIGHT:")
    print("-" * 80)
    print(f"  ✓ Classifier uses ONLY:  Encoder + Bottleneck + FC")
    print(f"  ✓ Classifier has NO decoder")
    print(f"  ✓ Encoder/Bottleneck are {(encoder_params + bottleneck_params)/total_classifier*100:.1f}% of classifier")
    print(f"  ✓ Most params are in the encoder ({encoder_params/total_classifier*100:.1f}%)")
    
    # Show encoder layer breakdown
    print("\nENCODER LAYER BREAKDOWN:")
    print("-" * 80)
    for i, layer in enumerate(classifier.encoder.layers):
        layer_params = sum(p.numel() for p in layer.parameters())
        print(f"  Layer {i}: {layer.__class__.__name__:25s} {layer_params:>10,} params")
    
    print("\n" + "=" * 80)
    

if __name__ == "__main__":
    show_detailed_breakdown()
