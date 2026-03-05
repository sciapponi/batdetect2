import torch
from batdetect2.models.vad import build_vad_1d_model

# Build default model
channels_list = [32, 64, 128, 256]
model = build_vad_1d_model(channels_list=channels_list)

print("=" * 80)
print("1D VAD MODEL ARCHITECTURE (Raw Audio)")
print("=" * 80)
print(model)
print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Model size: {total_params * 4 / 1024:.2f} KB (float32)")
print(f"Model size: {total_params * 2 / 1024:.2f} KB (float16)")
print(f"Model size: {total_params / 1024:.2f} KB (int8 quantized)")

print("\n" + "=" * 80)
print("DATA FLOW THROUGH NETWORK")
print("=" * 80)
print()
print("Input: Raw audio waveform (200ms @ 256kHz)")
print("   → 51,200 samples (0.2s * 256,000 Hz)")
print()

# Simulate forward pass
dummy_input = torch.randn(1, 1, 51200)  # Batch=1, Channels=1, Samples=51200
print(f"Input shape: {dummy_input.shape} (B, C, T)")
print()

# Track through encoder
x = dummy_input
for i, layer in enumerate(model.encoder.layers):
    x_before = x
    x = layer(x)
    downsampling = x_before.shape[2] // x.shape[2]
    print(f"Layer {i}: DepthwiseSeparableConv1D")
    print(f"  Input:  {x_before.shape}")
    print(f"  Output: {x.shape}")
    print(f"  Temporal downsampling: {downsampling}x")
    print(f"  Channels: {x_before.shape[1]} → {x.shape[1]}")
    print()

print(f"Encoder output shape: {x.shape}")
print("   ↓")

# Through head
pooled = model.head.pool(x)
print(f"After Global Average Pooling: {pooled.shape}")
print("   ↓")

flattened = pooled.flatten(1)
print(f"After Flatten: {flattened.shape}")
print("   ↓")

output = model.head.fc(flattened)
print(f"Final Output (logit): {output.shape}")
print()

# Apply sigmoid
prob = torch.sigmoid(output)
print(f"After Sigmoid (probability): {prob.shape}")
print()

print("=" * 80)
print("PARAMETER BREAKDOWN")
print("=" * 80)

encoder_params = sum(p.numel() for p in model.encoder.parameters())
head_params = sum(p.numel() for p in model.head.parameters())

print(f"Encoder:     {encoder_params:6,} params ({encoder_params/total_params*100:.1f}%)")
print(f"Head:        {head_params:6,} params ({head_params/total_params*100:.1f}%)")
print(f"Total:       {total_params:6,} params")
print()

print("=" * 80)
print("LAYER-BY-LAYER BREAKDOWN")
print("=" * 80)
for i, layer in enumerate(model.encoder.layers):
    num_params = sum(p.numel() for p in layer.parameters())
    print(f"Layer {i}: {num_params:6,} params")

print()
print("=" * 80)
print("COMPARISON WITH 2D VERSION")
print("=" * 80)
print("2D Spectrogram model: 5,131 params")
print(f"1D Raw audio model:   {total_params:,} params")
print(f"Ratio: {total_params/5131:.1f}x larger")
print()
print("Trade-off:")
print("  + No spectrogram computation needed")
print("  + Learns features directly from raw audio")
print("  + Can capture temporal patterns at sample level")
print("  - Slightly larger model")
print("  - More compute during training")
