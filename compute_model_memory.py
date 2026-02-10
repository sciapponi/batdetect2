"""
Compute RAM usage for the model with and without skip connections.
Based on the current config.yaml architecture.
"""

import math

# Model configuration from config.yaml
INPUT_HEIGHT = 128
RESIZE_FACTOR = 0.5
DURATION = 0.256  # seconds
SAMPLERATE = 256000
WINDOW_DURATION = 0.002
WINDOW_OVERLAP = 0.75

# Calculate input dimensions
n_samples = int(DURATION * SAMPLERATE)
window_samples = int(WINDOW_DURATION * SAMPLERATE)
hop_samples = int(window_samples * (1 - WINDOW_OVERLAP))
time_bins = 1 + (n_samples - window_samples) // hop_samples

# After resize
height = int(INPUT_HEIGHT * RESIZE_FACTOR)  # 64
width = int(time_bins * RESIZE_FACTOR)

print("="*70)
print("MODEL MEMORY CALCULATION")
print("="*70)
print(f"\nInput dimensions: {height}H × {width}W")
print(f"Batch size: 1 (inference)")

# Encoder layers (downsampling by 2 each time)
encoder_layers = [
    (height, width, 32, "FreqCoordConvDown 32"),
    (height//2, width//2, 64, "FreqCoordConvDown 64"),
    (height//4, width//4, 128, "FreqCoordConvDown 128"),
    (height//8, width//8, 256, "ConvBlock 256"),
]

bottleneck = (height//8, width//8, 256, "LiteMLA 256")

# Decoder layers (upsampling by 2 each time)
decoder_layers = [
    (height//4, width//4, 64, "FreqCoordConvUp 64"),
    (height//2, width//2, 32, "FreqCoordConvUp 32"),
    (height, width, 32, "FreqCoordConvUp 32"),
    (height, width, 32, "ConvBlock 32"),
]

# Calculate memory for each layer
def calculate_memory(h, w, c, dtype='float32'):
    """Calculate memory in MB for a feature map."""
    elements = h * w * c
    bytes_per_element = 4 if dtype == 'float32' else 1  # INT8
    return elements * bytes_per_element / (1024 * 1024)

print("\n" + "="*70)
print("ENCODER ACTIVATIONS")
print("="*70)
encoder_memory = []
for h, w, c, name in encoder_layers:
    mem_f32 = calculate_memory(h, w, c, 'float32')
    mem_int8 = calculate_memory(h, w, c, 'int8')
    encoder_memory.append((h, w, c, mem_f32, mem_int8))
    print(f"{name:30s} {h:3d}×{w:3d}×{c:3d} = {mem_f32:7.3f} MB (float32) | {mem_int8:7.3f} MB (int8)")

print("\n" + "="*70)
print("BOTTLENECK")
print("="*70)
h, w, c, name = bottleneck
mem_f32 = calculate_memory(h, w, c, 'float32')
mem_int8 = calculate_memory(h, w, c, 'int8')
print(f"{name:30s} {h:3d}×{w:3d}×{c:3d} = {mem_f32:7.3f} MB (float32) | {mem_int8:7.3f} MB (int8)")

print("\n" + "="*70)
print("DECODER ACTIVATIONS")
print("="*70)
for h, w, c, name in decoder_layers:
    mem_f32 = calculate_memory(h, w, c, 'float32')
    mem_int8 = calculate_memory(h, w, c, 'int8')
    print(f"{name:30s} {h:3d}×{w:3d}×{c:3d} = {mem_f32:7.3f} MB (float32) | {mem_int8:7.3f} MB (int8)")

# Calculate total memory requirements
print("\n" + "="*70)
print("MEMORY COMPARISON: WITH vs WITHOUT SKIP CONNECTIONS")
print("="*70)

# WITHOUT skip connections (current config: use_skip: false)
# Only need to store the current layer activation
max_activation_f32 = max(mem for _, _, _, mem, _ in encoder_memory)
max_activation_int8 = max(mem for _, _, _, _, mem in encoder_memory)

print("\nWITHOUT SKIP CONNECTIONS (current config):")
print(f"  Architecture: LiteMLA (lightweight multi-head linear attention)")
print(f"  Peak activation memory (float32): {max_activation_f32:.3f} MB")
print(f"  Peak activation memory (int8):    {max_activation_int8:.3f} MB")

# WITH skip connections (use_skip: true)
# Must store all encoder activations for concatenation in decoder
total_skip_f32 = sum(mem for _, _, _, mem, _ in encoder_memory)
total_skip_int8 = sum(mem for _, _, _, _, mem in encoder_memory)

print("\nWITH SKIP CONNECTIONS (U-Net architecture):")
print(f"  Architecture: SelfAttention (standard multi-head attention)")
print(f"  Total encoder memory to store (float32): {total_skip_f32:.3f} MB")
print(f"  Total encoder memory to store (int8):    {total_skip_int8:.3f} MB")

# Calculate savings
savings_f32 = total_skip_f32 - max_activation_f32
savings_int8 = total_skip_int8 - max_activation_int8
savings_pct = 100 * savings_f32 / total_skip_f32

print("\n" + "="*70)
print("MEMORY SAVINGS")
print("="*70)
print(f"\nRAM saved by disabling skip connections:")
print(f"  Float32: {savings_f32:.3f} MB ({savings_pct:.1f}% reduction)")
print(f"  Int8:    {savings_int8:.3f} MB ({savings_pct:.1f}% reduction)")

# Microcontroller feasibility
print("\n" + "="*70)
print("MICROCONTROLLER DEPLOYMENT FEASIBILITY")
print("="*70)
print("\nCommon MCU RAM capacities:")
print(f"  STM32H7 (high-end):    2048 KB = {2048/1024:.1f} MB")
print(f"  ESP32-S3:               512 KB = {512/1024:.1f} MB")
print(f"  Raspberry Pi Pico:      264 KB = {264/1024:.1f} MB")
print(f"  STM32F4 (mid-range):    192 KB = {192/1024:.1f} MB")

print("\nMinimum RAM needed (activation memory only, no model weights):")
print(f"  Without skips + float32: {max_activation_f32:.3f} MB = {max_activation_f32*1024:.0f} KB")
print(f"  Without skips + int8:    {max_activation_int8:.3f} MB = {max_activation_int8*1024:.0f} KB")
print(f"  With skips + float32:    {total_skip_f32:.3f} MB = {total_skip_f32*1024:.0f} KB")
print(f"  With skips + int8:       {total_skip_int8:.3f} MB = {total_skip_int8*1024:.0f} KB")

print("\nVerdict:")
if max_activation_int8 * 1024 < 512:
    print(f"  ✓ Fits on ESP32-S3 (512KB) with int8 + no skips")
else:
    print(f"  ✗ Does NOT fit on ESP32-S3 (512KB) even with int8 + no skips")

if total_skip_int8 * 1024 < 512:
    print(f"  ✓ Fits on ESP32-S3 (512KB) with int8 + skips")
else:
    print(f"  ✗ Does NOT fit on ESP32-S3 (512KB) with int8 + skips")

print("\nNote: This calculation includes only activation memory.")
print("Real deployment also needs:")
print("  - Model weights (typically 0.5-2 MB even with int8)")
print("  - Input/output buffers")
print("  - Stack and heap space")
print("="*70)
