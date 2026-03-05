"""Test if ESP32 model can learn on a simple task."""

import torch
import torch.nn as nn
import torch.optim as optim
from batdetect2.models.detectors import build_model
from batdetect2.core.configs import ModelConfig
import yaml

# Load configs
with open("example_data/config_esp32_optimized.yaml") as f:
    esp32_config = yaml.safe_load(f)

with open("example_data/config_with_skips.yaml") as f:
    standard_config = yaml.safe_load(f)

# Build models
esp32_model = build_model(ModelConfig(**esp32_config["model"]))
standard_model = build_model(ModelConfig(**standard_config["model"]))

print("=" * 70)
print("QUICK CONVERGENCE TEST")
print("=" * 70)

# Create simple synthetic task: detect calls in bottom half of spectrogram
batch_size = 4
x = torch.randn(batch_size, 1, 128, 200)

# Target: detection=1 in bottom half, 0 in top half
# Size: 32x50 after 2x downsampling
target_det = torch.zeros(batch_size, 1, 32, 50)
target_det[:, :, 16:, :] = 1  # Bottom half = detect

target_class = torch.zeros(batch_size, 18, 32, 50)  # 17 classes + background
target_class[:, 0, 16:, :] = 1  # Bottom half = class 0

# Train for 50 steps
print("\n" + "=" * 70)
print("ESP32 Model (ConvTranspose2d)")
print("=" * 70)

esp32_model.train()
optimizer_esp = optim.Adam(esp32_model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

for step in range(50):
    optimizer_esp.zero_grad()
    det, cls, _ = esp32_model(x)
    
    loss_det = loss_fn(det, target_det)
    loss_cls = loss_fn(cls, target_class)
    loss = loss_det + loss_cls
    
    loss.backward()
    
    # Check gradient norms
    decoder_grads = []
    for name, param in esp32_model.named_parameters():
        if 'decoder' in name and param.grad is not None:
            decoder_grads.append(param.grad.norm().item())
    
    optimizer_esp.step()
    
    if step % 10 == 0:
        avg_grad = sum(decoder_grads) / len(decoder_grads) if decoder_grads else 0
        print(f"Step {step:2d}: Loss={loss.item():.4f}, Avg decoder grad={avg_grad:.6f}")

# Test convergence
esp32_model.eval()
with torch.no_grad():
    det, cls, _ = esp32_model(x)
    det_sig = torch.sigmoid(det)
    print(f"\nFinal detection (top half):    {det_sig[:, :, :16, :].mean():.4f}")
    print(f"Final detection (bottom half): {det_sig[:, :, 16:, :].mean():.4f}")
    print(f"Separation: {(det_sig[:, :, 16:, :].mean() - det_sig[:, :, :16, :].mean()).item():.4f}")

print("\n" + "=" * 70)
print("Standard Model (Bilinear)")
print("=" * 70)

standard_model.train()
optimizer_std = optim.Adam(standard_model.parameters(), lr=0.001)

for step in range(50):
    optimizer_std.zero_grad()
    det, cls, _ = standard_model(x)
    
    loss_det = loss_fn(det, target_det)
    loss_cls = loss_fn(cls, target_class)
    loss = loss_det + loss_cls
    
    loss.backward()
    
    # Check gradient norms
    decoder_grads = []
    for name, param in standard_model.named_parameters():
        if 'decoder' in name and param.grad is not None:
            decoder_grads.append(param.grad.norm().item())
    
    optimizer_std.step()
    
    if step % 10 == 0:
        avg_grad = sum(decoder_grads) / len(decoder_grads) if decoder_grads else 0
        print(f"Step {step:2d}: Loss={loss.item():.4f}, Avg decoder grad={avg_grad:.6f}")

# Test convergence
standard_model.eval()
with torch.no_grad():
    det, cls, _ = standard_model(x)
    det_sig = torch.sigmoid(det)
    print(f"\nFinal detection (top half):    {det_sig[:, :, :16, :].mean():.4f}")
    print(f"Final detection (bottom half): {det_sig[:, :, 16:, :].mean():.4f}")
    print(f"Separation: {(det_sig[:, :, 16:, :].mean() - det_sig[:, :, :16, :].mean()).item():.4f}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
print("Both models should learn to detect calls in bottom half.")
print("If ESP32 has much worse separation, the architecture has issues.")
