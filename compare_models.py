#!/usr/bin/env python3
"""Compare original vs tiny model parameter counts."""

import torch
from pathlib import Path
from batdetect2.models import build_model
from batdetect2.config import validate_config
from batdetect2.targets import build_targets
import yaml

# Load tiny config
tiny_config_path = Path("example_data/config_phinet_unet_tiny.yaml")
with open(tiny_config_path) as f:
    tiny_config_dict = yaml.safe_load(f)
tiny_config = validate_config(tiny_config_dict)
targets = build_targets(config=tiny_config.targets)
tiny_model = build_model(config=tiny_config.model, targets=targets)

# Load original config
orig_config_path = Path("example_data/config.yaml")
with open(orig_config_path) as f:
    orig_config_dict = yaml.safe_load(f)
orig_config = validate_config(orig_config_dict)
orig_model = build_model(config=orig_config.model, targets=targets)

print(f"Tiny model: {sum(p.numel() for p in tiny_model.parameters()):,} params")
print(f"Original model: {sum(p.numel() for p in orig_model.parameters()):,} params")
print(f"Reduction: {sum(p.numel() for p in orig_model.parameters()) / sum(p.numel() for p in tiny_model.parameters()):.1f}x")
