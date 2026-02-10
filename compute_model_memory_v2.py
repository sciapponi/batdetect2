"""
Compute RAM usage for models by actually loading configurations and tracing through the architecture.
More realistic than hardcoded calculations.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from batdetect2.config import validate_config
from batdetect2.models import build_model
from batdetect2.preprocess import build_preprocessor
from batdetect2.postprocess import build_postprocessor
from batdetect2.targets import build_targets

def plot_ram_utilization(result1, result2):
    """Create cute barplots showing RAM utilization percentages."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle
    
    # RAM budgets to visualize against (in KB)
    budgets = {
        'ESP32-S3': 512,
        'STM32H7': 2048,
        'Big Boi': 8192,
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    results = [result1, result2]
    colors = ['#FF6B9D', '#4ECDC4', '#FFE66D', '#95E1D3']
    
    for idx, (budget_name, budget_kb) in enumerate(budgets.items()):
        ax = axes[idx]
        
        bar_width = 0.35
        x_positions = np.arange(len(results))
        
        for i, result in enumerate(results):
            ram_kb = result['activation_mb_int8'] * 1024
            utilization = (ram_kb / budget_kb) * 100
            
            # Determine color based on utilization
            if utilization <= 100:
                color = colors[0]  # Pink - fits!
                edge_color = colors[0]
            else:
                color = colors[1]  # Teal - doesn't fit
                edge_color = colors[1]
            
            # Cap at 150% for visualization
            display_util = min(utilization, 150)
            
            # Draw the bar
            bar = ax.barh(i, display_util, bar_width*2, 
                         color=color, alpha=0.7, 
                         edgecolor=edge_color, linewidth=2)
            
            # Add percentage label
            config_name = Path(result['config_path']).stem
            label_x = display_util + 3 if display_util < 140 else display_util - 10
            ax.text(label_x, i, f"{utilization:.0f}%", 
                   va='center', ha='left' if display_util < 140 else 'right',
                   fontweight='bold', fontsize=11)
            
            # Add marker if it fits
            if utilization <= 100:
                ax.text(-5, i, '✓', va='center', ha='right', fontsize=14, color='green', fontweight='bold')
            else:
                ax.text(-5, i, '✗', va='center', ha='right', fontsize=14, color='red', fontweight='bold')
        
        # Draw the 100% reference line
        ax.axvline(100, color='#2C3E50', linestyle='--', linewidth=2, alpha=0.5)
        ax.text(100, len(results) - 0.5, '← budget', 
               va='center', ha='right', fontsize=9, alpha=0.7,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Styling
        ax.set_xlim(0, 155)
        ax.set_ylim(-0.5, len(results) - 0.5)
        ax.set_yticks(range(len(results)))
        ax.set_yticklabels([Path(r['config_path']).stem for r in results], fontsize=10)
        ax.set_xlabel('RAM Utilization (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{budget_name}\n({budget_kb} KB)', fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.2, linestyle=':')
        ax.invert_yaxis()
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Add overall title
    fig.suptitle('RAM Utilization: Can These Models Fit?', 
                fontsize=16, fontweight='bold', y=1.02)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=colors[0], edgecolor=colors[0], label='✓ Fits!'),
        mpatches.Patch(facecolor=colors[1], edgecolor=colors[1], label='✗ Too big'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False, fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'ram_utilization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n🎨 RAM utilization plot saved to: {output_path}")
    plt.close()

def plot_memory_comparison(result1, result2):
    """Create a two-sided barplot comparing live RAM usage at each operation."""
    import matplotlib.pyplot as plt
    
    # Compute live RAM for each model
    def compute_live_ram(layer_stats, is_unet):
        live_rams = []
        pinned_skips = []
        
        for i, layer in enumerate(layer_stats):
            out_kb = layer['bytes'] / 1024
            
            if is_unet:
                if i < len(layer_stats) // 2:
                    pinned_skips.append(out_kb)
                    prev_kb = layer_stats[i-1]['bytes']/1024 if i > 0 else 0
                    live_kb = out_kb + prev_kb + sum(pinned_skips[:-1])
                else:
                    current_skip = pinned_skips.pop() if pinned_skips else 0
                    prev_kb = layer_stats[i-1]['bytes']/1024 if i > 0 else 0
                    live_kb = out_kb + prev_kb + current_skip + sum(pinned_skips)
            else:
                prev_kb = layer_stats[i-1]['bytes']/1024 if i > 0 else 0
                live_kb = out_kb + prev_kb
            
            live_rams.append(live_kb)
        
        return live_rams
    
    # Get live RAM for both models
    live_ram1 = compute_live_ram(result1['layer_stats'], result1['use_skip'])
    live_ram2 = compute_live_ram(result2['layer_stats'], result2['use_skip'])
    
    # Pad to same length
    max_len = max(len(live_ram1), len(live_ram2))
    if len(live_ram1) < max_len:
        live_ram1 = live_ram1 + [0] * (max_len - len(live_ram1))
    if len(live_ram2) < max_len:
        live_ram2 = live_ram2 + [0] * (max_len - len(live_ram2))
    
    # Create two-sided barplot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(max_len)
    
    # Plot left side (config1)
    ax.barh(y_pos, [-x for x in live_ram1], color='#FF6B9D', alpha=0.8, label=Path(result1['config_path']).stem)
    
    # Plot right side (config2)
    ax.barh(y_pos, live_ram2, color='#4ECDC4', alpha=0.8, label=Path(result2['config_path']).stem)
    
    ax.set_xlabel('Live RAM (KB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_title('Memory Usage Comparison\n(Two-sided barplot of live RAM at each operation)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis to show absolute values
    ax.set_xticks([-4000, -3000, -2000, -1000, 0, 1000, 2000, 3000, 4000])
    ax.set_xticklabels(['4000', '3000', '2000', '1000', '0', '1000', '2000', '3000', '4000'])
    
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    ax.set_ylim(-0.5, max_len - 0.5)
    
    # Invert y-axis so layer 0 is at top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'memory_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Memory comparison plot saved to: {output_path}")
    plt.close()

class DeepMCUProfiler:
    def __init__(self, model, input_size=(1, 1, 128, 128), is_unet=False):
        self.model = model
        self.input_size = input_size
        self.is_unet = is_unet
        self.stats = []
        self.hooks = []
        self.pinned_skips = []
        self.peak_ram = 0

    def _get_leaf_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Calculate size in bytes (simulating INT8)
                out_bytes = output.numel() 
                
                # Special Case: Attention Matrix Overhead
                attn_extra = 0
                if "Attention" in str(type(module)):
                    # seq_len^2 * num_heads
                    # Assumes input is (B, N, C) or (B, C, H, W)
                    n = input[0].shape[1] if input[0].dim() == 3 else input[0].shape[2] * input[0].shape[3]
                    heads = getattr(module, 'num_heads', 8)
                    attn_extra = (n * n * heads) 

                self.stats.append({
                    "name": name,
                    "out_kb": out_bytes / 1024,
                    "extra_kb": attn_extra / 1024,
                    "shape": list(output.shape),
                    "bytes": out_bytes,  # For compatibility with plotting
                })
        return hook

    def profile(self):
        # Register hooks ONLY on leaf modules (no double counting parents)
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                self.hooks.append(module.register_forward_hook(self._get_leaf_hook(name)))

        with torch.no_grad():
            self.model(torch.randn(*self.input_size))

        for h in self.hooks: h.remove()
        self._combine_bottleneck_layers()
        return self._analyze()
    
    def _combine_bottleneck_layers(self):
        """Combine all bottleneck sub-layers into a single layer entry."""
        bottleneck_layers = [s for s in self.stats if 'bottleneck' in s['name']]
        if not bottleneck_layers:
            return
        
        # Remove all bottleneck layers from stats
        self.stats = [s for s in self.stats if 'bottleneck' not in s['name']]
        
        # Find the position where bottleneck should be inserted (after last encoder layer)
        insert_idx = 0
        for i, s in enumerate(self.stats):
            if 'encoder' in s['name']:
                insert_idx = i + 1
        
        # Create combined bottleneck entry with max output size
        max_bytes = max(layer['bytes'] for layer in bottleneck_layers)
        combined = {
            'name': 'backbone.bottleneck',
            'out_kb': max_bytes / 1024,
            'extra_kb': sum(layer['extra_kb'] for layer in bottleneck_layers),
            'shape': next(layer['shape'] for layer in bottleneck_layers if layer['bytes'] == max_bytes),
            'bytes': max_bytes,
        }
        
        self.stats.insert(insert_idx, combined)

    def _analyze(self):
        print(f"\n{'MODULE PATH':<40} | {'OUT (KB)':<10} | {'PEAK (KB)':<10}")
        print("-" * 65)
        
        running_peak = 0
        
        # Microcontroller Logic: 
        # Total RAM = Current Output + Input + Pinned Skips + Internal Op Buffer
        for i, op in enumerate(self.stats):
            out_kb = op['out_kb']
            extra_kb = op['extra_kb'] # The Attention Matrix "spike"
            
            # Get previous layer output (which is current layer's input)
            if i == 0:
                # First layer: use actual model input size
                # Input is typically [B, C, H, W] - we stored it during profiling
                prev_kb = self.input_size[0] * self.input_size[1] * self.input_size[2] * self.input_size[3] / 1024
            else:
                prev_kb = self.stats[i-1]['out_kb']
            
            # Logic for U-Net persistence
            # Simpler heuristic: encoder is first half, decoder is second half
            if self.is_unet:
                mid_point = len(self.stats) // 2
                
                # In encoder: accumulate all outputs as skips
                if i < mid_point:
                    self.pinned_skips.append(out_kb)
                    # Live = Current + Previous + All accumulated skips (except current)
                    current_live = out_kb + prev_kb + extra_kb + sum(self.pinned_skips[:-1])
                else:
                    # In decoder: pop a skip when we use it
                    # Pop happens when we move to a new decoder layer
                    if i == mid_point or (i > mid_point and self.pinned_skips):
                        skip_kb = self.pinned_skips.pop() if self.pinned_skips else 0
                    else:
                        skip_kb = 0
                    
                    current_live = out_kb + prev_kb + extra_kb + skip_kb + sum(self.pinned_skips)
            else:
                # Autoencoder: no skip persistence
                current_live = out_kb + prev_kb + extra_kb

            running_peak = max(running_peak, current_live)
            
            # Visualizing the spike if it's an attention layer
            name_str = f"⚠️ {op['name']}" if extra_kb > 0 else op['name']
            print(f"{name_str[:40]:<40} | {out_kb:>10.1f} | {current_live:>10.1f}")

        print("-" * 65)
        print(f"TRUE PEAK RAM (INT8 SIM): {running_peak:.2f} KB")
        
        return running_peak / 1024, self.stats  # Return in MB + detailed stats

def trace_model_memory(config_path, batch_size=1):
    """
    Trace through model to measure actual peak memory usage.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {config_path}")
    print(f"{'='*70}")
    
    # Load config
    import yaml
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = validate_config(config_dict)
    
    # Build components
    targets = build_targets(config.targets)
    preprocessor = build_preprocessor(config=config.preprocess, input_samplerate=config.audio.samplerate)
    postprocessor = build_postprocessor(preprocessor=preprocessor, config=config.postprocess, targets=targets)
    
    # Build model
    from batdetect2.models.detectors import build_detector
    num_genera = len(targets.genus_names) if targets.genus_names else None
    detector = build_detector(
        num_classes=len(targets.class_names),
        config=config.model,
        num_genera=num_genera,
    )
    
    # Get input dimensions
    # Typical audio clip duration from config
    if hasattr(config.train, 'train_loader') and hasattr(config.train.train_loader, 'clipping_strategy'):
        duration = config.train.train_loader.clipping_strategy.duration
    else:
        duration = 0.256  # default
    
    # Calculate spectrogram dimensions
    n_samples = int(duration * config.audio.samplerate)
    window_samples = int(config.preprocess.stft.window_duration * config.audio.samplerate)
    hop_samples = int(window_samples * (1 - config.preprocess.stft.window_overlap))
    time_bins = 1 + (n_samples - window_samples) // hop_samples
    
    height = config.preprocess.size.height
    resize_factor = config.preprocess.size.resize_factor
    
    # After preprocessing
    final_height = int(height * resize_factor)
    final_width = int(time_bins * resize_factor)
    
    # The model expects input_height from config, not final_height
    model_input_height = config.model.input_height if hasattr(config.model, 'input_height') else height
    
    print(f"\nInput Configuration:")
    print(f"  Audio duration: {duration}s @ {config.audio.samplerate} Hz")
    print(f"  Spectrogram: {height}H × {time_bins}W bins")
    print(f"  After resize ({resize_factor}x): {final_height}H × {final_width}W")
    print(f"  Model expects: {model_input_height}H")
    print(f"  Batch size: {batch_size}")
    
    # Create dummy input to trace through model - use model's expected height
    dummy_input = torch.randn(batch_size, 1, model_input_height, final_width)
    
    # Calculate model parameter memory
    total_params = sum(p.numel() for p in detector.parameters())
    param_memory_mb = (total_params * 4) / (1024 * 1024)  # float32
    param_memory_int8_mb = (total_params * 1) / (1024 * 1024)  # int8
    
    print(f"\nModel parameters: {total_params:,}")
    print(f"Parameter memory: {param_memory_mb:.3f} MB (float32) | {param_memory_int8_mb:.3f} MB (int8)")
    
    # Determine if U-Net or Autoencoder
    has_skip = config.model.decoder.use_skip if hasattr(config.model.decoder, 'use_skip') else False
    
    # Run MCU profiler
    print(f"\nArchitecture: {'U-Net (skip connections)' if has_skip else 'Autoencoder (no skip)'}")
    profiler = DeepMCUProfiler(detector, input_size=dummy_input.shape, is_unet=has_skip)
    activation_memory_int8_mb, layer_stats = profiler.profile()
    activation_memory_mb = activation_memory_int8_mb * 4  # float32 would be 4x
    
    total_memory_mb = param_memory_mb + activation_memory_mb
    total_memory_int8_mb = param_memory_int8_mb + activation_memory_int8_mb
    
    print(f"\n{'='*70}")
    print("MEMORY BREAKDOWN")
    print(f"{'='*70}")
    print(f"Model parameters:     {param_memory_mb:.3f} MB (f32) | {param_memory_int8_mb:.3f} MB (i8)")
    print(f"Peak activations:     {activation_memory_mb:.3f} MB (f32) | {activation_memory_int8_mb:.3f} MB (i8)")
    print(f"Total (training):     {total_memory_mb:.3f} MB (f32) | {total_memory_int8_mb:.3f} MB (i8)")
    
    print(f"\n{'='*70}")
    print("MICROCONTROLLER DEPLOYMENT (inference only)")
    print(f"{'='*70}")
    print(f"FLASH (parameters):   {param_memory_int8_mb:.3f} MB (INT8 quantized)")
    print(f"RAM (activations):    {activation_memory_int8_mb:.3f} MB = {activation_memory_int8_mb*1024:.0f} KB")
    
    # Also show bottleneck type
    bottleneck_layers = config.model.bottleneck.layers if hasattr(config.model.bottleneck, 'layers') else []
    bottleneck_type = bottleneck_layers[0].name if bottleneck_layers else 'unknown'
    print(f"\nBottleneck type: {bottleneck_type}")
    has_skip = config.model.decoder.use_skip if hasattr(config.model.decoder, 'use_skip') else False
    print(f"Skip connections: {'YES' if has_skip else 'NO'}")
    
    return {
        'config_path': config_path,
        'use_skip': has_skip,
        'bottleneck_type': bottleneck_type,
        'param_mb': param_memory_mb,
        'param_mb_int8': param_memory_int8_mb,
        'activation_mb': activation_memory_mb,
        'activation_mb_int8': activation_memory_int8_mb,
        'total_mb': total_memory_mb,
        'total_mb_int8': total_memory_int8_mb,
        'layer_stats': layer_stats,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config1', default='example_data/config.yaml', help='Config with genus (no skip)')
    parser.add_argument('--config2', default='example_data/config_baseline.yaml', help='Baseline config (no skip)')
    args = parser.parse_args()
    
    results = []
    
    for config_path in [args.config1, args.config2]:
        result = trace_model_memory(config_path)
        if result:
            results.append(result)
    
    # Compare results
    if len(results) >= 1:
        print(f"\n{'='*70}")
        print("MEMORY COMPARISON")
        print(f"{'='*70}")
        
        for i, result in enumerate(results, 1):
            config_name = Path(result['config_path']).stem
            print(f"\nConfig {i}: {config_name}")
            print(f"  Bottleneck: {result['bottleneck_type']}")
            print(f"  Skip connections: {'YES' if result['use_skip'] else 'NO'}")
            print(f"  FLASH (params):  {result['param_mb_int8']:.3f} MB = {result['param_mb_int8']*1024:.0f} KB")
            print(f"  RAM (activations): {result['activation_mb_int8']:.3f} MB = {result['activation_mb_int8']*1024:.0f} KB")
        
        print(f"\n{'='*70}")
        
        # Create comparison plot if we have 2 results
        if len(results) == 2:
            plot_memory_comparison(results[0], results[1])
