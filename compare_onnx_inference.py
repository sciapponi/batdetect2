#!/usr/bin/env python3
"""Compare ONNX inference time between different model configurations."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from batdetect2.config import validate_config
from batdetect2.models import build_model
from batdetect2.targets import build_targets


def export_model(config_path: Path, output_path: Path):
    """Export a model to ONNX format."""
    print(f"\nLoading config from: {config_path}")
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    
    config = validate_config(config_dict)
    
    # Create export directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build targets and model
    targets = build_targets(config=config.targets)
    model = build_model(config=config.model, targets=targets)
    model.eval()
    
    detector = model.detector
    num_params = sum(p.numel() for p in detector.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 1, 128, 400)
    
    # Test forward pass
    with torch.no_grad():
        output = detector(dummy_input)
    
    # Prepare output names
    output_names = ['detection_probs', 'size_preds', 'class_probs', 'features']
    if output.genus_probs is not None:
        output_names.append('genus_probs')
    
    # Export to ONNX using legacy exporter
    print(f"Exporting to: {output_path}")
    with torch.onnx.select_model_mode_for_export(detector, torch.onnx.TrainingMode.EVAL):
        torch.onnx.export(
            detector,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['spectrogram'],
            output_names=output_names,
            dynamic_axes={
                'spectrogram': {0: 'batch_size', 3: 'time'},
                'detection_probs': {0: 'batch_size', 3: 'time'},
                'size_preds': {0: 'batch_size', 3: 'time'},
                'class_probs': {0: 'batch_size', 3: 'time'},
                'features': {0: 'batch_size', 3: 'time'},
            },
            dynamo=False,
        )
    print(f"✓ Exported successfully")
    
    return num_params


def benchmark_onnx_inference(onnx_path: Path, warmup_runs: int = 10, benchmark_runs: int = 100):
    """Benchmark ONNX inference time."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime is not installed!")
        print("Please install it with: uv pip install onnxruntime")
        return None
    
    print(f"\nBenchmarking: {onnx_path.name}")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    
    # Create random input data
    input_data = np.random.randn(1, 1, 128, 400).astype(np.float32)
    
    # Warmup runs
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        session.run(None, {input_name: input_data})
    
    # Benchmark runs
    print(f"Running {benchmark_runs} benchmark iterations...")
    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        session.run(None, {input_name: input_data})
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    times = np.array(times)
    
    results = {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare ONNX inference time between model configurations"
    )
    parser.add_argument(
        "--config1",
        type=Path,
        default=Path("example_data/config_half_the_size.yaml"),
        help="First config file (default: example_data/config_half_the_size.yaml)"
    )
    parser.add_argument(
        "--config2",
        type=Path,
        default=Path("example_data/config_with_skips.yaml"),
        help="Second config file (default: example_data/config_with_skips.yaml)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export models without benchmarking"
    )
    
    args = parser.parse_args()
    
    # Export models
    print("=" * 70)
    print("EXPORTING MODELS")
    print("=" * 70)
    
    onnx1_path = Path("export") / f"{args.config1.stem}.onnx"
    onnx2_path = Path("export") / f"{args.config2.stem}.onnx"
    
    params1 = export_model(args.config1, onnx1_path)
    params2 = export_model(args.config2, onnx2_path)
    
    print("\n" + "=" * 70)
    print("MODEL SIZES")
    print("=" * 70)
    print(f"{args.config1.name:50s} {params1:>12,} params")
    print(f"{args.config2.name:50s} {params2:>12,} params")
    print(f"Ratio: {params2/params1:.2f}x")
    
    if args.export_only:
        print("\n✓ Export complete. Use --runs to benchmark inference time.")
        return
    
    # Benchmark inference
    print("\n" + "=" * 70)
    print("BENCHMARKING ONNX INFERENCE")
    print("=" * 70)
    
    results1 = benchmark_onnx_inference(onnx1_path, args.warmup, args.runs)
    results2 = benchmark_onnx_inference(onnx2_path, args.warmup, args.runs)
    
    if results1 is None or results2 is None:
        return
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{args.config1.name}:")
    print(f"  Mean:   {results1['mean']:6.2f} ms  (± {results1['std']:5.2f} ms)")
    print(f"  Median: {results1['median']:6.2f} ms")
    print(f"  Min:    {results1['min']:6.2f} ms")
    print(f"  Max:    {results1['max']:6.2f} ms")
    print(f"  P95:    {results1['p95']:6.2f} ms")
    print(f"  P99:    {results1['p99']:6.2f} ms")
    
    print(f"\n{args.config2.name}:")
    print(f"  Mean:   {results2['mean']:6.2f} ms  (± {results2['std']:5.2f} ms)")
    print(f"  Median: {results2['median']:6.2f} ms")
    print(f"  Min:    {results2['min']:6.2f} ms")
    print(f"  Max:    {results2['max']:6.2f} ms")
    print(f"  P95:    {results2['p95']:6.2f} ms")
    print(f"  P99:    {results2['p99']:6.2f} ms")
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    speedup = results2['mean'] / results1['mean']
    
    if speedup > 1:
        print(f"{args.config1.name} is {speedup:.2f}x FASTER")
    else:
        print(f"{args.config2.name} is {1/speedup:.2f}x FASTER")
    
    print(f"\nMean difference: {abs(results2['mean'] - results1['mean']):.2f} ms")
    print(f"Parameter ratio: {params2/params1:.2f}x")
    print(f"Speed ratio:     {speedup:.2f}x")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
