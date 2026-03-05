"""
Analyze how different clip durations affect VAD model performance.

This script evaluates a trained 2-class VAD model using various clip durations
and plots how metrics (FP rate, precision, recall, etc.) change.
"""
import argparse
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from loguru import logger
from tqdm import tqdm

from batdetect2.audio import build_audio_loader
from batdetect2.models import VADConfig, build_vad_model
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor


def sliding_window_predict(
    audio_path: Path,
    model: torch.nn.Module,
    preprocessor,
    clip_duration: float = 0.2,
    stride: float = 0.1,
    device: str = "cpu"
) -> List[dict]:
    """Run sliding window prediction on an audio file."""
    # Load audio
    wav, sr = sf.read(str(audio_path))
    duration = len(wav) / sr
    
    # Convert to mono if stereo
    if len(wav.shape) > 1:
        wav = wav.mean(axis=1)
    
    predictions = []
    
    # Sliding window
    start_time = 0.0
    sample_duration = int(clip_duration * sr)
    stride_samples = int(stride * sr)
    
    while start_time + clip_duration <= duration:
        # Extract clip
        start_sample = int(start_time * sr)
        end_sample = start_sample + sample_duration
        clip_wav = wav[start_sample:end_sample]
        
        # Convert to tensor
        wav_tensor = torch.tensor(clip_wav, dtype=torch.float32).unsqueeze(0)
        
        # Preprocess
        spec = preprocessor(wav_tensor).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            features = model.encoder(spec)
            bottleneck = features if isinstance(features, torch.Tensor) else features[-1]
            logits = model.head(bottleneck)
            probs = torch.softmax(logits, dim=1)
        
        # Get bat probability (class 1)
        bat_prob = probs[0, 1].item()
        pred_class = torch.argmax(probs, dim=1).item()
        
        predictions.append({
            'start_time': start_time,
            'end_time': start_time + clip_duration,
            'bat_prob': bat_prob,
            'pred_class': pred_class,
            'is_bat': pred_class == 1
        })
        
        start_time += stride
    
    return predictions


def evaluate_non_bat(
    model: torch.nn.Module,
    preprocessor,
    non_bat_dir: Path,
    clip_duration: float,
    stride: float,
    device: str
) -> Dict:
    """Evaluate model on non-bat data and return metrics."""
    audio_files = list(non_bat_dir.glob("*.wav")) + list(non_bat_dir.glob("*.WAV"))
    
    if not audio_files:
        return None
    
    model.eval()
    
    total_windows = 0
    total_false_positives = 0
    all_probs = []
    errors = 0
    
    for audio_file in tqdm(audio_files, desc=f"Non-bat (clip={clip_duration}s)", leave=False):
        try:
            predictions = sliding_window_predict(
                audio_file,
                model,
                preprocessor,
                clip_duration=clip_duration,
                stride=stride,
                device=device
            )
            
            fp_count = sum(1 for p in predictions if p['is_bat'])
            total_windows += len(predictions)
            total_false_positives += fp_count
            all_probs.extend([p['bat_prob'] for p in predictions])
            
        except Exception as e:
            errors += 1
            if errors <= 3:  # Only log first few errors
                logger.warning(f"Error processing {audio_file.name}: {e}")
            continue
    
    if errors > 3:
        logger.warning(f"Total errors: {errors} files failed")
    
    # If too many errors, this clip duration doesn't work
    if errors > len(audio_files) * 0.5:
        logger.error(f"Clip duration {clip_duration}s failed on >50% of files - skipping")
        return None
    
    fp_rate = total_false_positives / total_windows if total_windows > 0 else 0.0
    
    return {
        'total_windows': total_windows,
        'false_positives': total_false_positives,
        'fp_rate': fp_rate,
        'mean_prob': np.mean(all_probs) if all_probs else 0,
        'median_prob': np.median(all_probs) if all_probs else 0,
        'max_prob': np.max(all_probs) if all_probs else 0,
        'errors': errors,
    }


def analyze_clip_durations(
    model: torch.nn.Module,
    preprocessor,
    non_bat_dir: Path,
    clip_durations: List[float],
    stride: float,
    device: str,
    output_csv: str,
    output_plot: str
):
    """Analyze performance across different clip durations."""
    
    results = []
    
    logger.info(f"Testing {len(clip_durations)} different clip durations...")
    
    for clip_duration in clip_durations:
        logger.info(f"\nEvaluating with clip_duration={clip_duration}s")
        
        # Evaluate on non-bat data
        non_bat_metrics = evaluate_non_bat(
            model, preprocessor, non_bat_dir, clip_duration, stride, device
        )
        
        if non_bat_metrics:
            result = {
                'clip_duration': clip_duration,
                **non_bat_metrics
            }
            results.append(result)
            
            logger.info(f"  FP Rate: {non_bat_metrics['fp_rate']:.2%}")
            logger.info(f"  Mean Prob: {non_bat_metrics['mean_prob']:.3f}")
            logger.info(f"  Total Windows: {non_bat_metrics['total_windows']}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"\nResults saved to {output_csv}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('VAD Performance vs Clip Duration', fontsize=16, fontweight='bold')
    
    # Plot 1: False Positive Rate
    ax = axes[0, 0]
    ax.plot(df['clip_duration'], df['fp_rate'] * 100, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('Clip Duration (seconds)', fontsize=12)
    ax.set_ylabel('False Positive Rate (%)', fontsize=12)
    ax.set_title('False Positive Rate vs Clip Duration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Plot 2: Mean Probability
    ax = axes[0, 1]
    ax.plot(df['clip_duration'], df['mean_prob'], 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.set_xlabel('Clip Duration (seconds)', fontsize=12)
    ax.set_ylabel('Mean Bat Probability', fontsize=12)
    ax.set_title('Mean Probability vs Clip Duration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # Plot 3: Max Probability
    ax = axes[1, 0]
    ax.plot(df['clip_duration'], df['max_prob'], 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax.set_xlabel('Clip Duration (seconds)', fontsize=12)
    ax.set_ylabel('Max Bat Probability', fontsize=12)
    ax.set_title('Max Probability vs Clip Duration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim([0, 1])
    
    # Plot 4: Total Windows Analyzed
    ax = axes[1, 1]
    ax.plot(df['clip_duration'], df['total_windows'], 'o-', linewidth=2, markersize=8, color='#9b59b6')
    ax.set_xlabel('Clip Duration (seconds)', fontsize=12)
    ax.set_ylabel('Total Windows Analyzed', fontsize=12)
    ax.set_title('Number of Windows vs Clip Duration', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    logger.info(f"Plots saved to {output_plot}")
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("SUMMARY TABLE")
    logger.info("="*80)
    logger.info(f"{'Duration (s)':<12} {'FP Rate':<10} {'Mean Prob':<12} {'Max Prob':<12} {'Windows':<10}")
    logger.info("-"*80)
    for _, row in df.iterrows():
        logger.info(f"{row['clip_duration']:<12.2f} "
                   f"{row['fp_rate']*100:<10.2f} "
                   f"{row['mean_prob']:<12.3f} "
                   f"{row['max_prob']:<12.3f} "
                   f"{int(row['total_windows']):<10d}")
    
    # Find optimal duration (lowest FP rate)
    optimal_idx = df['fp_rate'].idxmin()
    optimal = df.loc[optimal_idx]
    logger.info("\n" + "="*80)
    logger.info(f"OPTIMAL CLIP DURATION: {optimal['clip_duration']:.2f}s")
    logger.info(f"  FP Rate: {optimal['fp_rate']*100:.2f}%")
    logger.info(f"  Mean Prob: {optimal['mean_prob']:.3f}")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Analyze VAD performance vs clip duration")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pt file)")
    parser.add_argument("--non-bat-dir", type=str, required=True, help="Directory containing non-bat audio files")
    parser.add_argument("--clip-durations", type=str, default="0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0", 
                       help="Comma-separated list of clip durations to test (default: 0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0)")
    parser.add_argument("--stride", type=float, default=0.1, help="Stride for sliding window in seconds")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu/cuda)")
    parser.add_argument("--output-csv", type=str, default="clip_duration_analysis.csv", help="Output CSV file")
    parser.add_argument("--output-plot", type=str, default="clip_duration_analysis.png", help="Output plot file")
    args = parser.parse_args()
    
    # Parse clip durations
    clip_durations = [float(d.strip()) for d in args.clip_durations.split(',')]
    logger.info(f"Testing clip durations: {clip_durations}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    
    config_dict = checkpoint['config_dict']
    vad_config = checkpoint['config']
    
    # Build model
    model = build_vad_model(vad_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    logger.info(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Build preprocessor
    samplerate = config_dict.get("audio", {}).get("samplerate", 256000)
    preproc_config = None
    if "preprocess" in config_dict:
        preproc_config = PreprocessingConfig(**config_dict["preprocess"])
    
    preprocessor = build_preprocessor(config=preproc_config, input_samplerate=samplerate)
    
    # Check directories exist
    non_bat_dir = Path(args.non_bat_dir)
    if not non_bat_dir.exists():
        logger.error(f"Directory not found: {non_bat_dir}")
        return
    
    # Run analysis
    analyze_clip_durations(
        model,
        preprocessor,
        non_bat_dir,
        clip_durations,
        args.stride,
        args.device,
        args.output_csv,
        args.output_plot
    )


if __name__ == "__main__":
    main()
