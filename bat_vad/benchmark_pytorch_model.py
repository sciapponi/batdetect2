"""
Run a PyTorch VAD checkpoint on a validation set and report metrics.
Tests the model against an int8 statically quantized version of itself.
Now quantizes convolutions as well.
"""
import copy
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.quantization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from loguru import logger
import csv
import os
from batdetect2.data import load_dataset_from_config
from batdetect2.audio import build_audio_loader
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.models import VADConfig, build_vad_model


try:
    import onnx
    from onnxsim import simplify
    from esp_ppq.api import espdl_quantize_onnx
    from esp_ppq.api import espdl_quantize_torch
    from esp_ppq.executor import TorchExecutor
    HAS_ESPDL = True
except ImportError:
    logger.warning("esp-ppq not found. ESP-DL simulation will be skipped.")
    HAS_ESPDL = False



# ----------------------------------------------------------------------
# Wrapper to insert QuantStub / DeQuantStub for static quantization
# ----------------------------------------------------------------------
class VADModelWrapper(torch.nn.Module):
    """Wraps encoder and head, adds quantization stubs, and handles
    possible multi‑output from encoder (e.g., feature pyramids)."""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        features = self.encoder(x)
        # If encoder returns a list/tuple (e.g., multiple feature maps),
        # use the last one as the bottleneck (same as original code).
        if isinstance(features, (list, tuple)):
            features = features[-1]
        x = self.head(features)
        x = self.dequant(x)
        return x

#ESPDL

def espdl_collate_fn(batch):
    """Matches your reference code's collate logic."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        return batch[0]
    return batch

def run_espdl_simulated_model(fp32_model, dataloader, val_loader_for_calib, quiet=False, calib_steps=32, 
                             non_bat_loader=None, mixed_calibration=False, clip_outliers=False, outlier_percentile=1.0,
                             keep_fp32_layers=""):
    """Export the provided PyTorch model to ONNX, quantize it with ESP-PPQ,
    and run the resulting fixed‑point graph on ``dataloader``.

    ``quiet`` disables PPQ/ESP-DL error reporting and verbose output,
    which can be useful when you just care about accuracy metrics and
    want to avoid the big noise‑to‑signal tables cluttering the log.

    Returns a tuple ``(labels, probs, executor, run_executor)`` where
    ``executor`` is the :class:`~esp_ppq.executor.TorchExecutor` instance and
    ``run_executor`` is a small helper that takes a single-batch input tensor
    and returns the raw logits (already dequantized).
    """
    fp32_model.eval().cpu()
    
    # ensure export directory exists (caller should have created it too)
    export_dir = Path("export")
    export_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get shape from data
    sample_batch = next(iter(dataloader))
    sample_tensor = sample_batch["spec"].cpu()
    input_shape = [1] + list(sample_tensor.shape[1:])
    dummy_input = torch.zeros(input_shape) # Using zeros for a clean trace

    onnx_path = str(export_dir / "temp_model.onnx")
    
    # 2. Export - we don't want the softmax baked into the graph because
    #    quantizing Softmax often collapses the outputs to zero.  Instead we
    #    export a tiny wrapper that returns *logits*; softmax will be applied
    #    later in Python exactly the same way we do for the FP32/PyTorch path.
    logger.info(f"Exporting ONNX with shape {input_shape}...")

    # wrapper that only runs encoder+head and returns raw logits
    class _LogitWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.encoder = base_model.encoder
            self.head = base_model.head
        def forward(self, x):
            x = self.encoder(x)
            if isinstance(x, (list, tuple)):
                x = x[-1]
            return self.head(x)

    logit_model = _LogitWrapper(fp32_model)
    logit_model.eval()

    with torch.no_grad():
        torch.onnx.export(
            logit_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13, # 13 is usually a safe middle ground
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )

    # 3. Simplify AND Prune (This kills the 'zeros' KeyError)
    model_onnx = onnx.load(onnx_path)
    model_simp, check = simplify(model_onnx)
    
    # --- MANUAL PRUNE START ---
    # We only want the node named 'output' in our graph outputs
    valid_outputs = [out for out in model_simp.graph.output if out.name == 'output']
    
    # Clear and reset outputs
    while len(model_simp.graph.output) > 0:
        model_simp.graph.output.pop()
    model_simp.graph.output.extend(valid_outputs)
    # --- MANUAL PRUNE END ---

    onnx.save(model_simp, onnx_path)
    logger.info("ONNX simplified and pruned of orphaned outputs.")

    # 4. Prepare Calibration Tensors - SIMPLE approach with debugging
    calib_tensors = []
    
    # Simple approach: just collect more samples and log what we're getting
    from itertools import cycle
    loader_iter = cycle(val_loader_for_calib)
    
    bat_count = 0
    non_bat_count = 0
    all_min_vals = []
    all_max_vals = []
    
    logger.info("Collecting calibration samples (simple approach)...")
    
    # Collect samples first
    raw_calib_tensors = []
    for i in range(calib_steps):
        batch = next(loader_iter)
        spec = batch["spec"].cpu()
        label = batch["label"].cpu()
        
        # Ensure single sample
        if spec.shape[0] > 1:
            spec = spec[0:1]
            label = label[0:1]
        
        raw_calib_tensors.append((spec, label))
        
        # Track value ranges for debugging
        min_val = spec.min().item()
        max_val = spec.max().item()
        all_min_vals.append(min_val)
        all_max_vals.append(max_val)
    
    # Apply outlier clipping if requested
    if clip_outliers and len(all_min_vals) > 10:
        logger.info("Applying outlier clipping to calibration data...")
        
        # Calculate percentile thresholds
        sorted_mins = sorted(all_min_vals)
        sorted_maxs = sorted(all_max_vals)
        
        low_percentile = outlier_percentile
        high_percentile = 100.0 - outlier_percentile
        
        low_idx = int(len(sorted_mins) * low_percentile / 100.0)
        high_idx = int(len(sorted_maxs) * high_percentile / 100.0)
        
        min_threshold = sorted_mins[low_idx]
        max_threshold = sorted_maxs[high_idx]
        
        logger.info(f"  Clipping thresholds: [{min_threshold:.6f}, {max_threshold:.6f}]")
        logger.info(f"  Original range: [{min(all_min_vals):.6f}, {max(all_max_vals):.6f}]")
        
        # Apply clipping and rebuild tensors
        clipped_count = 0
        for spec, label in raw_calib_tensors:
            # Clip the tensor values
            clipped_spec = torch.clamp(spec, min_threshold, max_threshold)
            
            if not torch.equal(spec, clipped_spec):
                clipped_count += 1
            
            calib_tensors.append(clipped_spec)
            
            # Track class balance
            if label.item() == 1:
                bat_count += 1
            else:
                non_bat_count += 1
        
        logger.info(f"  Clipped {clipped_count} out of {len(raw_calib_tensors)} samples")
        
        # Recalculate stats after clipping
        all_min_vals = [t.min().item() for t in calib_tensors]
        all_max_vals = [t.max().item() for t in calib_tensors]
        
    else:
        # No clipping, just use raw tensors
        for spec, label in raw_calib_tensors:
            calib_tensors.append(spec)
            
            # Track class balance
            if label.item() == 1:
                bat_count += 1
            else:
                non_bat_count += 1
    
    # Log calibration statistics for debugging
    global_min = min(all_min_vals)
    global_max = max(all_max_vals)
    mean_min = sum(all_min_vals) / len(all_min_vals)
    mean_max = sum(all_max_vals) / len(all_max_vals)
    
    logger.info(f"Final calibration stats:")
    logger.info(f"  Samples: {len(calib_tensors)} ({bat_count} bat, {non_bat_count} non-bat)")
    logger.info(f"  Value range: [{global_min:.6f}, {global_max:.6f}]")
    logger.info(f"  Average range: [{mean_min:.6f}, {mean_max:.6f}]")
    
    # Sort to find percentiles
    sorted_mins = sorted(all_min_vals)
    sorted_maxs = sorted(all_max_vals)
    p1_min = sorted_mins[len(sorted_mins) // 100] if len(sorted_mins) > 100 else sorted_mins[0]
    p99_max = sorted_maxs[99 * len(sorted_maxs) // 100] if len(sorted_maxs) > 100 else sorted_maxs[-1]
    
    logger.info(f"  1st percentile min: {p1_min:.6f}")
    logger.info(f"  99th percentile max: {p99_max:.6f}")
    
    # Check for outliers that might be causing problems
    outlier_threshold = 3 * (mean_max - mean_min) if mean_max != mean_min else 1.0
    outliers = [(i, val) for i, val in enumerate(all_max_vals) if abs(val - mean_max) > outlier_threshold]
    if outliers:
        logger.warning(f"Still have {len(outliers)} potential outlier samples")
        for i, val in outliers[:3]:  # Show first 3
            logger.warning(f"  Sample {i}: max_val = {val:.6f}")
    
    # no warning needed now – we always fill the list by cycling
    # no warning needed now – we always fill the list by cycling

    # 5. Quantize with problematic layers in FP32 if requested 
    fp32_layer_list = []
    if keep_fp32_layers:
        fp32_layer_list = [layer.strip() for layer in keep_fp32_layers.split(",") if layer.strip()]
        logger.info(f"Keeping layers in FP32: {fp32_layer_list}")
    
    # Add debugging hook to capture activation ranges during calibration
    activation_stats = {}
    
    def debug_hook(module, input, output, name):
        """Hook to capture activation statistics during calibration"""
        if hasattr(output, 'data'):
            data = output.data
        else:
            data = output
            
        if isinstance(data, torch.Tensor):
            stats = {
                'min': data.min().item(),
                'max': data.max().item(),
                'mean': data.mean().item(),
                'std': data.std().item(),
                'shape': list(data.shape)
            }
            
            if name not in activation_stats:
                activation_stats[name] = []
            activation_stats[name].append(stats)
    
    # Register hooks on the model for debugging (only if not quiet)
    hooks = []
    if not quiet:
        for name, module in fp32_model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hook = module.register_forward_hook(lambda m, i, o, n=name: debug_hook(m, i, o, n))
                hooks.append(hook)
    
    # Run a few samples through the model to collect activation stats
    if not quiet and len(calib_tensors) > 0:
        logger.info("Analyzing activation ranges during calibration...")
        fp32_model.eval()
        with torch.no_grad():
            for i in range(min(10, len(calib_tensors))):  # Check first 10 samples
                sample = calib_tensors[i]
                if sample.dim() == 3:
                    sample = sample.unsqueeze(0)
                _ = fp32_model(sample)
        
        # Log activation statistics for problematic layers
        for layer_name in ['encoder.conv1', 'encoder.conv2', 'head.linear', 'head']:
            if layer_name in activation_stats:
                stats_list = activation_stats[layer_name]
                if stats_list:
                    min_vals = [s['min'] for s in stats_list]
                    max_vals = [s['max'] for s in stats_list]
                    logger.info(f"  {layer_name}: range [{min(min_vals):.6f}, {max(max_vals):.6f}], "
                               f"avg_range [{sum(min_vals)/len(min_vals):.6f}, {sum(max_vals)/len(max_vals):.6f}]")
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
    
    # Prepare quantization settings
    quantization_settings = {
        "target": "esp32s3",
        "num_of_bits": 8,
        "device": "cpu",
        "error_report": not quiet,
        "verbose": 0 if quiet else 1,
    }
    
    # Add FP32 layer settings if specified
    if fp32_layer_list:
        quantization_settings["fp32_layers"] = fp32_layer_list
        logger.info(f"Quantization will keep {len(fp32_layer_list)} layers in FP32")
    
    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=str(Path("export") / "vad_final.espdl"),
        calib_dataloader=calib_tensors,
        calib_steps=len(calib_tensors),
        input_shape=[input_shape],
        collate_fn=lambda x: x,
        **quantization_settings
    )
    executor = TorchExecutor(graph=quant_ppq_graph)
    
    # helper that runs the executor and dequantizes results
    def run_executor(inp: torch.Tensor):
        # call with explicit output names, this avoids ambiguity if the
        # quantized graph has multiple outputs (some hidden initializers etc.)
        out = executor(inp, output_names=['output'])
        if isinstance(out, list):
            out = out[0]
        # PPQ sometimes returns a quantized tensor; convert to float
        if hasattr(out, 'dtype') and out.dtype in (torch.qint8, torch.quint8):
            out = out.dequantize()
        return out

    all_labels = []
    all_probs = []
    
    logger.info("Running ESP-DL simulation on validation set...")

    for batch in dataloader:
        # 1. Extract the spectrogram and label
        spec = batch["spec"]
        label = batch["label"]

        # 2. ESP-DL executor expects [1, C, H, W]. 
        # If your batch has 32 samples, we must iterate through them.
        if spec.shape[0] > 1:
            for i in range(spec.shape[0]):
                single_spec = spec[i:i+1] # Keep the batch dim as 1

                # Ensure it's a tensor
                if isinstance(single_spec, np.ndarray):
                    single_spec = torch.from_numpy(single_spec).float()

                output = run_executor(single_spec.cpu())

                # 3. Handle logits -> probabilities
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs.detach().cpu().numpy())
                all_labels.append(label[i].item())
        else:
            # Already batch size 1
            if isinstance(spec, np.ndarray):
                spec = torch.from_numpy(spec).float()

            output = run_executor(spec.cpu())
            probs = torch.softmax(output, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(label.item())

    return np.array(all_labels), np.vstack(all_probs), executor, run_executor

# ----------------------------------------------------------------------
# Dataset and collate (unchanged)
# ----------------------------------------------------------------------
class TwoClassVADDataset(torch.utils.data.Dataset):
    # ... (identical to original, omitted for brevity)
    def __init__(self, bat_annotations, non_bat_annotations, audio_loader, preprocessor, clip_duration=0.2, samples_per_epoch=1000, positive_ratio=0.5, return_wav=False):
        self.bat_annotations = bat_annotations
        self.non_bat_annotations = non_bat_annotations
        self.audio_loader = audio_loader
        self.preprocessor = preprocessor
        self.clip_duration = clip_duration
        self.samples_per_epoch = samples_per_epoch
        self.positive_ratio = positive_ratio
        self.return_wav = return_wav
        self.bat_recording_anns = {}
        for ann in bat_annotations:
            rec_id = ann.clip.recording.uuid
            if rec_id not in self.bat_recording_anns:
                self.bat_recording_anns[rec_id] = {"recording": ann.clip.recording, "events": []}
            for event in ann.sound_events:
                self.bat_recording_anns[rec_id]["events"].append(event)
        self.bat_recording_ids = list(self.bat_recording_anns.keys())
        self.non_bat_recordings = []
        if non_bat_annotations:
            for ann in non_bat_annotations:
                rec = ann.clip.recording
                if rec not in self.non_bat_recordings:
                    self.non_bat_recordings.append(rec)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_positive(self):
        if not self.bat_recording_ids:
            return None
        valid_recs = [r for r in self.bat_recording_ids if self.bat_recording_anns[r]["events"]]
        if not valid_recs:
            return None
        rec_id = random.choice(valid_recs)
        rec_data = self.bat_recording_anns[rec_id]
        event = random.choice(rec_data["events"])
        recording = rec_data["recording"]
        from soundevent import geometry
        event_start, _, event_end, _ = geometry.compute_bounds(event.sound_event.geometry)
        event_center = (event_start + event_end) / 2
        half_clip = self.clip_duration / 2
        start_time = event_center - half_clip + random.uniform(-0.05, 0.05)
        start_time = max(0.0, min(start_time, recording.duration - self.clip_duration))
        return recording, start_time, 1

    def _sample_negative_from_bat_recordings(self):
        if not self.bat_recording_ids:
            return None
        max_attempts = 10
        from soundevent import geometry
        for _ in range(max_attempts):
            rec_id = random.choice(self.bat_recording_ids)
            rec_data = self.bat_recording_anns[rec_id]
            recording = rec_data["recording"]
            duration = recording.duration
            if duration <= self.clip_duration:
                start_time = 0.0
            else:
                start_time = random.uniform(0, duration - self.clip_duration)
            end_time = start_time + self.clip_duration
            overlap = False
            for event in rec_data["events"]:
                e_start, _, e_end, _ = geometry.compute_bounds(event.sound_event.geometry)
                if max(start_time, e_start) < min(end_time, e_end):
                    overlap = True
                    break
            if not overlap:
                return recording, start_time, 0
        return None

    def _sample_negative_from_non_bat(self):
        if not self.non_bat_recordings:
            return None
        recording = random.choice(self.non_bat_recordings)
        duration = recording.duration
        if duration <= self.clip_duration:
            start_time = 0.0
        else:
            start_time = random.uniform(0, duration - self.clip_duration)
        return recording, start_time, 0

    def __getitem__(self, idx):
        is_positive = random.random() < self.positive_ratio
        max_attempts = 5
        sample = None
        for _ in range(max_attempts):
            if is_positive:
                sample = self._sample_positive()
            else:
                if self.non_bat_recordings and random.random() < 0.5:
                    sample = self._sample_negative_from_non_bat()
                else:
                    sample = self._sample_negative_from_bat_recordings()
            if sample is not None:
                break
            is_positive = not is_positive
        if sample is None:
            if not self.bat_recording_ids:
                raise RuntimeError("Dataset has no bat recordings to sample from, check your annotation paths")
            rec_id = random.choice(self.bat_recording_ids)
            recording = self.bat_recording_anns[rec_id]["recording"]
            start_time = 0.0
            label = 0
        else:
            recording, start_time, label = sample
        from soundevent import data
        clip = data.Clip(recording=recording, start_time=start_time, end_time=start_time + self.clip_duration)
        wav = self.audio_loader.load_clip(clip)
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        spec = self.preprocessor(wav_tensor)
        item = {"spec": spec, "label": torch.tensor(label, dtype=torch.long)}
        if self.return_wav:
            # retain original waveform for baseline gating
            item["wav"] = wav
        return item


def collate_fn(batch):
    max_width = max(item["spec"].shape[-1] for item in batch)
    specs = []
    labels = []
    for item in batch:
        spec = item["spec"]
        if spec.shape[-1] < max_width:
            pad_width = max_width - spec.shape[-1]
            spec = F.pad(spec, (0, pad_width), mode='constant', value=0)
        specs.append(spec)
        labels.append(item["label"])
    return {"spec": torch.stack(specs), "label": torch.stack(labels)}


# ----------------------------------------------------------------------
# Evaluation functions (one for original, one for quantized model)
# ----------------------------------------------------------------------
def run_pytorch_model(model, dataloader, device):
    """Run original model (encoder + head called separately)."""
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            spec = batch["spec"].to(device)
            label = batch["label"].to(device)
            features = model.encoder(spec)
            bottleneck = features if isinstance(features, torch.Tensor) else features[-1]
            logits = model.head(bottleneck)
            probs = torch.softmax(logits, dim=1)
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    return all_labels, all_probs


def run_quantized_model(model, dataloader, device):
    """Run quantized model (forward call returns logits directly)."""
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            spec = batch["spec"].to(device)
            label = batch["label"].to(device)
            logits = model(spec)               # quantized forward
            probs = torch.softmax(logits, dim=1)
            all_labels.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    return all_labels, all_probs



# ----------------------------------------------------------------------
# post‑processing utilities
# ----------------------------------------------------------------------

def apply_amp_gate(labels: np.ndarray, probs: np.ndarray, dataset, amp_thresh: float):
    """Modify ``probs`` by applying an amplitude gate using ``dataset``.

    ``dataset`` items must include a ``wav`` entry (i.e. ``return_wav=True``).
    For each sample, if the maximum absolute amplitude falls below
    ``amp_thresh`` we override the network probabilities with ``[1,0]``
    (negative prediction).

    Returns a tuple ``(new_labels, new_probs, stats)`` where ``stats`` is a
    dictionary containing the number of gated clips and the minimum/maximum
    amplitude observed. This helps users chose a sensible threshold.
    """
    gated = []
    min_amp = float("inf")
    max_amp = 0.0
    gated_count = 0
    for i in range(len(dataset)):
        item = dataset[i]
        wav = item.get("wav")
        if wav is None:
            gated.append(probs[i])
            continue
        if isinstance(wav, torch.Tensor):
            amp = wav.abs().max().item()
        else:
            amp = np.max(np.abs(wav))
        min_amp = min(min_amp, amp)
        max_amp = max(max_amp, amp)
        if amp < amp_thresh:
            gated.append(np.array([1.0, 0.0]))
            gated_count += 1
        else:
            gated.append(probs[i])
    stats = {"min_amp": min_amp if min_amp != float("inf") else 0.0,
             "max_amp": max_amp,
             "gated": gated_count,
             "total": len(dataset)}
    return labels, np.vstack(gated), stats


def compute_metrics(labels, probs):
    # ... (unchanged)
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        auc_roc = roc_auc_score(labels, probs[:, 1])
        avg_precision = average_precision_score(labels, probs[:, 1])
    except Exception:
        auc_roc = 0.0
        avg_precision = 0.0
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp + 1e-8) if (tn + fp) > 0 else 0.0
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "avg_precision": avg_precision,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(metrics, title="Metrics"):
    print(f"\n{'='*30} {title} {'='*30}")
    print(f"\n{'Metric':<18}{'Value':<10}")
    print("-" * 28)
    for k in ["acc", "precision", "recall", "f1", "auc_roc", "avg_precision", "specificity", "tp", "tn", "fp", "fn"]:
        print(f"{k:<18}{metrics[k]:<10.4f}")
    
    # Add class-wise breakdown for quantization debugging
    tn, fp, fn, tp = metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']
    
    print(f"\n{'Class Breakdown':<30}")
    print("-" * 40)
    
    # Non-bat class metrics
    total_actual_nonbat = tn + fp
    total_actual_bat = fn + tp
    
    if total_actual_nonbat > 0:
        nonbat_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        nonbat_recall = tn / total_actual_nonbat
        false_pos_rate = fp / total_actual_nonbat
        print(f"{'Non-bat precision:':<22}{nonbat_precision:<10.4f}")
        print(f"{'Non-bat recall:':<22}{nonbat_recall:<10.4f}")
        print(f"{'False positive rate:':<22}{false_pos_rate:<10.4f}")
    
    # Bat class metrics  
    if total_actual_bat > 0:
        bat_precision = tp / (fp + tp) if (fp + tp) > 0 else 0
        bat_recall = tp / total_actual_bat
        false_neg_rate = fn / total_actual_bat
        print(f"{'Bat precision:':<22}{bat_precision:<10.4f}")
        print(f"{'Bat recall:':<22}{bat_recall:<10.4f}")
        print(f"{'False negative rate:':<22}{false_neg_rate:<10.4f}")
    
    print(f"\n{'Confusion Matrix:':<30}")
    print(f"           Pred Non-bat  Pred Bat")
    print(f"Actual Non-bat:  {tn:4.0f}        {fp:4.0f}")
    print(f"Actual Bat:      {fn:4.0f}        {tp:4.0f}")


def append_metrics_csv(csv_path: str, model_name: str, metrics: dict):
    """Append a row of metrics to a CSV file.

    The file will be created with a header if it doesn't already exist.
    Columns are: model,acc,precision,recall,f1,auc_roc,avg_precision,specificity,tp,tn,fp,fn
    """
    header = ["model", "acc", "precision", "recall", "f1", "auc_roc",
              "avg_precision", "specificity", "tp", "tn", "fp", "fn"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        row = [model_name] + [metrics[k] for k in header if k != "model"]
        writer.writerow(row)


# ----------------------------------------------------------------------
# Baseline gating pipeline helpers
# ----------------------------------------------------------------------

def goertzel(samples: np.ndarray, samplerate: float, freq: float) -> float:
    """Return power at a single frequency using the Goertzel algorithm."""
    N = len(samples)
    if N == 0:
        return 0.0
    k = int(0.5 + (N * freq) / samplerate)
    omega = (2.0 * np.pi * k) / N
    coeff = 2.0 * np.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for x in samples:
        s = x + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2 ** 2 + s_prev ** 2 - coeff * s_prev * s_prev2
    return power


def baseline_classifier(wav: np.ndarray,
                        samplerate: float,
                        amp_thresh: float,
                        freqs: list[float],
                        goertzel_thresh: float) -> int:
    """Amplitude gate + Goertzel energy detector."""
    if np.max(np.abs(wav)) < amp_thresh:
        return 0
    energies = [goertzel(wav, samplerate, f) for f in freqs]
    if max(energies) > goertzel_thresh:
        return 1
    return 0


def run_baseline(dataset, samplerate, amp_thresh, freqs, goertzel_thresh):
    """Evaluate baseline detector on ``dataset`` (must return ``wav`` key)."""
    all_labels = []
    all_probs = []
    for i in range(len(dataset)):
        item = dataset[i]
        wav = item.get("wav")
        label = item["label"].item()
        pred = baseline_classifier(wav, samplerate, amp_thresh, freqs, goertzel_thresh)
        all_labels.append(label)
        all_probs.append([1.0 - pred, float(pred)])
    return np.array(all_labels), np.array(all_probs)


# ----------------------------------------------------------------------
# Static quantization (instead of dynamic)
# ----------------------------------------------------------------------
def quantize_model_static(model, calibration_loader):
    """
    Applies post‑training static quantization to the model.
    Quantizes both convolutional and linear layers.
    Expects model to be on CPU and in eval mode.
    """
    logger.info("Applying static quantization to INT8...")

    # Move model to CPU (required for static quantization)
    model = model.cpu()
    model.eval()

    # Wrap the model to insert QuantStub / DeQuantStub
    wrapped_model = VADModelWrapper(model.encoder, model.head)
    wrapped_model.eval()

    # ------------------------------------------------------------------
    # OPTIONAL FUSION: Uncomment and adjust the module paths to match
    # your encoder architecture. For example, if your encoder has
    # sequential blocks like: conv -> bn -> relu, you can fuse them.
    # torch.quantization.fuse_modules(
    #     wrapped_model,
    #     [['encoder.conv1', 'encoder.bn1', 'encoder.relu1']],  # example
    #     inplace=True
    # )
    # ------------------------------------------------------------------

    # Set quantization configuration for x86 CPUs
    wrapped_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare the model for calibration (inserts observers)
    torch.quantization.prepare(wrapped_model, inplace=True)

    # Calibrate with a subset of the validation data
    logger.info("Calibrating with validation data...")
    
    # Collect calibration samples with balanced class distribution
    calib_samples = []
    calib_labels = []
    min_vals, max_vals = [], []
    
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= 300:  # collect more samples for analysis
                break
            spec = batch["spec"]   # already on CPU after loader
            label = batch["label"]
            calib_samples.append(spec)
            calib_labels.append(label)
            min_vals.append(spec.min().item())
            max_vals.append(spec.max().item())
    
    if calib_samples:
        # Ensure balanced representation
        bat_indices = []
        non_bat_indices = []
        
        for i, label in enumerate(calib_labels):
            if label.shape[0] > 1:
                # Handle batch - take first sample
                label_val = label[0].item()
            else:
                label_val = label.item()
            
            if label_val == 1:
                bat_indices.append(i)
            else:
                non_bat_indices.append(i)
        
        logger.info(f"Static quantization - Available: {len(bat_indices)} bat, {len(non_bat_indices)} non-bat samples")
        
        # Balance the classes for calibration
        target_per_class = 50  # 50 samples per class
        import random
        random.shuffle(bat_indices)
        random.shuffle(non_bat_indices)
        
        selected_bat = bat_indices[:min(target_per_class, len(bat_indices))]
        selected_non_bat = non_bat_indices[:min(target_per_class, len(non_bat_indices))]
        
        # Run calibration on balanced samples
        bat_count = 0
        non_bat_count = 0
        
        for idx in selected_bat + selected_non_bat:
            spec = calib_samples[idx]
            label = calib_labels[idx]
            _ = wrapped_model(spec)
            
            if label.shape[0] > 1:
                label_val = label[0].item()
            else:
                label_val = label.item()
            
            if label_val == 1:
                bat_count += 1
            else:
                non_bat_count += 1
        
        logger.info(f"Static quantization calibration balance: {bat_count} bat, {non_bat_count} non-bat")
        logger.info(f"Static quantization range: [{min(min_vals):.3f}, {max(max_vals):.3f}]")
    
    # Fallback to original method if no samples collected
    if not calib_samples:
        for i, batch in enumerate(calibration_loader):
            if i >= 100:  # use first 100 batches for calibration
                break
            spec = batch["spec"]   # already on CPU after loader
            _ = wrapped_model(spec)

    # Convert to quantized model
    torch.quantization.convert(wrapped_model, inplace=True)
    logger.info("Static quantization complete.")

    return wrapped_model

def print_model_stats(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    dtype_counts = {}
    for p in model.parameters():
        dtype_counts[p.dtype] = dtype_counts.get(p.dtype, 0) + p.numel()
    print(f"\n{name} parameter stats:")
    print(f"  Total parameters: {total_params}")
    for dtype, count in dtype_counts.items():
        bytes_per = 2 if dtype == torch.qint8 else (1 if dtype == torch.quint8 else (4 if dtype == torch.float32 else '?'))
        print(f"  {dtype}: {count} params -> {count * bytes_per / 1e6:.2f} MB")

def print_state_dict_size(model, name):
    state_dict = model.state_dict()
    total_bytes = 0
    dtype_counts = {}
    print(f"\n{name} state_dict keys: {list(state_dict.keys())}")
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            dtype = value.dtype
            numel = value.numel()
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + numel
            total_bytes += numel * value.element_size()
        else:
            print(f"  Non-tensor entry: {key} ({type(value)})")
    print(f"\n{name} state_dict stats (tensors only):")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} elements")
    print(f"  Total bytes from tensors: {total_bytes} -> {total_bytes / 1e6:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch VAD checkpoint with static quantization")
    parser.add_argument("--config", type=str, required=True, help="Path to VAD config YAML")
    parser.add_argument("--bat-val", type=str, required=True, help="Path to bat val dataset yaml")
    parser.add_argument("--non-bat", type=str, help="Path to non-bat dataset yaml (optional)")
    parser.add_argument(
        "--non-bat-split-seed",
        type=int,
        default=None,
        help="Optional deterministic seed for splitting non-bat data. "
             "When omitted the list is sorted by path; provide an integer "
             "to shuffle reproducibly."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--clip-duration", type=float, default=0.2, help="Clip duration (s)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Ratio of positive (bat) samples")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--amp-gate", type=float, default=None,
                        help="If specified, apply an amplitude threshold on the raw waveform before passing to the CNN (negative if below)")
    # baseline options
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run traditional amplitude+Goertzel gating baseline on the same validation samples"
    )
    parser.add_argument("--amp-thresh", type=float, default=1e-3,
                        help="Amplitude threshold for baseline gate (absolute)")
    parser.add_argument("--goertzel-thresh", type=float, default=1e-6,
                        help="Energy threshold for Goertzel detector")
    parser.add_argument("--goertzel-freqs", type=float, nargs="+",
                        default=[25000.0, 50000.0],
                        help="Frequencies (Hz) used by the Goertzel filter")
    parser.add_argument("--espdl-examples", type=int, default=6,
                        help="Number of example clips to plot when ESP-DL simulation is run")
    parser.add_argument("--espdl-quiet", action="store_true",
                        help="Suppress verbose PPQ/ESP-DL quantization error tables")
    parser.add_argument("--espdl-calib-steps", type=int, default=512,
                        help="Number of batches from validation loader to use for ESP-DL calibration (more = better accuracy, slower)")
    parser.add_argument("--clip-outliers", action="store_true", 
                        help="Clip extreme outliers from calibration data (may help with quantization)")
    parser.add_argument("--outlier-percentile", type=float, default=1.0,
                        help="Percentile for outlier clipping (1.0 = clip to 1%-99% range)")
    parser.add_argument("--keep-fp32-layers", type=str, default="",
                        help="Comma-separated list of layer names to keep in FP32 (e.g. 'node_linear,node_conv2d_2')")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    model_config = config_dict.get("model", config_dict)
    model_config["type"] = "vad"
    if model_config.get("out_channels", 1) != 2:
        model_config["out_channels"] = 2
    vad_config = VADConfig(**model_config)
    samplerate = config_dict.get("audio", {}).get("samplerate", 256000)
    audio_loader = build_audio_loader()
    preproc_config = None
    if "preprocess" in config_dict:
        preproc_config = PreprocessingConfig(**config_dict["preprocess"])
    preprocessor = build_preprocessor(config=preproc_config, input_samplerate=samplerate)

    bat_val_anns = load_dataset_from_config(args.bat_val)
    non_bat_val_anns = []
    if args.non_bat:
        from train_vad_2class import load_non_bat_dataset
        all_non_bat_anns = load_non_bat_dataset(args.non_bat)
        # deterministic split like training script
        if args.non_bat_split_seed is not None:
            rng = random.Random(args.non_bat_split_seed)
            rng.shuffle(all_non_bat_anns)
        else:
            all_non_bat_anns.sort(key=lambda ann: str(ann.clip.recording.path))
        split_idx = int(0.8 * len(all_non_bat_anns))
        non_bat_val_anns = all_non_bat_anns[split_idx:]
        logger.info(f"Using {len(non_bat_val_anns)} non-bat validation recordings")

    val_ds = TwoClassVADDataset(
        bat_val_anns,
        non_bat_val_anns,
        audio_loader,
        preprocessor,
        clip_duration=args.clip_duration,
        samples_per_epoch=args.samples,
        positive_ratio=args.positive_ratio
    )
    # if any operation requires raw waveform, enable return_wav
    if args.baseline or args.amp_gate is not None:
        val_ds.return_wav = True
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_vad_model(vad_config)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model_for_espdl = copy.deepcopy(model)

    # --- Run FP32 Model ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    logger.info("Running FP32 (Original) model on validation set...")
    labels, probs = run_pytorch_model(model, val_loader, device)
    metrics_fp32 = compute_metrics(labels, probs)
    print_metrics(metrics_fp32, "FP32 (Original) Model")
    # record to CSV
    # metrics CSV in export folder
    append_metrics_csv(str(Path("export") / "benchmark_metrics.csv"), "FP32", metrics_fp32)

    # optionally apply amplitude gating before the network
    if args.amp_gate is not None:
        # reseed so the sample order matches the FP32 evaluation
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        logger.info(f"Running FP32 model with amplitude gate {args.amp_gate}...")
        labels_gate, probs_gate = run_pytorch_model(model, val_loader, device)
        labels_gate, probs_gate, stats = apply_amp_gate(labels_gate, probs_gate, val_ds, args.amp_gate)
        pct = 100.0 * stats['gated'] / stats['total'] if stats['total'] > 0 else 0.0
        print(
            f"amp gate stats: min_amp={stats['min_amp']:.6g} max_amp={stats['max_amp']:.6g}; "
            f"discarded {stats['gated']} of {stats['total']} clips ({pct:.1f}%)"
        )
        metrics_gate = compute_metrics(labels_gate, probs_gate)
        print_metrics(metrics_gate, f"FP32 + AmpGate {args.amp_gate}")
        append_metrics_csv(str(Path("export") / "benchmark_metrics.csv"),
                           f"FP32_amp{args.amp_gate}", metrics_gate)

    # optionally run traditional baseline gating
    if args.baseline:
        # reseed so sampling order matches the model evaluation above
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        logger.info("Running amplitude+Goertzel baseline on validation set...")
        # run on the dataset directly (no DataLoader needed)
        labels_base, probs_base = run_baseline(
            val_ds,
            samplerate,
            args.amp_thresh,
            args.goertzel_freqs,
            args.goertzel_thresh,
        )
        metrics_base = compute_metrics(labels_base, probs_base)
        print_metrics(metrics_base, "Baseline (Amplitude+Goertzel)")
        append_metrics_csv(str(Path("export") / "benchmark_metrics.csv"),
                           "Baseline", metrics_base)

    # --- Static Quantization ---
    # Move model to CPU and quantize
    logger.info("Moving model to CPU and applying static quantization...")
    model = model.cpu()

    print_state_dict_size(model, "FP32 model (before)")
    # Use the same dataloader for calibration (it yields CPU tensors already)
    model_int8 = quantize_model_static(model, val_loader)
    print_state_dict_size(model_int8, "INT8 model (after)")
    # Save state dicts for size comparison (optional)
    import os
    export_dir = Path("export")
    export_dir.mkdir(parents=True, exist_ok=True)
    fp32_file = export_dir / "fp32.pt"
    int8_file = export_dir / "int8.pt"
    torch.save(model.state_dict(), fp32_file)
    torch.save(model_int8.state_dict(), int8_file)
    print(f"FP32 Size: {os.path.getsize(fp32_file)/1e6:.2f} MB")
    print(f"INT8 Size: {os.path.getsize(int8_file)/1e6:.2f} MB")

    # --- Run INT8 Quantized Model ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    logger.info("Running INT8 (Quantized) model on validation set (CPU)...")
    labels_int8, probs_int8 = run_quantized_model(model_int8, val_loader, "cpu")
    metrics_int8 = compute_metrics(labels_int8, probs_int8)
    print_metrics(metrics_int8, "INT8 (Statically Quantized) Model")
    append_metrics_csv(str(Path("export") / "benchmark_metrics.csv"), "INT8", metrics_int8)

    # if amp gate requested also evaluate quantized version
    if args.amp_gate is not None:
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        logger.info(f"Running INT8 model with amplitude gate {args.amp_gate}...")
        labels_qgate, probs_qgate = run_quantized_model(model_int8, val_loader, "cpu")
        labels_qgate, probs_qgate, stats = apply_amp_gate(labels_qgate, probs_qgate, val_ds, args.amp_gate)
        pct = 100.0 * stats['gated'] / stats['total'] if stats['total'] > 0 else 0.0
        print(
            f"amp gate stats: min_amp={stats['min_amp']:.6g} max_amp={stats['max_amp']:.6g}; "
            f"discarded {stats['gated']} of {stats['total']} clips ({pct:.1f}%)"
        )
        metrics_qgate = compute_metrics(labels_qgate, probs_qgate)
        print_metrics(metrics_qgate, f"INT8 + AmpGate {args.amp_gate}")
        append_metrics_csv(str(Path("export") / "benchmark_metrics.csv"),
                           f"INT8_amp{args.amp_gate}", metrics_qgate)

    if HAS_ESPDL:
        # Use a small subset for calibration as ESP-PPQ is sensitive to this
        # Re-seeding to ensure identical validation samples
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Run simulation with enhanced calibration diversity but using original data distribution
        # The val_loader already contains both bat and non-bat samples in the correct ratio
        labels_espdl, probs_espdl, espdl_executor, espdl_run = run_espdl_simulated_model(
            model_for_espdl, val_loader, val_loader,
            quiet=args.espdl_quiet,
            calib_steps=args.espdl_calib_steps,
            non_bat_loader=None,  # Don't use separate loader
            mixed_calibration=False,  # Use original validation distribution
            clip_outliers=args.clip_outliers,
            outlier_percentile=args.outlier_percentile,
            keep_fp32_layers=args.keep_fp32_layers)
        metrics_espdl = compute_metrics(labels_espdl, probs_espdl)
        print_metrics(metrics_espdl, "ESP-DL (Simulated Fixed-Point) Model")
        append_metrics_csv(str(Path("export") / "benchmark_metrics.csv"), "ESP-DL", metrics_espdl)

        # ------------------------------------------------------------------
        # Generate a small collection of example clips and dump a figure
        # illustrating the model's decision on each one.  Draw from ``val_loader``
        # (not the raw dataset) so each spectrogram has the same padding used for
        # metrics, which avoids max‑pool padding errors in the ESP-DL executor.
        try:
            import matplotlib.pyplot as plt

            num_examples = args.espdl_examples
            samples = []  # list of (spec_tensor, label)
            for batch in val_loader:
                spec_batch = batch["spec"]
                label_batch = batch["label"]
                for i in range(spec_batch.shape[0]):
                    samples.append((spec_batch[i:i+1], label_batch[i].item()))
                    if len(samples) >= num_examples:
                        break
                if len(samples) >= num_examples:
                    break

            samples = samples[:num_examples]
            # layout: five columns per row
            ncols = 5
            nrows = (len(samples) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
            axes = axes.flatten()
            # iterate through available samples and assign axes
            for idx, (spec_tensor, label) in enumerate(samples):
                ax = axes[idx]
                display = spec_tensor.cpu().numpy()
                while display.ndim > 2:
                    display = display.squeeze(0)
                if display.size > 0:
                    nonzero_cols = np.where(display.std(axis=0) != 0)[0]
                    if nonzero_cols.size > 0:
                        display = display[:, nonzero_cols[0]:nonzero_cols[-1]+1]
                inp = spec_tensor.clone().float()
                pred = None
                try:
                    probs = torch.softmax(espdl_run(inp), dim=1).detach().cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                except Exception as e:
                    logger.warning(f"ESP-DL executor failed on example plot: {e}")
                title = f"pred={pred if pred is not None else 'ERR'} label={label}"
                color = 'red' if (pred is not None and pred != label) else 'black'
                ax.imshow(display, origin='lower', aspect='auto', cmap='viridis')
                ax.set_title(title, color=color)
                ax.axis('off')
            # hide any unused axes
            for j in range(len(samples), len(axes)):
                axes[j].axis('off')
            fig.tight_layout()
            outpath = Path("export") / "espdl_examples.png"
            fig.savefig(outpath)
            logger.info(f"Saved ESP-DL example plot to {outpath}")
        except ImportError:
            logger.warning("matplotlib not available, skipping example plot")

if __name__ == "__main__":
    main()

    """
        uv run python benchmark_pytorch_model.py --config configs/config_vad_2class.yaml --bat-val ../example_data/brazil_val.yaml     --non-bat configs/non_bat_dataset.yaml --ckpt vad_2class_model.pt --clip-duration 0.1 
    """
