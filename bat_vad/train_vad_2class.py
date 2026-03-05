"""
Train a 2-class VAD model (bat vs non-bat) using spectrogram-based classification.

This script loads both positive (bat) and negative (non-bat) examples, and trains
a binary classifier to distinguish between them.
"""
import argparse
import csv
import random
from pathlib import Path
from typing import List, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from loguru import logger
from sklearn.metrics import roc_auc_score, average_precision_score
from soundevent import data, geometry
from torch.utils.data import DataLoader, Dataset

from batdetect2.audio import SoundEventAudioLoader as AudioLoader, build_audio_loader, AudioConfig
from batdetect2.data import load_dataset_from_config
from batdetect2.models import VADConfig, VADModel, build_vad_model
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor, Preprocessor
from batdetect2.typing import ModelOutput


class TwoClassVADDataset(Dataset):
    """
    Dataset for 2-class Voice Activity Detection (bat vs non-bat).
    Generates short clips (e.g. 200ms) with binary class labels.
    """
    def __init__(
        self,
        bat_annotations: List[data.ClipAnnotation],
        non_bat_annotations: List[data.ClipAnnotation],
        audio_loader: AudioLoader,
        preprocessor: Preprocessor,
        clip_duration: float = 0.2,  # 200ms clips
        samples_per_epoch: int = 10000,
        positive_ratio: float = 0.5,  # Balance between bat and non-bat
    ):
        self.bat_annotations = bat_annotations
        self.non_bat_annotations = non_bat_annotations
        self.audio_loader = audio_loader
        self.preprocessor = preprocessor
        self.clip_duration = clip_duration
        self.samples_per_epoch = samples_per_epoch
        self.positive_ratio = positive_ratio
        
        # Group bat annotations by recording
        self.bat_recording_anns = {}
        for ann in bat_annotations:
            rec_id = ann.clip.recording.uuid
            if rec_id not in self.bat_recording_anns:
                self.bat_recording_anns[rec_id] = {
                    "recording": ann.clip.recording,
                    "events": []
                }
            for event in ann.sound_events:
                self.bat_recording_anns[rec_id]["events"].append(event)
        
        self.bat_recording_ids = list(self.bat_recording_anns.keys())
        
        # Non-bat recordings (no annotations, entire audio is negative)
        self.non_bat_recordings = []
        if non_bat_annotations:
            for ann in non_bat_annotations:
                rec = ann.clip.recording
                if rec not in self.non_bat_recordings:
                    self.non_bat_recordings.append(rec)

    def __len__(self):
        return self.samples_per_epoch

    def _sample_positive(self):
        """Sample a clip containing a bat call."""
        # Filter recordings with events
        if not self.bat_recording_ids:
            return None
        valid_recs = [r for r in self.bat_recording_ids 
                     if self.bat_recording_anns[r]["events"]]
        
        if not valid_recs:
            # No positive examples available, return None
            return None

        rec_id = random.choice(valid_recs)
        rec_data = self.bat_recording_anns[rec_id]
        event = random.choice(rec_data["events"])
        recording = rec_data["recording"]
        
        # Center roughly around the event, but add jitter
        event_start, _, event_end, _ = geometry.compute_bounds(event.sound_event.geometry)
        event_center = (event_start + event_end) / 2
        
        # Random offset
        half_clip = self.clip_duration / 2
        start_time = event_center - half_clip + random.uniform(-0.05, 0.05)
        
        # Ensure within bounds
        start_time = max(0.0, min(start_time, recording.duration - self.clip_duration))
        
        return recording, start_time, 1

    def _sample_negative_from_bat_recordings(self):
        """Sample a clip from bat recordings but without overlap with bat calls."""
        if not self.bat_recording_ids:
            # no bat recordings available
            return None

        max_attempts = 10
        for _ in range(max_attempts):
            rec_id = random.choice(self.bat_recording_ids)
            rec_data = self.bat_recording_anns[rec_id]
            recording = rec_data["recording"]
            
            duration = recording.duration
            if duration <= self.clip_duration:
                start_time = 0.0
            else:
                start_time = random.uniform(0, duration - self.clip_duration)
            
            # Check overlap
            end_time = start_time + self.clip_duration
            overlap = False
            for event in rec_data["events"]:
                e_start, _, e_end, _ = geometry.compute_bounds(event.sound_event.geometry)
                if max(start_time, e_start) < min(end_time, e_end):
                    overlap = True
                    break
            
            if not overlap:
                return recording, start_time, 0
        
        # If we couldn't find a non-overlapping region, return None
        return None

    def _sample_negative_from_non_bat(self):
        """Sample a clip from dedicated non-bat recordings."""
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
        # Determine if we want a positive or negative sample
        is_positive = random.random() < self.positive_ratio
        
        # Try to get a sample
        max_attempts = 5
        sample = None
        
        for _ in range(max_attempts):
            if is_positive:
                sample = self._sample_positive()
            else:
                # Sample from non-bat recordings 50% of the time, bat recordings 50%
                if self.non_bat_recordings and random.random() < 0.5:
                    sample = self._sample_negative_from_non_bat()
                else:
                    sample = self._sample_negative_from_bat_recordings()
            
            if sample is not None:
                break
            
            # Switch to the other class if we failed
            is_positive = not is_positive
        
        if sample is None:
            # Fallback: return a random clip from any bat recording
            if not self.bat_recording_ids:
                raise RuntimeError("Dataset has no bat recordings to sample from, check your annotation paths")
            rec_id = random.choice(self.bat_recording_ids)
            recording = self.bat_recording_anns[rec_id]["recording"]
            start_time = 0.0
            label = 0  # Assume no bat for safety
        else:
            recording, start_time, label = sample
        
        # Load audio
        clip = data.Clip(
            recording=recording,
            start_time=start_time,
            end_time=start_time + self.clip_duration
        )
        
        # Load audio waveform
        wav = self.audio_loader.load_clip(clip)
        
        # Add channel dimension and convert to tensor
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        
        # Preprocess (Audio -> Spectrogram)
        spec = self.preprocessor(wav_tensor)
        
        return {
            "spec": spec,
            "label": torch.tensor(label, dtype=torch.long),  # Class index for CrossEntropyLoss
            "recording_id": str(recording.uuid),
            "time": start_time
        }


class TwoClassVADTrainingModule(L.LightningModule):
    """Training module for 2-class VAD (bat vs non-bat)."""
    
    def __init__(self, model: VADModel, learning_rate: float = 1e-3, label_smoothing: float = 0.0, csv_log_path: Optional[str] = None):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.csv_log_path = csv_log_path
        
        # For validation metrics
        self.validation_step_outputs = []
        
        # Initialize CSV logging
        if self.csv_log_path:
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_acc', 'train_bat_acc', 'train_non_bat_acc',
                    'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1',
                    'val_specificity', 'val_fpr', 'val_bat_acc', 'val_non_bat_acc',
                    'val_auc_roc', 'val_avg_precision',
                    'val_tp', 'val_tn', 'val_fp', 'val_fn'
                ])
        
        # Track metrics for CSV logging
        self.epoch_metrics = {}

        # Track best metrics and epoch
        self.best_val_metric = None
        self.best_val_epoch = None
        self.best_val_metrics_dict = None

    def training_step(self, batch, batch_idx):
        spec = batch["spec"]
        label = batch["label"]
        
        # Forward pass
        features = self.model.encoder(spec)
        bottleneck = features if isinstance(features, torch.Tensor) else features[-1]
        logits = self.model.head(bottleneck)
        
        # Loss
        loss = self.criterion(logits, label)
        
        self.log("train_loss", loss, prog_bar=True)
        
        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == label).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        
        # Log class-wise accuracy
        if label.sum() > 0 and label.sum() < len(label):  # Both classes present
            bat_mask = label == 1
            non_bat_mask = label == 0
            bat_acc = (preds[bat_mask] == label[bat_mask]).float().mean() if bat_mask.sum() > 0 else 0
            non_bat_acc = (preds[non_bat_mask] == label[non_bat_mask]).float().mean() if non_bat_mask.sum() > 0 else 0
            self.log("train_bat_acc", bat_acc)
            self.log("train_non_bat_acc", non_bat_acc)
            
            # Store for epoch-end CSV logging
            self.epoch_metrics['train_bat_acc'] = bat_acc.item() if isinstance(bat_acc, torch.Tensor) else float(bat_acc)
            self.epoch_metrics['train_non_bat_acc'] = non_bat_acc.item() if isinstance(non_bat_acc, torch.Tensor) else float(non_bat_acc)
        
        # Store for CSV logging
        self.epoch_metrics['train_loss'] = loss.item()
        self.epoch_metrics['train_acc'] = acc.item()
        
        return loss

    def validation_step(self, batch, batch_idx):
        spec = batch["spec"]
        label = batch["label"]
        
        features = self.model.encoder(spec)
        bottleneck = features if isinstance(features, torch.Tensor) else features[-1]
        logits = self.model.head(bottleneck)
        
        loss = self.criterion(logits, label)
        
        # Get predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'preds': preds.detach(),
            'labels': label.detach(),
            'probs': probs.detach(),
            'loss': loss.detach()
        })
        
        self.log("val_loss", loss, prog_bar=True, batch_size=spec.size(0))
        
        return loss
    
    def on_validation_epoch_end(self):
        # Gather all predictions and labels
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        
        # Compute confusion matrix for bat class (class 1)
        # Treat class 1 as positive, class 0 as negative
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        tn = ((all_preds == 0) & (all_labels == 0)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
        
        # Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Precision, Recall, F1 (for bat class)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp + 1e-8)
        
        # False Positive Rate
        fpr_metric = fp / (fp + tn + 1e-8)
        
        # Class-wise accuracy
        bat_mask = all_labels == 1
        non_bat_mask = all_labels == 0
        bat_acc = (all_preds[bat_mask] == all_labels[bat_mask]).float().mean() if bat_mask.sum() > 0 else 0
        non_bat_acc = (all_preds[non_bat_mask] == all_labels[non_bat_mask]).float().mean() if non_bat_mask.sum() > 0 else 0
        
        # Compute AUC-ROC and Average Precision
        # Convert to numpy for sklearn
        all_labels_np = all_labels.cpu().numpy()
        all_probs_np = all_probs[:, 1].cpu().numpy()  # Probability of bat class
        
        try:
            auc_roc = roc_auc_score(all_labels_np, all_probs_np)
            avg_precision = average_precision_score(all_labels_np, all_probs_np)
        except Exception as e:
            logger.warning(f"Could not compute AUC-ROC or AP: {e}")
            auc_roc = 0.0
            avg_precision = 0.0
        
        # Log all metrics
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_specificity", specificity)
        self.log("val_fpr", fpr_metric)
        self.log("val_bat_acc", bat_acc)
        self.log("val_non_bat_acc", non_bat_acc)
        self.log("val_auc_roc", auc_roc, prog_bar=True)
        self.log("val_avg_precision", avg_precision, prog_bar=True)
        
        # Log confusion matrix components
        self.log("val_tp", tp)
        self.log("val_tn", tn)
        self.log("val_fp", fp)
        self.log("val_fn", fn)
        
        # Log class distribution
        pos_rate = (all_labels == 1).sum().float() / len(all_labels)
        self.log("val_pos_rate", pos_rate)
        
        # Print summary
        logger.info(f"Validation Metrics:")
        logger.info(f"  Overall Acc: {acc:.4f}")
        logger.info(f"  Bat Class - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Acc: {bat_acc:.4f}")
        logger.info(f"  Non-Bat Class - Specificity: {specificity:.4f}, Acc: {non_bat_acc:.4f}")
        logger.info(f"  AUC-ROC: {auc_roc:.4f}, Average Precision: {avg_precision:.4f}")
        logger.info(f"  Confusion Matrix - TP: {tp:.0f}, TN: {tn:.0f}, FP: {fp:.0f}, FN: {fn:.0f}")
        
        # Calculate average validation loss
        avg_val_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        
        # Update best validation metric (accuracy)
        val_metric = acc.item() if isinstance(acc, torch.Tensor) else float(acc)
        if self.best_val_metric is None or val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_val_epoch = self.current_epoch
            self.best_val_metrics_dict = {
                'epoch': self.current_epoch,
                'val_acc': acc.item() if isinstance(acc, torch.Tensor) else float(acc),
                'val_precision': precision.item() if isinstance(precision, torch.Tensor) else float(precision),
                'val_recall': recall.item() if isinstance(recall, torch.Tensor) else float(recall),
                'val_f1': f1.item() if isinstance(f1, torch.Tensor) else float(f1),
                'val_specificity': specificity.item() if isinstance(specificity, torch.Tensor) else float(specificity),
                'val_auc_roc': auc_roc,
                'val_avg_precision': avg_precision,
                'val_bat_acc': bat_acc.item() if isinstance(bat_acc, torch.Tensor) else float(bat_acc),
                'val_non_bat_acc': non_bat_acc.item() if isinstance(non_bat_acc, torch.Tensor) else float(non_bat_acc),
                'val_tp': tp.item(),
                'val_tn': tn.item(),
                'val_fp': fp.item(),
                'val_fn': fn.item(),
            }
        
        # Write to CSV
        if self.csv_log_path:
            with open(self.csv_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.current_epoch,
                    self.epoch_metrics.get('train_loss', 0),
                    self.epoch_metrics.get('train_acc', 0),
                    self.epoch_metrics.get('train_bat_acc', 0),
                    self.epoch_metrics.get('train_non_bat_acc', 0),
                    avg_val_loss.item(),
                    acc.item(),
                    precision.item(),
                    recall.item(),
                    f1.item(),
                    specificity.item(),
                    fpr_metric.item(),
                    bat_acc.item() if isinstance(bat_acc, torch.Tensor) else float(bat_acc),
                    non_bat_acc.item() if isinstance(non_bat_acc, torch.Tensor) else float(non_bat_acc),
                    auc_roc,
                    avg_precision,
                    tp.item(),
                    tn.item(),
                    fp.item(),
                    fn.item()
                ])
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()
        self.epoch_metrics = {}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def load_non_bat_dataset(dataset_yaml: str) -> List[data.ClipAnnotation]:
    """
    Load non-bat dataset and create dummy annotations.
    
    For non-bat data, we don't have annotations, so we create ClipAnnotations
    that span entire recordings with no sound events.
    """
    with open(dataset_yaml, "r") as f:
        dataset_config = yaml.safe_load(f)
    
    annotations = []
    
    for source in dataset_config["sources"]:
        audio_dir = Path(source["audio_dir"])
        
        if not audio_dir.exists():
            logger.warning(f"Audio directory not found: {audio_dir}")
            continue
        
        # Find all audio files
        audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.WAV"))
        
        logger.info(f"Found {len(audio_files)} audio files in {audio_dir}")
        
        for audio_file in audio_files:
            # Create a recording
            import soundfile as sf
            try:
                info = sf.info(str(audio_file))
                duration = info.duration
                samplerate = info.samplerate
            except Exception as e:
                logger.warning(f"Could not read {audio_file}: {e}")
                continue
            
            recording = data.Recording(
                path=audio_file,
                duration=duration,
                samplerate=samplerate,
                channels=1,
            )
            
            # Create a clip annotation spanning the entire recording
            clip = data.Clip(
                recording=recording,
                start_time=0.0,
                end_time=duration,
            )
            
            # No sound events (negative sample)
            clip_ann = data.ClipAnnotation(
                clip=clip,
                sound_events=[],
            )
            
            annotations.append(clip_ann)
    
    logger.info(f"Loaded {len(annotations)} non-bat recordings")
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Train 2-class VAD model (bat vs non-bat)")
    parser.add_argument("--config", type=str, required=True, help="Path to VAD config (use config_vad_2class.yaml)")
    parser.add_argument("--bat-train", type=str, required=True, help="Path to bat train dataset yaml")
    parser.add_argument("--bat-val", type=str, required=True, help="Path to bat val dataset yaml")
    parser.add_argument("--non-bat", type=str, help="Path to non-bat dataset yaml (optional)")
    parser.add_argument(
        "--non-bat-split-seed",
        type=int,
        default=None,
        help="Optional deterministic seed for splitting non-bat data. "
             "When omitted the list is sorted by path to obtain a stable "
             "train/val split; provide an integer to shuffle reproducibly."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing for regularization")
    parser.add_argument("--save-path", type=str, default="vad_2class_model.pt", help="Path to save trained model")
    parser.add_argument("--csv-log", type=str, default=None, help="Path to save training metrics as CSV")
    parser.add_argument("--clip-duration", type=float, default=0.2, help="Duration of each training clip in seconds (default: 0.2)")
    parser.add_argument("--samples-per-epoch", type=int, default=10000, help="Number of samples per training epoch (default: 10000)")
    parser.add_argument("--val-samples", type=int, default=1000, help="Number of validation samples per epoch (default: 1000)")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Ratio of positive (bat) samples (default: 0.5)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Extract model config
    model_config = config_dict.get("model", config_dict)
    model_config["type"] = "vad"
    
    # Ensure 2-class output
    if model_config.get("out_channels", 1) != 2:
        logger.warning(f"Config has out_channels={model_config.get('out_channels')}, but 2-class training needs 2. Setting to 2.")
        model_config["out_channels"] = 2
    
    vad_config = VADConfig(**model_config)
    
    # Build Model
    model = build_vad_model(vad_config)
    logger.info(f"Built 2-class VAD model with {vad_config.out_channels} output classes")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params:,} parameters")

    # Load Bat Annotations
    logger.info("Loading bat datasets...")
    bat_train_anns = load_dataset_from_config(args.bat_train)
    bat_val_anns = load_dataset_from_config(args.bat_val)
    logger.info(f"Loaded {len(bat_train_anns)} bat train annotations, {len(bat_val_anns)} bat val annotations")
    
    # Load Non-Bat Annotations
    non_bat_train_anns = []
    non_bat_val_anns = []
    
    if args.non_bat:
        logger.info("Loading non-bat dataset...")
        all_non_bat_anns = load_non_bat_dataset(args.non_bat)

        # Determine a deterministic train/val partition.  We avoid using
        # `random.shuffle` without a seed because that would produce a
        # different split each time the script is run.  Instead we sort the
        # annotations by file path (stable across machines) and then slice.
        # An optional seed parameter is provided in case the user wants to
        # introduce randomness but still be able to reproduce the split.
        if args.non_bat_split_seed is not None:
            rng = random.Random(args.non_bat_split_seed)
            rng.shuffle(all_non_bat_anns)
        else:
            # sort by the recording path to get a stable order
            all_non_bat_anns.sort(key=lambda ann: str(ann.clip.recording.path))

        split_idx = int(0.8 * len(all_non_bat_anns))
        non_bat_train_anns = all_non_bat_anns[:split_idx]
        non_bat_val_anns = all_non_bat_anns[split_idx:]

        logger.info(f"Split non-bat data: {len(non_bat_train_anns)} train, {len(non_bat_val_anns)} val")
    else:
        logger.warning("No non-bat dataset provided. Training will only use negative samples from bat recordings.")
    
    # Audio Loader & Preprocessor
    samplerate = config_dict.get("audio", {}).get("samplerate", 256000)
    audio_loader = build_audio_loader()
    
    preproc_config = None
    if "preprocess" in config_dict:
        preproc_config = PreprocessingConfig(**config_dict["preprocess"])
    
    preprocessor = build_preprocessor(config=preproc_config, input_samplerate=samplerate)
    
    # Custom collate function
    def collate_fn(batch):
        max_width = max(item["spec"].shape[-1] for item in batch)
        
        specs = []
        labels = []
        recording_ids = []
        times = []
        
        for item in batch:
            spec = item["spec"]
            if spec.shape[-1] < max_width:
                pad_width = max_width - spec.shape[-1]
                spec = F.pad(spec, (0, pad_width), mode='constant', value=0)
            specs.append(spec)
            labels.append(item["label"])
            recording_ids.append(item["recording_id"])
            times.append(item["time"])
        
        return {
            "spec": torch.stack(specs),
            "label": torch.stack(labels),
            "recording_id": recording_ids,
            "time": torch.tensor(times)
        }
    
    # Datasets
    logger.info(f"Creating datasets with clip_duration={args.clip_duration}s, positive_ratio={args.positive_ratio}")
    train_ds = TwoClassVADDataset(
        bat_train_anns, 
        non_bat_train_anns,
        audio_loader, 
        preprocessor, 
        clip_duration=args.clip_duration,
        samples_per_epoch=args.samples_per_epoch,
        positive_ratio=args.positive_ratio
    )
    
    val_ds = TwoClassVADDataset(
        bat_val_anns,
        non_bat_val_anns,
        audio_loader, 
        preprocessor, 
        clip_duration=args.clip_duration,
        samples_per_epoch=args.val_samples,
        positive_ratio=args.positive_ratio
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4, collate_fn=collate_fn)
    
    # Determine CSV log path
    csv_log_path = args.csv_log
    if csv_log_path is None:
        # Auto-generate CSV log path based on save path
        csv_log_path = str(Path(args.save_path).with_suffix('.csv'))
        logger.info(f"No CSV log path specified, using: {csv_log_path}")
    
    # Lightning Module
    module = TwoClassVADTrainingModule(
        model, 
        learning_rate=args.lr, 
        label_smoothing=args.label_smoothing,
        csv_log_path=csv_log_path
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        log_every_n_steps=10,
    )
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    trainer.fit(module, train_loader, val_loader)
    
    # After training, if best_val_metrics_dict is None, set it to the last epoch's metrics
    if module.best_val_metrics_dict is None and hasattr(module, 'validation_step_outputs') and module.validation_step_outputs:
        # Compute metrics from the last validation outputs
        all_preds = torch.cat([x['preds'] for x in module.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in module.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in module.validation_step_outputs])
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        tn = ((all_preds == 0) & (all_labels == 0)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        bat_mask = all_labels == 1
        non_bat_mask = all_labels == 0
        bat_acc = (all_preds[bat_mask] == all_labels[bat_mask]).float().mean() if bat_mask.sum() > 0 else 0
        non_bat_acc = (all_preds[non_bat_mask] == all_labels[non_bat_mask]).float().mean() if non_bat_mask.sum() > 0 else 0
        all_labels_np = all_labels.cpu().numpy()
        all_probs_np = all_probs[:, 1].cpu().numpy()
        try:
            auc_roc = roc_auc_score(all_labels_np, all_probs_np)
            avg_precision = average_precision_score(all_labels_np, all_probs_np)
        except Exception:
            auc_roc = 0.0
            avg_precision = 0.0
        module.best_val_metrics_dict = {
            'epoch': getattr(module, 'current_epoch', 0),
            'val_acc': acc.item() if isinstance(acc, torch.Tensor) else float(acc),
            'val_precision': precision.item() if isinstance(precision, torch.Tensor) else float(precision),
            'val_recall': recall.item() if isinstance(recall, torch.Tensor) else float(recall),
            'val_f1': f1.item() if isinstance(f1, torch.Tensor) else float(f1),
            'val_specificity': specificity.item() if isinstance(specificity, torch.Tensor) else float(specificity),
            'val_auc_roc': auc_roc,
            'val_avg_precision': avg_precision,
            'val_bat_acc': bat_acc.item() if isinstance(bat_acc, torch.Tensor) else float(bat_acc),
            'val_non_bat_acc': non_bat_acc.item() if isinstance(non_bat_acc, torch.Tensor) else float(non_bat_acc),
            'val_tp': tp.item(),
            'val_tn': tn.item(),
            'val_fp': fp.item(),
            'val_fn': fn.item(),
        }
    # Print best metrics summary if available
    if module.best_val_metrics_dict is not None:
        logger.info("\n================ BEST VALIDATION METRICS ================")
        logger.info(f"Best epoch: {module.best_val_metrics_dict['epoch']}")
        logger.info(f"  Overall Acc: {module.best_val_metrics_dict['val_acc']:.4f}")
        logger.info(f"  Bat Class - Precision: {module.best_val_metrics_dict['val_precision']:.4f}, Recall: {module.best_val_metrics_dict['val_recall']:.4f}, F1: {module.best_val_metrics_dict['val_f1']:.4f}, Acc: {module.best_val_metrics_dict['val_bat_acc']:.4f}")
        logger.info(f"  Non-Bat Class - Specificity: {module.best_val_metrics_dict['val_specificity']:.4f}, Acc: {module.best_val_metrics_dict['val_non_bat_acc']:.4f}")
        logger.info(f"  AUC-ROC: {module.best_val_metrics_dict['val_auc_roc']:.4f}, Average Precision: {module.best_val_metrics_dict['val_avg_precision']:.4f}")
        logger.info(f"  Confusion Matrix - TP: {module.best_val_metrics_dict['val_tp']:.0f}, TN: {module.best_val_metrics_dict['val_tn']:.0f}, FP: {module.best_val_metrics_dict['val_fp']:.0f}, FN: {module.best_val_metrics_dict['val_fn']:.0f}")
        logger.info("========================================================\n")
    # Save model
    logger.info(f"Saving model to {args.save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vad_config,
        'config_dict': config_dict,
    }, args.save_path)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
