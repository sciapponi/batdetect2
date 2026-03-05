import argparse
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
from soundevent import data, geometry
from torch.utils.data import DataLoader, Dataset

from batdetect2.audio import SoundEventAudioLoader as AudioLoader, build_audio_loader, AudioConfig
from batdetect2.data import load_dataset_from_config
from batdetect2.models.vad import VAD1DModel, build_vad_1d_model

class VAD1DDataset(Dataset):
    """
    Dataset for 1D Voice Activity Detection on raw audio.
    Generates short clips (e.g. 50ms) and binary labels.
    """
    def __init__(
        self,
        annotations: List[data.ClipAnnotation],
        audio_loader: AudioLoader,
        clip_duration: float = 0.05,  # 50ms
        samplerate: int = 256000,
        samples_per_epoch: int = 10000,
        positive_ratio: float = 0.5,
    ):
        self.annotations = annotations
        self.audio_loader = audio_loader
        self.clip_duration = clip_duration
        self.samplerate = samplerate
        self.clip_samples = int(clip_duration * samplerate)
        self.samples_per_epoch = samples_per_epoch
        self.positive_ratio = positive_ratio
        
        # Group annotations by recording
        self.recording_anns = {}
        for ann in annotations:
            rec_id = ann.clip.recording.uuid
            if rec_id not in self.recording_anns:
                self.recording_anns[rec_id] = {
                    "recording": ann.clip.recording,
                    "events": []
                }
            for event in ann.sound_events:
                self.recording_anns[rec_id]["events"].append(event)

        self.recording_ids = list(self.recording_anns.keys())

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Determine if we want a positive or negative sample
        is_positive = random.random() < self.positive_ratio
        
        if is_positive:
            # Sample from a recording that has events
            valid_recs = [r for r in self.recording_ids if self.recording_anns[r]["events"]]
            if not valid_recs:
                # Fallback to random sample
                is_positive = False
            else:
                rec_id = random.choice(valid_recs)
                rec_data = self.recording_anns[rec_id]
                event = random.choice(rec_data["events"])
                recording = rec_data["recording"]
                
                # Center roughly around the event
                event_start, _, event_end, _ = geometry.compute_bounds(event.sound_event.geometry)
                event_center = (event_start + event_end) / 2
                
                # Random offset
                half_clip = self.clip_duration / 2
                start_time = event_center - half_clip + random.uniform(-0.05, 0.05)
                
                # Ensure within bounds
                start_time = max(0.0, min(start_time, recording.duration - self.clip_duration))
                
                label = 1.0
        
        if not is_positive:
            # Negative sample
            rec_id = random.choice(self.recording_ids)
            rec_data = self.recording_anns[rec_id]
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
            
            label = 1.0 if overlap else 0.0

        # Load audio - create a Clip object
        clip = data.Clip(
            recording=recording,
            start_time=start_time,
            end_time=start_time + self.clip_duration
        )
        
        # Load audio waveform
        wav = self.audio_loader.load_clip(clip)
        
        # Ensure fixed length
        if len(wav) < self.clip_samples:
            # Pad with zeros
            wav = np.pad(wav, (0, self.clip_samples - len(wav)), mode='constant')
        elif len(wav) > self.clip_samples:
            # Truncate
            wav = wav[:self.clip_samples]
        
        # Convert to tensor and add channel dimension
        wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        
        return {
            "audio": wav_tensor,
            "label": torch.tensor([label], dtype=torch.float32),
            "recording_id": str(rec_id),
            "time": start_time
        }


class VAD1DTrainingModule(L.LightningModule):
    def __init__(self, model: VAD1DModel, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        
        # For validation metrics
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        label = batch["label"]
        
        # Forward
        logits = self.model(audio)
        
        loss = self.criterion(logits, label)
        
        self.log("train_loss", loss, prog_bar=True, batch_size=audio.size(0))
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == label).float().mean()
        self.log("train_acc", acc, prog_bar=True, batch_size=audio.size(0))
        
        return loss

    def validation_step(self, batch, batch_idx):
        audio = batch["audio"]
        label = batch["label"]
        
        logits = self.model(audio)
        
        loss = self.criterion(logits, label)
        
        # Get probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Store for epoch-end metrics
        self.validation_step_outputs.append({
            'preds': preds.detach(),
            'labels': label.detach(),
            'probs': probs.detach(),
            'loss': loss.detach()
        })
        
        self.log("val_loss", loss, prog_bar=True, batch_size=audio.size(0))
        
        return loss
    
    def on_validation_epoch_end(self):
        # Gather all predictions and labels
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.validation_step_outputs])
        
        # Compute metrics
        tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
        tn = ((all_preds == 0) & (all_labels == 0)).sum().float()
        fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
        fn = ((all_preds == 0) & (all_labels == 1)).sum().float()
        
        # Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp + 1e-8)
        
        # False Positive Rate
        fpr = fp / (fp + tn + 1e-8)
        
        # Log all metrics
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_specificity", specificity)
        self.log("val_fpr", fpr)
        
        # Log confusion matrix components
        self.log("val_tp", tp)
        self.log("val_tn", tn)
        self.log("val_fp", fp)
        self.log("val_fn", fn)
        
        # Log positive/negative counts
        pos_rate = all_labels.sum() / len(all_labels)
        self.log("val_pos_rate", pos_rate)
        
        # Print summary
        logger.info(f"Validation Metrics - Acc: {acc:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}, Specificity: {specificity:.4f}")
        logger.info(f"Confusion Matrix - TP: {tp:.0f}, TN: {tn:.0f}, FP: {fp:.0f}, FN: {fn:.0f}")
        
        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=str, required=True, help="Path to train dataset yaml")
    parser.add_argument("--val-dataset", type=str, required=True, help="Path to val dataset yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--channels", type=str, default="16,32,64,128,256", 
                       help="Comma-separated list of channel counts")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--clip-duration", type=float, default=0.2, help="Clip duration in seconds")
    parser.add_argument("--samplerate", type=int, default=256000)
    args = parser.parse_args()

    # Parse channels
    channels_list = [int(x) for x in args.channels.split(',')]
    
    # Build Model
    model = build_vad_1d_model(channels_list=channels_list, num_classes=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Built 1D VAD model with {total_params:,} parameters")
    logger.info(f"Channel progression: 1 → {' → '.join(map(str, channels_list))}")

    # Load Annotations
    train_anns = load_dataset_from_config(args.train_dataset)
    val_anns = load_dataset_from_config(args.val_dataset)
    
    # Audio Loader
    audio_loader = build_audio_loader()
    
    # Datasets
    train_ds = VAD1DDataset(
        train_anns, audio_loader, 
        clip_duration=args.clip_duration, 
        samplerate=args.samplerate
    )
    val_ds = VAD1DDataset(
        val_anns, audio_loader, 
        clip_duration=args.clip_duration, 
        samplerate=args.samplerate,
        samples_per_epoch=1000
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)
    
    # Lightning Module
    module = VAD1DTrainingModule(model)
    
    # Trainer
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        default_root_dir=script_dir
    )
    trainer.fit(module, train_loader, val_loader)

if __name__ == "__main__":
    main()
