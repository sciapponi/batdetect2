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
from batdetect2.models import VADConfig, VADModel, build_vad_model
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor, Preprocessor
from batdetect2.typing import ModelOutput

class VADDataset(Dataset):
    """
    Dataset for Voice Activity Detection.
    Generates short clips (e.g. 200ms) and binary labels.
    """
    def __init__(
        self,
        annotations: List[data.ClipAnnotation],
        audio_loader: AudioLoader,
        preprocessor: Preprocessor,
        clip_duration: float = 0.2,  # 200ms
        samples_per_epoch: int = 10000,
        positive_ratio: float = 0.5,
    ):
        self.annotations = annotations
        self.audio_loader = audio_loader
        self.preprocessor = preprocessor
        self.clip_duration = clip_duration
        self.samples_per_epoch = samples_per_epoch
        self.positive_ratio = positive_ratio
        
        # Pre-calculate positive intervals for faster sampling
        self.positive_intervals = []
        self.negative_intervals = [] # We'll sample negatives randomly from non-positive areas
        
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
            # Filter recordings with events
            valid_recs = [r for r in self.recording_ids if self.recording_anns[r]["events"]]
            if not valid_recs:
                 # Fallback if no positive events
                 return self._get_random_sample()

            rec_id = random.choice(valid_recs)
            rec_data = self.recording_anns[rec_id]
            event = random.choice(rec_data["events"])
            recording = rec_data["recording"]
            
            # Center roughly around the event, but add jitter
            # Use geometry.compute_bounds to handle any geometry type
            event_start, _, event_end, _ = geometry.compute_bounds(event.sound_event.geometry)
            event_center = (event_start + event_end) / 2
            
            # Random offset
            half_clip = self.clip_duration / 2
            start_time = event_center - half_clip + random.uniform(-0.05, 0.05)
            
            # Ensure within bounds
            start_time = max(0.0, min(start_time, recording.duration - self.clip_duration))
            
            label = 1.0
            
        else:
            # Negative sample (random part of a random recording, assuming sparse events)
            # Ideally checks for overlap, simplified here for speed: 
            # Bat calls are sparse, random sampling is usually negative.
            # But better to check.
            rec_id = random.choice(self.recording_ids)
            rec_data = self.recording_anns[rec_id]
            recording = rec_data["recording"]
            
            duration = recording.duration
            if duration <= self.clip_duration:
                start_time = 0.0
            else:
                start_time = random.uniform(0, duration - self.clip_duration)
            
            # Check overlap to be sure
            end_time = start_time + self.clip_duration
            overlap = False
            for event in rec_data["events"]:
                e_start, _, e_end, _ = geometry.compute_bounds(event.sound_event.geometry)
                if max(start_time, e_start) < min(end_time, e_end):
                    overlap = True
                    break
            
            if overlap:
                # Accidentally hit a bat, mark as positive
                label = 1.0
            else:
                label = 0.0

        recording = self.recording_anns[rec_id]["recording"]
        
        # Load audio - create a Clip object
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
            "label": torch.tensor([label], dtype=torch.float32),
            "recording_id": str(rec_id),
            "time": start_time
        }

    def _get_random_sample(self):
        # Fallback
        pass

class VADTrainingModule(L.LightningModule):
    def __init__(self, model: VADModel, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        
        # For validation metrics
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        spec = batch["spec"]
        label = batch["label"]
        
        # Forward
        features = self.model.encoder(spec)
        # Handle both single tensor and list of tensors
        bottleneck = features if isinstance(features, torch.Tensor) else features[-1]
        logits = self.model.head(bottleneck)
        
        loss = self.criterion(logits, label)
        
        self.log("train_loss", loss, prog_bar=True)
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == label).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        spec = batch["spec"]
        label = batch["label"]
        
        features = self.model.encoder(spec)
        # Handle both single tensor and list of tensors  
        bottleneck = features if isinstance(features, torch.Tensor) else features[-1]
        logits = self.model.head(bottleneck)
        
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
        
        self.log("val_loss", loss, prog_bar=True, batch_size=spec.size(0))
        
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
    parser.add_argument("--config", type=str, required=True, help="Path to VAD config")
    parser.add_argument("--train-dataset", type=str, required=True, help="Path to train dataset yaml")
    parser.add_argument("--val-dataset", type=str, required=True, help="Path to val dataset yaml")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Extract model config if nested
    model_config = config_dict.get("model", config_dict)
    
    # Create VADConfig (ensure type is vad)
    model_config["type"] = "vad"
    # Basic validation/filling defaults
    vad_config = VADConfig(**model_config)
    
    # Build Model
    model = build_vad_model(vad_config)
    logger.info(f"Built VAD model with encoder: {vad_config.encoder.type if hasattr(vad_config.encoder, 'type') else 'custom'}")

    # Load Annotations
    train_anns = load_dataset_from_config(args.train_dataset)
    val_anns = load_dataset_from_config(args.val_dataset)
    
    # Audio Loader & Preprocessor
    # We need to extract samplerate from config or use default
    samplerate = 256000 # Default for batdetect2
    if "audio" in config_dict and "samplerate" in config_dict["audio"]:
        samplerate = config_dict["audio"]["samplerate"]

    audio_loader = build_audio_loader() # Default loader
    
    # Preprocessor config
    preproc_config = None
    if "preprocess" in config_dict:
        preproc_config = PreprocessingConfig(**config_dict["preprocess"])
    
    preprocessor = build_preprocessor(config=preproc_config, input_samplerate=samplerate)
    
    # Custom collate function to handle variable-width spectrograms
    def collate_fn(batch):
        # Find max width
        max_width = max(item["spec"].shape[-1] for item in batch)
        
        # Pad spectrograms to max width
        specs = []
        labels = []
        recording_ids = []
        times = []
        
        for item in batch:
            spec = item["spec"]
            # Pad if needed
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
    train_ds = VADDataset(train_anns, audio_loader, preprocessor, clip_duration=0.2)
    val_ds = VADDataset(val_anns, audio_loader, preprocessor, clip_duration=0.2, samples_per_epoch=1000)
    
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4, collate_fn=collate_fn)
    
    # Lightning Module
    module = VADTrainingModule(model)
    
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
