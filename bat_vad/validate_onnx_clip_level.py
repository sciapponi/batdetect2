"""
Validate an ONNX VAD model at the clip level, using the same preprocessing and batching as the training dataset.
Compares ONNX model predictions to ground truth labels from a validation YAML file.
"""
import argparse
from pathlib import Path
import yaml
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
from tqdm import tqdm
from batdetect2.preprocess import PreprocessingConfig, build_preprocessor
from batdetect2.audio import build_audio_loader


def load_validation_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    # Expecting a list of dicts with keys: audio_path, label, start_time, end_time
    return data


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX VAD model at clip level using validation YAML.")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--val-yaml", type=str, required=True, help="Validation YAML file (as used in training)")
    parser.add_argument("--config", type=str, required=True, help="Config YAML for preprocessing")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ONNX inference")
    parser.add_argument("--device", type=str, default="cpu", help="Device (for preprocessing)")
    args = parser.parse_args()

    # Load config and build preprocessor
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    samplerate = config_dict.get("audio", {}).get("samplerate", 256000)
    preproc_config = PreprocessingConfig(**config_dict["preprocess"])
    preprocessor = build_preprocessor(config=preproc_config, input_samplerate=samplerate)

    # Load ONNX model
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Load validation clips
    val_clips = load_validation_yaml(args.val_yaml)
    y_true = []
    y_pred = []
    for i in tqdm(range(0, len(val_clips), args.batch_size), desc="Validating"):
        batch = val_clips[i:i+args.batch_size]
        specs = []
        for clip in batch:
            wav, sr = sf.read(clip['audio_path'])
            if len(wav.shape) > 1:
                wav = wav.mean(axis=1)
            start_sample = int(clip['start_time'] * sr)
            end_sample = int(clip['end_time'] * sr)
            clip_wav = wav[start_sample:end_sample]
            wav_tensor = torch.tensor(clip_wav, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                spec = preprocessor(wav_tensor)
            # Crop/pad to [1, 64, 100] if needed
            spec_np = spec.squeeze().cpu().numpy().astype(np.float32)
            height, width = 64, 100
            if spec_np.shape[0] < height:
                pad = np.zeros((height - spec_np.shape[0], spec_np.shape[1]), dtype=np.float32)
                spec_np = np.vstack([spec_np, pad])
            elif spec_np.shape[0] > height:
                spec_np = spec_np[:height, :]
            if spec_np.shape[1] < width:
                pad = np.zeros((spec_np.shape[0], width - spec_np.shape[1]), dtype=np.float32)
                spec_np = np.hstack([spec_np, pad])
            elif spec_np.shape[1] > width:
                spec_np = spec_np[:, :width]
            specs.append(spec_np[None, :, :])  # [1, H, W]
            y_true.append(clip['label'])
        specs_np = np.stack(specs).astype(np.float32)  # [B, 1, H, W]
        outputs = session.run(None, {input_name: specs_np})
        logits = outputs[0]
        probs = softmax(logits)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
    # Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = np.mean(y_true == y_pred)
    print(f"Clip-level accuracy: {acc:.4f}")
    # Optionally print confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

if __name__ == "__main__":
    main()
