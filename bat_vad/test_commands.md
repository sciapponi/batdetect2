```bash
    uv run python train_vad_2class.py     --config configs/config_vad_2class_ema.yaml     --bat-train ../example_data/uk_same_train.yaml     --bat-val ../example_data/uk_same_val.yaml     --non-bat configs/non_bat_dataset.yaml     --clip-duration 0.1     --epochs 20     --non-bat-split-seed 123
```
```bash
    uv run python benchmark_pytorch_model.py --config configs/config_vad_2class_ema.yaml --bat-val ../example_data/uk_same_val.yaml --non-bat configs/non_bat_dataset.yaml --ckpt vad_2class_model.pt --clip-duration 0.1     --non-bat-split-seed 123 
```

```bash
    uv run python benchmark_pytorch_model.py --config configs/config_vad_2class_ema.yaml --bat-val ../example_data/uk_same_val.yaml --non-bat configs/non_bat_dataset.yaml --ckpt vad_2class_model.pt --clip-duration 0.1     --non-bat-split-seed 123 --baseline \
        --amp-thresh 0.002 \
        --goertzel-freqs 20000 40000 \
        --goertzel-thresh 5e-7
```

```bash
    uv run python tune_baseline.py --config configs/config_vad_2class.yaml --bat-val ../example_data/brazil_val.yaml --non-bat configs/non_bat_dataset.yaml --clip-duration 0.1 --samples 100 --method optuna --n-trials 100 --goer-freqs 20000 40000 --metric specificity --verbose --out baseline_opt.csv
```

uv run python tune_baseline.py \
    --config configs/config_vad_2class.yaml \
    --bat-val ../example_data/uk_same_val.yaml \
    --non-bat configs/non_bat_dataset.yaml \
    --clip-duration 0.1 \
    --samples 1000 \
    --method optuna \
    --n-trials 200 \
    --goer-freqs 20000 40000 60000 \
    --search-freqs \
    --metric specificity \
    --verbose \
    --out baseline_freq_optuna.csv

uv run python tune_baseline.py \
    --config configs/config_vad_2class.yaml \
    --bat-val ../example_data/brazil_val.yaml \
    --non-bat configs/non_bat_dataset.yaml \
    --clip-duration 0.1 \
    --samples 1000 \
    --method optuna \
    --n-trials 200 \
    --goer-freqs 20000 40000 60000 \
    --search-freqs \
    --freq-range 20000 60000 20 \
    --metric specificity \
    --verbose \
    --out baseline_freq_optuna.csv

uv run python benchmark_pytorch_model.py \
    --config configs/config_vad_2class_ema.yaml \
    --bat-val ../example_data/brazil_val.yaml \
    --non-bat configs/non_bat_dataset.yaml \
    --ckpt vad_2class_model.pt \
    --clip-duration 0.1 \
    --samples 1000 \
    --batch-size 32 \
    --amp-gate 0.002

uv run python benchmark_pytorch_model.py --config configs/config_vad_2class_ema.yaml --bat-val ../example_data/uk_same_val.yaml --non-bat configs/non_bat_dataset.yaml --ckpt vad_2class_model.pt --clip-duration 0.1 --batch-size 32 --samples 50 --espdl-examples 8

# quieter run without the big PPQ noise/signal diagnostics
uv run python benchmark_pytorch_model.py --config configs/config_vad_2class_ema.yaml --bat-val ../example_data/uk_same_val.yaml --non-bat configs/non_bat_dataset.yaml --ckpt vad_2class_model.pt --clip-duration 0.1 --batch-size 32 --samples 50 --espdl-examples 8 --espdl-quiet

# use more calibration batches (loops if necessary) to improve quant accuracy
uv run python benchmark_pytorch_model.py --config configs/config_vad_2class_ema.yaml --bat-val ../example_data/brazil_val.yaml --non-bat configs/non_bat_dataset.yaml --ckpt vad_2class_model.pt --clip-duration 0.1 --batch-size 32  --espdl-calib-steps 512

# improved ESP-DL quantization
uv run python benchmark_pytorch_model.py \
    --config configs/config_vad_2class_ema.yaml \
    --bat-val ../example_data/uk_same_val.yaml \
    --non-bat configs/non_bat_dataset.yaml \
    --ckpt vad_2class_model.pt \
    --clip-duration 0.1 \
    --batch-size 32 \
    --espdl-calib-steps 512 \
    --keep-fp32-layers "node_linear,node_conv2d_2"  # for problematic datasets like Brazil
