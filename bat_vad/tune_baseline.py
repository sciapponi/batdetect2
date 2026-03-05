"""Grid-search helper for the traditional amplitude+Goertzel baseline

This script builds the same validation dataset used by
``benchmark_pytorch_model.py`` but forces ``return_wav=True`` so that raw
waveforms are available.  It then evaluates the hand-crafted detector over a
user-specified grid of amplitude and Goertzel thresholds, optionally with a
fixed list of frequencies, and reports the best settings according to a
selected metric (default: f1).

Usage example::

    python bat_vad/tune_baseline.py \
        --config bat_vad/configs/config_vad_2class.yaml \
        --bat-val ../example_data/brazil_val.yaml \
        --non-bat configs/non_bat_dataset.yaml \
        --clip-duration 0.1 \
        --samples 2000 \
        --amp-range 1e-4 1e-2 20 \
        --goer-range 1e-7 1e-5 20 \
        --goer-freqs 20000 40000 \
        --metric f1 \
        --out results.csv

The resulting CSV contains one row per parameter tuple along with all
computed performance metrics.
"""

import argparse
import csv
import itertools
import os
import random
import sys

# when running from inside bat_vad directory the package isn't on sys.path
# so add the repository root to ensure imports resolve correctly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from bat_vad.benchmark_pytorch_model import (
    TwoClassVADDataset,
    run_baseline,
    compute_metrics,
    build_audio_loader,
    PreprocessingConfig,
    build_preprocessor,
    load_dataset_from_config,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_grid(low, high, count, logscale=True):
    """Return ``count`` values between ``low`` and ``high``.

    If ``logscale`` is True (the default), points are spaced evenly in
    logarithmic space; otherwise they are linearly spaced.
    """
    if logscale:
        return list(np.logspace(np.log10(low), np.log10(high), count))
    else:
        return list(np.linspace(low, high, count))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tuning script for simple amplitude+Goertzel baseline"
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to VAD config YAML (same as benchmark script)")
    parser.add_argument("--bat-val", type=str, required=True,
                        help="Path to bat validation dataset yaml")
    parser.add_argument("--non-bat", type=str,
                        help="Path to non-bat dataset yaml (optional)")
    parser.add_argument(
        "--non-bat-split-seed",
        type=int,
        default=None,
        help="Seed used when splitting the non-bat set; see benchmark_pytorch_model"
    )
    parser.add_argument("--clip-duration", type=float, default=0.2,
                        help="Clip duration (seconds)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of validation samples to generate")
    parser.add_argument("--positive-ratio", type=float, default=0.5,
                        help="Positive sample ratio in the dataset")

    parser.add_argument(
        "--amp-range",
        type=float,
        nargs=3,
        metavar=("LOW", "HIGH", "COUNT"),
        default=[1e-4, 1e-2, 20],
        help="Low/high/count for amplitude threshold grid"
    )
    parser.add_argument(
        "--goer-range",
        type=float,
        nargs=3,
        metavar=("LOW", "HIGH", "COUNT"),
        default=[1e-7, 1e-5, 20],
        help="Low/high/count for Goertzel energy threshold"
    )
    parser.add_argument("--goer-freqs", type=float, nargs="+",
                        default=[25000.0, 50000.0],
                        help="Frequencies (Hz) used by the Goertzel filter")
    parser.add_argument("--metric", type=str, default="f1",
                        choices=["acc", "precision", "recall", "f1",
                                 "specificity"],
                        help="Metric used to choose the best parameters")
    parser.add_argument("--method", type=str, default="grid",
                        choices=["grid", "optuna"],
                        help="Search method: brute-force grid or Bayesian optuna")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of trials for optuna search (ignored in grid mode)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output (optuna progress bar)")
    parser.add_argument("--search-freqs", action="store_true",
                        help="Also search over Goertzel frequencies instead of using the entire list")
    parser.add_argument("--freq-range", type=float, nargs=3, metavar=("LOW","HIGH","COUNT"),
                        help="When searching frequencies, optionally treat them as a continuous grid between LOW and HIGH with COUNT points (log scale)")
    parser.add_argument("--out", type=str, default="baseline_grid.csv",
                        help="CSV file to write results to")

    args = parser.parse_args()

    # load config & build dataset
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    samplerate = cfg.get("audio", {}).get("samplerate", 256000)
    audio_loader = build_audio_loader()
    preproc_conf = None
    if "preprocess" in cfg:
        preproc_conf = PreprocessingConfig(**cfg["preprocess"])
    preprocessor = build_preprocessor(config=preproc_conf, input_samplerate=samplerate)

    bat_val_anns = load_dataset_from_config(args.bat_val)
    non_bat_val_anns = []
    if args.non_bat:
        from train_vad_2class import load_non_bat_dataset
        all_non_bat = load_non_bat_dataset(args.non_bat)
        if args.non_bat_split_seed is not None:
            rng = random.Random(args.non_bat_split_seed)
            rng.shuffle(all_non_bat)
        else:
            all_non_bat.sort(key=lambda ann: str(ann.clip.recording.path))
        split = int(0.8 * len(all_non_bat))
        non_bat_val_anns = all_non_bat[split:]
        print(f"Using {len(non_bat_val_anns)} non-bat validation recordings")

    val_ds = TwoClassVADDataset(
        bat_val_anns,
        non_bat_val_anns,
        audio_loader,
        preprocessor,
        clip_duration=args.clip_duration,
        samples_per_epoch=args.samples,
        positive_ratio=args.positive_ratio,
        return_wav=True,
    )

    best_score = -np.inf
    best_params = None

    header = ["amp_thresh", "goer_thresh"]
    if args.search_freqs:
        header.insert(0, "freq")
    header += ["acc", "precision", "recall", "f1", "specificity", "tp", "tn", "fp", "fn"]
    write_header = not os.path.exists(args.out)

    def evaluate_pair(a: float, g: float, freq: float | None = None):
        """Run baseline at given thresholds (and optional single frequency).

        If ``freq`` is provided we call the classifier with a one-element
        frequency list; otherwise the full ``args.goer_freqs`` list is used.
        """
        freqs = [freq] if (args.search_freqs and freq is not None) else args.goer_freqs
        labels, probs = run_baseline(val_ds, samplerate, a, freqs, g)
        return compute_metrics(labels, probs)

    if args.method == "grid":
        # build threshold grids
        amp_low, amp_high, amp_count = args.amp_range
        goer_low, goer_high, goer_count = args.goer_range
        amp_vals = make_grid(amp_low, amp_high, int(amp_count))
        goer_vals = make_grid(goer_low, goer_high, int(goer_count))

        # frequency values for search
        freq_vals = None
        if args.search_freqs:
            if args.freq_range:
                f_low, f_high, f_count = args.freq_range
                freq_vals = make_grid(f_low, f_high, int(f_count))
            else:
                freq_vals = args.goer_freqs

        with open(args.out, "w", newline="") as fout:
            writer = csv.writer(fout)
            if write_header:
                writer.writerow(header)

            if freq_vals is not None:
                for freq in freq_vals:
                    for a, g in itertools.product(amp_vals, goer_vals):
                        metrics = evaluate_pair(a, g, freq)
                        row = [freq, a, g] + [metrics[k] for k in header[3:]]
                        writer.writerow(row)
                        score = metrics[args.metric]
                        if score > best_score:
                            best_score = score
                            best_params = (a, g, metrics, freq)
            else:
                for a, g in itertools.product(amp_vals, goer_vals):
                    metrics = evaluate_pair(a, g)
                    writer.writerow([a, g] + [metrics[k] for k in header[2:]])
                    score = metrics[args.metric]
                    if score > best_score:
                        best_score = score
                        best_params = (a, g, metrics)
    else:
        # optuna mode
        # attempt to import optuna for Bayesian search; if unavailable fall back
        # to a simple random search using the same number of trials.
        use_optuna = True
        try:
            import optuna
        except ImportError:
            use_optuna = False
            print("WARNING: optuna not installed; falling back to random search")

        if use_optuna:
            study = optuna.create_study(direction="maximize")

            def objective(trial):
                # optuna 3.x prefers suggest_float with log=True
                a = trial.suggest_float("amp_thresh", args.amp_range[0], args.amp_range[1], log=True)
                g = trial.suggest_float("goer_thresh", args.goer_range[0], args.goer_range[1], log=True)
                freq = None
                if args.search_freqs:
                    if args.freq_range:
                        # continuous frequency
                        f_low, f_high, _ = args.freq_range
                        freq = trial.suggest_float("freq", f_low, f_high, log=True)
                    else:
                        freq = trial.suggest_categorical("freq", args.goer_freqs)
                metrics = evaluate_pair(a, g, freq)
                score = metrics[args.metric]
                # record results to CSV incrementally
                nonlocal write_header
                with open(args.out, "a", newline="") as fout:
                    writer = csv.writer(fout)
                    if write_header:
                        writer.writerow(header)
                        write_header = False
                    row = []
                    if args.search_freqs:
                        row.append(freq)
                    row += [a, g] + [metrics[k] for k in header[len(row)+2:]]
                    writer.writerow(row)
                return score

            study.optimize(
                objective,
                n_trials=args.n_trials,
                show_progress_bar=args.verbose,
            )
            best = study.best_params
            best_score = study.best_value
            if args.search_freqs:
                best_metrics = evaluate_pair(best["amp_thresh"], best["goer_thresh"], best["freq"])
                best_params = (best["amp_thresh"], best["goer_thresh"], best_metrics, best["freq"])
            else:
                best_metrics = evaluate_pair(best["amp_thresh"], best["goer_thresh"])
                best_params = (best["amp_thresh"], best["goer_thresh"], best_metrics)
        else:
            # random search fallback: sample uniformly in log space
            import math
            rng = random.Random(42)
            with open(args.out, "a", newline="") as fout:
                writer = csv.writer(fout)
                if write_header:
                    writer.writerow(header)
                    write_header = False
                for _ in range(args.n_trials):
                    loga = rng.uniform(math.log(args.amp_range[0]), math.log(args.amp_range[1]))
                    logg = rng.uniform(math.log(args.goer_range[0]), math.log(args.goer_range[1]))
                    a = math.exp(loga)
                    g = math.exp(logg)
                    freq = None
                    if args.search_freqs:
                        if args.freq_range:
                            f_low, f_high, _ = args.freq_range
                            freq = math.exp(rng.uniform(math.log(f_low), math.log(f_high)))
                        else:
                            freq = rng.choice(args.goer_freqs)
                    metrics = evaluate_pair(a, g, freq)
                    row = []
                    if args.search_freqs:
                        row.append(freq)
                    row += [a, g] + [metrics[k] for k in header[len(row)+2:]]
                    writer.writerow(row)
                    score = metrics[args.metric]
                    if score > best_score:
                        best_score = score
                        if args.search_freqs:
                            best_params = (a, g, metrics, freq)
                        else:
                            best_params = (a, g, metrics)

    print("\nBest parameters:")
    if args.search_freqs:
        # best_params = (a, g, metrics, freq)
        a_val, g_val, metrics_dict, freq_val = best_params
        print(f"  freq = {freq_val:.6g}")
    else:
        a_val, g_val, metrics_dict = best_params
    print(f"  amp_thresh = {a_val:.6g}")
    print(f"  goer_thresh = {g_val:.6g}")
    print("with metrics:")
    for k, v in metrics_dict.items():
        print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()
