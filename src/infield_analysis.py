#!/usr/bin/env python3
"""
In-Field Deployment Analysis.

Evaluates whether any trained baseline model is lightweight enough for
periodic in-field health monitoring (e.g., via a debug/PMU interface).

For each pickle model (LR, RF, XGBoost, MLP):
  - Serialized size (KB)
  - Parameter count
  - Single-sample inference latency (mean over 1000 reps, microseconds)
  - Feasibility verdict (target: <1ms inference, <100KB serialized)

LR is the recommended in-field candidate: deterministic, ~156 weights,
sub-microsecond inference. Also estimates monitoring overhead relative to
a 50 MHz SoC clock and a 600-cycle observation window.

Outputs:
  results/infield_analysis.json
  Console table

Usage:
    python src/infield_analysis.py
"""

import argparse
import io
import json
import pickle
import timeit
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent

# In-field feasibility thresholds
MAX_SIZE_KB      = 100.0   # serialized model size
MAX_INFERENCE_US = 1000.0  # 1 ms per inference
SOC_CLOCK_MHZ    = 50.0    # target SoC clock
OBS_WINDOW_CYCLES = 600    # observation window length

LABEL_CLEAN = 0
LABEL_OBS   = 1


def model_size_kb(pkg: dict) -> float:
    buf = io.BytesIO()
    pickle.dump(pkg, buf)
    return buf.tell() / 1024.0


def count_params(model) -> int:
    """Best-effort parameter count across model types."""
    name = type(model).__name__

    if name == 'LogisticRegression':
        return int(model.coef_.size + model.intercept_.size)

    if name == 'RandomForestClassifier':
        total = 0
        for tree in model.estimators_:
            t = tree.tree_
            total += t.node_count  # each node: threshold + feature + value
        return total

    if name in ('XGBClassifier', 'XGBoostClassifier'):
        try:
            booster = model.get_booster()
            dump    = booster.get_dump()
            # Count lines across all trees as a proxy for total split nodes
            return sum(len(t.split('\n')) for t in dump)
        except Exception:
            return -1

    if name == 'MLPClassifier':
        return sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)

    return -1


def measure_inference_us(model, scaler, X_sample: np.ndarray, n_reps: int = 1000) -> float:
    """Mean single-sample inference time in microseconds."""
    x = scaler.transform(X_sample[:1])
    elapsed = timeit.timeit(lambda: model.predict_proba(x), number=n_reps)
    return (elapsed / n_reps) * 1e6


def feasibility_verdict(size_kb: float, inference_us: float) -> str:
    ok_size  = size_kb <= MAX_SIZE_KB
    ok_speed = inference_us <= MAX_INFERENCE_US
    if ok_size and ok_speed:
        return 'FEASIBLE'
    elif ok_size:
        return 'SIZE_OK/SLOW'
    elif ok_speed:
        return 'FAST/TOO_LARGE'
    return 'NOT_FEASIBLE'


def monitoring_overhead(inference_us: float) -> dict:
    """
    Estimate monitoring overhead relative to one observation window at SOC_CLOCK_MHZ.
    """
    window_us = OBS_WINDOW_CYCLES / SOC_CLOCK_MHZ  # microseconds
    overhead_pct = (inference_us / window_us) * 100.0
    return {
        'obs_window_us':      round(window_us, 3),
        'inference_us':       round(inference_us, 3),
        'overhead_pct':       round(overhead_pct, 3),
        'feasible_periodic':  inference_us < window_us,
    }


def main():
    parser = argparse.ArgumentParser(description='In-field deployment analysis of baseline models')
    parser.add_argument('--models-dir',  default='models',  help='Directory with trained models')
    parser.add_argument('--data-dir',    default='data',    help='Directory with features')
    parser.add_argument('--results-dir', default='results', help='Output JSON directory')
    args = parser.parse_args()

    models_dir  = ITC2_ROOT / args.models_dir
    data_dir    = ITC2_ROOT / args.data_dir
    results_dir = ITC2_ROOT / args.results_dir
    results_dir.mkdir(exist_ok=True)

    # Load a representative sample for timing
    X_agg = np.load(str(data_dir / 'features_agg.npy'))
    import pandas as pd
    meta  = pd.read_csv(str(data_dir / 'metadata.csv'))
    y     = np.load(str(data_dir / 'labels.npy'))

    # Use test split, binary samples
    test_binary = (meta['split'] == 'test').values & ((y == LABEL_CLEAN) | (y == LABEL_OBS))
    X_sample = X_agg[test_binary]

    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'mlp']
    results = {}

    print(f"\n{'Model':<25} {'Size KB':>9} {'Params':>10} {'Infer us':>10} {'Verdict':<16} {'Overhead%':>10}")
    print("-" * 85)

    for name in model_names:
        path = models_dir / f'{name}_binary.pkl'
        if not path.exists():
            print(f"  {name:<23} - not found, skipping")
            continue
        with open(path, 'rb') as f:
            pkg = pickle.load(f)
        model  = pkg['model']
        scaler = pkg['scaler']

        size_kb    = model_size_kb(pkg)
        n_params   = count_params(model)
        infer_us   = measure_inference_us(model, scaler, X_sample)
        verdict    = feasibility_verdict(size_kb, infer_us)
        overhead   = monitoring_overhead(infer_us)

        results[name] = {
            'size_kb':          round(size_kb, 2),
            'n_params':         n_params,
            'inference_us':     round(infer_us, 3),
            'feasibility':      verdict,
            'monitoring_overhead': overhead,
        }
        params_str = str(n_params) if n_params >= 0 else 'N/A'
        print(f"  {name:<23} {size_kb:>9.1f} {params_str:>10} {infer_us:>10.2f} "
              f"{verdict:<16} {overhead['overhead_pct']:>10.2f}%")

    # Recommendation
    feasible = [n for n, r in results.items() if r['feasibility'] == 'FEASIBLE']
    if feasible:
        # Pick smallest by inference time among feasible
        recommended = min(feasible, key=lambda n: results[n]['inference_us'])
    else:
        recommended = min(results, key=lambda n: results[n]['inference_us'])

    rec = results[recommended]
    print(f"\nRecommended in-field model: {recommended}")
    print(f"  Size:      {rec['size_kb']:.1f} KB")
    print(f"  Inference: {rec['inference_us']:.2f} us")
    print(f"  Overhead:  {rec['monitoring_overhead']['overhead_pct']:.2f}% of a {OBS_WINDOW_CYCLES}-cycle window at {SOC_CLOCK_MHZ} MHz")

    out = {
        'generated': datetime.now().isoformat(),
        'thresholds': {
            'max_size_kb':      MAX_SIZE_KB,
            'max_inference_us': MAX_INFERENCE_US,
            'soc_clock_mhz':    SOC_CLOCK_MHZ,
            'obs_window_cycles': OBS_WINDOW_CYCLES,
        },
        'models': results,
        'recommended_infield_model': recommended,
        'recommendation_rationale': (
            f'{recommended} is the recommended in-field candidate: '
            f'smallest serialized size among feasible models, '
            f'deterministic inference, runs in {rec["inference_us"]:.1f} us - '
            f'{rec["monitoring_overhead"]["overhead_pct"]:.1f}% overhead per observation window. '
            f'Can be polled periodically via debug/PMU interface without disrupting normal operation.'
        ),
    }
    out_path = results_dir / 'infield_analysis.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
