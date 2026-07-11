#!/usr/bin/env python3
"""
Position-Confound Audit (across all 5 workloads).

Faulty samples are all injected at a fixed fault_cycle=200, while clean samples
span fault_cycle 50-500. Any feature that varies with window POSITION is
therefore a confounded clean-vs-faulty discriminator: it reports *when* the
window was taken, not *whether* a fault occurred.

The aggregate feature vector has three 52-dim blocks:
    window   (indices   0:52 )  computed on the POST-fault window (200-240)
    baseline (indices  52:104)  computed on the PRE-fault window  (160-200)
    delta    (indices 104:156)  window - baseline

The `baseline` block is pre-injection, so it is identical across all faulty
samples (verified per workload below) and carries ZERO fault information; any
separation it provides is pure fault_cycle position. `delta` inherits the
baseline. Only the `window` block can contain genuine post-fault signal (and
even it partly encodes position).

This script, per workload, compares the LOOCV-fold AUROC using the full 156-dim
vector vs. the window-only 52-dim vector. A large drop = the headline number was
inflated by the position confound.

Usage:
    venv/bin/python src/confound_audit.py --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent
WORKLOADS  = ['counting_loop', 'alu_heavy', 'branch_heavy', 'mem_intensive', 'irq_test']

WINDOW_SLICE   = slice(0, 52)      # post-fault block
BASELINE_SLICE = slice(52, 104)    # pre-fault block (confounded)
DELTA_SLICE    = slice(104, 156)


def _fit_auc(Xtr, ytr, Xte, yte, cols):
    sc = StandardScaler().fit(Xtr[:, cols])
    a, b = sc.transform(Xtr[:, cols]), sc.transform(Xte[:, cols])
    out = {}
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                n_jobs=-1, random_state=42).fit(a, ytr)
    out['RF'] = roc_auc_score(yte, rf.predict_proba(b)[:, 1])
    if HAS_XGB:
        xg = xgb.XGBClassifier(n_estimators=200, eval_metric='logloss',
                               random_state=42).fit(a, ytr)
        out['XGB'] = roc_auc_score(yte, xg.predict_proba(b)[:, 1])
    return out


def main(args):
    data_dir = Path(args.data_dir).resolve()
    X = np.load(str(data_dir / 'features_agg.npy'))
    y = np.load(str(data_dir / 'labels.npy'))
    meta = pd.read_csv(str(data_dir / 'metadata.csv'))
    bn = (y == 0) | (y == 1)

    full_cols = np.arange(156)
    win_cols  = np.arange(WINDOW_SLICE.start, WINDOW_SLICE.stop)

    print(f"\n  {'workload':<15} {'model':<5} {'full156':>8} {'window52':>9} "
          f"{'drop':>7} {'base_const':>11} {'best_base_AUC':>14}")
    print("  " + "-" * 74)

    rows = []
    for w in WORKLOADS:
        is_w = (meta['workload'] == w).values
        tr = bn & ~is_w
        te = bn & is_w
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        if len(np.unique(yte)) < 2:
            continue

        full = _fit_auc(Xtr, ytr, Xte, yte, full_cols)
        win  = _fit_auc(Xtr, ytr, Xte, yte, win_cols)

        # confound magnitude on the test workload
        fault = yte == 1
        base = X[te][:, BASELINE_SLICE]
        base_const = np.mean([np.allclose(base[fault, j], base[fault, j][0])
                              for j in range(base.shape[1])])
        # best single baseline feature = pure position artifact ceiling
        best_base = 0.5
        for j in range(base.shape[1]):
            col = X[te][:, BASELINE_SLICE.start + j]
            if np.allclose(col, col[0]):
                continue
            a = roc_auc_score(yte, col)
            best_base = max(best_base, a, 1 - a)

        for mdl in full:
            drop = full[mdl] - win[mdl]
            print(f"  {w:<15} {mdl:<5} {full[mdl]:>8.3f} {win[mdl]:>9.3f} "
                  f"{drop:>+7.3f} {base_const*100:>10.0f}% {best_base:>14.3f}")
            rows.append(dict(workload=w, model=mdl, full=full[mdl],
                             window_only=win[mdl], drop=drop,
                             baseline_const=base_const, best_baseline_auc=best_base))
        print()

    df = pd.DataFrame(rows)
    print("  " + "=" * 74)
    print("  Interpretation: 'drop' = how much AUROC the leaky baseline/delta")
    print("  blocks were adding. 'best_base_AUC' = AUROC of the single best")
    print("  pre-fault (zero-information) feature - a pure position artifact.")
    print(f"\n  Mean window-only AUROC (honest-ish signal): "
          f"RF={df[df.model=='RF'].window_only.mean():.3f}", end='')
    if HAS_XGB:
        print(f"  XGB={df[df.model=='XGB'].window_only.mean():.3f}")
    else:
        print()

    out = ITC2_ROOT / 'results' / 'confound_audit.json'
    out.parent.mkdir(exist_ok=True)
    df.to_json(str(out), orient='records', indent=2)
    print(f"  Saved: {out}")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description='Position-confound audit (all workloads)')
    p.add_argument('--data-dir', default=str(ITC2_ROOT / 'data'))
    return p.parse_args()


if __name__ == '__main__':
    import sys
    sys.exit(main(_parse_args()))
