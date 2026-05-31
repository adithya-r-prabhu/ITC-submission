#!/usr/bin/env python3
"""
Feature-Selection Sweep (post-confound-fix).

The 156-dim aggregate feature vector was designed BEFORE the position-confound
audit. Its three 52-dim blocks are:
    window   (  0: 52)  post-fault  - genuine signal
    baseline ( 52:104)  pre-fault   - ZERO fault information
    delta    (104:156)  window - baseline

The `baseline` block is computed on cycles *before* the fault, so it only ever
discriminated clean-vs-faulty via the old fixed-`fault_cycle=200` position
confound (now fixed by the shared position grid). Many of the 156 features are
therefore likely noise for the linear/MLP models, which - unlike trees - cannot
ignore irrelevant inputs.

This script ranks the 156 features by three independent measures, keeps the top
{20, 40, 80}, retrains all four baseline models on each subset, and compares
detection AUROC to the full-156 baseline, under both:
    Exp A - held-out fault sites + positions (default split)
    Exp C - leave-one-workload-out cross-validation

All rankings are computed on the TRAIN split only (no test leakage).

Outputs:
    results/feature_selection.json
    reports/figures/feature_selection_sweep.png

Usage:
    venv/bin/python src/feature_selection.py --data-dir data
    venv/bin/python src/feature_selection.py --data-dir data \
        --top-k 20 40 80 --methods rf xgb mutual_info
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

# Reuse the existing baseline pipeline (no copy-paste).
from train_real_fault_baselines import (AGG_FEATURE_NAMES, HAS_XGB, LABEL_CLEAN,
                                        LABEL_OBS, WORKLOADS, compute_metrics,
                                        filter_binary, get_split_mask,
                                        load_data, make_models)

if HAS_XGB:
    import xgboost as xgb

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT = SCRIPT_DIR.parent

N_FEATURES = 156
# Block boundaries (must match the aggregate feature layout).
BLOCKS = {'window': (0, 52), 'baseline': (52, 104), 'delta': (104, 156)}


def _block_of(idx: int) -> str:
    for name, (lo, hi) in BLOCKS.items():
        if lo <= idx < hi:
            return name
    return 'unknown'


def block_composition(cols: np.ndarray) -> dict:
    """Count how many selected feature indices fall in each block."""
    comp = {name: 0 for name in BLOCKS}
    for i in cols:
        comp[_block_of(int(i))] += 1
    return comp


# ---------------------------------------------------------------------------
# Ranking methods - fit on the TRAIN split only
# ---------------------------------------------------------------------------

def compute_rankings(X_tr: np.ndarray, y_tr: np.ndarray, methods: list) -> dict:
    """Return {method: ranked feature indices (best first)} for each method.

    X_tr is the binary-filtered, UNSCALED train matrix. RF/XGB are scale-
    invariant; mutual information is computed on standardized inputs.
    """
    rankings = {}

    if 'rf' in methods:
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                    n_jobs=-1, random_state=42)
        rf.fit(X_tr, y_tr)
        rankings['rf'] = {
            'order': np.argsort(rf.feature_importances_)[::-1],
            'scores': rf.feature_importances_,
        }

    if 'xgb' in methods:
        if not HAS_XGB:
            print("  WARNING: xgboost not installed; skipping 'xgb' ranking.")
        else:
            xg = xgb.XGBClassifier(n_estimators=200, eval_metric='logloss',
                                   random_state=42)
            xg.fit(X_tr, y_tr)
            rankings['xgb'] = {
                'order': np.argsort(xg.feature_importances_)[::-1],
                'scores': xg.feature_importances_,
            }

    if 'mutual_info' in methods:
        X_std = StandardScaler().fit_transform(X_tr)
        mi = mutual_info_classif(X_std, y_tr, random_state=42)
        rankings['mutual_info'] = {
            'order': np.argsort(mi)[::-1],
            'scores': mi,
        }

    return rankings


# ---------------------------------------------------------------------------
# Subset evaluation
# ---------------------------------------------------------------------------

def eval_exp_a(X_agg, y, meta, cols) -> dict:
    """Fit each model on train[:, cols], score on test[:, cols] (binary)."""
    Xtr, ytr, _ = filter_binary(X_agg, y, meta, get_split_mask(meta, 'train'))
    Xte, yte, _ = filter_binary(X_agg, y, meta, get_split_mask(meta, 'test'))
    Xtr, Xte = Xtr[:, cols], Xte[:, cols]

    scaler = StandardScaler().fit(Xtr)
    Xtr_sc, Xte_sc = scaler.transform(Xtr), scaler.transform(Xte)

    out = {}
    for name, clf in make_models('binary'):
        clf.fit(Xtr_sc, ytr)
        prob = clf.predict_proba(Xte_sc)
        pred = clf.predict(Xte_sc)
        m = compute_metrics(yte, pred, prob, 'binary')
        out[name] = {'auroc': m['auroc'], 'f1_macro': m['f1_macro']}
    return out


def eval_exp_c(X_agg, y, meta, cols) -> dict:
    """Leave-one-workload-out CV on the selected cols (binary). Returns
    per-model auroc_mean/auroc_std across the 5 folds + per-fold AUROCs."""
    per_fold = {wl: {} for wl in WORKLOADS}
    for holdout in WORKLOADS:
        tr_mask = (meta['workload'] != holdout).values
        te_mask = (meta['workload'] == holdout).values
        Xtr, ytr, _ = filter_binary(X_agg, y, meta, tr_mask)
        Xte, yte, _ = filter_binary(X_agg, y, meta, te_mask)
        if len(np.unique(yte)) < 2:
            continue
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
        scaler = StandardScaler().fit(Xtr)
        Xtr_sc, Xte_sc = scaler.transform(Xtr), scaler.transform(Xte)
        for name, clf in make_models('binary'):
            clf.fit(Xtr_sc, ytr)
            prob = clf.predict_proba(Xte_sc)
            pred = clf.predict(Xte_sc)
            m = compute_metrics(yte, pred, prob, 'binary')
            per_fold[holdout][name] = m['auroc']

    out = {}
    model_names = list(next(v for v in per_fold.values() if v).keys())
    for name in model_names:
        aurocs = [per_fold[wl][name] for wl in WORKLOADS
                  if name in per_fold[wl] and not np.isnan(per_fold[wl][name])]
        out[name] = {
            'auroc_mean': float(np.mean(aurocs)) if aurocs else float('nan'),
            'auroc_std': float(np.std(aurocs)) if aurocs else float('nan'),
            'per_fold': {wl: per_fold[wl].get(name, float('nan')) for wl in WORKLOADS},
        }
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def generate_figure(sweep: dict, full_a: dict, k_values: list, reports_dir: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available; skipping figure.")
        return

    methods = [m for m in sweep if sweep[m]]
    if not methods:
        return
    model_names = list(full_a.keys())
    x_all = k_values + [N_FEATURES]

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5),
                             sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        for name in model_names:
            ys = [sweep[method][k]['exp_a'][name]['auroc'] for k in k_values]
            ys.append(full_a[name]['auroc'])
            ax.plot(x_all, ys, marker='o', label=name)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'Ranking: {method}')
        ax.set_xlabel('# features kept (Exp A AUROC)')
        ax.set_xticks(x_all)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('AUROC')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Feature-selection sweep - Exp A detection AUROC vs. # features')
    plt.tight_layout()
    fig_dir = reports_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / 'feature_selection_sweep.png'
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"  Figure saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args) -> int:
    data_dir = Path(args.data_dir).resolve()
    results_dir = ITC2_ROOT / 'results'
    reports_dir = ITC2_ROOT / 'reports'
    results_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print("  ITC2 - Feature-Selection Sweep (binary: clean vs observable)")
    print(f"{'='*70}")

    X_seq, X_agg, y, meta = load_data(data_dir)
    k_values = sorted(args.top_k)
    methods = args.methods
    print(f"\n  Loaded {len(y)} samples; features_agg {X_agg.shape}")
    print(f"  top-k = {k_values}   methods = {methods}")

    # --- Rankings (train split, binary-filtered, unscaled) -----------------
    Xtr, ytr, _ = filter_binary(X_agg, y, meta, get_split_mask(meta, 'train'))
    rankings = compute_rankings(Xtr, ytr, methods)

    # --- Full-156 reference (same harness) ---------------------------------
    print("\n  full-156 baseline ...")
    all_cols = np.arange(N_FEATURES)
    full_a = eval_exp_a(X_agg, y, meta, all_cols)
    full_c = eval_exp_c(X_agg, y, meta, all_cols)
    for name in full_a:
        print(f"    [{name:<20}] ExpA AUROC={full_a[name]['auroc']:.4f}  "
              f"ExpC AUROC={full_c[name]['auroc_mean']:.4f}"
              f"+/-{full_c[name]['auroc_std']:.3f}")

    # --- Sweep --------------------------------------------------------------
    sweep = {m: {} for m in rankings}
    rankings_report = {}
    for method, info in rankings.items():
        order = info['order']
        rankings_report[method] = {
            'top_feature_names': [AGG_FEATURE_NAMES[int(i)] for i in order[:max(k_values)]],
            'top_feature_indices': [int(i) for i in order[:max(k_values)]],
        }
        print(f"\n  -- ranking: {method} ---------------------------------")
        for k in k_values:
            cols = order[:k]
            comp = block_composition(cols)
            exp_a = eval_exp_a(X_agg, y, meta, cols)
            exp_c = eval_exp_c(X_agg, y, meta, cols)
            sweep[method][k] = {
                'block_composition': comp,
                'exp_a': exp_a,
                'exp_c': {n: {'auroc_mean': exp_c[n]['auroc_mean'],
                              'auroc_std': exp_c[n]['auroc_std'],
                              'per_fold': exp_c[n]['per_fold']} for n in exp_c},
            }
            lr = exp_a.get('logistic_regression', {}).get('auroc', float('nan'))
            mlp = exp_a.get('mlp', {}).get('auroc', float('nan'))
            print(f"    top-{k:<3}  blocks(win/base/delta)="
                  f"{comp['window']}/{comp['baseline']}/{comp['delta']}  "
                  f"ExpA: LR={lr:.4f} MLP={mlp:.4f}")

    # --- Save JSON ----------------------------------------------------------
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if np.isnan(o) else float(o)
        if isinstance(o, float) and np.isnan(o):
            return None
        return o

    out = {
        'timestamp': datetime.now().isoformat(),
        'task': 'binary',
        'top_k': k_values,
        'methods': list(rankings.keys()),
        'block_layout': {k: list(v) for k, v in BLOCKS.items()},
        'rankings': rankings_report,
        'full_156': {'exp_a': full_a, 'exp_c': full_c},
        'sweep': sweep,
    }
    out_path = results_dir / 'feature_selection.json'
    with open(out_path, 'w') as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\n  Saved {out_path}")

    # --- Figure -------------------------------------------------------------
    generate_figure(sweep, full_a, k_values, reports_dir)

    print("\n  Done.")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description='Feature-selection sweep (binary)')
    p.add_argument('--data-dir', default=str(ITC2_ROOT / 'data'))
    p.add_argument('--top-k', type=int, nargs='+', default=[20, 40, 80])
    p.add_argument('--methods', nargs='+', default=['rf', 'xgb', 'mutual_info'],
                   choices=['rf', 'xgb', 'mutual_info'])
    p.add_argument('--task', default='binary', choices=['binary'],
                   help='Only binary is supported for now.')
    return p.parse_args()


if __name__ == '__main__':
    sys.exit(main(_parse_args()))
