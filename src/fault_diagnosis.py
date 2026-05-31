#!/usr/bin/env python3
"""
Fault Category Diagnosis Model - Improving Diagnostic Resolution.

The binary detector in train_real_fault_baselines.py answers *whether* a fault
is observable. It does not say *what kind* of fault it is. Stuck-at (SA) testing
gives zero diagnostic information beyond pass/fail. This model takes the next
step required by the competition topic ("improve diagnostic resolution"):

    given an OBSERVABLE fault, predict which circuit category produced it
    (ALU / Control / Decode / Memory / PC / RegFile).

This turns an opaque "fault detected" verdict into an actionable
"fault detected -> likely ALU or RegFile" diagnosis, narrowing the physical
search space for failure analysis.

Method
------
  * Filter the dataset to observable-only samples (label == 1).
  * Target = `category` from metadata.csv. The IRQ category (n=2, both in
    train) is dropped: too few samples to train or evaluate.
  * Train XGBoost (fallback: RandomForest) on the 156-dim aggregate features,
    using the same fault-site train/test split as Experiment A, so a fault
    site never appears in both train and test.
  * Evaluate on the test split: per-class one-vs-rest AUROC, confusion matrix,
    macro F1, accuracy.

Baseline comparison
-------------------
  SA testing provides 0 diagnostic resolution (binary pass/fail, no category).
  A uniform random guesser over 6 classes scores AUROC 0.5 per class and
  accuracy ~1/6 = 0.167. Both are reported alongside the model for context.

Outputs
-------
  results/fault_diagnosis.json
  reports/figures/diagnosis_auroc.png
  reports/figures/diagnosis_confusion.png
  models/fault_diagnosis.pkl

Usage
-----
    python src/fault_diagnosis.py --data-dir data
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent

LABEL_OBS    = 1
RANDOM_STATE = 42

# Categories with enough observable samples to model. IRQ (n=2) is excluded.
DROP_CATEGORIES = {'IRQ', 'clean'}

# 156-dim aggregate feature names: 13 signals x 4 stats x 3 blocks
_OBS_SIGNAL_NAMES = [
    'if_active', 'mem_active', 'mem_read', 'mem_write', 'pipeline_stall',
    'mem_stall', 'branch_taken', 'opcode_alu', 'opcode_branch',
    'opcode_load', 'opcode_store', 'illegal_instr', 'pmp_region',
]
_STAT_NAMES  = ['mean', 'std', 'max', 'nonzero_frac']
_BLOCK_NAMES = ['window', 'baseline', 'delta']

AGG_FEATURE_NAMES = [
    f'{sig}_{stat}_{blk}'
    for blk in _BLOCK_NAMES
    for stat in _STAT_NAMES
    for sig in _OBS_SIGNAL_NAMES
]


# ---------------------------------------------------------------------------
# Data loading and filtering
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    X_agg = np.load(str(data_dir / 'features_agg.npy'))
    y     = np.load(str(data_dir / 'labels.npy'))
    meta  = pd.read_csv(str(data_dir / 'metadata.csv'))
    return X_agg, y, meta


def filter_observable(X_agg: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
    """Keep observable faults whose category is modellable."""
    meta = meta.reset_index(drop=True)
    mask = (y == LABEL_OBS) & (~meta['category'].isin(DROP_CATEGORIES)).values
    return X_agg[mask], meta[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_diagnosis_model(n_classes: int):
    if HAS_XGB:
        return 'xgboost', xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            objective='multi:softprob', num_class=n_classes,
            eval_metric='mlogloss', random_state=RANDOM_STATE)
    return 'random_forest', RandomForestClassifier(
        n_estimators=300, class_weight='balanced', n_jobs=-1,
        random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_class_auroc(y_true: np.ndarray, y_prob: np.ndarray,
                    classes: list[str]) -> dict:
    """One-vs-rest AUROC per category. NaN where a class is absent in test."""
    result = {}
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            result[cls] = float('nan')
            continue
        try:
            result[cls] = float(roc_auc_score(y_bin, y_prob[:, i]))
        except Exception:
            result[cls] = float('nan')
    return result


def macro_auroc(per_class: dict) -> float:
    vals = [v for v in per_class.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float('nan')


def bootstrap_auroc_ci(y_true: np.ndarray, y_prob: np.ndarray,
                       classes: list[str], n_boot: int = 2000,
                       alpha: float = 0.05, seed: int = RANDOM_STATE):
    """Percentile bootstrap CIs for per-class one-vs-rest and macro AUROC.

    The model is fixed; only the held-out test set is resampled with
    replacement (case resampling). This is the standard way to attach a
    confidence interval to a point AUROC on a small test set, and it directly
    answers the reviewer concern about three-decimal AUROCs on n=5--10 classes.

    A bootstrap replicate can omit every positive of a rare class; that
    replicate yields NaN for that class and is dropped from *its* CI (the
    surviving count is reported as ``n_valid_boot``). The macro per replicate
    averages whatever classes remain valid in that replicate.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    per_class_samples = {c: [] for c in classes}
    macro_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        pc = per_class_auroc(yt, yp, classes)
        for c in classes:
            if not np.isnan(pc[c]):
                per_class_samples[c].append(pc[c])
        vals = [v for v in pc.values() if not np.isnan(v)]
        if vals:
            macro_samples.append(float(np.mean(vals)))
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def _ci(samples):
        if len(samples) >= 20:
            return {'lo': float(np.percentile(samples, lo_q)),
                    'hi': float(np.percentile(samples, hi_q)),
                    'median': float(np.median(samples)),
                    'n_valid_boot': len(samples)}
        return {'lo': float('nan'), 'hi': float('nan'),
                'median': float('nan'), 'n_valid_boot': len(samples)}

    per_class_ci = {c: _ci(per_class_samples[c]) for c in classes}
    macro_ci = _ci(macro_samples)
    return per_class_ci, macro_ci


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_per_class_auroc(per_class: dict, test_counts: dict, path: Path,
                         per_class_ci: dict | None = None):
    classes = list(per_class.keys())
    vals    = [per_class[c] for c in classes]
    fig, ax = plt.subplots(figsize=(8, 5))
    yerr = None
    if per_class_ci is not None:
        lo, hi = [], []
        for c, v in zip(classes, vals):
            ci = per_class_ci.get(c, {})
            if np.isnan(v) or np.isnan(ci.get('lo', float('nan'))):
                lo.append(0.0); hi.append(0.0)
            else:
                lo.append(max(0.0, v - ci['lo'])); hi.append(max(0.0, ci['hi'] - v))
        yerr = np.array([lo, hi])
    bars = ax.bar(range(len(classes)), vals, color='steelblue',
                  yerr=yerr, capsize=4,
                  error_kw={'ecolor': 'black', 'elinewidth': 1.2})
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6,
               label='random / SA diagnostic limit (0.5)')
    for i, (b, c) in enumerate(zip(bars, classes)):
        h = b.get_height()
        if not np.isnan(h):
            top = h + (yerr[1][i] if yerr is not None else 0.0)
            ax.text(i, top + 0.015, f'{h:.2f}\n(n={test_counts.get(c, 0)})',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=20, ha='right')
    ax.set_ylabel('One-vs-rest AUROC (test split)')
    ax.set_ylim(0, 1.12)
    ax.set_title('Fault category diagnosis - per-class AUROC')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, classes: list[str], path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=20, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted category')
    ax.set_ylabel('True category')
    ax.set_title('Fault category diagnosis - confusion matrix (test)')
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def _serialise(d):
    if isinstance(d, dict):
        return {k: _serialise(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_serialise(v) for v in d]
    if isinstance(d, np.integer):
        return int(d)
    if isinstance(d, np.floating):
        return float(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args) -> int:
    data_dir    = Path(args.data_dir).resolve()
    models_dir  = ITC2_ROOT / 'models'
    results_dir = ITC2_ROOT / 'results'
    fig_dir     = ITC2_ROOT / 'reports' / 'figures'
    for d in [models_dir, results_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print("  ITC2 - Fault Category Diagnosis (diagnostic resolution)")
    print(f"{'='*64}")

    X_agg, y, meta = load_data(data_dir)
    X_obs, m_obs = filter_observable(X_agg, y, meta)
    print(f"\n  Observable, modellable samples: {len(m_obs)}")
    print(f"  Dropped categories: {sorted(DROP_CATEGORIES)} "
          f"(IRQ has too few observable samples)")

    # Encode categories from the full observable set so train/test share a
    # consistent class index even if test is missing a rare class.
    le = LabelEncoder().fit(m_obs['category'].values)
    classes = list(le.classes_)
    print(f"  Classes ({len(classes)}): {classes}")

    train_mask = (m_obs['split'] == 'train').values
    test_mask  = (m_obs['split'] == 'test').values

    Xtr = X_obs[train_mask]
    Xte = X_obs[test_mask]
    ytr = le.transform(m_obs.loc[train_mask, 'category'].values)
    yte = le.transform(m_obs.loc[test_mask, 'category'].values)
    print(f"  Train: {len(ytr)}   Test: {len(yte)}")

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)

    name, clf = make_diagnosis_model(len(classes))
    print(f"\n  Model: {name}")
    clf.fit(Xtr_sc, ytr)
    pred = clf.predict(Xte_sc)
    prob = clf.predict_proba(Xte_sc)

    per_class = per_class_auroc(yte, prob, classes)
    cm        = confusion_matrix(yte, pred, labels=range(len(classes)))
    test_counts = {c: int((yte == i).sum()) for i, c in enumerate(classes)}

    # Bootstrap 95% CIs over the held-out test set (model fixed, cases resampled)
    per_class_ci, macro_ci = bootstrap_auroc_ci(
        yte, prob, classes, n_boot=args.n_boot, seed=args.seed)

    metrics = {
        'accuracy':       float(accuracy_score(yte, pred)),
        'f1_macro':       float(f1_score(yte, pred, average='macro', zero_division=0)),
        'precision_macro': float(precision_score(yte, pred, average='macro', zero_division=0)),
        'recall_macro':   float(recall_score(yte, pred, average='macro', zero_division=0)),
        'auroc_macro':    macro_auroc(per_class),
    }

    print(f"\n  {'category':<10} {'AUROC':>7}  {'95% CI':>16}  {'n_test':>6}")
    for c in classes:
        au = per_class[c]
        au_s = f"{au:.3f}" if not np.isnan(au) else "  nan"
        ci = per_class_ci[c]
        ci_s = (f"[{ci['lo']:.3f}, {ci['hi']:.3f}]"
                if not np.isnan(ci['lo']) else "      ---")
        print(f"  {c:<10} {au_s:>7}  {ci_s:>16}  {test_counts[c]:>6}")
    print(f"\n  macro AUROC : {metrics['auroc_macro']:.3f}  "
          f"95% CI [{macro_ci['lo']:.3f}, {macro_ci['hi']:.3f}]")
    print(f"  macro F1    : {metrics['f1_macro']:.3f}")
    print(f"  accuracy    : {metrics['accuracy']:.3f}")
    print(f"  bootstrap   : {args.n_boot} replicates, seed {args.seed}")

    # Baselines for context: SA gives no category; random over K classes.
    n_classes = len(classes)
    baseline = {
        'sa_testing': {
            'description': 'Stuck-at pass/fail gives no fault category.',
            'diagnostic_resolution': 'none (binary pass/fail only)',
            'per_class_auroc': 0.5,
        },
        'random_guess': {
            'description': f'Uniform guess over {n_classes} categories.',
            'accuracy': 1.0 / n_classes,
            'per_class_auroc': 0.5,
        },
    }

    # Figures
    auroc_fig = fig_dir / 'diagnosis_auroc.png'
    conf_fig  = fig_dir / 'diagnosis_confusion.png'
    plot_per_class_auroc(per_class, test_counts, auroc_fig, per_class_ci)
    plot_confusion(cm, classes, conf_fig)
    print(f"\n  Figures: {auroc_fig.name}, {conf_fig.name}")

    # Save model
    ckpt = models_dir / 'fault_diagnosis.pkl'
    with open(ckpt, 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler, 'label_encoder': le,
                     'classes': classes}, f)

    # Save results JSON
    out = {
        'generated':  datetime.now().isoformat(timespec='seconds'),
        'model':      name,
        'task':       'fault_category_diagnosis',
        'classes':    classes,
        'dropped_categories': sorted(DROP_CATEGORIES),
        'n_train':    int(len(ytr)),
        'n_test':     int(len(yte)),
        'test_counts': test_counts,
        'metrics':    metrics,
        'per_class_auroc': per_class,
        'bootstrap': {
            'n_boot': int(args.n_boot),
            'seed':   int(args.seed),
            'ci_level': 0.95,
            'method': 'percentile bootstrap, test-set case resampling, model fixed',
            'per_class_auroc_ci': per_class_ci,
            'macro_auroc_ci': macro_ci,
        },
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': classes,
        'baseline':   baseline,
    }
    out_path = results_dir / 'fault_diagnosis.json'
    with open(out_path, 'w') as f:
        json.dump(_serialise(out), f, indent=2)
    print(f"  Results: {out_path}")
    print(f"\n{'='*64}\n")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data',
                   help='Directory with features_agg.npy, labels.npy, metadata.csv')
    p.add_argument('--n-boot', type=int, default=2000,
                   help='Bootstrap replicates for AUROC confidence intervals')
    p.add_argument('--seed', type=int, default=RANDOM_STATE,
                   help='RNG seed for the bootstrap resampling')
    return p.parse_args()


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
