#!/usr/bin/env python3
"""
IRQ Diagnostic (pre-Fix-4 investigation).

Before implementing IRQ-aware feature normalization, this script answers a
prior question: *why* does XGBoost reach 0.69 LOOCV AUROC on irq_test while
RF/LR/MLP collapse below chance on the SAME 156-dim features?

It trains the irq_test LOOCV fold (train on the 4 non-irq workloads, test on
irq_test, binary clean-vs-observable) and produces:

  reports/figures/irq_diagnostic.png
    Panel A - XGB vs RF importance rank for every feature, colored by each
              feature's per-feature clean-vs-faulty AUROC *within* irq_test.
              Surfaces features XGB ranks high / RF ranks low that "separate"
              irq (the model-disagreement smoking gun).
    Panel B - distributions (clean vs faulty) of the top irq-separating
              features, annotated with which window block they come from.

It also prints a CONFOUND CHECK: baseline-block features are computed on the
pre-injection window (cycles 160-200, before the fault at 200), so they are
IDENTICAL across all faulty samples (fixed fault_cycle=200) and vary only for
clean samples (varied fault_cycle positions). Any separation they provide is a
fault_cycle-POSITION artifact, not fault signal.

Usage:
    venv/bin/python src/irq_diagnostic.py --data-dir data
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent

_SIG = ['if_active', 'mem_active', 'mem_read', 'mem_write', 'pipeline_stall',
        'mem_stall', 'branch_taken', 'opcode_alu', 'opcode_branch',
        'opcode_load', 'opcode_store', 'illegal_instr', 'pmp_region']
_STAT  = ['mean', 'std', 'max', 'nzfrac']
_BLOCK = ['window', 'baseline', 'delta']
NAMES  = [f'{s}_{st}_{b}' for b in _BLOCK for st in _STAT for s in _SIG]
BLOCK_OF = [b for b in _BLOCK for _ in _STAT for _ in _SIG]   # block per feature index


def _per_feature_irq_auc(X_irq, y_irq):
    """clean-vs-faulty AUROC of each feature alone, within irq_test."""
    aucs = np.full(X_irq.shape[1], 0.5)
    for j in range(X_irq.shape[1]):
        col = X_irq[:, j]
        if np.allclose(col, col[0]):
            continue
        try:
            a = roc_auc_score(y_irq, col)
            aucs[j] = max(a, 1 - a)
        except Exception:
            pass
    return aucs


def main(args):
    data_dir = Path(args.data_dir).resolve()
    X = np.load(str(data_dir / 'features_agg.npy'))
    y = np.load(str(data_dir / 'labels.npy'))
    meta = pd.read_csv(str(data_dir / 'metadata.csv'))

    bn = (y == 0) | (y == 1)            # binary: clean vs observable
    is_irq = (meta['workload'] == 'irq_test').values
    tr = bn & ~is_irq
    te = bn & is_irq
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]
    print(f"  train(non-irq) n={len(ytr)}  | irq_test n={len(yte)} "
          f"(clean={int((yte==0).sum())} obs={int((yte==1).sum())})")

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                n_jobs=-1, random_state=42).fit(Xtr_s, ytr)
    lr = LogisticRegression(max_iter=1000, class_weight='balanced',
                            random_state=42).fit(Xtr_s, ytr)
    models = {'RF': rf, 'LR': lr}
    if HAS_XGB:
        xg = xgb.XGBClassifier(n_estimators=200, eval_metric='logloss',
                               random_state=42).fit(Xtr_s, ytr)
        models['XGB'] = xg
    for nm, m in models.items():
        print(f"  {nm} irq AUROC = {roc_auc_score(yte, m.predict_proba(Xte_s)[:,1]):.3f}")

    xi = (models['XGB'].feature_importances_ if HAS_XGB
          else rf.feature_importances_)
    ri = rf.feature_importances_
    xgb_rank = np.argsort(np.argsort(-xi))   # 0 = most important
    rf_rank  = np.argsort(np.argsort(-ri))
    feat_auc = _per_feature_irq_auc(Xte, yte)

    # ---- CONFOUND CHECK: baseline features constant across faulty samples ----
    print("\n  CONFOUND CHECK - baseline block is pre-injection (cycles 160-200):")
    fault_mask = (yte == 1)
    clean_mask = (yte == 0)
    base_idx = [j for j, b in enumerate(BLOCK_OF) if b == 'baseline']
    win_idx  = [j for j, b in enumerate(BLOCK_OF) if b == 'window']
    base_const_frac = np.mean([
        np.allclose(Xte[fault_mask, j], Xte[fault_mask, j][0]) for j in base_idx
    ])
    print(f"    baseline features constant across ALL faulty irq samples: "
          f"{base_const_frac*100:.0f}%  (expected ~100% - fault not yet injected)")
    print(f"    mean per-feature irq-AUC  baseline-block = "
          f"{feat_auc[base_idx].mean():.3f}  (confounded by fault_cycle position)")
    print(f"    mean per-feature irq-AUC  window-block   = "
          f"{feat_auc[win_idx].mean():.3f}  (legit post-fault signal)")
    top_conf = sorted(base_idx, key=lambda j: -feat_auc[j])[:3]
    print("    top baseline 'separators' (position artifacts):")
    for j in top_conf:
        print(f"      {NAMES[j]:<28} irqAUC={feat_auc[j]:.3f}  "
              f"XGBrank={xgb_rank[j]:>3}  RFrank={rf_rank[j]:>3}")

    # ---------------------------- FIGURE --------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 6))
    gsA = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])

    # Panel A: XGB rank vs RF rank, colored by per-feature irq AUC
    axA = fig.add_subplot(gsA[0, 0])
    sca = axA.scatter(xgb_rank, rf_rank, c=feat_auc, cmap='viridis',
                      s=30, vmin=0.5, vmax=1.0, edgecolor='none')
    axA.plot([0, 155], [0, 155], '--', color='gray', alpha=0.5, lw=1,
             label='equal rank')
    # annotate the smoking-gun features: high XGB importance, low RF importance,
    # high irq-separation
    interesting = np.where((xgb_rank < 12) & (rf_rank > 20) & (feat_auc > 0.7))[0]
    for j in interesting:
        axA.annotate(NAMES[j], (xgb_rank[j], rf_rank[j]), fontsize=7,
                     xytext=(4, 3), textcoords='offset points')
        axA.scatter([xgb_rank[j]], [rf_rank[j]], s=90, facecolor='none',
                    edgecolor='red', lw=1.4)
    axA.set_xlabel('XGB importance rank (0 = most important)')
    axA.set_ylabel('RF importance rank')
    axA.set_title('A. Feature reliance: XGB vs RF (irq_test LOOCV fold)\n'
                  'red = high XGB / low RF reliance & separates irq')
    cb = fig.colorbar(sca, ax=axA)
    cb.set_label('per-feature clean-vs-faulty AUROC within irq_test')
    axA.legend(loc='lower right', fontsize=8)
    axA.grid(alpha=0.25)

    # Panel B: distributions of the top irq-separating features, clean vs faulty
    axB = fig.add_subplot(gsA[0, 1])
    top_sep = np.argsort(-feat_auc)[:4]
    pos = np.arange(len(top_sep))
    for k, j in enumerate(top_sep):
        cv = Xte[clean_mask, j]
        fv = Xte[fault_mask, j]
        # normalize each feature to [0,1] across irq for shared axis
        lo, hi = Xte[te_local := slice(None), j].min(), Xte[:, j].max()
        rng = (hi - lo) or 1.0
        cvn = (cv - lo) / rng
        fvn = (fv - lo) / rng
        axB.scatter(np.full_like(cvn, pos[k]-0.15) + np.random.uniform(-0.05, 0.05, len(cvn)),
                    cvn, s=10, color='#2ca02c', alpha=0.5,
                    label='clean' if k == 0 else None)
        axB.scatter(np.full_like(fvn, pos[k]+0.15) + np.random.uniform(-0.05, 0.05, len(fvn)),
                    fvn, s=10, color='#d62728', alpha=0.5,
                    label='faulty' if k == 0 else None)
    axB.set_xticks(pos)
    axB.set_xticklabels(
        [f'{NAMES[j]}\n[{BLOCK_OF[j]}] AUC={feat_auc[j]:.2f}' for j in top_sep],
        rotation=20, ha='right', fontsize=7)
    axB.set_ylabel('feature value (min-max normalized within irq)')
    axB.set_title('B. Top irq-separating features: clean vs faulty\n'
                  '(note: [baseline] = pre-fault -> position artifact)')
    axB.legend(fontsize=8)
    axB.grid(alpha=0.25, axis='y')

    fig.tight_layout()
    out = ITC2_ROOT / 'reports' / 'figures' / 'irq_diagnostic.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"\n  Figure saved: {out}")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description='IRQ diagnostic (pre-Fix-4)')
    p.add_argument('--data-dir', default=str(ITC2_ROOT / 'data'))
    return p.parse_args()


if __name__ == '__main__':
    import sys
    sys.exit(main(_parse_args()))
