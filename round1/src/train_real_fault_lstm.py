#!/usr/bin/env python3
"""
Real-Fault LSTM Training.

Reuses HierarchicalBiLSTMClassifier from ITC/src/model.py (unchanged)
and trains it on the real-fault dataset produced by build_real_fault_dataset.py.

Tasks:
  binary  - class 0 (clean) vs class 1 (observable_fault)
  3class  - class 0, 1, 2 (clean / observable / silent)
  all     - both

Generalization experiments (same A/B/C as baselines):
  Exp A - held-out fault sites (train/val/test split)
  Exp B - held-out workload
  Exp C - leave-one-workload-out CV

Outputs:
  models/lstm_binary.pt
  models/lstm_3class.pt
  results/lstm_binary.json
  results/lstm_3class.json
  results/lstm_generalization.json

Usage:
    python src/train_real_fault_lstm.py --data-dir data
    python src/train_real_fault_lstm.py --data-dir data --task binary --epochs 30
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent
ITC_SRC    = ITC2_ROOT.parent / 'ITC' / 'src'

if str(ITC_SRC) not in sys.path:
    sys.path.insert(0, str(ITC_SRC))

from model import HierarchicalBiLSTMClassifier, device

LABEL_CLEAN  = 0
LABEL_OBS    = 1
LABEL_SILENT = 2
WORKLOADS    = ['counting_loop', 'alu_heavy', 'branch_heavy', 'mem_intensive', 'irq_test']


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealFaultDataset(Dataset):
    """
    Wraps the .npy arrays produced by build_real_fault_dataset.py.
    mask: boolean array selecting which samples to include.
    remap: dict mapping original labels to contiguous 0-based labels (for binary task).
    """
    def __init__(self, X_seq: np.ndarray, X_agg: np.ndarray,
                 y: np.ndarray, mask: np.ndarray, remap: dict | None = None):
        self.X_seq = torch.tensor(X_seq[mask], dtype=torch.float32)
        self.X_agg = torch.tensor(X_agg[mask], dtype=torch.float32)
        raw_y = y[mask].copy()
        if remap:
            raw_y = np.vectorize(remap.get)(raw_y)
        self.y = torch.tensor(raw_y, dtype=torch.long)
        self.window_size = X_seq.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        y_loc = torch.zeros(self.window_size, dtype=torch.float32)
        return self.X_seq[idx], self.X_agg[idx], self.y[idx], y_loc


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts  = Counter(labels.tolist())
    weights = [1.0 / counts[int(lbl)] for lbl in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def _forward_loss(model, x_seq, x_agg, y, y_loc, num_classes, device,
                  crit_spec, crit_domain, crit_binary, crit_loc):
    x_seq = x_seq.to(device); x_agg = x_agg.to(device)
    y     = y.to(device);     y_loc = y_loc.to(device)

    spec_l, dom_l, bin_l, loc_l, _, fused = model(x_seq, x_agg)

    # For real-fault training we only have the primary class label.
    # Domain and binary labels are derived from the class label.
    if num_classes == 2:
        y_domain = y.clone()          # same as binary: 0=clean, 1=fault
        y_binary = y.clone()
    else:
        # 3-class: domain = same as label (0/1/2 -> 3 domains)
        y_domain = y.clone()
        y_binary = (y > 0).long()

    loss = (
        crit_spec(spec_l, y)
        + 0.50 * crit_domain(dom_l, y_domain)
        + 0.30 * crit_binary(bin_l, y_binary)
        + 0.50 * crit_loc(loc_l, y_loc)
    )
    return loss, spec_l, fused


def train_model(X_seq, X_agg, y, train_mask, val_mask,
                num_classes, args, ckpt_path: str) -> tuple:
    """Train one model, return (best_val_loss, best_val_acc, final model)."""
    remap = {LABEL_CLEAN: 0, LABEL_OBS: 1} if num_classes == 2 else None
    if remap:
        train_mask = train_mask & ((y == LABEL_CLEAN) | (y == LABEL_OBS))
        val_mask   = val_mask   & ((y == LABEL_CLEAN) | (y == LABEL_OBS))

    train_ds = RealFaultDataset(X_seq, X_agg, y, train_mask, remap)
    val_ds   = RealFaultDataset(X_seq, X_agg, y, val_mask,   remap)

    sampler      = make_weighted_sampler(train_ds.y.numpy())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    num_domains = num_classes  # one domain per class in this simplified setup
    num_binary  = 2
    model = HierarchicalBiLSTMClassifier(
        input_dim=13, hidden_dim=64,
        num_classes=num_classes, agg_dim=156, num_domains=num_domains,
    ).to(device)

    # Class weights
    counts = Counter(train_ds.y.tolist())
    raw_w  = [1.0 / counts.get(c, 1) for c in range(num_classes)]
    mean_w = sum(raw_w) / len(raw_w)
    loss_w = torch.tensor([w / mean_w for w in raw_w], dtype=torch.float32).to(device)

    crit_spec   = nn.CrossEntropyLoss(weight=loss_w, label_smoothing=0.05)
    crit_domain = nn.CrossEntropyLoss(label_smoothing=0.05)
    crit_binary = nn.CrossEntropyLoss()
    crit_loc    = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-5)

    best_val_loss = float('inf')
    best_val_acc  = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x_seq, x_agg, y_b, y_loc in train_loader:
            optimizer.zero_grad()
            loss, spec_l, _ = _forward_loss(
                model, x_seq, x_agg, y_b, y_loc, num_classes, device,
                crit_spec, crit_domain, crit_binary, crit_loc)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss    += loss.item() * x_seq.size(0)
            t_correct += (spec_l.argmax(1) == y_b.to(device)).sum().item()
            t_total   += x_seq.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x_seq, x_agg, y_b, y_loc in val_loader:
                loss, spec_l, _ = _forward_loss(
                    model, x_seq, x_agg, y_b, y_loc, num_classes, device,
                    crit_spec, crit_domain, crit_binary, crit_loc)
                v_loss    += loss.item() * x_seq.size(0)
                v_correct += (spec_l.argmax(1) == y_b.to(device)).sum().item()
                v_total   += x_seq.size(0)

        t_loss /= t_total; v_loss /= v_total
        v_acc   = v_correct / v_total

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{args.epochs} | "
                  f"Train loss={t_loss:.4f} | Val loss={v_loss:.4f} acc={v_acc*100:.1f}%")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_val_acc  = v_acc
            torch.save(model.state_dict(), ckpt_path)

        scheduler.step(v_loss)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_val_loss, best_val_acc, model


def evaluate_model(model, X_seq, X_agg, y, mask, num_classes, remap=None) -> dict:
    if remap:
        mask = mask & ((y == LABEL_CLEAN) | (y == LABEL_OBS))
    ds     = RealFaultDataset(X_seq, X_agg, y, mask, remap)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for x_seq, x_agg, y_b, _ in loader:
            x_seq = x_seq.to(device); x_agg = x_agg.to(device)
            spec_l, _, _, _, _, _ = model(x_seq, x_agg)
            probs  = torch.softmax(spec_l, dim=1).cpu().numpy()
            preds  = spec_l.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())
            all_probs.extend(probs)

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    probs   = np.array(all_probs)

    acc = float(np.mean(preds == targets))
    try:
        if num_classes == 2:
            auroc = float(roc_auc_score(targets, probs[:, 1]))
        else:
            auroc = float(roc_auc_score(targets, probs, multi_class='ovr', average='macro'))
    except Exception:
        auroc = float('nan')

    return {'accuracy': acc, 'auroc': auroc,
            'n_samples': len(targets), 'num_classes': num_classes}


def workload_breakdown(model, X_seq, X_agg, y, meta, num_classes, remap=None) -> dict:
    result = {}
    for wl in WORKLOADS:
        wl_mask = (meta['workload'] == wl).values
        m = evaluate_model(model, X_seq, X_agg, y, wl_mask, num_classes, remap)
        result[wl] = m
    return result


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_task(X_seq, X_agg, y, meta, num_classes, task_name,
             args, models_dir, results_dir) -> dict:
    print(f"\n{'-'*64}")
    print(f"  LSTM task: {task_name.upper()}  ({num_classes}-class  device={device})")
    print(f"{'-'*64}")

    remap = {LABEL_CLEAN: 0, LABEL_OBS: 1} if num_classes == 2 else None

    train_mask = (meta['split'] == 'train').values
    val_mask   = (meta['split'] == 'val').values
    test_mask  = (meta['split'] == 'test').values

    ckpt = str(models_dir / f'lstm_{task_name}.pt')

    best_loss, best_acc, model = train_model(
        X_seq, X_agg, y, train_mask, val_mask, num_classes, args, ckpt)

    print(f"\n  Best val loss={best_loss:.4f}  acc={best_acc*100:.1f}%")
    print("  Evaluating on test split ...")

    test_metrics = evaluate_model(model, X_seq, X_agg, y, test_mask, num_classes, remap)
    print(f"  Test AUROC={test_metrics['auroc']:.4f}  Acc={test_metrics['accuracy']*100:.1f}%")

    wl_br = workload_breakdown(model, X_seq, X_agg, y, meta, num_classes, remap)

    # Exp B: held-out workload
    print(f"\n  Exp B - held-out workload: {args.holdout_workload}")
    holdout_train = (meta['workload'] != args.holdout_workload).values
    holdout_val   = (meta['workload'] == args.holdout_workload).values
    _, _, model_b = train_model(
        X_seq, X_agg, y, holdout_train, holdout_val, num_classes, args,
        str(models_dir / f'lstm_{task_name}_expb.pt'))
    exp_b = evaluate_model(model_b, X_seq, X_agg, y, holdout_val, num_classes, remap)
    print(f"  Held-out AUROC={exp_b['auroc']:.4f}")

    # Exp C: leave-one-workload-out CV
    print("\n  Exp C - leave-one-workload-out CV")
    exp_c_folds = {}
    for holdout in WORKLOADS:
        tr_m = (meta['workload'] != holdout).values
        te_m = (meta['workload'] == holdout).values
        _, _, m_c = train_model(
            X_seq, X_agg, y, tr_m, te_m, num_classes, args,
            str(models_dir / f'lstm_{task_name}_loocv_{holdout}.pt'))
        r = evaluate_model(m_c, X_seq, X_agg, y, te_m, num_classes, remap)
        exp_c_folds[holdout] = r
        print(f"    held-out={holdout:16s}  AUROC={r['auroc']:.4f}")

    aurocs = [v['auroc'] for v in exp_c_folds.values() if not np.isnan(v['auroc'])]
    exp_c_summary = {
        'auroc_mean': float(np.mean(aurocs)) if aurocs else float('nan'),
        'auroc_std':  float(np.std(aurocs))  if aurocs else float('nan'),
        'per_fold':   exp_c_folds,
    }
    print(f"  LOOCV AUROC = {exp_c_summary['auroc_mean']:.4f} +/- {exp_c_summary['auroc_std']:.4f}")

    return {
        'task': task_name,
        'num_classes': num_classes,
        'best_val_loss': float(best_loss),
        'best_val_acc':  float(best_acc),
        'test': test_metrics,
        'auroc_by_workload': {wl: v['auroc'] for wl, v in wl_br.items()},
        'exp_b': {'holdout': args.holdout_workload, **exp_b},
        'exp_c': exp_c_summary,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args) -> int:
    data_dir    = Path(args.data_dir).resolve()
    models_dir  = ITC2_ROOT / 'models'
    results_dir = ITC2_ROOT / 'results'
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print(f"\n{'='*64}")
    print("  ITC2 - Real Fault LSTM Training")
    print(f"{'='*64}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch_size}")

    X_seq = np.load(str(data_dir / 'features_seq.npy'))
    X_agg = np.load(str(data_dir / 'features_agg.npy'))
    y     = np.load(str(data_dir / 'labels.npy'))
    meta  = pd.read_csv(str(data_dir / 'metadata.csv'))

    print(f"\n  Loaded: {len(y)} samples  seq:{X_seq.shape}  agg:{X_agg.shape}")
    for lbl, name in [(0, 'clean'), (1, 'observable'), (2, 'silent')]:
        print(f"    {name}: {int((y == lbl).sum())}")

    tasks_to_run = []
    if args.task in ('binary', 'all'):
        tasks_to_run.append(('binary', 2))
    if args.task in ('3class', 'all'):
        tasks_to_run.append(('3class', 3))

    all_results = {}
    ts = datetime.now().isoformat()

    for task_name, num_classes in tasks_to_run:
        r = run_task(X_seq, X_agg, y, meta, num_classes, task_name,
                     args, models_dir, results_dir)
        all_results[task_name] = r

        def _ser(d):
            if isinstance(d, dict):
                return {k: _ser(v) for k, v in d.items()}
            if isinstance(d, list):
                return [_ser(v) for v in d]
            if isinstance(d, (np.integer,)):
                return int(d)
            if isinstance(d, (np.floating,)):
                return float(d)
            if isinstance(d, float) and np.isnan(d):
                return None
            return d

        out = {'timestamp': ts, **_ser(r)}
        path = results_dir / f'lstm_{task_name}.json'
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved {path.name}")

    # Combined generalization summary
    gen_path = results_dir / 'lstm_generalization.json'
    with open(gen_path, 'w') as f:
        gen = {task: {
            'exp_b': _ser(all_results[task]['exp_b']),
            'exp_c': _ser(all_results[task]['exp_c']),
        } for task in all_results}
        json.dump({'timestamp': ts, **gen}, f, indent=2)
    print(f"  Saved {gen_path.name}")

    print("\n  Done.")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description='Train real-fault LSTM classifier')
    p.add_argument('--data-dir',         default=str(ITC2_ROOT / 'data'))
    p.add_argument('--task',             default='all',
                   choices=['binary', '3class', 'all'])
    p.add_argument('--epochs',           type=int, default=50)
    p.add_argument('--batch-size',       type=int, default=32)
    p.add_argument('--lr',               type=float, default=1e-3)
    p.add_argument('--holdout-workload', default='irq_test')
    return p.parse_args()


if __name__ == '__main__':
    sys.exit(main(_parse_args()))
