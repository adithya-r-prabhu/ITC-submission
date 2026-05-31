#!/usr/bin/env python3
"""
Position-Matched Re-Evaluation (confound control for Fix 4 investigation).

The main dataset injects every fault at a fixed fault_cycle=200 while clean
samples span fault_cycle 50-500. confound_audit.py showed this lets any
position-varying feature "detect" faults - inflating LOOCV AUROC well above the
genuine post-fault signal (e.g. counting_loop 0.995 -> 0.531 window-only).

This script removes the confound by construction: it generates BOTH clean and
faulty samples at the SAME set of fault_cycle positions, on a stratified site
subset. With position balanced across classes, the pre-fault `baseline` block
becomes non-discriminative (best single baseline feature should fall to ~0.5),
and the full-156 AUROC should collapse to the honest window-only level.

It reuses the exact simulation + feature pipeline from build_real_fault_dataset.

Outputs:
    results/position_matched_reeval.json

Usage:
    venv/bin/python src/position_matched_reeval.py --itc-root /home/atomic_zuccini/ITC \
        --n-sites 100 --n-positions 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import build_real_fault_dataset as B  # sibling module - reuse sim + feature code

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent
WINDOW_SLICE   = slice(0, 52)
BASELINE_SLICE = slice(52, 104)


def _select_sites(sites, obs_map, n_sites, seed=42):
    """Stratified subset of n_sites preserving the observable fraction."""
    ids = [s['id'] for s in sites]
    obs_flag = [int(any(obs_map.get(w, {}).get(i, {}).get('observable', False)
                        for w in obs_map)) for i in ids]
    if n_sites >= len(ids):
        return sites
    keep_idx, _ = train_test_split(
        np.arange(len(ids)), train_size=n_sites,
        stratify=obs_flag, random_state=seed)
    keep = set(keep_idx.tolist())
    return [s for k, s in enumerate(sites) if k in keep]


def build_matched(args):
    itc_root = Path(args.itc_root).resolve()
    B._setup_itc_imports(itc_root)
    sim     = str(itc_root / 'build' / 'sim_fault_oracle')
    fw_dir  = itc_root / 'src' / 'firmware'
    sites_f = itc_root / 'src' / 'fault_sites_picorv32.json'
    cache_dir = ITC2_ROOT / '.cache'
    cache_dir.mkdir(exist_ok=True)

    with open(sites_f) as f:
        sites = json.load(f)
    obs_map = B._load_observability(itc_root)
    sites = _select_sites(sites, obs_map, args.n_sites)

    positions = np.linspace(120, 440, args.n_positions, dtype=int).tolist()
    firmwares = {fw_dir / f'{w}.bin': w for w in B.WORKLOADS
                 if (fw_dir / f'{w}.bin').exists()}

    print(f"  sites={len(sites)}  positions={positions}  workloads={list(firmwares.values())}")

    rows, X_list = [], []

    def _sim_features(fw_path, fw_name, site, sa, fc, irq):
        key = B._cache_key(fw_name, site, sa, fc)
        csv = B._load_cached(cache_dir, key)
        if csv is None:
            csv = B._run_sim(sim, str(fw_path), B.SIM_CYCLES, site, sa, fc, irq)
            if csv:
                B._save_cached(cache_dir, key, csv)
        if not csv:
            return None, None
        arr = B._parse_csv(csv)
        if len(arr) == 0:
            return None, None
        return arr, B._extract_features(arr, fc)[1]   # agg only

    for fw_path, fw_name in firmwares.items():
        irq = 80 if fw_name == 'irq_test' else 0
        for fc in positions:
            # clean reference at this position
            clean_arr, clean_agg = _sim_features(fw_path, fw_name, -1, 0, fc, irq)
            if clean_arr is None:
                continue
            X_list.append(clean_agg)
            rows.append(dict(workload=fw_name, position=fc, label=0))
            # faulty samples at the SAME position
            for s in sites:
                arr, agg = _sim_features(fw_path, fw_name, s['id'], s['sa_value'], fc, irq)
                if arr is None:
                    continue
                n = min(len(clean_arr), len(arr))
                c = clean_arr[fc:n][:, B.OBS_COLS]
                fl = arr[fc:n][:, B.OBS_COLS]
                observable = bool(np.any(c != fl))
                X_list.append(agg)
                rows.append(dict(workload=fw_name, position=fc,
                                 label=1 if observable else 2))

    X = np.stack(X_list).astype(np.float32)
    meta = pd.DataFrame(rows)
    print(f"  built matched set: {X.shape[0]} samples "
          f"(clean={int((meta.label==0).sum())}, obs={int((meta.label==1).sum())}, "
          f"silent={int((meta.label==2).sum())})")
    return X, meta


def _auc(Xtr, ytr, Xte, yte, cols):
    sc = StandardScaler().fit(Xtr[:, cols])
    a, b = sc.transform(Xtr[:, cols]), sc.transform(Xte[:, cols])
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                n_jobs=-1, random_state=42).fit(a, ytr)
    out = {'RF': roc_auc_score(yte, rf.predict_proba(b)[:, 1])}
    if HAS_XGB:
        xg = xgb.XGBClassifier(n_estimators=200, eval_metric='logloss',
                               random_state=42).fit(a, ytr)
        out['XGB'] = roc_auc_score(yte, xg.predict_proba(b)[:, 1])
    return out


def evaluate(X, meta):
    bn = meta['label'].isin([0, 1]).values
    full = np.arange(156)
    win  = np.arange(WINDOW_SLICE.start, WINDOW_SLICE.stop)
    base_cols = np.arange(BASELINE_SLICE.start, BASELINE_SLICE.stop)

    print(f"\n  {'workload':<15} {'model':<5} {'full156':>8} {'window52':>9} "
          f"{'best_base':>10}")
    print("  " + "-" * 52)
    results = {}
    for w in B.WORKLOADS:
        is_w = (meta['workload'] == w).values
        tr = bn & ~is_w
        te = bn & is_w
        ytr, yte = meta['label'].values[tr], meta['label'].values[te]
        if len(np.unique(yte)) < 2 or len(np.unique(ytr)) < 2:
            continue
        full_auc = _auc(X[tr], ytr, X[te], yte, full)
        win_auc  = _auc(X[tr], ytr, X[te], yte, win)
        # best single baseline feature on test - should now be ~0.5 (confound gone)
        best_base = 0.5
        for j in base_cols:
            col = X[te][:, j]
            if np.allclose(col, col[0]):
                continue
            a = roc_auc_score(yte, col)
            best_base = max(best_base, a, 1 - a)
        results[w] = dict(full=full_auc, window=win_auc, best_baseline=best_base,
                          n_clean=int((yte == 0).sum()), n_obs=int((yte == 1).sum()))
        for m in full_auc:
            print(f"  {w:<15} {m:<5} {full_auc[m]:>8.3f} {win_auc[m]:>9.3f} "
                  f"{best_base:>10.3f}")
        print()
    return results


def main(args):
    X, meta = build_matched(args)
    results = evaluate(X, meta)

    # summary vs confounded headline
    rf_full = np.mean([results[w]['full']['RF'] for w in results])
    rf_win  = np.mean([results[w]['window']['RF'] for w in results])
    base_mean = np.mean([results[w]['best_baseline'] for w in results])
    print("  " + "=" * 52)
    print(f"  Position-MATCHED mean RF AUROC: full156={rf_full:.3f}  window={rf_win:.3f}")
    print(f"  Mean best-single-baseline-feature AUROC: {base_mean:.3f}  "
          f"(~0.5 confirms the confound is removed)")

    out = ITC2_ROOT / 'results' / 'position_matched_reeval.json'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'positions': args.n_positions, 'n_sites': args.n_sites,
                   'per_workload': results,
                   'summary': {'rf_full_mean': rf_full, 'rf_window_mean': rf_win,
                               'best_baseline_mean': base_mean}}, f, indent=2)
    print(f"  Saved: {out}")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description='Position-matched confound-controlled re-eval')
    p.add_argument('--itc-root', default='/home/atomic_zuccini/ITC')
    p.add_argument('--n-sites', type=int, default=100)
    p.add_argument('--n-positions', type=int, default=8)
    return p.parse_args()


if __name__ == '__main__':
    import sys
    sys.exit(main(_parse_args()))
