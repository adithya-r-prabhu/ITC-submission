#!/usr/bin/env python3
"""
Functional Fault Taxonomy - Three-Tier Observability Model.

Builds a principled functional fault taxonomy on three axes:

  Tier 1 - UNIVERSAL:   observable in >=4/5 workloads.
                         Any single ATE test program will detect these.
  Tier 2 - WORKLOAD-SENSITIVE: observable in 1-3/5 workloads.
                         Require targeted test selection - the ML model's
                         primary value. Sub-clustered by workload exposure
                         pattern (K-means on 5-dim binary exposure vector).
  Tier 3 - SILENT:      not observable in any workload via external trace.
                         Sub-divided by circuit category.

Minimum cluster size = MIN_CLUSTER_SIZE (default 20).
Any sub-cluster below the threshold is merged into its parent tier.

This replaces the previous XGBoost-feature K-means approach, which placed
91% of sites into one cluster because its primary axis was the
observable/silent discriminant - not circuit behavior.

Outputs:
  results/fault_taxonomy.json
  reports/figures/fault_taxonomy_tsne.png

Usage:
    python src/fault_abstraction.py --data-dir data
"""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent

LABEL_CLEAN      = 0
LABEL_OBS        = 1
LABEL_SILENT     = 2
RANDOM_STATE     = 42
MIN_CLUSTER_SIZE = 20
T2_MAX_K         = 4    # max sub-clusters within workload-sensitive tier

WORKLOADS = ['counting_loop', 'alu_heavy', 'branch_heavy', 'mem_intensive', 'irq_test']

# 156-dim aggregate feature names (mirrors train_real_fault_baselines.py)
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
# Per-site observability profile
# ---------------------------------------------------------------------------

def build_per_site_profile(meta: pd.DataFrame) -> pd.DataFrame:
    """
    One row per fault site.
    Columns: fault_site, category, sa_value, split, n_obs_workloads,
             obs_<workload> (bool per workload), mean_latency.
    """
    faults = meta[meta['label'] != LABEL_CLEAN].copy()
    rows = []
    for site, grp in faults.groupby('fault_site'):
        obs_per_wl = {wl: bool(grp[grp['workload'] == wl]['observable'].values[0])
                      if wl in grp['workload'].values else False
                      for wl in WORKLOADS}
        n_obs   = sum(obs_per_wl.values())
        lat_pos = grp[grp['latency'] >= 0]['latency']
        row = {
            'fault_site':        int(site),
            'category':          grp['category'].iloc[0],
            'sa_value':          int(grp['sa_value'].iloc[0]),
            'split':             grp['split'].iloc[0],
            'n_obs_workloads':   n_obs,
            'mean_latency':      float(lat_pos.mean()) if len(lat_pos) > 0 else -1.0,
        }
        for wl in WORKLOADS:
            row[f'obs_{wl}'] = obs_per_wl[wl]
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def assign_tiers(profile: pd.DataFrame) -> pd.DataFrame:
    """Assign primary tier to each fault site."""
    profile = profile.copy()
    profile['tier'] = profile['n_obs_workloads'].map(
        lambda n: 'T1_universal' if n >= 4 else ('T3_silent' if n == 0 else 'T2_sensitive')
    )
    return profile


# ---------------------------------------------------------------------------
# T2 sub-clustering: workload exposure patterns
# ---------------------------------------------------------------------------

def _name_t2_subcluster(obs_vec_mean: np.ndarray) -> str:
    """
    Name a T2 sub-cluster by its top-2 workload exposure pattern.
    obs_vec_mean[i] = fraction of sites in cluster exposed by workload i.
    Using top-2 ensures two clusters with the same leading workload get
    distinct names (e.g. compute+memory vs compute+branch).
    """
    wl_short = {
        'counting_loop': 'compute',
        'alu_heavy':     'ALU',
        'branch_heavy':  'branch',
        'mem_intensive': 'memory',
        'irq_test':      'interrupt',
    }
    sorted_idx = np.argsort(obs_vec_mean)[::-1]
    # Include workloads with >=20% exposure rate, up to 2
    active = [WORKLOADS[i] for i in sorted_idx if obs_vec_mean[i] >= 0.20][:2]
    if not active:
        active = [WORKLOADS[sorted_idx[0]]]
    short_labels = [wl_short.get(wl, wl) for wl in active]
    return '+'.join(short_labels) + '-sensitive faults'


def subcluster_t2(profile: pd.DataFrame) -> pd.Series:
    """
    K-means on 5-dim binary workload-exposure vectors within T2 tier.
    Returns a Series of sub-cluster IDs (0-based), indexed like profile.
    Merges any sub-cluster below MIN_CLUSTER_SIZE into its nearest neighbour.
    """
    t2_mask = profile['tier'] == 'T2_sensitive'
    t2_idx  = profile[t2_mask].index
    n_t2    = t2_mask.sum()

    sub_ids = pd.Series(-1, index=profile.index)
    if n_t2 < 2:
        sub_ids.loc[t2_idx] = 0
        return sub_ids

    obs_cols = [f'obs_{wl}' for wl in WORKLOADS]
    X_t2 = profile.loc[t2_idx, obs_cols].values.astype(float)

    # Choose k: at most T2_MAX_K, but ensure each cluster has >= MIN_CLUSTER_SIZE
    k = min(T2_MAX_K, max(2, n_t2 // MIN_CLUSTER_SIZE))
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=30)
    raw_labels = km.fit_predict(X_t2)
    centroids  = km.cluster_centers_

    # Merge any sub-cluster smaller than MIN_CLUSTER_SIZE into nearest centroid
    final_labels = raw_labels.copy()
    changed = True
    while changed:
        changed = False
        unique, counts = np.unique(final_labels, return_counts=True)
        for cid, cnt in zip(unique, counts):
            if cnt < MIN_CLUSTER_SIZE and len(unique) > 1:
                # Reassign these points to nearest other centroid
                small_mask = final_labels == cid
                other_ids  = [c for c in unique if c != cid]
                other_centroids = np.array([centroids[c] for c in other_ids])
                for pt_idx in np.where(small_mask)[0]:
                    dists = np.linalg.norm(other_centroids - X_t2[pt_idx], axis=1)
                    final_labels[pt_idx] = other_ids[int(np.argmin(dists))]
                changed = True
                break  # restart loop after one merge

    # Re-index to contiguous 0-based labels
    remap = {old: new for new, old in enumerate(sorted(set(final_labels)))}
    final_labels = np.array([remap[l] for l in final_labels])

    sub_ids.loc[t2_idx] = final_labels
    return sub_ids


# ---------------------------------------------------------------------------
# T3 sub-grouping: circuit category
# ---------------------------------------------------------------------------

def subgroup_t3(profile: pd.DataFrame) -> pd.Series:
    """
    Group silent (T3) sites by circuit category.
    Any category with fewer than MIN_CLUSTER_SIZE sites is merged into the
    largest T3 category (rather than creating a tiny 'other' group).
    Returns sub-group name Series indexed like profile.
    """
    t3_mask = profile['tier'] == 'T3_silent'
    sub_grp = pd.Series('', index=profile.index)

    if not t3_mask.any():
        return sub_grp

    cat_counts  = profile[t3_mask]['category'].value_counts()
    large_cats  = set(cat_counts[cat_counts >= MIN_CLUSTER_SIZE].index)
    # Fallback: absorb small categories into the single largest T3 category
    fallback    = cat_counts.index[0]

    for idx in profile[t3_mask].index:
        cat = profile.loc[idx, 'category']
        sub_grp.loc[idx] = cat if cat in large_cats else fallback
    return sub_grp


# ---------------------------------------------------------------------------
# Final taxonomy assembly
# ---------------------------------------------------------------------------

def _tier_description(tier: str) -> str:
    return {
        'T1_universal':  'Observable in >=4/5 workloads. Any single ATE test program detects these.',
        'T2_sensitive':  'Observable in 1-3/5 workloads. Require targeted workload selection.',
        'T3_silent':     'Not observable in any workload via external trace signals.',
    }.get(tier, '')


def assemble_taxonomy(profile: pd.DataFrame,
                      t2_sub: pd.Series,
                      t3_sub: pd.Series,
                      xgb_importances: np.ndarray) -> tuple[dict, np.ndarray]:
    """
    Returns (taxonomy dict, integer label array indexed like profile).
    """
    top_signals = [AGG_FEATURE_NAMES[i].split('_')[0]
                   for i in np.argsort(xgb_importances)[::-1][:5]]
    top_signals = list(dict.fromkeys(top_signals))[:3]  # deduplicate, keep order

    taxonomy = {}
    label_arr = np.full(len(profile), -1, dtype=int)
    next_id = 0

    for tier in ['T1_universal', 'T2_sensitive', 'T3_silent']:
        tier_mask = (profile['tier'] == tier).values

        if tier == 'T1_universal':
            groups = {'': tier_mask}

        elif tier == 'T2_sensitive':
            unique_sub = sorted(set(t2_sub[profile['tier'] == tier]))
            groups = {}
            for sub in unique_sub:
                gmask = tier_mask & (t2_sub == sub).values
                groups[str(int(sub))] = gmask

        else:  # T3_silent
            unique_sub = sorted(set(t3_sub[profile['tier'] == tier]))
            groups = {}
            for sub in unique_sub:
                gmask = tier_mask & (t3_sub == sub).values
                groups[sub] = gmask

        for key, gmask in groups.items():
            rows = profile[gmask]
            if len(rows) == 0:
                continue

            obs_rate   = float(rows['n_obs_workloads'].gt(0).mean())
            cat_counts = rows['category'].value_counts().to_dict()
            dom_cat    = rows['category'].value_counts().index[0]
            lat_pos    = rows[rows['mean_latency'] >= 0]['mean_latency']
            mean_lat   = float(lat_pos.mean()) if len(lat_pos) > 0 else -1.0

            # Name
            if tier == 'T1_universal':
                name = f'universal propagating faults ({dom_cat}-dominant)'
            elif tier == 'T2_sensitive':
                obs_cols = [f'obs_{wl}' for wl in WORKLOADS]
                mean_vec = rows[obs_cols].values.mean(axis=0)
                name = _name_t2_subcluster(mean_vec)
            else:
                cat_label = key if key != 'other' else 'mixed-category'
                name = f'architecturally silent {cat_label} faults'

            # Best workload (for T2 only)
            best_wl = None
            if tier == 'T2_sensitive':
                obs_cols = [f'obs_{wl}' for wl in WORKLOADS]
                wl_rates = rows[obs_cols].mean()
                best_wl  = wl_rates.idxmax().replace('obs_', '')

            # N-obs distribution
            n_obs_dist = rows['n_obs_workloads'].value_counts().sort_index().to_dict()

            cid_str = str(next_id)
            taxonomy[cid_str] = {
                'cluster_id':             next_id,
                'tier':                   tier,
                'tier_description':       _tier_description(tier),
                'name':                   name,
                'size':                   int(len(rows)),
                'observability_rate':     obs_rate,
                'n_obs_workloads_dist':   {int(k): int(v) for k, v in n_obs_dist.items()},
                'dominant_category':      dom_cat,
                'category_breakdown':     cat_counts,
                'mean_latency_cycles':    mean_lat,
                'best_exposing_workload': best_wl,
                'top_diagnostic_signals': top_signals,
                'sa_value_split': {
                    'sa0': int((rows['sa_value'] == 0).sum()),
                    'sa1': int((rows['sa_value'] == 1).sum()),
                },
            }
            label_arr[gmask] = next_id
            next_id += 1

    return taxonomy, label_arr


# ---------------------------------------------------------------------------
# Feature aggregation for t-SNE
# ---------------------------------------------------------------------------

def get_site_features(X_agg: np.ndarray, y: np.ndarray,
                      meta: pd.DataFrame, profile: pd.DataFrame) -> np.ndarray:
    """Average agg features across workloads for each fault site, in profile order."""
    fault_mask = y != LABEL_CLEAN
    Xf    = X_agg[fault_mask]
    metaf = meta[fault_mask].reset_index(drop=True)
    site_ids = metaf['fault_site'].values

    X_site = np.array([
        Xf[site_ids == row['fault_site']].mean(axis=0)
        for _, row in profile.iterrows()
    ])
    return X_site


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_tsne(X_site: np.ndarray, label_arr: np.ndarray,
              taxonomy: dict, profile: pd.DataFrame, out_path: Path):
    print("  Running t-SNE (~30s)...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X_site)

    n_classes = len(taxonomy)
    cmap = plt.cm.get_cmap('tab10', n_classes)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: colour by taxonomy class
    ax = axes[0]
    for cid_str, info in taxonomy.items():
        cid  = info['cluster_id']
        mask = label_arr == cid
        label_str = f"C{cid}: {info['name'][:30]}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=[cmap(cid)], label=label_str, s=15, alpha=0.75)
    ax.set_title('t-SNE: Functional Fault Taxonomy')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.legend(fontsize=7, markerscale=1.5, loc='best')
    ax.grid(alpha=0.2)

    # Right: colour by n_obs_workloads (0=silent, 1-3=sensitive, 4-5=universal)
    ax2 = axes[1]
    tier_colors = {
        'T3_silent':    '#4878cf',
        'T2_sensitive': '#f5a623',
        'T1_universal': '#d65f5f',
    }
    for tier, color in tier_colors.items():
        mask = (profile['tier'] == tier).values
        ax2.scatter(X_2d[mask, 0], X_2d[mask, 1],
                    c=color, label=tier.replace('_', ' '), s=15, alpha=0.75)
    ax2.set_title('t-SNE: Observability Tiers\n(T1=universal, T2=sensitive, T3=silent)')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Functional fault taxonomy - three-tier observability model')
    parser.add_argument('--data-dir',         default='data',    help='features + metadata')
    parser.add_argument('--models-dir',       default='models',  help='trained models')
    parser.add_argument('--results-dir',      default='results', help='JSON output')
    parser.add_argument('--figures-dir',      default='reports/figures', help='figure output')
    parser.add_argument('--min-cluster-size', type=int, default=MIN_CLUSTER_SIZE,
                        help='Minimum sites per cluster before merging')
    args = parser.parse_args()

    data_dir    = ITC2_ROOT / args.data_dir
    models_dir  = ITC2_ROOT / args.models_dir
    results_dir = ITC2_ROOT / args.results_dir
    figures_dir = ITC2_ROOT / args.figures_dir
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X_agg = np.load(str(data_dir / 'features_agg.npy'))
    y     = np.load(str(data_dir / 'labels.npy'))
    meta  = pd.read_csv(str(data_dir / 'metadata.csv'))

    print("Loading XGBoost model (for diagnostic signal names)...")
    with open(models_dir / 'xgboost_binary.pkl', 'rb') as f:
        pkg = pickle.load(f)
    importances = pkg['model'].feature_importances_

    print("Building per-site observability profiles...")
    profile = build_per_site_profile(meta)
    profile = assign_tiers(profile)

    tier_counts = profile['tier'].value_counts()
    print(f"  T1 universal:       {tier_counts.get('T1_universal', 0):>4} sites")
    print(f"  T2 workload-sensitive: {tier_counts.get('T2_sensitive', 0):>4} sites")
    print(f"  T3 silent:          {tier_counts.get('T3_silent', 0):>4} sites")

    print(f"Sub-clustering T2 (min cluster size={args.min_cluster_size})...")
    t2_sub = subcluster_t2(profile)

    print("Sub-grouping T3 by circuit category...")
    t3_sub = subgroup_t3(profile)

    print("Assembling taxonomy...")
    taxonomy, label_arr = assemble_taxonomy(profile, t2_sub, t3_sub, importances)

    print("\nFunctional Fault Taxonomy:")
    print(f"{'ID':<3} {'Tier':<18} {'Name':<45} {'Size':>6} {'Obs%':>6}")
    print("-" * 84)
    for cid_str, info in taxonomy.items():
        tier_short = info['tier'].replace('T1_universal', 'T1-Universal') \
                                 .replace('T2_sensitive', 'T2-Sensitive') \
                                 .replace('T3_silent',    'T3-Silent')
        print(f"  {info['cluster_id']:<3} {tier_short:<18} {info['name']:<45} "
              f"{info['size']:>6} {info['observability_rate']:>5.1%}")

    # Cluster assignment map: fault_site -> cluster_id
    assignments = {
        str(int(profile.iloc[i]['fault_site'])): int(label_arr[i])
        for i in range(len(profile))
    }

    out = {
        'generated':             datetime.now().isoformat(),
        'method':                'three-tier observability taxonomy',
        'tiers': {
            'T1_universal':  'observable in >=4/5 workloads',
            'T2_sensitive':  'observable in 1-3/5 workloads (K-means on exposure vector)',
            'T3_silent':     'not observable in any workload (grouped by circuit category)',
        },
        'min_cluster_size':      args.min_cluster_size,
        'n_clusters':            len(taxonomy),
        'clusters':              taxonomy,
        'cluster_assignments':   assignments,
    }
    out_path = results_dir / 'fault_taxonomy.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("Building site feature matrix for t-SNE...")
    X_site   = get_site_features(X_agg, y, meta, profile)
    fig_path = figures_dir / 'fault_taxonomy_tsne.png'
    plot_tsne(X_site, label_arr, taxonomy, profile, fig_path)

    print("Done.")


if __name__ == '__main__':
    main()
