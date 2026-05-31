#!/usr/bin/env python3
"""
ATE Test Program Mapping.

Maps each functional fault class (from fault_taxonomy.json) to the minimal
workload needed to expose it, providing a concrete ATE test reduction argument.

For each fault cluster:
  - Compute per-workload observability rate
  - Greedy minimal cover: which single workload covers the most faults?
  - Compute test reduction ratio vs exhaustive 5-workload testing

Also computes workload recommendation at the global level (no clustering):
  - Which workload exposes the most faults per category?
  - Test reduction: 1 optimal workload vs all 5

Outputs:
  results/ate_mapping.json
  reports/figures/ate_heatmap.png

Usage:
    python src/ate_mapping.py --data-dir data
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent

WORKLOADS    = ['counting_loop', 'alu_heavy', 'branch_heavy', 'mem_intensive', 'irq_test']
LABEL_CLEAN  = 0


def load_metadata(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(str(data_dir / 'metadata.csv'))


def load_taxonomy(results_dir: Path) -> dict:
    path = results_dir / 'fault_taxonomy.json'
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fault_abstraction.py first.")
    with open(path) as f:
        return json.load(f)


def workload_coverage_matrix(faults: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame: fault_site x workload, value = observable (0/1).
    """
    pivot = faults.pivot_table(
        index='fault_site', columns='workload', values='observable', aggfunc='max'
    ).fillna(0).astype(bool)
    # Ensure all workloads are present
    for wl in WORKLOADS:
        if wl not in pivot.columns:
            pivot[wl] = False
    return pivot[WORKLOADS]


def greedy_minimal_cover(coverage_matrix: pd.DataFrame) -> dict:
    """
    Greedy: at each step pick the workload that covers the most uncovered sites.
    Returns order of workloads chosen and cumulative coverage at each step.
    """
    uncovered = set(coverage_matrix[coverage_matrix.any(axis=1)].index)
    total_coverable = len(uncovered)
    order = []
    cumulative = []

    remaining_wls = list(WORKLOADS)
    covered_so_far = set()

    while remaining_wls and uncovered:
        # Pick workload covering most uncovered sites
        best_wl   = max(remaining_wls,
                        key=lambda w: coverage_matrix.loc[
                            list(uncovered), w].sum())
        newly_covered = set(
            coverage_matrix[coverage_matrix[best_wl]].index) & uncovered
        covered_so_far |= newly_covered
        uncovered -= newly_covered
        order.append(best_wl)
        cumulative.append(len(covered_so_far) / total_coverable if total_coverable > 0 else 0.0)
        remaining_wls.remove(best_wl)

    return {
        'workload_order': order,
        'cumulative_coverage': [round(c, 4) for c in cumulative],
        'total_coverable_sites': total_coverable,
    }


def per_cluster_ate(faults: pd.DataFrame, taxonomy: dict) -> dict:
    """For each cluster: per-workload coverage + best single-workload recommendation."""
    assignments = taxonomy.get('cluster_assignments', {})
    # site_id (str) -> cluster_id
    site_to_cluster = {int(k): int(v) for k, v in assignments.items()}

    result = {}
    for cid_str, info in taxonomy['clusters'].items():
        cid = int(cid_str)
        cluster_sites = [s for s, c in site_to_cluster.items() if c == cid]

        cluster_faults = faults[faults['fault_site'].isin(cluster_sites)]
        if len(cluster_faults) == 0:
            continue

        per_wl = {}
        for wl in WORKLOADS:
            wl_rows = cluster_faults[cluster_faults['workload'] == wl]
            per_wl[wl] = float(wl_rows['observable'].mean()) if len(wl_rows) > 0 else 0.0

        best_wl        = max(per_wl, key=per_wl.get)
        best_coverage  = per_wl[best_wl]
        all_wl_coverage = float(
            cluster_faults.groupby('fault_site')['observable'].any().mean())

        result[cid_str] = {
            'cluster_name':         info['name'],
            'cluster_size':         info['size'],
            'per_workload_coverage': per_wl,
            'recommended_workload':  best_wl,
            'single_wl_coverage':    round(best_coverage, 4),
            'all_wl_coverage':       round(all_wl_coverage, 4),
            'test_reduction_ratio':  round(
                best_coverage / all_wl_coverage if all_wl_coverage > 0 else 0.0, 4),
        }
    return result


def per_category_ate(faults: pd.DataFrame) -> dict:
    """Per fault category: best workload and coverage ratio."""
    result = {}
    for cat, grp in faults.groupby('category'):
        per_wl = {}
        for wl in WORKLOADS:
            wl_rows = grp[grp['workload'] == wl]
            per_wl[wl] = float(wl_rows['observable'].mean()) if len(wl_rows) > 0 else 0.0

        best_wl        = max(per_wl, key=per_wl.get)
        best_cov       = per_wl[best_wl]
        all_wl_cov     = float(grp.groupby('fault_site')['observable'].any().mean())

        result[cat] = {
            'per_workload_coverage':  per_wl,
            'recommended_workload':   best_wl,
            'single_wl_coverage':     round(best_cov, 4),
            'all_wl_coverage':        round(all_wl_cov, 4),
            'test_reduction_ratio':   round(
                best_cov / all_wl_cov if all_wl_cov > 0 else 0.0, 4),
        }
    return result


def global_test_reduction(faults: pd.DataFrame) -> dict:
    """
    Overall test reduction: best single workload vs all 5 vs greedy cover.
    """
    cov_matrix = workload_coverage_matrix(faults)

    per_wl_global = {}
    for wl in WORKLOADS:
        per_wl_global[wl] = float(faults[faults['workload'] == wl]['observable'].mean())

    best_wl       = max(per_wl_global, key=per_wl_global.get)
    best_single   = per_wl_global[best_wl]
    all_wl        = float(faults.groupby('fault_site')['observable'].any().mean())
    greedy        = greedy_minimal_cover(cov_matrix)

    # Coverage with just 1 workload vs all 5
    reduction_ratio = round(best_single / all_wl, 4) if all_wl > 0 else 0.0
    # How many workloads needed to reach 95% of all-wl coverage?
    wls_for_95 = None
    for i, c in enumerate(greedy['cumulative_coverage']):
        if c >= 0.95 * all_wl:
            wls_for_95 = i + 1
            break

    return {
        'per_workload_coverage': per_wl_global,
        'best_single_workload':  best_wl,
        'best_single_coverage':  round(best_single, 4),
        'all_workload_coverage': round(all_wl, 4),
        'test_reduction_1_wl':   reduction_ratio,
        'greedy_cover':          greedy,
        'workloads_for_95pct':   wls_for_95,
        'interpretation': (
            f"Running only '{best_wl}' achieves {best_single:.1%} coverage "
            f"vs {all_wl:.1%} exhaustive - {reduction_ratio:.1%} efficiency. "
            f"Greedy 2-workload cover reaches "
            f"{greedy['cumulative_coverage'][1] if len(greedy['cumulative_coverage']) > 1 else 'N/A':.1%}."
        ),
    }


def plot_heatmap(faults: pd.DataFrame, cluster_data: dict,
                 category_data: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: cluster x workload heatmap
    ax = axes[0]
    cluster_ids   = sorted(cluster_data.keys(), key=int)
    cluster_names = [f"C{cid}: {cluster_data[cid]['cluster_name'][:25]}" for cid in cluster_ids]
    data_matrix = np.array([
        [cluster_data[cid]['per_workload_coverage'].get(wl, 0) for wl in WORKLOADS]
        for cid in cluster_ids
    ])
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels([w.replace('_', '\n') for w in WORKLOADS], fontsize=8)
    ax.set_yticks(range(len(cluster_names)))
    ax.set_yticklabels(cluster_names, fontsize=8)
    ax.set_title('Observability Rate\nFault Cluster x Workload')
    plt.colorbar(im, ax=ax, label='Observable fraction')
    for i in range(len(cluster_ids)):
        for j in range(len(WORKLOADS)):
            val = data_matrix[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if val < 0.6 else 'white')

    # Right: category x workload heatmap
    ax2 = axes[1]
    cats = sorted(category_data.keys())
    data_matrix2 = np.array([
        [category_data[cat]['per_workload_coverage'].get(wl, 0) for wl in WORKLOADS]
        for cat in cats
    ])
    im2 = ax2.imshow(data_matrix2, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(WORKLOADS)))
    ax2.set_xticklabels([w.replace('_', '\n') for w in WORKLOADS], fontsize=8)
    ax2.set_yticks(range(len(cats)))
    ax2.set_yticklabels(cats, fontsize=9)
    ax2.set_title('Observability Rate\nFault Category x Workload')
    plt.colorbar(im2, ax=ax2, label='Observable fraction')
    for i in range(len(cats)):
        for j in range(len(WORKLOADS)):
            val = data_matrix2[i, j]
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                     fontsize=7, color='black' if val < 0.6 else 'white')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='ATE test program mapping from fault taxonomy')
    parser.add_argument('--data-dir',    default='data',    help='Directory with metadata.csv')
    parser.add_argument('--results-dir', default='results', help='JSON input/output directory')
    parser.add_argument('--figures-dir', default='reports/figures', help='Output figures directory')
    args = parser.parse_args()

    data_dir    = ITC2_ROOT / args.data_dir
    results_dir = ITC2_ROOT / args.results_dir
    figures_dir = ITC2_ROOT / args.figures_dir
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata and taxonomy...")
    meta     = load_metadata(data_dir)
    taxonomy = load_taxonomy(results_dir)
    faults   = meta[meta['label'] != LABEL_CLEAN].copy()

    print("Computing per-cluster ATE mapping...")
    cluster_map = per_cluster_ate(faults, taxonomy)

    print("Computing per-category ATE mapping...")
    cat_map = per_category_ate(faults)

    print("Computing global test reduction...")
    global_red = global_test_reduction(faults)

    # Print summary table
    print("\nATE Recommendation Table (by cluster):")
    print(f"{'Cluster':<35} {'Best WL':<20} {'1-WL Cov':>9} {'All-WL Cov':>11} {'Ratio':>6}")
    print("-" * 85)
    for cid, info in cluster_map.items():
        print(f"  {info['cluster_name']:<33} {info['recommended_workload']:<20} "
              f"{info['single_wl_coverage']:>9.3f} {info['all_wl_coverage']:>11.3f} "
              f"{info['test_reduction_ratio']:>6.3f}")

    print(f"\nGlobal: best single workload = '{global_red['best_single_workload']}' "
          f"({global_red['best_single_coverage']:.1%} coverage vs "
          f"{global_red['all_workload_coverage']:.1%} exhaustive)")
    print(f"Test reduction with 1 workload: {global_red['test_reduction_1_wl']:.1%} of exhaustive coverage")
    greedy = global_red['greedy_cover']
    print(f"Greedy cover order: {' -> '.join(greedy['workload_order'])}")
    print(f"Cumulative coverage: {[f'{c:.2%}' for c in greedy['cumulative_coverage']]}")

    out = {
        'generated':     datetime.now().isoformat(),
        'per_cluster':   cluster_map,
        'per_category':  cat_map,
        'global':        global_red,
        'ate_note': (
            'Recommended workload = single ATE test program achieving highest '
            'observable-fault coverage for this fault class. '
            'Test reduction ratio = single-workload coverage / all-workload coverage.'
        ),
    }
    out_path = results_dir / 'ate_mapping.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")

    fig_path = figures_dir / 'ate_heatmap.png'
    plot_heatmap(faults, cluster_map, cat_map, fig_path)

    print("Done.")


if __name__ == '__main__':
    main()
