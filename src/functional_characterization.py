#!/usr/bin/env python3
"""
Signal-Divergence Functional Characterization - making the taxonomy functional.

The observability taxonomy (src/fault_abstraction.py) clusters faults by *where*
and *under which workloads* they become observable. It does not say *what the
fault does* once it manifests. The competition topic asks for a "functional fault
model": an abstraction keyed on the functional consequence of a fault, not just
its detectability.

This script supplies that missing layer. For every observable fault it replays
the cached architectural trace, finds the first bus signal that diverges from the
clean run after the injection cycle, and maps that signal to a functional
consequence type:

    Type A - Control-flow corruption : fetch_pc / mem_addr diverge first, or a
              trap is raised  -> execution is redirected or the core crashes.
    Type B - Data corruption         : mem_wdata / mem_rdata / mem_wstrb diverge
              first  -> a wrong value is computed or moved; control flow survives.
    Type C - Decode corruption       : mem_instr diverges first  -> the access
              changes instruction/data character (wrong instruction stream).
    Type D - Stall / hazard          : mem_valid / mem_ready / stall diverge
              first  -> a pipeline timing or handshake perturbation.

When several signals diverge on the same first cycle (common: a redirected fetch
moves fetch_pc and mem_addr together), a fixed precedence picks the most specific
consequence - control-flow over decode over data over timing.

Method
------
  * Faulty trace cache key : MD5(f"{workload}_{site}_{sa_value}_{fault_cycle}")
    where fault_cycle is the site's per-sample injection position read from
    metadata.csv (position-matched grid; no longer a fixed 200).
  * Clean reference         : any cached clean trace for that workload
    (clean runs inject no fault, so the trace is independent of fault_cycle).
  * Divergence is measured from each sample's injection cycle inclusive; the
    resulting latency reproduces the precomputed `latency` in metadata.csv,
    which validates the comparison against the dataset build.

Outputs
-------
  results/functional_characterization.json   per-fault records + breakdowns
  reports/figures/consequence_by_cluster.png  stacked consequence mix per cluster
  results/fault_taxonomy.json (augmented)     each cluster gains a
                                              `consequence_breakdown` field (Fix 3)

Fix 3 is performed here rather than in fault_abstraction.py to avoid a circular
dependency: functional characterization needs the cluster assignments produced by
fault_abstraction.py, so it runs after the taxonomy and writes the consequence
breakdown back into fault_taxonomy.json. Pass --no-taxonomy-augment to skip this.

Usage
-----
    python src/functional_characterization.py --data-dir data
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent

LABEL_OBS   = 1
FAULT_CYCLE = 200

# Bus-trace columns, excluding the leading `cycle` index.
SIGNALS = [
    'mem_valid', 'mem_instr', 'mem_ready', 'mem_addr', 'mem_wstrb',
    'mem_wdata', 'mem_rdata', 'trap', 'stall', 'fetch_pc',
]

CONSEQUENCE_TYPES = {
    'A': 'Control-flow corruption',
    'B': 'Data corruption',
    'C': 'Decode corruption',
    'D': 'Stall/hazard',
}

# Precedence for resolving the consequence type when multiple signals diverge on
# the same first cycle. Most-specific functional consequence first.
SIGNAL_PRECEDENCE = [
    ('trap',      'A'),   # raised trap -> crash / forced control transfer
    ('fetch_pc',  'A'),   # PC redirected
    ('mem_addr',  'A'),   # access address redirected
    ('mem_instr', 'C'),   # instruction/data character of the access flipped
    ('mem_wdata', 'B'),   # wrong value written
    ('mem_rdata', 'B'),   # wrong value read back
    ('mem_wstrb', 'B'),   # write-enable pattern altered
    ('mem_valid', 'D'),   # request handshake perturbed
    ('mem_ready', 'D'),   # response handshake perturbed
    ('stall',     'D'),   # pipeline stall perturbed
]
_PRECEDENCE_ORDER = [s for s, _ in SIGNAL_PRECEDENCE]
_SIGNAL_TYPE      = dict(SIGNAL_PRECEDENCE)


# ---------------------------------------------------------------------------
# Trace cache access
# ---------------------------------------------------------------------------

def cache_key(workload: str, site: int, sa: int, fault_cycle: int) -> str:
    raw = f'{workload}_{site}_{sa}_{fault_cycle}'
    return hashlib.md5(raw.encode()).hexdigest()


def find_clean_trace(cache_dir: Path, workload: str):
    """Return the clean-reference signal array for a workload, or None.

    A clean run (fault_site = -1) injects no fault, so its trace is identical
    for every fault_cycle. The dataset builder cached clean runs at 50
    evenly-spaced positions in [50, 500]; any one of them is a valid reference.
    """
    for fc in np.linspace(50, 500, 50, dtype=int):
        p = cache_dir / f'{cache_key(workload, -1, 0, int(fc))}.csv'
        if p.exists():
            return pd.read_csv(p)[SIGNALS].to_numpy()
    return None


def load_fault_trace(cache_dir: Path, workload: str, site: int, sa: int,
                     fault_cycle: int):
    p = cache_dir / f'{cache_key(workload, site, sa, fault_cycle)}.csv'
    if not p.exists():
        return None
    return pd.read_csv(p)[SIGNALS].to_numpy()


# ---------------------------------------------------------------------------
# Per-fault characterization
# ---------------------------------------------------------------------------

def characterize(fault_arr: np.ndarray, clean_arr: np.ndarray,
                 fault_cycle: int) -> dict | None:
    """First post-injection divergence -> consequence type. None if identical."""
    n = min(len(fault_arr), len(clean_arr))
    f = fault_arr[fault_cycle:n]
    c = clean_arr[fault_cycle:n]
    diff = f != c                                   # (cycles, n_signals)
    diverging_rows = np.where(diff.any(axis=1))[0]
    # trap appearing anywhere post-fault is a definitive control-flow signature.
    trap_idx = SIGNALS.index('trap')
    trap_observed = bool((f[:, trap_idx] == 1).any() and (c[:, trap_idx] == 0).any())
    if len(diverging_rows) == 0:
        return None                                 # silent under this comparison

    first = int(diverging_rows[0])
    diverging_signals = [SIGNALS[i] for i in np.where(diff[first])[0]]
    # Pick the highest-precedence signal among those diverging on the first cycle.
    primary = next(s for s in _PRECEDENCE_ORDER if s in diverging_signals)
    return {
        'latency': first,                           # matches metadata.csv latency
        'first_div_cycle': fault_cycle + first,
        'diverging_signals': diverging_signals,
        'primary_signal': primary,
        'consequence_type': _SIGNAL_TYPE[primary],
        'trap_observed': trap_observed,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _type_counts(records, key) -> dict:
    """For each value of `key`, count consequence types A/B/C/D + dominant."""
    buckets = defaultdict(Counter)
    for r in records:
        buckets[r[key]][r['consequence_type']] += 1
    out = {}
    for grp, ctr in buckets.items():
        total = sum(ctr.values())
        counts = {t: int(ctr.get(t, 0)) for t in CONSEQUENCE_TYPES}
        fracs  = {t: counts[t] / total for t in CONSEQUENCE_TYPES}
        dominant = max(counts, key=counts.get)
        out[str(grp)] = {
            'n': total,
            'counts': counts,
            'fractions': {t: round(v, 4) for t, v in fracs.items()},
            'dominant_type': dominant,
            'dominant_label': CONSEQUENCE_TYPES[dominant],
        }
    return out


def augment_taxonomy(tax_path: Path, by_cluster: dict) -> int:
    """Write the per-cluster consequence breakdown back into fault_taxonomy.json.

    This is Fix 3: the observability taxonomy (Addition 2) gains a functional
    axis. Each cluster gets a `consequence_breakdown` field with its A/B/C/D
    counts, fractions, and dominant type. T3 (architecturally silent) clusters
    have no observable faults, so their breakdown is null. The merge is
    idempotent - re-running overwrites the field rather than appending.

    Returns the number of clusters annotated with a non-null breakdown.
    """
    tax = json.loads(tax_path.read_text())
    n_annotated = 0
    for cid, cluster in tax['clusters'].items():
        v = by_cluster.get(str(cid))
        if v is None:
            cluster['consequence_breakdown'] = None
            continue
        cluster['consequence_breakdown'] = {
            'n_observable_samples': v['n'],
            'counts':        v['counts'],
            'fractions':     v['fractions'],
            'dominant_type': v['dominant_type'],
            'dominant_label': v['dominant_label'],
        }
        n_annotated += 1
    tax['consequence_source'] = 'functional_characterization.py (Addition 7 / Fix 3)'
    tax_path.write_text(json.dumps(tax, indent=2))
    return n_annotated


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_consequence_by_cluster(by_cluster: dict, cluster_names: dict, path: Path):
    cids = sorted(by_cluster, key=lambda x: int(x))
    types = list(CONSEQUENCE_TYPES)                 # A, B, C, D
    colors = {'A': '#d62728', 'B': '#1f77b4', 'C': '#ff7f0e', 'D': '#2ca02c'}

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(cids))
    x = np.arange(len(cids))
    for t in types:
        vals = np.array([by_cluster[c]['fractions'][t] for c in cids])
        ax.bar(x, vals, bottom=bottom, color=colors[t], width=0.7,
               label=f'{t} - {CONSEQUENCE_TYPES[t]}')
        for i, v in enumerate(vals):
            if v > 0.06:
                ax.text(i, bottom[i] + v / 2, f'{v*100:.0f}%',
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
        bottom += vals

    labels = []
    for c in cids:
        nm = cluster_names.get(c, f'cluster {c}')
        labels.append(f'C{c}\n{nm}\n(n={by_cluster[c]["n"]})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Fraction of observable faults')
    ax.set_ylim(0, 1.0)
    ax.set_title('Functional consequence mix per taxonomy cluster\n'
                 '(observable faults, first post-injection signal divergence)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args) -> int:
    data_dir    = Path(args.data_dir).resolve()
    cache_dir   = ITC2_ROOT / '.cache'
    results_dir = ITC2_ROOT / 'results'
    fig_dir     = ITC2_ROOT / 'reports' / 'figures'
    for d in [results_dir, fig_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print("  ITC2 - Functional Characterization (signal divergence)")
    print(f"{'='*64}")

    meta = pd.read_csv(data_dir / 'metadata.csv')
    obs  = meta[meta['label'] == LABEL_OBS].reset_index(drop=True)
    print(f"\n  Observable faults to characterize : {len(obs)}")

    # Site -> cluster from the taxonomy (string-keyed JSON).
    tax_path = results_dir / 'fault_taxonomy.json'
    site_cluster, cluster_names = {}, {}
    if tax_path.exists():
        tax = json.loads(tax_path.read_text())
        site_cluster = {int(k): int(v) for k, v in tax['cluster_assignments'].items()}
        cluster_names = {str(cid): c['name'] for cid, c in tax['clusters'].items()}
        print(f"  Loaded cluster assignments for {len(site_cluster)} sites")
    else:
        print("  WARNING: fault_taxonomy.json not found - cluster breakdown skipped")

    # Cache one clean reference per workload.
    clean_cache = {}
    for w in obs['workload'].unique():
        clean_cache[w] = find_clean_trace(cache_dir, w)
        if clean_cache[w] is None:
            print(f"  WARNING: no cached clean trace for workload '{w}'")

    records, n_missing, n_silent, n_latency_match = [], 0, 0, 0
    for _, r in obs.iterrows():
        w, site, sa = r['workload'], int(r['fault_site']), int(r['sa_value'])
        fc = int(r['fault_cycle'])                   # per-sample injection position
        clean = clean_cache.get(w)
        fault = load_fault_trace(cache_dir, w, site, sa, fc)
        if clean is None or fault is None:
            n_missing += 1
            continue
        res = characterize(fault, clean, fc)
        if res is None:
            n_silent += 1
            continue
        if int(r['latency']) == res['latency']:
            n_latency_match += 1
        records.append({
            'fault_site':       site,
            'workload':         w,
            'category':         r['category'],
            'sa_value':         sa,
            'cluster':          site_cluster.get(site, -1),
            **res,
        })

    print(f"\n  Characterized          : {len(records)}")
    print(f"  Missing from cache     : {n_missing}")
    print(f"  Identical to clean ref : {n_silent}")
    if records:
        print(f"  Latency reproduces metadata.csv : "
              f"{n_latency_match}/{len(records)} "
              f"({100*n_latency_match/len(records):.1f}%)")

    # Overall consequence distribution.
    overall = Counter(r['consequence_type'] for r in records)
    total = sum(overall.values()) or 1
    consequence_distribution = {
        t: {'label': CONSEQUENCE_TYPES[t], 'count': int(overall.get(t, 0)),
            'fraction': round(overall.get(t, 0) / total, 4)}
        for t in CONSEQUENCE_TYPES
    }
    print(f"\n  {'type':<5} {'label':<26} {'count':>6} {'frac':>7}")
    for t in CONSEQUENCE_TYPES:
        d = consequence_distribution[t]
        print(f"  {t:<5} {d['label']:<26} {d['count']:>6} {d['fraction']:>7.3f}")

    by_category = _type_counts(records, 'category')
    by_workload = _type_counts(records, 'workload')
    by_cluster  = _type_counts(records, 'cluster')

    # How often each signal participates in the *first* divergence (regardless
    # of which one precedence selected). This exposes precedence effects: e.g.
    # the decode flag mem_instr (type C) may co-diverge but always alongside a
    # higher-precedence control-flow signal, so it never becomes the primary.
    first_div_cooccurrence = Counter()
    for r in records:
        for s in r['diverging_signals']:
            first_div_cooccurrence[s] += 1
    first_div_cooccurrence = {
        s: int(first_div_cooccurrence.get(s, 0)) for s in SIGNALS}

    # Figure (clusters only; skip the -1 "unassigned" bucket if present).
    by_cluster_named = {c: v for c, v in by_cluster.items() if c != '-1'}
    if by_cluster_named:
        fig_path = fig_dir / 'consequence_by_cluster.png'
        plot_consequence_by_cluster(by_cluster_named, cluster_names, fig_path)
        print(f"\n  Figure : {fig_path.name}")

    # Fix 3 - fold the consequence breakdown back into the taxonomy so
    # fault_taxonomy.json is self-contained (observability + functional axes).
    if tax_path.exists() and not args.no_taxonomy_augment:
        n_annotated = augment_taxonomy(tax_path, by_cluster)
        print(f"  Augmented {tax_path.name}: {n_annotated} clusters tagged with "
              f"consequence breakdown (T3-silent clusters set to null)")

    out = {
        'generated':   datetime.now().isoformat(timespec='seconds'),
        'method':      'first post-injection bus-signal divergence vs clean trace',
        'fault_cycle': 'per-sample (position-matched grid; see metadata.csv)',
        'consequence_type_definitions': CONSEQUENCE_TYPES,
        'signal_precedence': [{'signal': s, 'type': t} for s, t in SIGNAL_PRECEDENCE],
        'n_observable':          int(len(obs)),
        'n_characterized':       len(records),
        'n_missing_from_cache':  n_missing,
        'n_identical_to_clean':  n_silent,
        'latency_match_rate':    round(n_latency_match / len(records), 4) if records else None,
        'consequence_distribution': consequence_distribution,
        'first_divergence_signal_cooccurrence': first_div_cooccurrence,
        'by_category':           by_category,
        'by_workload':           by_workload,
        'by_cluster':            by_cluster,
        'cluster_names':         cluster_names,
        'per_fault':             records,
    }
    out_path = results_dir / 'functional_characterization.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  Results : {out_path}")
    print(f"\n{'='*64}\n")
    return 0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data',
                   help='Directory with metadata.csv')
    p.add_argument('--no-taxonomy-augment', action='store_true',
                   help='Do not write the consequence breakdown back into '
                        'fault_taxonomy.json (Fix 3 augmentation)')
    return p.parse_args()


if __name__ == '__main__':
    raise SystemExit(main(parse_args()))
