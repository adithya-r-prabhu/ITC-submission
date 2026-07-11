#!/usr/bin/env python3
"""
Real Fault Dataset Builder (POSITION-MATCHED).

The dataset is built on a shared grid of fault_cycle POSITIONS so that clean
and faulty samples have the same position distribution. This removes the
fault_cycle-position confound that previously inflated every workload's AUROC:
when all faults were injected at a fixed cycle (200) while clean samples spanned
many cycles, any position-varying feature could "detect" faults by reading off
*when* the window was taken rather than *whether* a fault occurred. (See
src/confound_audit.py and src/position_matched_reeval.py for the diagnosis.)

For each of the 580 architectural fault sites x 5 workloads:
  - Injects the fault at the position assigned to the site's split
  - Runs sim_fault_oracle to obtain the faulty trace CSV
  - Labels observability by comparing against the CLEAN reference trace at the
    SAME position (the v5 pre-computed flags are valid only at the former fixed
    cycle 200, so they are not used for labels here)
  - Extracts features via the exact v5 observable pipeline

CLEAN samples are one deterministic trace per (workload, position).

Two independent, aligned splits:
  - Fault SITES: stratified random split (random_state=42) so observable and
    silent sites are proportionally represented in every split.
  - POSITIONS: DISJOINT partition across train/val/test. Because a clean trace
    is deterministic per (workload, position), a shared position would leak an
    identical clean vector; disjoint positions prevent that AND keep position
    balanced between clean and faulty within each split.
Each site is injected only at positions belonging to its split's position set.

Outputs (in --out-dir):
  features_seq.npy   (N, 64, 13)
  features_agg.npy   (N, 156)
  labels.npy         (N,)
  metadata.csv

Usage:
    python src/build_real_fault_dataset.py --itc-root /home/atomic_zuccini/ITC
    python src/build_real_fault_dataset.py --itc-root /home/atomic_zuccini/ITC --max-sites 20
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ITC2_ROOT  = SCRIPT_DIR.parent

# Fallback contiguous ranges - used only when obs_map is unavailable.
# The primary split is computed via _compute_stratified_split().
_FALLBACK_SPLIT_BOUNDARIES = {
    'train': (0,   400),
    'val':   (400, 490),
    'test':  (490, 580),
}

WORKLOADS = ['counting_loop', 'alu_heavy', 'branch_heavy', 'mem_intensive', 'irq_test']
FAULT_CYCLE_DEFAULT = 200
SIM_CYCLES = 600
LABEL_CLEAN    = 0
LABEL_OBS      = 1
LABEL_SILENT   = 2


# ---------------------------------------------------------------------------
# ITC imports (injected after arg parsing so --itc-root can be overridden)
# ---------------------------------------------------------------------------

def _setup_itc_imports(itc_root: Path):
    src = str(itc_root / 'src')
    if src not in sys.path:
        sys.path.insert(0, src)


# ---------------------------------------------------------------------------
# Simulator helpers (ported from ITC/src/eval_random_faults.py)
# ---------------------------------------------------------------------------

CSV_COLS = ['cycle', 'mem_valid', 'mem_instr', 'mem_ready', 'mem_addr',
            'mem_wstrb', 'mem_wdata', 'mem_rdata', 'trap', 'stall', 'fetch_pc']
OBS_COLS = list(range(1, 11))


def _parse_csv(text: str) -> np.ndarray:
    rows = []
    for line in text.strip().splitlines():
        if line.startswith('cycle'):
            continue
        try:
            rows.append([float(x) for x in line.split(',')])
        except ValueError:
            continue
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 11), dtype=np.float32)


def _run_sim(sim: str, firmware: str, cycles: int,
             fault_site: int, fault_sa: int, fault_cycle: int,
             irq_period: int = 0) -> str:
    cmd = [sim, '--firmware', firmware, '--cycles', str(cycles),
           '--fault-site', str(fault_site), '--fault-sa', str(fault_sa),
           '--fault-cycle', str(fault_cycle)]
    if irq_period > 0:
        cmd += ['--irq-period', str(irq_period)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else ''
    except subprocess.TimeoutExpired:
        return ''


def _csv_to_trace(arr: np.ndarray, fault_cycle: int):
    """Convert raw CSV array to Trace object using v5-compatible signal mapping."""
    from trace_schema import Trace, TraceMetadata
    n = len(arr)
    md = TraceMetadata(
        trace_id='picorv32_real', fault_id='picorv32_real', fault_class='execution',
        injection_cycle=int(fault_cycle), cycle_count=n, program='picorv32',
    )
    tr = Trace.create_empty(n, md)
    if n == 0:
        return tr

    mem_valid = arr[:, 1]; mem_instr = arr[:, 2]; mem_ready = arr[:, 3]
    mem_addr  = arr[:, 4]; mem_wstrb = arr[:, 5]; mem_wdata = arr[:, 6]
    mem_rdata = arr[:, 7]; stall     = arr[:, 9]; fetch_pc  = arr[:, 10]

    valid      = mem_valid > 0
    instr_fetch = valid & (mem_instr > 0)
    data_acc   = valid & (mem_instr == 0)

    tr.set_signal('pc',            fetch_pc.astype(np.uint32))
    tr.set_signal('instr',         np.where(instr_fetch, mem_rdata, 0).astype(np.uint32))
    tr.set_signal('if_active',     instr_fetch.astype(np.uint8))
    tr.set_signal('mem_active',    data_acc.astype(np.uint8))
    tr.set_signal('mem_read',      (data_acc & (mem_wstrb == 0)).astype(np.uint8))
    tr.set_signal('mem_write',     (valid & (mem_wstrb > 0)).astype(np.uint8))
    tr.set_signal('mem_addr',      mem_addr.astype(np.uint32))
    tr.set_signal('mem_wdata',     mem_wdata.astype(np.uint32))
    tr.set_signal('mem_rdata',     mem_rdata.astype(np.uint32))
    tr.set_signal('pipeline_stall', (stall > 0).astype(np.uint8))
    tr.set_signal('mem_stall',     (valid & (mem_ready == 0)).astype(np.uint8))

    pc_int  = fetch_pc.astype(np.int64)
    pc_diff = np.diff(pc_int, prepend=pc_int[0])
    tr.set_signal('branch_taken', ((pc_diff != 0) & (pc_diff != 4)).astype(np.uint8))
    return tr


def _extract_features(arr: np.ndarray, fault_cycle: int):
    """Full v5 feature extraction from a raw CSV array."""
    from model import (extract_features_obs, extract_aggregate_features_obs,
                       WINDOW_SIZE)
    tr = _csv_to_trace(arr, fault_cycle)
    n = tr.num_cycles()
    win_start = max(0, fault_cycle - 8)
    feat_seq = extract_features_obs(tr, win_start, win_start + WINDOW_SIZE)
    feat_agg = extract_aggregate_features_obs(tr, fault_cycle, min(n, fault_cycle + 40))
    return feat_seq, feat_agg


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(firmware_stem: str, fault_site: int, fault_sa: int, fault_cycle: int) -> str:
    raw = f'{firmware_stem}_{fault_site}_{fault_sa}_{fault_cycle}'
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f'{key}.csv'


def _load_cached(cache_dir: Path, key: str) -> str | None:
    p = _cache_path(cache_dir, key)
    return p.read_text() if p.exists() else None


def _save_cached(cache_dir: Path, key: str, csv_text: str):
    _cache_path(cache_dir, key).write_text(csv_text)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _compute_stratified_split(
    site_ids: list,
    obs_map: dict,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Returns {site_id -> 'train'|'val'|'test'} via a stratified random split.

    Stratification key: a site is '1' (observable) if it is observable in
    at least one workload in obs_map, else '0' (always-silent).  This ensures
    each split receives a proportional share of observable fault sites so that
    Experiment A (fault-site split) yields a valid AUROC on the test set.

    Falls back to contiguous ranges if obs_map is empty.
    """
    if not obs_map:
        print("  WARNING: obs_map empty - falling back to contiguous site ranges.")
        result = {}
        for sid in site_ids:
            for split, (lo, hi) in _FALLBACK_SPLIT_BOUNDARIES.items():
                if lo <= sid < hi:
                    result[sid] = split
                    break
            else:
                result[sid] = 'train'
        return result

    from sklearn.model_selection import train_test_split as _tts

    # Build observability flag per site across all workloads.
    obs_flag = [
        int(any(
            obs_map.get(w, {}).get(sid, {}).get('observable', False)
            for w in obs_map
        ))
        for sid in site_ids
    ]

    n_obs    = sum(obs_flag)
    n_silent = len(site_ids) - n_obs
    print(f"  Stratified split: {n_obs} observable sites, {n_silent} always-silent sites")

    test_frac = 1.0 - train_frac - val_frac

    ids_train, ids_valtest, _, st_valtest = _tts(
        site_ids, obs_flag,
        test_size=(val_frac + test_frac),
        stratify=obs_flag,
        random_state=seed,
    )
    val_of_valtest = val_frac / (val_frac + test_frac)  # 0.5
    ids_val, ids_test = _tts(
        ids_valtest,
        test_size=(1.0 - val_of_valtest),
        stratify=st_valtest,
        random_state=seed,
    )

    result = {}
    for s in ids_train: result[s] = 'train'
    for s in ids_val:   result[s] = 'val'
    for s in ids_test:  result[s] = 'test'

    # Print summary for verification.
    for split_name, subset in [('train', ids_train), ('val', ids_val), ('test', ids_test)]:
        n_o = sum(1 for s in subset if obs_flag[site_ids.index(s)])
        print(f"    {split_name:5s}: {len(subset):3d} sites, "
              f"{n_o} observable ({100*n_o/len(subset):.1f}%)")

    return result


def _compute_position_split(
    positions: list,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Returns {position -> 'train'|'val'|'test'} as a DISJOINT partition.

    Positions are partitioned (not shared) across splits so that no clean
    reference trace - which is deterministic for a given (workload, position) -
    can appear in two splits and leak.  Positions are shuffled before slicing so
    each split's positions span the full cycle range (avoiding a contiguous
    early=train / late=test bias).

    Faulty samples are injected only at positions belonging to their site's
    split, so within every split clean and faulty share the same position
    distribution - removing the fault_cycle-position confound by construction.
    """
    rng = np.random.default_rng(seed)
    shuffled = list(positions)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr = max(1, int(round(train_frac * n)))
    n_va = max(1, int(round(val_frac * n)))
    n_tr = min(n_tr, n - 2)  # leave >=1 for val and test
    n_va = min(n_va, n - n_tr - 1)
    result = {}
    for p in shuffled[:n_tr]:            result[p] = 'train'
    for p in shuffled[n_tr:n_tr + n_va]: result[p] = 'val'
    for p in shuffled[n_tr + n_va:]:     result[p] = 'test'
    return result


# ---------------------------------------------------------------------------
# Load observability labels from existing evaluation results
# ---------------------------------------------------------------------------

def _load_observability(itc_root: Path) -> dict:
    """
    Returns dict: {workload -> {site_id -> {'observable': bool, 'latency': int|None}}}
    Prefers the v5 multiworkload file; falls back to v4.
    """
    candidates = [
        itc_root / 'results' / 'random_fault_v5_multiworkload.json',
        itc_root / 'results' / 'random_fault_multiworkload.json',
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print("WARNING: No pre-computed observability results found. "
              "Labels will be determined by re-running all simulations.")
        return {}

    print(f"  Loading observability labels from {path.name}")
    with open(path) as f:
        data = json.load(f)

    obs_map = {}
    for w in data.get('workloads', []):
        fw = w['firmware']
        obs_map[fw] = {}
        for r in w.get('results', []):
            obs_map[fw][r['id']] = {
                'observable': r.get('observable', False),
                'latency':    r.get('latency', None),
            }
    return obs_map


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def build_dataset(args) -> int:
    itc_root = Path(args.itc_root).resolve()
    out_dir  = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _setup_itc_imports(itc_root)

    sim      = str(itc_root / 'build' / 'sim_fault_oracle')
    fw_dir   = itc_root / 'src' / 'firmware'
    sites_f  = itc_root / 'src' / 'fault_sites_picorv32.json'
    cache_dir = (ITC2_ROOT / '.cache')
    if not args.no_cache:
        cache_dir.mkdir(exist_ok=True)

    for p, name in [(sim, 'sim'), (fw_dir, 'firmware dir'), (sites_f, 'fault sites')]:
        if not Path(p).exists():
            print(f"ERROR: {name} not found: {p}"); return 1

    with open(sites_f) as f:
        sites = json.load(f)
    if args.max_sites:
        sites = sites[:args.max_sites]

    obs_map = _load_observability(itc_root)

    firmwares = {Path(fw_dir / f'{w}.bin'): w for w in WORKLOADS
                 if (fw_dir / f'{w}.bin').exists()}
    if not firmwares:
        print("ERROR: no firmware .bin files found"); return 1

    # Position grid (replaces fixed fault_cycle=200). Faulty and clean samples
    # are both drawn from this grid so position cannot discriminate class.
    positions = sorted(set(int(p) for p in
                           np.linspace(args.position_min, args.position_max,
                                       args.n_positions)))
    pos_split_map = _compute_position_split(positions)
    pos_by_split = defaultdict(list)
    for p, s in pos_split_map.items():
        pos_by_split[s].append(p)
    for s in pos_by_split:
        pos_by_split[s].sort()

    print(f"\n{'='*64}")
    print("  ITC2 - Real Fault Dataset Builder (position-matched)")
    print(f"{'='*64}")
    print(f"  Fault sites : {len(sites)}")
    print(f"  Workloads   : {', '.join(firmwares.values())}")
    print(f"  Positions   : {len(positions)} in [{args.position_min}, {args.position_max}]")
    print(f"    train={len(pos_by_split['train'])}  val={len(pos_by_split['val'])}  "
          f"test={len(pos_by_split['test'])} (disjoint)")
    print(f"  Cache       : {'disabled' if args.no_cache else str(cache_dir)}")

    # Compute stratified split map once before the main loop.
    all_site_ids = [site['id'] for site in sites]
    print("\n  Computing stratified fault-site split...")
    site_split_map = _compute_stratified_split(all_site_ids, obs_map)

    # Assign each fault site one injection position drawn round-robin from its
    # split's position set, so faulty positions evenly cover (and stay within)
    # the same positions as that split's clean samples.
    site_fault_cycle = {}
    counters = defaultdict(int)
    for site in sites:
        sid = site['id']
        sp  = site_split_map.get(sid, 'train')
        plist = pos_by_split.get(sp) or positions
        site_fault_cycle[sid] = plist[counters[sp] % len(plist)]
        counters[sp] += 1

    feat_seq_list, feat_agg_list, labels_list, meta_rows = [], [], [], []
    sample_id = 0

    def _sim_arr(site_id, sa, fc, fw_path, fw_name, irq_period):
        """Run (or load cached) a simulation and return the parsed CSV array."""
        key = _cache_key(fw_name, site_id, sa, fc)
        csv_text = None if args.no_cache else _load_cached(cache_dir, key)
        if csv_text is None:
            csv_text = _run_sim(sim, str(fw_path), SIM_CYCLES,
                                site_id, sa, fc, irq_period)
            if csv_text and not args.no_cache:
                _save_cached(cache_dir, key, csv_text)
        if not csv_text:
            return None
        arr = _parse_csv(csv_text)
        return arr if len(arr) else None

    # ------------------------------------------------------------------
    # 1. Clean reference traces (one per workload x position) - also used to
    #    label faulty samples by trace comparison at the matching position.
    # ------------------------------------------------------------------
    print(f"\n[1/2] Generating clean samples ({len(positions)} positions x {len(firmwares)} workloads)...")
    clean_arrs = {}   # (fw_name, fc) -> array
    n_clean = 0
    for fw_path, fw_name in firmwares.items():
        irq_period = 80 if fw_name == 'irq_test' else 0
        for fc in positions:
            arr = _sim_arr(-1, 0, fc, fw_path, fw_name, irq_period)
            if arr is None:
                continue
            clean_arrs[(fw_name, fc)] = arr
            try:
                f_seq, f_agg = _extract_features(arr, fc)
            except Exception as e:
                print(f"  WARNING: clean feature extraction failed for {fw_name} fc={fc}: {e}")
                continue
            feat_seq_list.append(f_seq)
            feat_agg_list.append(f_agg)
            labels_list.append(LABEL_CLEAN)
            meta_rows.append({
                'sample_id': sample_id, 'fault_site': -1, 'workload': fw_name,
                'category': 'clean', 'sa_value': -1, 'observable': False,
                'latency': -1, 'fault_cycle': fc,
                'split': pos_split_map[fc], 'label': LABEL_CLEAN,
            })
            sample_id += 1
            n_clean += 1
    print(f"  {n_clean} clean samples generated")

    # ------------------------------------------------------------------
    # 2. Faulty samples - each site injected at its assigned position.
    #    Observability is derived by comparing against the clean reference at
    #    the SAME position (the v5 pre-computed labels are valid only at the
    #    former fixed cycle 200, so they are not used here).
    # ------------------------------------------------------------------
    print(f"\n[2/2] Generating faulty samples ({len(sites)} sites x {len(firmwares)} workloads)...")
    n_done = 0
    n_total_fault = len(sites) * len(firmwares)

    for fw_path, fw_name in firmwares.items():
        irq_period = 80 if fw_name == 'irq_test' else 0

        for site in sites:
            sid      = site['id']
            sa_val   = site['sa_value']
            category = site['category']
            split    = site_split_map.get(sid, 'train')
            fc       = site_fault_cycle[sid]

            arr = _sim_arr(sid, sa_val, fc, fw_path, fw_name, irq_period)
            if arr is None:
                n_done += 1
                continue

            clean_arr = clean_arrs.get((fw_name, fc))
            if clean_arr is None:
                n_done += 1
                continue
            n = min(len(clean_arr), len(arr))
            c = clean_arr[fc:n, :][:, OBS_COLS]
            f = arr[fc:n, :][:, OBS_COLS]
            observable = bool(np.any(c != f))
            diff_rows  = np.where(np.any(c != f, axis=1))[0]
            latency    = int(diff_rows[0]) if len(diff_rows) else None
            label = LABEL_OBS if observable else LABEL_SILENT

            try:
                f_seq, f_agg = _extract_features(arr, fc)
            except Exception as e:
                print(f"  WARNING: feature extraction failed for site {sid} / {fw_name}: {e}")
                n_done += 1
                continue

            feat_seq_list.append(f_seq)
            feat_agg_list.append(f_agg)
            labels_list.append(label)
            meta_rows.append({
                'sample_id': sample_id, 'fault_site': sid, 'workload': fw_name,
                'category': category, 'sa_value': sa_val, 'observable': observable,
                'latency': latency if latency is not None else -1, 'fault_cycle': fc,
                'split': split, 'label': label,
            })
            sample_id += 1
            n_done += 1

            if n_done % 200 == 0 or n_done == n_total_fault:
                print(f"  [{n_done}/{n_total_fault}] faulty done; total samples: {sample_id}")

    # ------------------------------------------------------------------
    # 3. Save outputs
    # ------------------------------------------------------------------
    if not feat_seq_list:
        print("ERROR: no samples collected"); return 1

    print(f"\nSaving dataset to {out_dir} ...")
    feat_seq_arr = np.stack(feat_seq_list, axis=0).astype(np.float32)
    feat_agg_arr = np.stack(feat_agg_list, axis=0).astype(np.float32)
    labels_arr   = np.array(labels_list, dtype=np.int64)

    np.save(str(out_dir / 'features_seq.npy'), feat_seq_arr)
    np.save(str(out_dir / 'features_agg.npy'), feat_agg_arr)
    np.save(str(out_dir / 'labels.npy'),        labels_arr)

    df = pd.DataFrame(meta_rows)
    df.to_csv(str(out_dir / 'metadata.csv'), index=False)

    # ------------------------------------------------------------------
    # 4. Summary and integrity check
    # ------------------------------------------------------------------
    _print_summary(df, labels_arr)
    _check_integrity(df)

    print(f"\n  features_seq.npy : {feat_seq_arr.shape}")
    print(f"  features_agg.npy : {feat_agg_arr.shape}")
    print(f"  labels.npy       : {labels_arr.shape}")
    print(f"  metadata.csv     : {len(df)} rows")
    print(f"  Timestamp        : {datetime.now().isoformat()}")
    print("\n  Done.")
    return 0


def _print_summary(df: pd.DataFrame, labels: np.ndarray):
    label_names = {LABEL_CLEAN: 'clean', LABEL_OBS: 'observable', LABEL_SILENT: 'silent'}
    print(f"\n{'='*52}")
    print("  Dataset Summary")
    print(f"{'='*52}")
    print(f"  {'Split':<8} {'Clean':>7} {'Obs':>7} {'Silent':>7} {'Total':>7}")
    print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for split in ['train', 'val', 'test']:
        mask = df['split'] == split
        n_c  = int((df.loc[mask, 'label'] == LABEL_CLEAN).sum())
        n_o  = int((df.loc[mask, 'label'] == LABEL_OBS).sum())
        n_s  = int((df.loc[mask, 'label'] == LABEL_SILENT).sum())
        print(f"  {split:<8} {n_c:>7} {n_o:>7} {n_s:>7} {n_c+n_o+n_s:>7}")
    print()
    print("  By workload:")
    for wl in df['workload'].unique():
        m  = df['workload'] == wl
        nc = int((df.loc[m, 'label'] == LABEL_CLEAN).sum())
        no = int((df.loc[m, 'label'] == LABEL_OBS).sum())
        ns = int((df.loc[m, 'label'] == LABEL_SILENT).sum())
        print(f"    {wl:<16} clean={nc}  obs={no}  silent={ns}")


def _check_integrity(df: pd.DataFrame):
    fault_df = df[df['fault_site'] >= 0]
    train_sites = set(fault_df.loc[fault_df['split'] == 'train', 'fault_site'])
    test_sites  = set(fault_df.loc[fault_df['split'] == 'test',  'fault_site'])
    leak = train_sites & test_sites
    if leak:
        print(f"ERROR: {len(leak)} fault sites appear in both train and test splits!")
    else:
        print("  Integrity check PASSED: no fault site appears in both train and test.")

    # Clean traces are deterministic per (workload, position); a shared position
    # across splits would leak an identical clean vector. Positions must be
    # disjoint across splits (this also keeps position balanced within a split).
    if 'fault_cycle' in df.columns:
        clean_df = df[df['fault_site'] < 0]
        pos_by_split = {s: set(clean_df.loc[clean_df['split'] == s, 'fault_cycle'])
                        for s in ('train', 'val', 'test')}
        overlap = (pos_by_split['train'] & pos_by_split['test']) | \
                  (pos_by_split['train'] & pos_by_split['val']) | \
                  (pos_by_split['val'] & pos_by_split['test'])
        if overlap:
            print(f"ERROR: clean positions shared across splits: {sorted(overlap)}")
        else:
            print("  Integrity check PASSED: clean positions are disjoint across splits.")

        # Position balance: within each split, clean and faulty must share the
        # same set of fault_cycle positions (else the confound returns).
        for s in ('train', 'val', 'test'):
            cpos = set(df.loc[(df['split'] == s) & (df['fault_site'] < 0), 'fault_cycle'])
            fpos = set(df.loc[(df['split'] == s) & (df['fault_site'] >= 0), 'fault_cycle'])
            if fpos - cpos:
                print(f"  WARNING: split '{s}' has faulty positions with no clean "
                      f"reference: {sorted(fpos - cpos)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Build real fault dataset from RTL simulation')
    p.add_argument('--itc-root',       default='/home/atomic_zuccini/ITC')
    p.add_argument('--out-dir',        default=str(ITC2_ROOT / 'data'))
    p.add_argument('--n-positions',    type=int, default=50,
                   help='Number of fault_cycle positions in the shared grid '
                        '(used for both clean and faulty samples)')
    p.add_argument('--position-min',   type=int, default=120,
                   help='Lowest fault_cycle in the grid (needs pre-window room)')
    p.add_argument('--position-max',   type=int, default=440,
                   help='Highest fault_cycle in the grid (needs post-window room)')
    p.add_argument('--max-sites',      type=int, default=None,
                   help='Limit fault sites (for quick testing)')
    p.add_argument('--no-cache',       action='store_true',
                   help='Disable simulation output caching')
    return p.parse_args()


if __name__ == '__main__':
    sys.exit(build_dataset(_parse_args()))
