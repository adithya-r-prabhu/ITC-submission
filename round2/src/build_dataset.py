import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


_FALLBACK_SPLIT_BOUNDARIES = {
    "train": (0, 400),
    "val": (400, 490),
    "test": (490, 580),
}

WORKLOADS = ["counting_loop", "alu_heavy", "branch_heavy", "mem_intensive", "irq_test"]
FAULT_CYCLE_DEFAULT = 200
SIM_CYCLES = 600
LABEL_CLEAN = 0
LABEL_OBS = 1
LABEL_SILENT = 2


def _setup_itc_imports(itc_root):
    src = str(itc_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


CSV_COLS = [
    "cycle",
    "mem_valid",
    "mem_instr",
    "mem_ready",
    "mem_addr",
    "mem_wstrb",
    "mem_wdata",
    "mem_rdata",
    "trap",
    "stall",
    "fetch_pc",
]
OBS_COLS = list(range(1, 11))


def _parse_csv(text):
    rows = []
    for line in text.strip().splitlines():
        if line.startswith("cycle"):
            continue
        try:
            rows.append([float(x) for x in line.split(",")])
        except ValueError:
            continue
    return (
        np.array(rows, dtype=np.float32)
        if rows
        else np.zeros((0, 11), dtype=np.float32)
    )


def _run_sim(
    sim,
    firmware,
    cycles,
    fault_site,
    fault_sa,
    fault_cycle,
    irq_period = 0,
):
    cmd = [
        sim,
        "--firmware",
        firmware,
        "--cycles",
        str(cycles),
        "--fault-site",
        str(fault_site),
        "--fault-sa",
        str(fault_sa),
        "--fault-cycle",
        str(fault_cycle),
    ]
    if irq_period > 0:
        cmd += ["--irq-period", str(irq_period)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        return ""


def _csv_to_trace(arr, fault_cycle):

    from trace_schema import Trace, TraceMetadata

    n = len(arr)
    md = TraceMetadata(
        trace_id="picorv32_real",
        fault_id="picorv32_real",
        fault_class="execution",
        injection_cycle=int(fault_cycle),
        cycle_count=n,
        program="picorv32",
    )
    tr = Trace.create_empty(n, md)
    if n == 0:
        return tr

    mem_valid = arr[:, 1]
    mem_instr = arr[:, 2]
    mem_ready = arr[:, 3]
    mem_addr = arr[:, 4]
    mem_wstrb = arr[:, 5]
    mem_wdata = arr[:, 6]
    mem_rdata = arr[:, 7]
    stall = arr[:, 9]
    fetch_pc = arr[:, 10]

    valid = mem_valid > 0
    instr_fetch = valid & (mem_instr > 0)
    data_acc = valid & (mem_instr == 0)

    tr.set_signal("pc", fetch_pc.astype(np.uint32))
    tr.set_signal("instr", np.where(instr_fetch, mem_rdata, 0).astype(np.uint32))
    tr.set_signal("if_active", instr_fetch.astype(np.uint8))
    tr.set_signal("mem_active", data_acc.astype(np.uint8))
    tr.set_signal("mem_read", (data_acc & (mem_wstrb == 0)).astype(np.uint8))
    tr.set_signal("mem_write", (valid & (mem_wstrb > 0)).astype(np.uint8))
    tr.set_signal("mem_addr", mem_addr.astype(np.uint32))
    tr.set_signal("mem_wdata", mem_wdata.astype(np.uint32))
    tr.set_signal("mem_rdata", mem_rdata.astype(np.uint32))
    tr.set_signal("pipeline_stall", (stall > 0).astype(np.uint8))
    tr.set_signal("mem_stall", (valid & (mem_ready == 0)).astype(np.uint8))

    pc_int = fetch_pc.astype(np.int64)
    pc_diff = np.diff(pc_int, prepend=pc_int[0])
    tr.set_signal("branch_taken", ((pc_diff != 0) & (pc_diff != 4)).astype(np.uint8))
    return tr


def _extract_features(arr, fault_cycle):

    from model import extract_features_obs, extract_aggregate_features_obs, WINDOW_SIZE

    tr = _csv_to_trace(arr, fault_cycle)
    n = tr.num_cycles()
    win_start = max(0, fault_cycle - 8)
    feat_seq = extract_features_obs(tr, win_start, win_start + WINDOW_SIZE)
    feat_agg = extract_aggregate_features_obs(tr, fault_cycle, min(n, fault_cycle + 40))
    return feat_seq, feat_agg


def _cache_key(
    firmware_stem, fault_site, fault_sa, fault_cycle
):
    raw = f"{firmware_stem}_{fault_site}_{fault_sa}_{fault_cycle}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_path(cache_dir, key):
    return cache_dir / f"{key}.csv"


def _load_cached(cache_dir, key):
    p = _cache_path(cache_dir, key)
    return p.read_text() if p.exists() else None


def _save_cached(cache_dir, key, csv_text):
    _cache_path(cache_dir, key).write_text(csv_text)


def _compute_stratified_split(
    site_ids,
    obs_map,
    train_frac = 0.70,
    val_frac = 0.15,
    seed = 42,
):

    if not obs_map:
        print("  WARNING: obs_map empty — falling back to contiguous site ranges.")
        result = {}
        for sid in site_ids:
            for split, (lo, hi) in _FALLBACK_SPLIT_BOUNDARIES.items():
                if lo <= sid < hi:
                    result[sid] = split
                    break
            else:
                result[sid] = "train"
        return result

    from sklearn.model_selection import train_test_split as _tts

    obs_flag = [
        int(
            any(
                obs_map.get(w, {}).get(sid, {}).get("observable", False)
                for w in obs_map
            )
        )
        for sid in site_ids
    ]

    n_obs = sum(obs_flag)
    n_silent = len(site_ids) - n_obs
    print(
        f"  Stratified split: {n_obs} observable sites, {n_silent} always-silent sites"
    )

    test_frac = 1.0 - train_frac - val_frac

    ids_train, ids_valtest, _, st_valtest = _tts(
        site_ids,
        obs_flag,
        test_size=(val_frac + test_frac),
        stratify=obs_flag,
        random_state=seed,
    )
    val_of_valtest = val_frac / (val_frac + test_frac)
    ids_val, ids_test = _tts(
        ids_valtest,
        test_size=(1.0 - val_of_valtest),
        stratify=st_valtest,
        random_state=seed,
    )

    result = {}
    for s in ids_train:
        result[s] = "train"
    for s in ids_val:
        result[s] = "val"
    for s in ids_test:
        result[s] = "test"

    for split_name, subset in [
        ("train", ids_train),
        ("val", ids_val),
        ("test", ids_test),
    ]:
        n_o = sum(1 for s in subset if obs_flag[site_ids.index(s)])
        print(
            f"    {split_name:5s}: {len(subset):3d} sites, "
            f"{n_o} observable ({100 * n_o / len(subset):.1f}%)"
        )

    return result


def _compute_position_split(
    positions,
    train_frac = 0.70,
    val_frac = 0.15,
    seed = 42,
):

    rng = np.random.default_rng(seed)
    shuffled = list(positions)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr = max(1, int(round(train_frac * n)))
    n_va = max(1, int(round(val_frac * n)))
    n_tr = min(n_tr, n - 2)
    n_va = min(n_va, n - n_tr - 1)
    result = {}
    for p in shuffled[:n_tr]:
        result[p] = "train"
    for p in shuffled[n_tr : n_tr + n_va]:
        result[p] = "val"
    for p in shuffled[n_tr + n_va :]:
        result[p] = "test"
    return result


def _load_observability(itc_root):

    candidates = [
        itc_root / "results" / "random_fault_v5_multiworkload.json",
        itc_root / "results" / "random_fault_multiworkload.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print(
            "WARNING: No pre-computed observability results found. "
            "Labels will be determined by re-running all simulations."
        )
        return {}

    print(f"  Loading observability labels from {path.name}")
    with open(path) as f:
        data = json.load(f)

    obs_map = {}
    for w in data.get("workloads", []):
        fw = w["firmware"]
        obs_map[fw] = {}
        for r in w.get("results", []):
            obs_map[fw][r["id"]] = {
                "observable": r.get("observable", False),
                "latency": r.get("latency", None),
            }
    return obs_map


def build_dataset(args):
    itc_root = Path(args.itc_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.positions_per_site < 1:
        print("ERROR: --positions-per-site must be >= 1")
        return 1

    _setup_itc_imports(itc_root)

    sim = str(itc_root / "build" / "sim_fault_oracle")
    fw_dir = (
        Path(args.fw_dir).resolve() if args.fw_dir else itc_root / "src" / "firmware"
    )
    sites_f = itc_root / "src" / "fault_sites_picorv32.json"
    cache_dir = REPO_ROOT / ".cache"
    if not args.no_cache:
        cache_dir.mkdir(exist_ok=True)

    for p, name in [(sim, "sim"), (fw_dir, "firmware dir"), (sites_f, "fault sites")]:
        if not Path(p).exists():
            print(f"ERROR: {name} not found: {p}")
            return 1

    with open(sites_f) as f:
        sites = json.load(f)
    if args.max_sites:
        sites = sites[: args.max_sites]

    obs_map = _load_observability(itc_root)

    workloads = (
        [w.strip() for w in args.workloads.split(",") if w.strip()]
        if args.workloads
        else WORKLOADS
    )
    firmwares = {
        Path(fw_dir / f"{w}.bin"): w
        for w in workloads
        if (fw_dir / f"{w}.bin").exists()
    }
    if not firmwares:
        print("ERROR: no firmware .bin files found")
        return 1

    positions = sorted(
        set(
            int(p)
            for p in np.linspace(args.position_min, args.position_max, args.n_positions)
        )
    )
    pos_split_map = _compute_position_split(positions)
    pos_by_split = defaultdict(list)
    for p, s in pos_split_map.items():
        pos_by_split[s].append(p)
    for s in pos_by_split:
        pos_by_split[s].sort()

    print(f"\n{'=' * 64}")
    print("  Position-matched fault dataset builder")
    print(f"{'=' * 64}")
    print(f"  Fault sites : {len(sites)}")
    print(f"  Workloads   : {', '.join(firmwares.values())}")
    print(
        f"  Positions   : {len(positions)} in [{args.position_min}, {args.position_max}]"
    )
    print(
        f"    train={len(pos_by_split['train'])}  val={len(pos_by_split['val'])}  "
        f"test={len(pos_by_split['test'])} (disjoint)"
    )
    print(f"  Cache       : {'disabled' if args.no_cache else str(cache_dir)}")

    all_site_ids = [site["id"] for site in sites]
    print("\n  Computing stratified fault-site split...")
    site_split_map = _compute_stratified_split(all_site_ids, obs_map)

    site_fault_cycles = {}
    counters = defaultdict(int)
    for site in sites:
        sid = site["id"]
        sp = site_split_map.get(sid, "train")
        plist = pos_by_split.get(sp) or positions
        if args.positions_per_site == 1:
            site_fault_cycles[sid] = [plist[counters[sp] % len(plist)]]
            counters[sp] += 1
        else:
            k = min(args.positions_per_site, len(plist))
            rng = np.random.default_rng(42 + int(sid))
            site_fault_cycles[sid] = sorted(
                rng.choice(plist, size=k, replace=False).tolist()
            )

    feat_seq_list, feat_agg_list, labels_list, meta_rows = [], [], [], []
    sample_id = 0

    def _sim_arr(site_id, sa, fc, fw_path, fw_name, irq_period):

        key = _cache_key(fw_name, site_id, sa, fc)
        csv_text = None if args.no_cache else _load_cached(cache_dir, key)
        if csv_text is None:
            csv_text = _run_sim(
                sim, str(fw_path), SIM_CYCLES, site_id, sa, fc, irq_period
            )
            if csv_text and not args.no_cache:
                _save_cached(cache_dir, key, csv_text)
        if not csv_text:
            return None
        arr = _parse_csv(csv_text)
        return arr if len(arr) else None

    print(
        f"\n[1/2] Generating clean samples ({len(positions)} positions × {len(firmwares)} workloads)..."
    )
    clean_arrs = {}
    n_clean = 0
    for fw_path, fw_name in firmwares.items():
        irq_period = 80 if fw_name == "irq_test" else 0
        for fc in positions:
            arr = _sim_arr(-1, 0, fc, fw_path, fw_name, irq_period)
            if arr is None:
                continue
            clean_arrs[(fw_name, fc)] = arr
            try:
                f_seq, f_agg = _extract_features(arr, fc)
            except Exception as e:
                print(
                    f"  WARNING: clean feature extraction failed for {fw_name} fc={fc}: {e}"
                )
                continue
            feat_seq_list.append(f_seq)
            feat_agg_list.append(f_agg)
            labels_list.append(LABEL_CLEAN)
            meta_rows.append(
                {
                    "sample_id": sample_id,
                    "fault_site": -1,
                    "workload": fw_name,
                    "category": "clean",
                    "sa_value": -1,
                    "observable": False,
                    "latency": -1,
                    "fault_cycle": fc,
                    "split": pos_split_map[fc],
                    "label": LABEL_CLEAN,
                }
            )
            sample_id += 1
            n_clean += 1
    print(f"  {n_clean} clean samples generated")

    print(
        f"\n[2/2] Generating faulty samples ({len(sites)} sites × "
        f"{len(firmwares)} workloads × up to {args.positions_per_site} positions/site)..."
    )
    n_done = 0
    n_total_fault = sum(len(site_fault_cycles[site["id"]]) for site in sites) * len(
        firmwares
    )

    for fw_path, fw_name in firmwares.items():
        irq_period = 80 if fw_name == "irq_test" else 0

        for site in sites:
            sid = site["id"]
            sa_val = site["sa_value"]
            category = site["category"]
            split = site_split_map.get(sid, "train")
            for fc in site_fault_cycles[sid]:
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
                diff_rows = np.where(np.any(c != f, axis=1))[0]
                latency = int(diff_rows[0]) if len(diff_rows) else None
                label = LABEL_OBS if observable else LABEL_SILENT

                try:
                    f_seq, f_agg = _extract_features(arr, fc)
                except Exception as e:
                    print(
                        f"  WARNING: feature extraction failed for site {sid} / {fw_name} "
                        f"fc={fc}: {e}"
                    )
                    n_done += 1
                    continue

                feat_seq_list.append(f_seq)
                feat_agg_list.append(f_agg)
                labels_list.append(label)
                meta_rows.append(
                    {
                        "sample_id": sample_id,
                        "fault_site": sid,
                        "workload": fw_name,
                        "category": category,
                        "sa_value": sa_val,
                        "observable": observable,
                        "latency": latency if latency is not None else -1,
                        "fault_cycle": fc,
                        "split": split,
                        "label": label,
                    }
                )
                sample_id += 1
                n_done += 1

                if n_done % 200 == 0 or n_done == n_total_fault:
                    print(
                        f"  [{n_done}/{n_total_fault}] faulty done; total samples: {sample_id}"
                    )

    if not feat_seq_list:
        print("ERROR: no samples collected")
        return 1

    print(f"\nSaving dataset to {out_dir} ...")
    feat_seq_arr = np.stack(feat_seq_list, axis=0).astype(np.float32)
    feat_agg_arr = np.stack(feat_agg_list, axis=0).astype(np.float32)
    labels_arr = np.array(labels_list, dtype=np.int64)

    np.save(str(out_dir / "features_seq.npy"), feat_seq_arr)
    np.save(str(out_dir / "features_agg.npy"), feat_agg_arr)
    np.save(str(out_dir / "labels.npy"), labels_arr)

    df = pd.DataFrame(meta_rows)
    df.to_csv(str(out_dir / "metadata.csv"), index=False)

    _print_summary(df, labels_arr)
    _check_integrity(df)

    print(f"\n  features_seq.npy : {feat_seq_arr.shape}")
    print(f"  features_agg.npy : {feat_agg_arr.shape}")
    print(f"  labels.npy       : {labels_arr.shape}")
    print(f"  metadata.csv     : {len(df)} rows")
    print("\n  Done.")
    return 0


def _print_summary(df, labels):
    _ = {LABEL_CLEAN: "clean", LABEL_OBS: "observable", LABEL_SILENT: "silent"}
    print(f"\n{'=' * 52}")
    print("  Dataset Summary")
    print(f"{'=' * 52}")
    print(f"  {'Split':<8} {'Clean':>7} {'Obs':>7} {'Silent':>7} {'Total':>7}")
    print(f"  {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")
    for split in ["train", "val", "test"]:
        mask = df["split"] == split
        n_c = int((df.loc[mask, "label"] == LABEL_CLEAN).sum())
        n_o = int((df.loc[mask, "label"] == LABEL_OBS).sum())
        n_s = int((df.loc[mask, "label"] == LABEL_SILENT).sum())
        print(f"  {split:<8} {n_c:>7} {n_o:>7} {n_s:>7} {n_c + n_o + n_s:>7}")
    print()
    print("  By workload:")
    for wl in df["workload"].unique():
        m = df["workload"] == wl
        nc = int((df.loc[m, "label"] == LABEL_CLEAN).sum())
        no = int((df.loc[m, "label"] == LABEL_OBS).sum())
        ns = int((df.loc[m, "label"] == LABEL_SILENT).sum())
        print(f"    {wl:<16} clean={nc}  obs={no}  silent={ns}")


def _check_integrity(df):
    fault_df = df[df["fault_site"] >= 0]
    train_sites = set(fault_df.loc[fault_df["split"] == "train", "fault_site"])
    test_sites = set(fault_df.loc[fault_df["split"] == "test", "fault_site"])
    leak = train_sites & test_sites
    if leak:
        print(f"ERROR: {len(leak)} fault sites appear in both train and test splits!")
    else:
        print("  Integrity check PASSED: no fault site appears in both train and test.")

    if "fault_cycle" in df.columns:
        clean_df = df[df["fault_site"] < 0]
        pos_by_split = {
            s: set(clean_df.loc[clean_df["split"] == s, "fault_cycle"])
            for s in ("train", "val", "test")
        }
        overlap = (
            (pos_by_split["train"] & pos_by_split["test"])
            | (pos_by_split["train"] & pos_by_split["val"])
            | (pos_by_split["val"] & pos_by_split["test"])
        )
        if overlap:
            print(f"ERROR: clean positions shared across splits: {sorted(overlap)}")
        else:
            print(
                "  Integrity check PASSED: clean positions are disjoint across splits."
            )

        for s in ("train", "val", "test"):
            cpos = set(
                df.loc[(df["split"] == s) & (df["fault_site"] < 0), "fault_cycle"]
            )
            fpos = set(
                df.loc[(df["split"] == s) & (df["fault_site"] >= 0), "fault_cycle"]
            )
            if fpos - cpos:
                print(
                    f"  WARNING: split '{s}' has faulty positions with no clean "
                    f"reference: {sorted(fpos - cpos)}"
                )


def _parse_args():
    p = argparse.ArgumentParser(
        description="Build real fault dataset from RTL simulation"
    )
    p.add_argument(
        "--itc-root", default=str(REPO_ROOT / "external" / "picorv32_fault_sim")
    )
    p.add_argument("--out-dir", default=str(REPO_ROOT / "data_picorv32"))
    p.add_argument(
        "--fw-dir",
        default=None,
        help="Firmware .bin directory (default: <itc-root>/src/firmware). "
        "Additive override for augmented-workload runs.",
    )
    p.add_argument(
        "--workloads",
        default=None,
        help="Comma-separated workload list (default: the five study "
        "workloads). Each <wl>.bin must exist in --fw-dir.",
    )
    p.add_argument(
        "--n-positions",
        type=int,
        default=50,
        help="Number of fault_cycle positions in the shared grid "
        "(used for both clean and faulty samples)",
    )
    p.add_argument(
        "--positions-per-site",
        type=int,
        default=1,
        help="Number of distinct split-local positions to inject per fault site.",
    )
    p.add_argument(
        "--position-min",
        type=int,
        default=120,
        help="Lowest fault_cycle in the grid (needs pre-window room)",
    )
    p.add_argument(
        "--position-max",
        type=int,
        default=440,
        help="Highest fault_cycle in the grid (needs post-window room)",
    )
    p.add_argument(
        "--max-sites",
        type=int,
        default=None,
        help="Limit fault sites (for quick testing)",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="Disable simulation output caching"
    )
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(build_dataset(_parse_args()))
