import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ORACLE = REPO_ROOT / "external" / "fault_model_simulator"
OBS_COLS = list(range(1, 11))
MODELS = ["stuck_at", "transient", "delay", "intermittent", "bridging"]
CACHE = REPO_ROOT / ".cache" / "fault_models"


def parse_csv(text):
    rows = []
    for line in text.strip().splitlines():
        if not line or line[0] == "c":
            continue
        parts = line.split(",")
        if len(parts) < 11:
            continue
        try:
            rows.append([int(x) for x in parts[:11]])
        except ValueError:
            continue
    return np.array(rows) if rows else None


def run_sim(fw, site, sa, pos, model, dur, per, irq, site2=-1, sa2=0):
    key = hashlib.md5(
        f"{fw}|{site}|{sa}|{pos}|{model}|{dur}|{per}|{irq}|{site2}|{sa2}".encode()
    ).hexdigest()
    cf = CACHE / f"{key}.csv"
    if cf.exists():
        return parse_csv(cf.read_text())
    cmd = [
        str(ORACLE),
        "--firmware",
        str(fw),
        "--cycles",
        "600",
        "--fault-site",
        str(site),
        "--fault-sa",
        str(sa),
        "--fault-cycle",
        str(pos),
        "--fault-model",
        model,
        "--fault-duration",
        str(dur),
        "--fault-period",
        str(per),
    ]
    if model == "bridging" and site2 >= 0:
        cmd += ["--fault-site2", str(site2), "--fault-sa2", str(sa2)]
    if irq:
        cmd += ["--irq-period", "80"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return None
    cf.write_text(r.stdout)
    return parse_csv(r.stdout)


def first_div(clean, faulted, pos):
    if clean is None or faulted is None:
        return None
    n = min(len(clean), len(faulted))
    for c in range(n):
        if clean[c, 0] < pos:
            continue
        if np.any(clean[c, OBS_COLS] != faulted[c, OBS_COLS]):
            return int(clean[c, 0] - pos)
    return -1


def run(args):
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(str(REPO_ROOT / "data_picorv32" / "metadata.csv"))
    positions = sorted(
        int(x) for x in meta.loc[meta["fault_site"] < 0, "fault_cycle"].unique()
    )
    if 0 < args.n_positions < len(positions):
        pick = np.linspace(0, len(positions) - 1, args.n_positions).round().astype(int)
        positions = [positions[i] for i in sorted(set(pick))]
    workloads = sorted(meta.loc[meta["fault_site"] >= 0, "workload"].unique())
    fw_dir = Path(args.itc_root) / "src" / "firmware"

    faults = meta[meta["fault_site"] >= 0]
    site_tbl = (
        faults.groupby("fault_site")
        .agg(sa_value=("sa_value", "first"), category=("category", "first"))
        .reset_index()
    )
    _ = np.random.default_rng(42)
    if 0 < args.n_sites < len(site_tbl):
        chosen = []
        for cat, grp in site_tbl.groupby("category"):
            k = max(1, int(round(args.n_sites * len(grp) / len(site_tbl))))
            chosen.append(grp.sample(min(k, len(grp)), random_state=42))
        site_tbl = pd.concat(chosen).reset_index(drop=True)

    clean_cache = {}

    b_rng = np.random.default_rng(42)
    all_sites_by_cat = (
        faults.groupby("category")["fault_site"]
        .apply(lambda s: sorted(int(x) for x in s.unique()))
        .to_dict()
    )
    site2_map = {}
    for cat, sids in all_sites_by_cat.items():
        for sid in sids:
            others = [s for s in sids if s // 2 != sid // 2]
            site2_map[sid] = int(b_rng.choice(others)) if others else sid

    def clean(wl, pos):
        k = (wl, pos)
        if k not in clean_cache:
            irq = wl == "irq_test"
            clean_cache[k] = run_sim(
                fw_dir / f"{wl}.bin", -1, 0, pos, "stuck_at", 0, 0, irq
            )
        return clean_cache[k]

    results = {}
    for model in MODELS:
        site_obs = {}
        lat_list = []
        n_obs_samples = 0
        n_samples = 0
        for _, r in site_tbl.iterrows():
            sid, sa = int(r["fault_site"]), int(r["sa_value"])
            any_obs = False
            for wl in workloads:
                irq = wl == "irq_test"
                for pos in positions:
                    cl = clean(wl, pos)
                    s2 = site2_map.get(sid, -1) if model == "bridging" else -1
                    fl = run_sim(
                        fw_dir / f"{wl}.bin",
                        sid,
                        sa,
                        pos,
                        model,
                        args.duration,
                        args.period,
                        irq,
                        site2=s2,
                        sa2=sa,
                    )
                    d = first_div(cl, fl, pos)
                    if d is None:
                        continue
                    n_samples += 1
                    if d >= 0:
                        n_obs_samples += 1
                        lat_list.append(d)
                        any_obs = True
            site_obs[sid] = any_obs
        n_sites = len(site_obs)
        n_obs_sites = sum(site_obs.values())
        results[model] = {
            "n_sites": n_sites,
            "observable_sites": n_obs_sites,
            "site_observable_rate": round(n_obs_sites / n_sites, 4),
            "silent_site_rate": round(1 - n_obs_sites / n_sites, 4),
            "sample_observable_frac": round(n_obs_samples / n_samples, 4)
            if n_samples
            else 0,
            "mean_first_div_latency": round(float(np.mean(lat_list)), 2)
            if lat_list
            else None,
        }
        print(
            f"  {model:13s} site-obs={results[model]['site_observable_rate']:.1%} "
            f"sample-obs={results[model]['sample_observable_frac']:.1%} "
            f"mean-lat={results[model]['mean_first_div_latency']}"
        )

    summary = {
        "n_sites": int(len(site_tbl)),
        "n_positions": len(positions),
        "n_workloads": len(workloads),
        "duration": args.duration,
        "period": args.period,
        "models": results,
        "denominators": {
            "site_observable_rate": "sites observable in at least one injection",
            "sample_observable_frac": "observable individual injections",
        },
    }
    with open(REPO_ROOT / "results" / "fault_models.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  wrote results/fault_models.json")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODELS))
    w = 0.35
    ax.bar(
        x - w / 2,
        [results[m]["site_observable_rate"] for m in MODELS],
        w,
        label="site observable rate",
        color="#2a9d8f",
    )
    ax.bar(
        x + w / 2,
        [results[m]["sample_observable_frac"] for m in MODELS],
        w,
        label="sample observable frac",
        color="#e9c46a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("observable fraction")
    ax.set_title("Observability by fault model (beyond stuck-at)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "reports" / "figures" / "fault_models.png", dpi=130)
    print("  wrote reports/figures/fault_models.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--itc-root", default=str(REPO_ROOT / "external" / "picorv32_fault_sim")
    )
    ap.add_argument("--n-sites", type=int, default=100)
    ap.add_argument("--n-positions", type=int, default=6)
    ap.add_argument("--duration", type=int, default=4)
    ap.add_argument("--period", type=int, default=40)
    args = ap.parse_args()
    if not ORACLE.exists():
        print(f"ERROR: extended oracle not found at {ORACLE}")
        sys.exit(1)
    print("=== Fault-model comparison ===")
    run(args)


if __name__ == "__main__":
    main()
