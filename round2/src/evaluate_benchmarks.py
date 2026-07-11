import argparse
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
sys.path.insert(0, str(SCRIPT_DIR))

SIM = REPO_ROOT / "external" / "benchmark_simulator"
FW_DIR = REPO_ROOT / "external" / "benchmark_firmware"
CACHE = REPO_ROOT / ".cache" / "bench_campaign"
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "reports" / "figures"

BENCHES = ["coremark", "dhrystone"]
MEM_WORDS = 65536
TOTAL_CYCLES = 400_000


POSITIONS = [5_000, 50_000, 95_000, 140_000, 185_000, 230_000, 275_000, 320_000]
OBS_HORIZON = 2_000
OBS_COLS = list(range(1, 11))
N_CLEAN_POS = 40


def run_sim(fw, cycles, site=-1, sa=0, fc=200):
    cmd = [
        str(SIM),
        "--firmware",
        str(fw),
        "--cycles",
        str(cycles),
        "--mem-words",
        str(MEM_WORDS),
        "--fault-model",
        "stuck_at",
    ]
    if site >= 0:
        cmd += [
            "--fault-site",
            str(site),
            "--fault-sa",
            str(sa),
            "--fault-cycle",
            str(fc),
        ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"sim failed: {r.stderr[:300]}")
    rows = [
        ln
        for ln in r.stdout.splitlines()
        if ln and (ln[0].isdigit() or ln.startswith("cycle"))
    ]
    from io import StringIO

    return pd.read_csv(StringIO("\n".join(rows))).values.astype(np.int64)


def load_sites():
    meta = pd.read_csv(REPO_ROOT / "data_picorv32" / "metadata.csv")
    f = meta[meta.fault_site >= 0][["fault_site", "sa_value", "category"]]
    return f.drop_duplicates("fault_site").sort_values("fault_site")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-sites", type=int, default=None)
    ap.add_argument("--benches", nargs="+", default=BENCHES)
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    sites = load_sites()
    if args.max_sites:
        sites = sites.iloc[: args.max_sites]
    print(
        f"{len(sites)} sites x {len(POSITIONS)} positions x {len(args.benches)} benches"
    )

    clean = {}
    for b in args.benches:
        print(f"  clean reference: {b} ({TOTAL_CYCLES} cycles) ...")
        clean[b] = run_sim(FW_DIR / f"{b}.bin", TOTAL_CYCLES)
        assert clean[b][:, 8].max() == 0, f"{b} clean run traps"

    per_bench = {}
    for b in args.benches:
        cpath = CACHE / f"{b}.csv"
        done = {}
        if cpath.exists():
            for ln in cpath.read_text().splitlines():
                s, p, o, d = ln.split(",")
                done[(int(s), int(p))] = (int(o), int(d))
        rows = []
        todo = [
            (int(r.fault_site), int(p)) for _, r in sites.iterrows() for p in POSITIONS
        ]
        todo = [t for t in todo if t not in done]
        print(f"  [{b}] {len(todo)} sims to run ({len(done)} cached)")
        with open(cpath, "a") as fh:
            for n, (site, fc) in enumerate(todo):
                sa = int(sites[sites.fault_site == site].sa_value.iloc[0])
                tr = run_sim(
                    FW_DIR / f"{b}.bin", fc + OBS_HORIZON, site=site, sa=sa, fc=fc
                )
                lim = min(len(tr), fc + OBS_HORIZON, len(clean[b]))
                dif = (tr[fc:lim][:, OBS_COLS] != clean[b][fc:lim][:, OBS_COLS]).any(
                    axis=1
                )
                obs = int(dif.any())
                first = int(np.argmax(dif)) if obs else -1
                fh.write(f"{site},{fc},{obs},{first}\n")
                fh.flush()
                done[(site, fc)] = (obs, first)
                if (n + 1) % 200 == 0:
                    print(f"    [{b}] {n + 1}/{len(todo)}")
        for (site, fc), (obs, first) in done.items():
            if site in set(sites.fault_site):
                rows.append(
                    {
                        "fault_site": site,
                        "fault_cycle": fc,
                        "observable": obs,
                        "first_div": first,
                    }
                )
        per_bench[b] = pd.DataFrame(rows)

    meta5 = pd.read_csv(REPO_ROOT / "data_picorv32" / "metadata.csv")
    meta8 = pd.read_csv(REPO_ROOT / "external" / "extended_workloads" / "metadata.csv")
    site_obs_5 = meta5[meta5.fault_site >= 0].groupby("fault_site").observable.max()
    site_obs_8 = meta8[meta8.fault_site >= 0].groupby("fault_site").observable.max()

    summary = {
        "protocol": {
            "benches": args.benches,
            "positions": POSITIONS,
            "obs_horizon_cycles": OBS_HORIZON,
            "total_cycles": TOTAL_CYCLES,
            "mem_words": MEM_WORDS,
            "firmware": "CoreMark (EEMBC, vendored; ITERATIONS=1, rv32i + "
            "soft muldiv) and Dhrystone (riscv-tests; 20 runs)",
        },
        "per_bench": {},
    }
    sobs = {}
    for b in args.benches:
        d = per_bench[b]
        so = d.groupby("fault_site").observable.max()
        sobs[b] = so

        d480 = d.assign(o480=(d.observable == 1) & (d.first_div <= 480))
        so480 = d480.groupby("fault_site").o480.max()
        cat = sites.set_index("fault_site").loc[so.index, "category"]
        summary["per_bench"][b] = {
            "n_sites": int(len(so)),
            "observable_sites": int(so.sum()),
            "site_observable_rate": round(float(so.mean()), 4),
            "site_observable_rate_480cy_horizon_control": round(float(so480.mean()), 4),
            "median_first_div_cycles": int(d[d.observable == 1].first_div.median())
            if int(so.sum())
            else None,
            "by_category_observable": {
                c: int(so[cat == c].sum()) for c in sorted(cat.unique())
            },
        }
        print(
            f"  [{b}] site-observable {so.mean() * 100:.1f}%  "
            f"({int(so.sum())}/{len(so)}; 480cy-control {so480.mean() * 100:.1f}%)"
        )

    idx = site_obs_5.index
    u5 = site_obs_5.astype(bool)
    u8 = site_obs_8.reindex(idx).fillna(False).astype(bool)
    ub = u8.copy()
    new_sites = {}
    for b in args.benches:
        sb = sobs[b].reindex(idx).fillna(False).astype(bool)
        new_sites[b] = int((sb & ~u8).sum())
        ub = ub | sb

    summary["union"] = {
        "short_workload_observable_sites": int(u5.sum()),
        "extended_workload_observable_sites": int(u8.sum()),
        "extended_plus_benchmark_observable_sites": int(ub.sum()),
        "new_sites_from_benchmarks_over_ext8": new_sites,
        "short_workload_silent_rate": round(float(1 - u5.mean()), 4),
        "extended_workload_silent_rate": round(float(1 - u8.mean()), 4),
        "extended_plus_benchmark_silent_rate": round(float(1 - ub.mean()), 4),
    }

    from train_detectors import make_models
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from build_dataset import _setup_itc_imports, _extract_features

    _setup_itc_imports(REPO_ROOT / "external" / "picorv32_fault_sim")

    def feat_window(arr, fc, pre=8, post=2_000):

        lo = max(0, fc - pre)
        sl = arr[lo : fc + post].copy()
        sl[:, 0] -= lo
        _, agg = _extract_features(sl, fc - lo)
        return agg

    det = {}
    Xagg = np.load(REPO_ROOT / "data_picorv32" / "features_agg.npy")
    y = np.load(REPO_ROOT / "data_picorv32" / "labels.npy")
    binm = (y == 0) | (y == 1)
    Xtr, ytr = Xagg[binm], (y[binm] == 1).astype(int)
    sc = StandardScaler().fit(Xtr)
    models = {n: m.fit(sc.transform(Xtr), ytr) for n, m in make_models("binary")}
    rng = np.random.default_rng(42)
    for b in args.benches:
        feats, labs = [], []
        cl_pos = sorted(
            rng.choice(
                np.arange(5_000, TOTAL_CYCLES - 3_000, 37), N_CLEAN_POS, replace=False
            )
        )
        for fc in cl_pos:
            feats.append(feat_window(clean[b], int(fc)))
            labs.append(0)
        dobs = per_bench[b][per_bench[b].observable == 1].drop_duplicates("fault_site")
        for _, r in dobs.iterrows():
            sa = int(sites[sites.fault_site == r.fault_site].sa_value.iloc[0])
            tr = run_sim(
                FW_DIR / f"{b}.bin",
                int(r.fault_cycle) + OBS_HORIZON,
                site=int(r.fault_site),
                sa=sa,
                fc=int(r.fault_cycle),
            )
            feats.append(feat_window(tr, int(r.fault_cycle)))
            labs.append(1)
        X = np.asarray(feats)
        yb = np.asarray(labs)

        mu = X[yb == 0].mean(axis=0)
        sd = X[yb == 0].std(axis=0) + 1e-9
        Xcal = (X - mu) / sd
        training_mean = Xtr.mean(axis=0)
        training_std = Xtr.std(axis=0) + 1e-9
        Xtr_cal = (Xtr - training_mean) / training_std
        models_cal = {n: m.fit(Xtr_cal, ytr) for n, m in make_models("binary")}
        det[b] = {
            "training_distribution_scaler": {
                n: round(
                    float(roc_auc_score(yb, m.predict_proba(sc.transform(X))[:, 1])), 4
                )
                for n, m in models.items()
            },
            "clean_calibrated": {
                n: round(float(roc_auc_score(yb, m.predict_proba(Xcal)[:, 1])), 4)
                for n, m in models_cal.items()
            },
            "n_clean": int((yb == 0).sum()),
            "n_observable": int((yb == 1).sum()),
        }
        print(
            f"  [detection {b}] naive: "
            + "  ".join(
                f"{n}={v}" for n, v in det[b]["training_distribution_scaler"].items()
            )
        )
        print(
            f"  [detection {b}] calib: "
            + "  ".join(f"{n}={v}" for n, v in det[b]["clean_calibrated"].items())
        )
    summary["short_workload_to_benchmark_transfer"] = det

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "benchmark_workloads.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out}")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    names, rates = [], []
    for wl in sorted(meta5[meta5.fault_site >= 0].workload.unique()):
        m = meta5[(meta5.fault_site >= 0) & (meta5.workload == wl)]
        names.append(wl)
        rates.append(m.groupby("fault_site").observable.max().mean())
    for b in args.benches:
        names.append(f"{b}*")
        rates.append(float(sobs[b].mean()))
    cols = ["#4878a8"] * 5 + ["#c0392b"] * len(args.benches)
    a1.barh(range(len(names)), [r * 100 for r in rates], color=cols)
    a1.set_yticks(range(len(names)))
    a1.set_yticklabels(names, fontsize=8)
    a1.set_xlabel("site-observable rate (%)")
    a1.set_title("Benchmark and short-workload observability")
    u = summary["union"]
    a2.bar(
        ["short workloads", "extended workloads", "+ benchmarks"],
        [
            u["short_workload_silent_rate"] * 100,
            u["extended_workload_silent_rate"] * 100,
            u["extended_plus_benchmark_silent_rate"] * 100,
        ],
        color=["#4878a8", "#e9c46a", "#2a9d8f"],
    )
    a2.set_ylabel("silent ceiling (% of 580 sites)")
    a2.set_title("Ceiling under workload union")
    for ax in (a1, a2):
        ax.grid(alpha=0.3, axis="x" if ax is a1 else "y")
    fig.tight_layout()
    fig.savefig(FIGS / "benchmark_workloads.png", dpi=130)
    print("  wrote reports/figures/benchmark_workloads.png")


if __name__ == "__main__":
    main()
