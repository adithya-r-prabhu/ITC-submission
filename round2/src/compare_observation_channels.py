import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from cluster_bootstrap import make_binary_groups, cluster_bootstrap_auroc_ci

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parent
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "reports" / "figures"
SEED = 42
WIN = 40
SIM_CYCLES = 600
ICOLS = [
    "cycle",
    "regwrite",
    "rd",
    "wrdata",
    "rs1",
    "rs2",
    "alu_out",
    "reg_out",
    "reg_pc",
    "trap",
]
CI = {c: i for i, c in enumerate(ICOLS)}
INT_SIGNALS = [
    "regwrite",
    "rd",
    "wrdata",
    "rs1",
    "rs2",
    "alu_out",
    "reg_out",
    "reg_pc",
    "trap",
]


def run_internal_log(sim, firmware, site, sa, fc, irq_period, cache_dir):
    stem = Path(firmware).stem
    key = hashlib.md5(f"{stem}_{site}_{sa}_{fc}_ilog".encode()).hexdigest()
    cpath = cache_dir / f"{key}.csv"
    if not cpath.exists():
        cmd = [
            sim,
            "--firmware",
            firmware,
            "--cycles",
            str(SIM_CYCLES),
            "--fault-site",
            str(site),
            "--fault-sa",
            str(sa),
            "--fault-cycle",
            str(fc),
            "--internal-log",
            str(cpath),
        ]
        if irq_period > 0:
            cmd += ["--irq-period", str(irq_period)]
        subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    if not cpath.exists():
        return None
    arr = np.genfromtxt(str(cpath), delimiter=",", skip_header=1, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _window(arr, fc):
    if arr is None:
        return None
    cyc = arr[:, CI["cycle"]]
    m = (cyc >= fc) & (cyc < fc + WIN)
    return arr[m]


def counter_feats(w):
    if w is None or len(w) == 0:
        return np.zeros(5)
    return np.array(
        [
            w[:, CI["regwrite"]].sum(),
            len(np.unique(w[:, CI["wrdata"]])),
            len(np.unique(w[:, CI["reg_pc"]])),
            int((np.diff(w[:, CI["alu_out"]]) != 0).sum()),
            w[:, CI["trap"]].max(),
        ],
        dtype=float,
    )


def buffer_feats(w):
    if w is None or len(w) == 0:
        return np.zeros(len(INT_SIGNALS) * 4)
    cols = w[:, [CI[s] for s in INT_SIGNALS]].astype(float)
    return np.concatenate([cols.mean(0), cols.std(0), cols.max(0), (cols != 0).mean(0)])


def build_trace_buffer_features(meta, itc_root, cache=None):
    itc_root = Path(itc_root)
    sim = str(itc_root / "build" / "sim_fault_oracle")
    fw_dir = itc_root / "src" / "firmware"
    if cache is None:
        cache = REPO_ROOT / ".cache" / "internal"
    cache.mkdir(parents=True, exist_ok=True)
    features = np.zeros((len(meta), len(INT_SIGNALS) * 4))
    missing = 0
    for i, row in meta.iterrows():
        workload = row["workload"]
        arr = run_internal_log(
            sim,
            str(fw_dir / f"{workload}.bin"),
            int(row["fault_site"]),
            int(row["sa_value"]),
            int(row["fault_cycle"]),
            80 if workload == "irq_test" else 0,
            cache,
        )
        if arr is None:
            missing += 1
            continue
        features[i] = buffer_feats(_window(arr, int(row["fault_cycle"])))
    return features, missing


def power_feats(w):

    if w is None or len(w) < 2:
        return np.zeros(9)
    cols = w[:, [CI[s] for s in INT_SIGNALS]].astype(np.int64)

    xor = np.bitwise_xor(cols[:-1], cols[1:])

    def popcount(a):
        a = a.astype(np.uint64)
        s = np.zeros_like(a)
        for _ in range(64):
            s += a & 1
            a = a >> 1
        return s.astype(np.int64)

    hd = popcount(xor).sum(axis=1)
    if len(hd) == 0:
        return np.zeros(9)
    return np.concatenate(
        [
            [hd.mean(), hd.std(), hd.max(), hd.sum(), (hd > 0).mean()],
            np.percentile(hd, [10, 50, 90, 99]),
        ]
    )


def run(args):
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    sim = str(Path(args.itc_root) / "build" / "sim_fault_oracle")
    fw_dir = Path(args.itc_root) / "src" / "firmware"
    cache = REPO_ROOT / ".cache" / "internal"
    cache.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(REPO_ROOT / "data_picorv32" / "metadata.csv")
    Xbus = np.load(REPO_ROOT / "data_picorv32" / "features_agg.npy")
    y = np.load(REPO_ROOT / "data_picorv32" / "labels.npy")
    assert len(meta) == len(Xbus) == len(y)

    n = len(meta)
    Ctr = np.zeros((n, 5))
    Buf = np.zeros((n, len(INT_SIGNALS) * 4))
    Pow = np.zeros((n, 9))
    missing = 0
    for i, r in meta.iterrows():
        site = int(r["fault_site"])
        sa = int(r["sa_value"])
        fc = int(r["fault_cycle"])
        wl = r["workload"]
        irq = 80 if wl == "irq_test" else 0
        arr = run_internal_log(sim, str(fw_dir / f"{wl}.bin"), site, sa, fc, irq, cache)
        if arr is None:
            missing += 1
            continue
        w = _window(arr, fc)
        Ctr[i] = counter_feats(w)
        Buf[i] = buffer_feats(w)
        Pow[i] = power_feats(w)
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{n} (missing={missing})")

    feature_sets = {
        "bus": Xbus,
        "+counters": np.hstack([Xbus, Ctr]),
        "internal_toggle_features": np.hstack([Xbus, Ctr, Pow]),
        "internal_trace": np.hstack([Xbus, Ctr, Pow, Buf]),
    }

    tr = (meta["split"] == "train").values
    te = (meta["split"] == "test").values
    binm = (y == 0) | (y == 1)
    trm, tem = tr & binm, te & binm
    ytr = (y[trm] == 1).astype(int)
    yte = (y[tem] == 1).astype(int)

    out = {
        "task": "held-out-site binary clean-versus-observable classification",
        "window": WIN,
        "n_test": int(tem.sum()),
        "tiers": {},
    }
    aurocs = {}
    for tier, X in feature_sets.items():
        sc = StandardScaler().fit(X[trm])
        clf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
        clf.fit(sc.transform(X[trm]), ytr)
        p = clf.predict_proba(sc.transform(X[tem]))[:, 1]

        meta_te = meta[tem].reset_index(drop=True)
        groups_te = make_binary_groups(meta_te, yte)
        a = float(roc_auc_score(yte, p))
        ci_result = cluster_bootstrap_auroc_ci(yte, p, groups_te)
        ci = [round(ci_result["lo"], 4), round(ci_result["hi"], 4)]
        out["tiers"][tier] = {
            "auroc": round(a, 4),
            "auroc_ci95": ci,
            "auprc": round(float(average_precision_score(yte, p)), 4),
            "n_features": int(X.shape[1]),
        }
        aurocs[tier] = a
        print(
            f"  {tier:14} AUROC={a:.4f} CI={ci} AUPRC={out['tiers'][tier]['auprc']} "
            f"({X.shape[1]} feat)"
        )

    gain_c = aurocs["+counters"] - aurocs["bus"]
    gain_p = aurocs["internal_toggle_features"] - aurocs["bus"]
    gain_b = aurocs["internal_trace"] - aurocs["bus"]
    out["auroc_gain_counters"] = round(gain_c, 4)
    out["auroc_gain_internal_toggle_features"] = round(gain_p, 4)
    out["auroc_gain_trace_buffer"] = round(gain_b, 4)
    (RESULTS / "observation_channels.json").write_text(json.dumps(out, indent=2))
    print("  wrote results/observation_channels.json")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    tiers = list(feature_sets)
    vals = [out["tiers"][t]["auroc"] for t in tiers]
    err = [
        [vals[i] - out["tiers"][t]["auroc_ci95"][0] for i, t in enumerate(tiers)],
        [out["tiers"][t]["auroc_ci95"][1] - vals[i] for i, t in enumerate(tiers)],
    ]
    bars = ax.bar(
        tiers,
        vals,
        yerr=err,
        capsize=6,
        color=["#4878a8", "#e9c46a", "#c0392b", "#2a9d8f"],
    )
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            fontsize=10,
        )
    ax.axhline(0.5, ls=":", color="red", label="chance")
    ax.set(
        ylabel="Held-out-site detection AUROC",
        ylim=(0.5, 1.0),
        title="Detection by observation channel",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "observation_channels.png", dpi=130)
    print("  wrote reports/figures/observation_channels.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--itc-root", default=str(REPO_ROOT / "external" / "picorv32_fault_sim")
    )
    run(ap.parse_args())
