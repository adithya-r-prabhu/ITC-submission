import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

from cluster_bootstrap import cluster_bootstrap
from compare_observation_channels import build_trace_buffer_features

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parent
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "reports" / "figures"
SEED = 42
FPR_BUDGETS = [0.01, 0.05, 0.10]


def recall_at_fpr(y, s, budget):

    order = np.argsort(-s)
    ys = y[order]
    P = int((y == 1).sum())
    N = int((y == 0).sum())
    tp = fp = 0
    best_recall, best_fpr = 0.0, 0.0
    for lab in ys:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / N
        if fpr <= budget:
            best_recall = tp / P
            best_fpr = fpr
        else:
            break
    return best_recall, best_fpr, N, P


def boot(y, s, groups, budget, n=2000, seed=SEED):

    def stat(idx):
        if len(np.unique(y[idx])) < 2:
            return None
        return recall_at_fpr(y[idx], s[idx], budget)[0]

    vals = cluster_bootstrap(stat, groups, n_boot=n, seed=seed)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def oof_scores(X, y, groups):

    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = np.zeros(len(y))
    for tr, te in sgk.split(X, y, groups):
        base = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=SEED
            ),
        )
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(X[tr], y[tr])
        scores[te] = cal.predict_proba(X[te])[:, 1]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--itc-root", default=str(REPO_ROOT / "external" / "picorv32_fault_sim")
    )
    args = ap.parse_args()
    _ = str(Path(args.itc_root) / "build" / "sim_fault_oracle")
    _ = Path(args.itc_root) / "src" / "firmware"
    cache = REPO_ROOT / ".cache" / "internal"
    cache.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(REPO_ROOT / "data_picorv32" / "metadata.csv")
    Xbus = np.load(REPO_ROOT / "data_picorv32" / "features_agg.npy")
    y = np.load(REPO_ROOT / "data_picorv32" / "labels.npy")

    print("  building enriched (internal trace-buffer) features from cache ...")
    Buf, missing = build_trace_buffer_features(meta, args.itc_root, cache)
    print(f"  built ({missing} missing)")

    binm = (y == 0) | (y == 1)
    yb = (y[binm] == 1).astype(int)
    Xb_bus = Xbus[binm]
    Xb_enr = np.hstack([Xbus, Buf])[binm]
    mb = meta[binm].reset_index(drop=True)
    groups = np.array(
        [
            str(int(fs)) if lab == 1 else f"c_{wl}_{fc}"
            for fs, lab, wl, fc in zip(
                mb["fault_site"], yb, mb["workload"], mb["fault_cycle"]
            )
        ]
    )
    print(
        f"  binary samples={len(yb)}  negatives(clean)={int((yb == 0).sum())}  "
        f"positives(obs)={int((yb == 1).sum())}  groups={len(set(groups))}"
    )

    channels = {}
    for name, X in [("bus", Xb_bus), ("bus_plus_internal_trace", Xb_enr)]:
        s = oof_scores(X, yb, groups)
        auroc = float(roc_auc_score(yb, s))
        rec = {}
        for bgt in FPR_BUDGETS:
            r, afpr, N, P = recall_at_fpr(yb, s, bgt)
            lo, hi = boot(yb, s, groups, bgt)
            rec[f"{bgt:.2f}"] = {
                "target_fpr": bgt,
                "recall": round(r, 4),
                "achieved_fpr": round(afpr, 4),
                "recall_ci95": [round(lo, 4), round(hi, 4)],
                "estimable": afpr >= bgt * 0.5,
            }
        channels[name] = {
            "oof_auroc": round(auroc, 4),
            "n_neg": int((yb == 0).sum()),
            "recall_at_fpr": rec,
        }
        print(
            f"  [{name:8}] OOF AUROC={auroc:.4f}  "
            + "  ".join(
                f"r@{int(b * 100)}%={rec[f'{b:.2f}']['recall']:.3f}"
                f"(fpr {rec[f'{b:.2f}']['achieved_fpr']:.3f})"
                for b in FPR_BUDGETS
            )
        )

    gains = {
        f"{b:.2f}": round(
            channels["bus_plus_internal_trace"]["recall_at_fpr"][f"{b:.2f}"]["recall"]
            - channels["bus"]["recall_at_fpr"][f"{b:.2f}"]["recall"],
            4,
        )
        for b in FPR_BUDGETS
    }
    out = {
        "method": (
            "group-disjoint 5-fold CV OOF scores over all binary samples "
            "(250 clean negatives); isotonic-"
            "calibrated RF; recall@FPR with achieved FPR + 2000x CLUSTER "
            "bootstrap CI (resampling fault-site / clean-position groups)."
        ),
        "channels": channels,
        "internal_trace_minus_bus_recall_gain": gains,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "deployment_evaluation.json").write_text(json.dumps(out, indent=2))
    print(f"  internal-trace-vs-bus recall gain: {gains}")
    print("  wrote results/deployment_evaluation.json")

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = FPR_BUDGETS
    for name, col in [
        ("bus", "#4878a8"),
        ("bus_plus_internal_trace", "#2a9d8f"),
    ]:
        r = channels[name]["recall_at_fpr"]
        ys = [r[f"{b:.2f}"]["recall"] for b in xs]
        lo = [r[f"{b:.2f}"]["recall_ci95"][0] for b in xs]
        hi = [r[f"{b:.2f}"]["recall_ci95"][1] for b in xs]
        ax.plot([b * 100 for b in xs], ys, "-o", color=col, label=name)
        ax.fill_between([b * 100 for b in xs], lo, hi, color=col, alpha=0.15)
    ax.set(
        xlabel="false-positive budget (%)",
        ylabel="recall (TPR)",
        title="In-field operating point: enriched observation vs bus\n"
        "(group-disjoint CV, 250 clean negatives, 95% bootstrap CI)",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "deployment_evaluation.png", dpi=130)
    print("  wrote reports/figures/deployment_evaluation.png")


if __name__ == "__main__":
    main()
