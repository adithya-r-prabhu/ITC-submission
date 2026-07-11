import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "reports" / "figures"


def core_metrics(name, uarch, data_dir, auroc, model, baseline_auroc, auroc_ci=None):
    meta = pd.read_csv(data_dir / "metadata.csv")
    fault = meta[meta["fault_site"] >= 0]
    n_fault = len(fault)
    lab = "label" if "label" in fault.columns else "observable"
    if lab == "label":
        n_obs = int((fault["label"] == 1).sum())
        n_sil = int((fault["label"] == 2).sum())
    else:
        n_obs = int((fault["observable"] > 0).sum())
        n_sil = n_fault - n_obs
    site_obs = fault.groupby("fault_site")["observable"].max()
    return {
        "core": name,
        "microarchitecture": uarch,
        "n_sites": int(fault["fault_site"].nunique()),
        "n_workloads": int(fault["workload"].nunique()),
        "n_faulty": n_fault,
        "observable_rate_sample": round(n_obs / n_fault, 4),
        "silent_rate_sample": round(n_sil / n_fault, 4),
        "site_observable_rate": round(float((site_obs > 0).mean()), 4),
        "site_silent_rate": round(float((site_obs == 0).mean()), 4),
        "heldout_site_auroc": round(auroc, 4),
        "model": model,
        "ml_auroc_ci95": auroc_ci,
        "pre_fault_feature_auroc": round(baseline_auroc, 4),
    }


def run(args):
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    base = json.load(open(RESULTS / "baselines_binary.json"))["heldout_sites"]
    pbest = max(base, key=lambda m: base[m]["auroc"])
    pre_fault = json.load(open(RESULTS / "pre_fault_feature_auroc.json"))
    pico_baseline = float(
        np.mean([r["max_pre_fault_feature_auroc"] for r in pre_fault])
    )
    pico = core_metrics(
        "PicoRV32",
        "multi-cycle FSM",
        REPO_ROOT / "data_picorv32",
        base[pbest]["auroc"],
        pbest,
        pico_baseline,
    )

    ib = json.load(open(RESULTS / "ibex_summary.json"))
    ibex = core_metrics(
        "Ibex",
        "2-stage pipeline",
        REPO_ROOT / "data_ibex",
        ib["heldout_sites_summary"]["auroc"],
        ib["heldout_sites_summary"]["model"],
        ib["pre_fault_features"]["max_single_feature_auroc"],
    )

    cv = json.load(open(RESULTS / "cv32e40p_summary.json"))
    ea = cv["heldout_sites_binary"]
    cbest = max(ea, key=lambda m: ea[m]["auroc"])
    cv32 = core_metrics(
        "CV32E40P",
        "4-stage pipeline",
        REPO_ROOT / "data_cv32e40p",
        ea[cbest]["auroc"],
        cbest,
        cv["pre_fault_features"]["max_single_feature_auroc"],
        auroc_ci=ea[cbest].get("auroc_ci95"),
    )

    cores = [pico, ibex, cv32]
    out = {
        "description": (
            "Three-core portability comparison. Same methodology, "
            "feature pipeline, splits, and models on three "
            "structurally distinct RISC-V cores."
        ),
        "cores": cores,
        "site_sets_are_core_specific": True,
    }
    (RESULTS / "core_comparison.json").write_text(json.dumps(out, indent=2))
    print("wrote results/core_comparison.json")
    hdr = f"{'metric':<26}{'PicoRV32':>11}{'Ibex':>11}{'CV32E40P':>11}"
    print("\n" + hdr)
    for k, lab in [
        ("microarchitecture", "uarch"),
        ("n_sites", "sites"),
        ("n_workloads", "workloads"),
        ("site_observable_rate", "site-observable"),
        ("site_silent_rate", "site-silent"),
        ("heldout_site_auroc", "held-out-site AUROC"),
        ("pre_fault_feature_auroc", "pre-fault feature AUROC"),
    ]:
        print(
            f"{lab:<26}{str(pico.get(k)):>11}{str(ibex.get(k)):>11}"
            f"{str(cv32.get(k)):>11}"
        )

    metrics = [
        ("site_observable_rate", "Site\nobservable"),
        ("site_silent_rate", "Site\nsilent"),
        ("heldout_site_auroc", "Detection\nAUROC"),
        ("pre_fault_feature_auroc", "Pre-fault feature\nAUROC"),
    ]
    x = np.arange(len(metrics))
    w = 0.26
    colors = ["#4878a8", "#e1812c", "#6acc64"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, c in enumerate(cores):
        vals = [c[m[0]] for m in metrics]
        bars = ax.bar(
            x + (i - 1) * w,
            vals,
            w,
            label=f"{c['core']} ({c['microarchitecture']}, "
            f"{c['n_sites']}s/{c['n_workloads']}wl)",
            color=colors[i],
        )
        for bb in bars:
            ax.text(
                bb.get_x() + bb.get_width() / 2,
                bb.get_height() + 0.01,
                f"{bb.get_height():.2f}",
                ha="center",
                fontsize=7,
            )
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("value")
    ax.set_title("Three-core portability: PicoRV32 vs Ibex vs CV32E40P")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGS / "core_comparison.png", dpi=130)
    print("wrote reports/figures/core_comparison.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    run(ap.parse_args())
