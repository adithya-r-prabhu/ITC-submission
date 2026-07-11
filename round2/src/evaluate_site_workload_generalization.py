import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from train_detectors import (
    LABEL_CLEAN,
    LABEL_OBS,
    filter_binary,
    load_data,
    make_models,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


FPR_TARGETS = [0.01, 0.05, 0.10]


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonify(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def binary_metrics(y_true, y_score, y_pred):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_test": int(len(y_true)),
        "n_clean": int((y_true == LABEL_CLEAN).sum()),
        "n_observable": int((y_true == LABEL_OBS).sum()),
    }
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def thresholds_from_validation(y_val, score_val, targets=FPR_TARGETS):

    clean_scores = np.sort(score_val[y_val == LABEL_CLEAN])
    if clean_scores.size == 0:
        return {str(t): 1.0 for t in targets}
    thresholds = {}
    for target in targets:
        q = max(0.0, min(1.0, 1.0 - target))
        thresholds[str(target)] = float(np.quantile(clean_scores, q))
    return thresholds


def recall_at_fpr(y_true, score, thresholds):
    out = {}
    for target, thr in thresholds.items():
        pred = (score >= thr).astype(int)
        clean = y_true == LABEL_CLEAN
        obs = y_true == LABEL_OBS
        fp = int(((pred == 1) & clean).sum())
        tn = int(((pred == 0) & clean).sum())
        tp = int(((pred == 1) & obs).sum())
        fn = int(((pred == 0) & obs).sum())
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        out[target] = {
            "threshold": float(thr),
            "test_fpr": float(fpr),
            "observable_recall": float(rec),
            "precision": float(prec),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
    return out


def run_fold(X, y, meta, holdout):
    tr_mask = ((meta["split"] == "train") & (meta["workload"] != holdout)).values
    va_mask = ((meta["split"] == "val") & (meta["workload"] != holdout)).values
    te_mask = ((meta["split"] == "test") & (meta["workload"] == holdout)).values

    Xtr, ytr, _ = filter_binary(X, y, meta, tr_mask)
    Xva, yva, _ = filter_binary(X, y, meta, va_mask)
    Xte, yte, _ = filter_binary(X, y, meta, te_mask)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return {}

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xva_s = scaler.transform(Xva) if len(Xva) else np.empty((0, Xtr.shape[1]))
    Xte_s = scaler.transform(Xte)

    fold = {}
    for name, clf in make_models("binary"):
        clf.fit(Xtr_s, ytr)
        score_te = clf.predict_proba(Xte_s)[:, 1]
        pred_te = clf.predict(Xte_s)
        m = binary_metrics(yte, score_te, pred_te)
        if len(Xva_s) and len(np.unique(yva)) >= 2:
            score_va = clf.predict_proba(Xva_s)[:, 1]
            th = thresholds_from_validation(yva, score_va)
            m["threshold_metrics"] = recall_at_fpr(yte, score_te, th)
        else:
            m["threshold_metrics"] = {}
        fold[name] = m
    return fold


def summarize(per_fold):
    model_names = sorted({m for fold in per_fold.values() for m in fold})
    out = {}
    for name in model_names:
        rows = [per_fold[wl][name] for wl in per_fold if name in per_fold[wl]]
        out[name] = {}
        for metric in ["auroc", "auprc", "f1_macro", "accuracy"]:
            vals = [r[metric] for r in rows if not np.isnan(r[metric])]
            out[name][f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[name][f"{metric}_std"] = float(np.std(vals)) if vals else float("nan")
            out[name][f"{metric}_min"] = float(np.min(vals)) if vals else float("nan")
            out[name][f"{metric}_max"] = float(np.max(vals)) if vals else float("nan")
        out[name]["per_fold_auroc"] = {
            wl: per_fold[wl].get(name, {}).get("auroc", float("nan")) for wl in per_fold
        }
        out[name]["per_fold_auprc"] = {
            wl: per_fold[wl].get(name, {}).get("auprc", float("nan")) for wl in per_fold
        }
    return out


def plot(summary, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summary.keys())
    if not models:
        return
    x = np.arange(len(models))
    auroc = [summary[m]["auroc_mean"] for m in models]
    auprc = [summary[m]["auprc_mean"] for m in models]
    auroc_err = [summary[m]["auroc_std"] for m in models]
    auprc_err = [summary[m]["auprc_std"] for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.36
    ax.bar(x - w / 2, auroc, w, yerr=auroc_err, capsize=3, label="AUROC")
    ax.bar(x + w / 2, auprc, w, yerr=auprc_err, capsize=3, label="AUPRC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Mean over held-out workloads")
    ax.set_ylim(0, 1.05)
    ax.set_title("Strict generalization: held-out sites + held-out workload")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data_picorv32"))
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"))
    ap.add_argument("--fig-dir", default=str(REPO_ROOT / "reports" / "figures"))
    ap.add_argument("--output-name", default="heldout_site_and_workload")
    args = ap.parse_args()

    _, X, y, meta = load_data(Path(args.data_dir))
    workloads = sorted(meta.loc[meta["fault_site"] >= 0, "workload"].unique())
    per_fold = {}
    for wl in workloads:
        print(f"  strict fold: holdout workload={wl}")
        per_fold[wl] = run_fold(X, y, meta, wl)

    summary = summarize(per_fold)
    out = {
        "data_dir": Path(args.data_dir).name,
        "definition": "train split + non-heldout workloads; test split + heldout workload",
        "task": "binary clean vs observable; silent faults excluded consistently with baseline binary task",
        "fpr_targets": FPR_TARGETS,
        "per_fold": per_fold,
        "summary": summary,
    }

    results_path = Path(args.results_dir) / f"{args.output_name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(_jsonify(out), f, indent=2)
    print(f"  wrote {results_path}")

    fig_path = Path(args.fig_dir) / f"{args.output_name}.png"
    plot(summary, fig_path)
    print(f"  wrote {fig_path}")


if __name__ == "__main__":
    main()
