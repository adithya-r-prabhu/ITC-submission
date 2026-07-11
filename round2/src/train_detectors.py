import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

LABEL_CLEAN = 0
LABEL_OBS = 1
LABEL_SILENT = 2
LABEL_NAMES = {0: "clean", 1: "observable", 2: "silent"}
WORKLOADS = ["counting_loop", "alu_heavy", "branch_heavy", "mem_intensive", "irq_test"]


_OBS_SIGNAL_NAMES = [
    "if_active",
    "mem_active",
    "mem_read",
    "mem_write",
    "pipeline_stall",
    "mem_stall",
    "branch_taken",
    "opcode_alu",
    "opcode_branch",
    "opcode_load",
    "opcode_store",
    "illegal_instr",
    "pmp_region",
]
_STAT_NAMES = ["mean", "std", "max", "nonzero_frac"]
_BLOCK_NAMES = ["window", "baseline", "delta"]

AGG_FEATURE_NAMES = [
    f"{sig}_{stat}_{blk}"
    for blk in _BLOCK_NAMES
    for stat in _STAT_NAMES
    for sig in _OBS_SIGNAL_NAMES
]


def load_data(data_dir):
    X_seq = np.load(str(data_dir / "features_seq.npy"))
    X_agg = np.load(str(data_dir / "features_agg.npy"))
    y = np.load(str(data_dir / "labels.npy"))
    meta = pd.read_csv(str(data_dir / "metadata.csv"))
    return X_seq, X_agg, y, meta


def get_split_mask(meta, split):
    return (meta["split"] == split).values


def get_workload_mask(meta, workload):
    return (meta["workload"] == workload).values


def filter_binary(X, y, meta, mask):

    valid = mask & ((y == LABEL_CLEAN) | (y == LABEL_OBS))
    return X[valid], y[valid], meta[valid]


def compute_metrics(
    y_true, y_pred, y_prob, task
):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        if task == "binary":
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            metrics["auroc"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except Exception:
        metrics["auroc"] = float("nan")
    return metrics


def make_models(task):
    models = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
            ),
        ),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=1000, class_weight="balanced", C=1.0, random_state=42
            ),
        ),
        (
            "mlp",
            MLPClassifier(
                hidden_layer_sizes=(256, 128),
                max_iter=300,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            ),
        ),
    ]
    if HAS_XGB:
        models.insert(
            1,
            (
                "xgboost",
                xgb.XGBClassifier(
                    n_estimators=200,
                    use_label_encoder=False,
                    eval_metric="mlogloss" if task == "multiclass" else "logloss",
                    random_state=42,
                ),
            ),
        )
    return models


def workload_auroc(
    model, scaler, X_agg, y, meta, task
):
    result = {}
    for wl in WORKLOADS:
        mask = get_workload_mask(meta.reset_index(drop=True), wl)
        if task == "binary":
            mask = mask & ((y == LABEL_CLEAN) | (y == LABEL_OBS))
        xi = X_agg[mask]
        yi = y[mask]
        if len(np.unique(yi)) < 2 or len(yi) == 0:
            result[wl] = float("nan")
            continue
        xi_sc = scaler.transform(xi)
        prob = model.predict_proba(xi_sc)
        try:
            if task == "binary":
                result[wl] = float(roc_auc_score(yi, prob[:, 1]))
            else:
                result[wl] = float(
                    roc_auc_score(yi, prob, multi_class="ovr", average="macro")
                )
        except Exception:
            result[wl] = float("nan")
    return result


def category_auroc(
    model, scaler, X_agg, y, meta, task
):
    result = {}
    for cat in meta["category"].unique():
        mask = (meta.reset_index(drop=True)["category"] == cat).values
        if task == "binary":
            mask = mask & ((y == LABEL_CLEAN) | (y == LABEL_OBS))
        xi = X_agg[mask]
        yi = y[mask]
        if len(np.unique(yi)) < 2 or len(yi) == 0:
            result[cat] = float("nan")
            continue
        xi_sc = scaler.transform(xi)
        prob = model.predict_proba(xi_sc)
        try:
            if task == "binary":
                result[cat] = float(roc_auc_score(yi, prob[:, 1]))
            else:
                result[cat] = float(
                    roc_auc_score(yi, prob, multi_class="ovr", average="macro")
                )
        except Exception:
            result[cat] = float("nan")
    return result


def evaluate_heldout_sites(
    X_agg, y, meta, models_dir, task
):
    print(f"\n  Experiment A — fault-site split ({task})")
    train_mask = get_split_mask(meta, "train")
    test_mask = get_split_mask(meta, "test")

    if task == "binary":
        Xtr, ytr, mtr = filter_binary(X_agg, y, meta, train_mask)
        Xte, yte, mte = filter_binary(X_agg, y, meta, test_mask)
    else:
        Xtr, ytr, _ = X_agg[train_mask], y[train_mask], meta[train_mask]
        Xte, yte, mte = X_agg[test_mask], y[test_mask], meta[test_mask]

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)

    results = {}
    for name, clf in make_models(task):
        print(f"    [{name}] fitting on {len(ytr)} samples ...", end=" ", flush=True)
        clf.fit(Xtr_sc, ytr)
        pred = clf.predict(Xte_sc)
        prob = clf.predict_proba(Xte_sc)
        m = compute_metrics(yte, pred, prob, task)
        wl_au = workload_auroc(clf, scaler, Xte, yte, mte.reset_index(drop=True), task)
        cat_au = category_auroc(clf, scaler, Xte, yte, mte.reset_index(drop=True), task)
        results[name] = {
            **m,
            "auroc_by_workload": wl_au,
            "auroc_by_category": cat_au,
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        }
        print(f"AUROC={m['auroc']:.4f}  F1={m['f1_macro']:.4f}")

        ckpt = models_dir / f"{name}_{task}.pkl"
        with open(ckpt, "wb") as f:
            pickle.dump({"model": clf, "scaler": scaler}, f)

        if name == "random_forest":
            importances = clf.feature_importances_
            results[name]["feature_importances"] = importances.tolist()

    return results


def evaluate_heldout_workload(
    X_agg, y, meta, holdout, task
):
    print(f"\n  Experiment B — held-out workload: {holdout} ({task})")
    train_mask = (meta["workload"] != holdout).values
    test_mask = (meta["workload"] == holdout).values

    if task == "binary":
        Xtr, ytr, _ = filter_binary(X_agg, y, meta, train_mask)
        Xte, yte, _ = filter_binary(X_agg, y, meta, test_mask)
    else:
        Xtr, ytr = X_agg[train_mask], y[train_mask]
        Xte, yte = X_agg[test_mask], y[test_mask]

    if len(np.unique(yte)) < 2:
        print(f"    SKIP: only one class in held-out workload {holdout}")
        return {}

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(Xtr)
    Xte_sc = scaler.transform(Xte)

    results = {}
    for name, clf in make_models(task):
        clf.fit(Xtr_sc, ytr)
        pred = clf.predict(Xte_sc)
        prob = clf.predict_proba(Xte_sc)
        m = compute_metrics(yte, pred, prob, task)
        results[name] = m
        print(f"    [{name}] AUROC={m['auroc']:.4f}  F1={m['f1_macro']:.4f}")
    return results


def evaluate_leave_one_workload_out(
    X_agg, y, meta, task
):
    print(f"\n  Experiment C — leave-one-workload-out CV ({task})")
    fold_results = {}
    for holdout in WORKLOADS:
        r = evaluate_heldout_workload(X_agg, y, meta, holdout, task)
        fold_results[holdout] = r

    summary = {}
    model_names = list(next(v for v in fold_results.values() if v).keys())
    for name in model_names:
        aurocs = [
            fold_results[wl][name]["auroc"]
            for wl in WORKLOADS
            if wl in fold_results
            and name in fold_results[wl]
            and not np.isnan(fold_results[wl][name]["auroc"])
        ]
        summary[name] = {
            "auroc_mean": float(np.mean(aurocs)) if aurocs else float("nan"),
            "auroc_std": float(np.std(aurocs)) if aurocs else float("nan"),
            "per_fold": {
                wl: fold_results[wl].get(name, {}).get("auroc", float("nan"))
                for wl in WORKLOADS
            },
        }
        print(
            f"    [{name}] AUROC={summary[name]['auroc_mean']:.4f} ± {summary[name]['auroc_std']:.4f}"
        )
    return {"per_fold": fold_results, "summary": summary}


def generate_figures(
    results_binary,
    results_multi,
    X_agg,
    y,
    meta,
    reports_dir,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("  WARNING: matplotlib not available; skipping figures.")
        return

    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results_binary.keys()) if results_binary else []

    if results_binary or results_multi:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(model_names))
        w = 0.35
        bin_aurocs = [results_binary.get(n, {}).get("auroc", 0) for n in model_names]
        multi_aurocs = [results_multi.get(n, {}).get("auroc", 0) for n in model_names]
        ax.bar(
            x - w / 2, bin_aurocs, w, label="Binary (clean vs obs)", color="steelblue"
        )
        ax.bar(x + w / 2, multi_aurocs, w, label="3-class", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0, 1.1)
        ax.axhline(
            0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline"
        )
        ax.set_title("AUROC by model — real fault training")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(fig_dir / "auroc_by_model.png"), dpi=150)
        plt.close()

    if results_binary:
        n = len(model_names)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, name in zip(axes, model_names):
            cm = np.array(results_binary[name].get("confusion_matrix", [[0]]))
            _ = ax.imshow(cm, cmap="Blues")
            ax.set_title(name)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            tick_labels = ["clean", "obs"]
            ax.set_xticks(range(len(cm)))
            ax.set_xticklabels(tick_labels[: len(cm)])
            ax.set_yticks(range(len(cm)))
            ax.set_yticklabels(tick_labels[: len(cm)])
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
        plt.tight_layout()
        plt.savefig(str(fig_dir / "confusion_matrices.png"), dpi=150)
        plt.close()

    rf_result = results_binary.get("random_forest", {}) or results_multi.get(
        "random_forest", {}
    )
    if "feature_importances" in rf_result:
        imp = np.array(rf_result["feature_importances"])
        top_n = 30
        idx = np.argsort(imp)[-top_n:][::-1]
        names = [
            AGG_FEATURE_NAMES[i] if i < len(AGG_FEATURE_NAMES) else f"feat_{i}"
            for i in idx
        ]
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(range(top_n), imp[idx][::-1], color="steelblue")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"RF feature importance (top {top_n})")
        plt.tight_layout()
        plt.savefig(str(fig_dir / "feature_importance.png"), dpi=150)
        plt.close()

    test_mask = (meta["split"] == "test").values
    Xte = X_agg[test_mask]
    yte = y[test_mask]
    if len(Xte) >= 10:
        n_tsne = min(1500, len(Xte))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xte), n_tsne, replace=False)
        try:
            emb = TSNE(
                n_components=2, random_state=42, perplexity=min(30, n_tsne - 1)
            ).fit_transform(Xte[idx])
            colors = ["#2ca02c", "#d62728", "#ff7f0e"]
            label_strs = ["clean", "observable", "silent"]
            fig, ax = plt.subplots(figsize=(8, 6))
            for lbl in [LABEL_CLEAN, LABEL_OBS, LABEL_SILENT]:
                m = yte[idx] == lbl
                if m.sum() > 0:
                    ax.scatter(
                        emb[m, 0],
                        emb[m, 1],
                        c=colors[lbl],
                        label=label_strs[lbl],
                        alpha=0.6,
                        s=15,
                    )
            ax.set_title("t-SNE of aggregate features (test split)")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(str(fig_dir / "tsne_visualization.png"), dpi=150)
            plt.close()
        except Exception as e:
            print(f"  WARNING: t-SNE failed: {e}")

    if results_binary:
        wl_data = {
            n: results_binary[n].get("auroc_by_workload", {}) for n in model_names
        }
        _plot_auroc_breakdown(
            wl_data,
            WORKLOADS,
            "AUROC by workload (binary)",
            str(fig_dir / "auroc_by_workload.png"),
            plt,
        )

    if results_binary:
        categories = list(
            next(
                v.get("auroc_by_category", {}).keys()
                for v in results_binary.values()
                if v.get("auroc_by_category")
            )
            if any(v.get("auroc_by_category") for v in results_binary.values())
            else iter([])
        )
        if categories:
            cat_data = {
                n: results_binary[n].get("auroc_by_category", {}) for n in model_names
            }
            _plot_auroc_breakdown(
                cat_data,
                categories,
                "AUROC by fault category (binary)",
                str(fig_dir / "auroc_by_category.png"),
                plt,
            )

    print(f"  Figures saved to {fig_dir}/")


def _plot_auroc_breakdown(data, keys, title, path, plt):
    model_names = list(data.keys())
    n_models = len(model_names)
    n_keys = len(keys)
    x = np.arange(n_keys)
    w = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(max(10, n_keys * 1.5), 5))
    for i, name in enumerate(model_names):
        vals = [data[name].get(k, float("nan")) for k in keys]
        ax.bar(x + i * w - (n_models - 1) * w / 2, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(args):
    data_dir = Path(args.data_dir).resolve()
    models_dir = REPO_ROOT / "models"
    results_dir = REPO_ROOT / "results"
    reports_dir = REPO_ROOT / "reports"
    for d in [models_dir, results_dir, reports_dir]:
        d.mkdir(exist_ok=True)

    print(f"\n{'=' * 64}")
    print("  Detector training")
    print(f"{'=' * 64}")

    X_seq, X_agg, y, meta = load_data(data_dir)
    print(f"\n  Loaded: {len(y)} samples  features_agg: {X_agg.shape}")

    label_counts = {LABEL_NAMES[k]: int((y == k).sum()) for k in [0, 1, 2]}
    print(f"  Labels: {label_counts}")

    tasks = ["binary", "multiclass"] if args.task == "all" else [args.task]
    results_by_task = {}
    heldout_workload_by_task = {}
    leave_one_workload_out_by_task = {}

    for task in tasks:
        print(f"\n{'─' * 64}")
        print(f"  Task: {task.upper()}")
        print(f"{'─' * 64}")

        res_a = evaluate_heldout_sites(X_agg, y, meta, models_dir, task)
        results_by_task[task] = res_a

        res_b = evaluate_heldout_workload(X_agg, y, meta, args.holdout_workload, task)
        heldout_workload_by_task[task] = {
            "holdout_workload": args.holdout_workload,
            "results": res_b,
        }

        res_c = evaluate_leave_one_workload_out(X_agg, y, meta, task)
        leave_one_workload_out_by_task[task] = res_c

    def _serialise(d):

        if isinstance(d, dict):
            return {k: _serialise(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_serialise(v) for v in d]
        if isinstance(d, np.integer):
            return int(d)
        if isinstance(d, np.floating):
            return float(d)
        if isinstance(d, float) and np.isnan(d):
            return None
        return d

    for task in tasks:
        fname = f"baselines_{task}.json"
        out = {
            "task": task,
            "heldout_sites": _serialise(results_by_task[task]),
        }
        with open(results_dir / fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Saved {fname}")

    with open(results_dir / "heldout_workload.json", "w") as f:
        json.dump(_serialise(heldout_workload_by_task), f, indent=2)
    with open(results_dir / "leave_one_workload_out.json", "w") as f:
        json.dump(_serialise(leave_one_workload_out_by_task), f, indent=2)

    print("\n  Generating figures ...")
    generate_figures(
        results_by_task.get("binary", {}),
        results_by_task.get("multiclass", {}),
        X_agg,
        y,
        meta,
        reports_dir,
    )

    print("\n  Done.")
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description="Train real-fault baseline classifiers")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "data_picorv32"))
    p.add_argument("--task", default="all", choices=["binary", "multiclass", "all"])
    p.add_argument(
        "--holdout-workload",
        default="irq_test",
        help="Workload to hold out",
    )
    return p.parse_args()


_OBS_SIGNAL_NAMES_IMPORT = [
    "if_active",
    "mem_active",
    "mem_read",
    "mem_write",
    "pipeline_stall",
    "mem_stall",
    "branch_taken",
    "opcode_alu",
    "opcode_branch",
    "opcode_load",
    "opcode_store",
    "illegal_instr",
    "pmp_region",
]
AGG_FEATURE_NAMES = [
    f"{sig}_{stat}_{blk}"
    for blk in ["window", "baseline", "delta"]
    for stat in ["mean", "std", "max", "nonzero_frac"]
    for sig in _OBS_SIGNAL_NAMES_IMPORT
]

if __name__ == "__main__":
    sys.exit(main(_parse_args()))
