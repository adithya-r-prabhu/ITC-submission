"""Generate all publication-quality figures for ITC2 research pipeline."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# -- paths ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -- palette / style ------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.1)
MODEL_COLORS = {
    "random_forest": "#4C72B0",
    "xgboost":       "#DD8452",
    "logistic_regression": "#55A868",
    "mlp":           "#C44E52",
    "lstm":          "#8172B2",
}
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Reg.",
    "mlp": "MLP",
    "lstm": "BiLSTM",
}
WORKLOAD_LABELS = {
    "counting_loop": "Counting Loop",
    "alu_heavy": "ALU Heavy",
    "branch_heavy": "Branch Heavy",
    "mem_intensive": "Mem Intensive",
    "irq_test": "IRQ Test",
}
CLASS_LABELS = ["Clean", "Observable Fault", "Silent Fault"]
CLASS_COLORS = ["#2ca02c", "#d62728", "#ff7f0e"]


def save(fig, name, dpi=150):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# -- load results ---------------------------------------------------------------
with open(RESULTS / "baselines_binary.json") as f:
    bin_res = json.load(f)
with open(RESULTS / "baselines_multiclass.json") as f:
    mc_res = json.load(f)
with open(RESULTS / "generalization_exp_b.json") as f:
    exp_b = json.load(f)
with open(RESULTS / "generalization_exp_c.json") as f:
    exp_c = json.load(f)
with open(RESULTS / "lstm_binary.json") as f:
    lstm_bin = json.load(f)
with open(RESULTS / "lstm_3class.json") as f:
    lstm_3c = json.load(f)
with open(RESULTS / "lstm_generalization.json") as f:
    lstm_gen = json.load(f)

labels = np.load(DATA / "labels.npy")
features_agg = np.load(DATA / "features_agg.npy")
meta = pd.read_csv(DATA / "metadata.csv")

MODELS = ["random_forest", "xgboost", "logistic_regression", "mlp"]


# ==============================================================================
# 1. DATASET OVERVIEW
# ==============================================================================
def fig_dataset_overview():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold", y=1.02)

    # 1a: class distribution
    ax = axes[0]
    counts = [np.sum(labels == i) for i in range(3)]
    bars = ax.bar(CLASS_LABELS, counts, color=CLASS_COLORS, edgecolor="white", linewidth=0.8)
    ax.set_title("Class Distribution (N=3,150)")
    ax.set_ylabel("Sample Count")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{cnt}\n({cnt/sum(counts)*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts) * 1.25)

    # 1b: per-workload observable rate
    ax = axes[1]
    wl_order = list(WORKLOAD_LABELS.keys())
    fault_meta = meta[meta["label"].isin([1, 2])]
    obs_rates = []
    total_counts = []
    for wl in wl_order:
        wl_df = fault_meta[fault_meta["workload"] == wl]
        obs_rates.append(wl_df["observable"].mean() * 100)
        total_counts.append(len(wl_df))
    bars = ax.bar([WORKLOAD_LABELS[w] for w in wl_order], obs_rates,
                  color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax.set_title("Observable Fault Rate by Workload")
    ax.set_ylabel("Observable %")
    ax.set_ylim(0, 55)
    ax.tick_params(axis="x", rotation=30)
    for bar, rate, cnt in zip(bars, obs_rates, total_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{rate:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=8)

    # 1c: split breakdown
    ax = axes[2]
    split_data = {}
    for split in ["train", "val", "test"]:
        split_df = meta[meta["split"] == split]
        split_data[split] = {
            "Clean": int((split_df["label"] == 0).sum()),
            "Observable": int((split_df["label"] == 1).sum()),
            "Silent": int((split_df["label"] == 2).sum()),
        }
    splits = list(split_data.keys())
    x = np.arange(len(splits))
    width = 0.25
    for i, (cls, color) in enumerate(zip(["Clean", "Observable", "Silent"], CLASS_COLORS)):
        vals = [split_data[s][cls] for s in splits]
        bars = ax.bar(x + i * width, vals, width, label=cls, color=color, edgecolor="white")
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Train", "Val", "Test"])
    ax.set_title("Split Label Distribution")
    ax.set_ylabel("Sample Count")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, "dataset_overview")


# ==============================================================================
# 2. CONFUSION MATRICES - BINARY (4 models)
# ==============================================================================
def fig_confusion_binary():
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Confusion Matrices - Binary Classification (Exp A, Test Set)",
                 fontsize=13, fontweight="bold")
    cls_names = ["Clean", "Fault"]
    for ax, model in zip(axes, MODELS):
        cm = np.array(bin_res["exp_a"][model]["confusion_matrix"])
        acc = bin_res["exp_a"][model]["accuracy"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=cls_names, yticklabels=cls_names,
                    cbar=False, linewidths=0.5)
        ax.set_title(f"{MODEL_LABELS[model]}\nAcc={acc:.3f}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.tight_layout()
    save(fig, "confusion_matrices_binary")


# ==============================================================================
# 3. CONFUSION MATRICES - MULTICLASS (4 models)
# ==============================================================================
def fig_confusion_multiclass():
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Confusion Matrices - 3-Class Classification (Exp A, Test Set)",
                 fontsize=13, fontweight="bold")
    cls_names = ["Clean", "Observable", "Silent"]
    for ax, model in zip(axes, MODELS):
        cm = np.array(mc_res["exp_a"][model]["confusion_matrix"])
        acc = mc_res["exp_a"][model]["accuracy"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=cls_names, yticklabels=cls_names,
                    cbar=False, linewidths=0.5)
        ax.set_title(f"{MODEL_LABELS[model]}\nAcc={acc:.3f}", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    save(fig, "confusion_matrices_multiclass")


# ==============================================================================
# 4. COMBINED CONFUSION MATRICES (binary + 3-class side by side)
# ==============================================================================
def fig_confusion_combined():
    fig = plt.figure(figsize=(20, 9))
    fig.suptitle("Confusion Matrices - All Models (Exp A, Test Set)",
                 fontsize=14, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.3)

    titles_row = ["Binary Classification", "3-Class Classification"]
    results_row = [bin_res, mc_res]
    cls_names_row = [["Clean", "Fault"], ["Clean", "Observable", "Silent"]]

    for row, (title, res, cls_names) in enumerate(zip(titles_row, results_row, cls_names_row)):
        for col, model in enumerate(MODELS):
            ax = fig.add_subplot(gs[row, col])
            cm = np.array(res["exp_a"][model]["confusion_matrix"])
            acc = res["exp_a"][model]["accuracy"]
            f1 = res["exp_a"][model]["f1_macro"]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=cls_names, yticklabels=cls_names,
                        cbar=False, linewidths=0.5)
            ax.set_title(f"{MODEL_LABELS[model]}\nAcc={acc:.3f}  F1={f1:.3f}",
                         fontsize=9.5)
            ax.set_xlabel("Predicted", fontsize=8)
            ax.set_ylabel("Actual", fontsize=8)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8)
            if col == 0:
                ax.annotate(title, xy=(-0.5, 0.5), xycoords="axes fraction",
                            fontsize=11, fontweight="bold", rotation=90,
                            ha="center", va="center")
    save(fig, "confusion_matrices")


# ==============================================================================
# 5. AUROC BY MODEL - bar chart (binary + multiclass)
# ==============================================================================
def fig_auroc_by_model():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("AUROC Comparison by Model (Exp A - Test Set)",
                 fontsize=13, fontweight="bold")

    for ax, (res, task_label) in zip(axes, [
        (bin_res, "Binary Classification"),
        (mc_res, "3-Class Classification"),
    ]):
        model_names = [MODEL_LABELS[m] for m in MODELS]
        aurocs = [res["exp_a"][m]["auroc"] for m in MODELS]

        # Add LSTM bar
        if task_label == "Binary Classification":
            model_names.append(MODEL_LABELS["lstm"])
            aurocs.append(lstm_bin["test"]["auroc"])
        else:
            model_names.append(MODEL_LABELS["lstm"])
            aurocs.append(lstm_3c["test"]["auroc"])

        colors = [MODEL_COLORS[m] for m in MODELS] + [MODEL_COLORS["lstm"]]
        bars = ax.bar(model_names, aurocs, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0.75, 1.02)
        ax.set_title(task_label)
        ax.set_ylabel("AUROC")
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.90 ref")
        ax.tick_params(axis="x", rotation=25)
        for bar, val in zip(bars, aurocs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    save(fig, "auroc_by_model")


# ==============================================================================
# 6. AUROC BY WORKLOAD - grouped bar
# ==============================================================================
def fig_auroc_by_workload():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AUROC by Workload (Exp A - Test Set)",
                 fontsize=13, fontweight="bold")

    workloads = list(WORKLOAD_LABELS.keys())
    x = np.arange(len(workloads))
    width = 0.15

    for ax, (res, task_label) in zip(axes, [
        (bin_res, "Binary Classification"),
        (mc_res, "3-Class Classification"),
    ]):
        for i, model in enumerate(MODELS):
            vals = [res["exp_a"][model]["auroc_by_workload"].get(wl, 0) or 0
                    for wl in workloads]
            ax.bar(x + i * width, vals, width,
                   label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads], rotation=25, ha="right")
        ax.set_ylim(0.7, 1.05)
        ax.set_title(task_label)
        ax.set_ylabel("AUROC")
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "auroc_by_workload")


# ==============================================================================
# 7. GENERALIZATION EXP B - holdout workload (binary)
# ==============================================================================
def fig_exp_b():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Generalization Experiment B: Train on 4 Workloads, Test on irq_test",
                 fontsize=12, fontweight="bold")

    for ax, (task_key, task_label, in_domain_res) in zip(axes, [
        ("binary", "Binary", bin_res),
        ("multiclass", "3-Class", mc_res),
    ]):
        models_all = MODELS + ["lstm"]
        in_domain = [in_domain_res["exp_a"][m]["auroc"] for m in MODELS]
        ood = [exp_b[task_key]["results"][m]["auroc"] for m in MODELS]

        if task_key == "binary":
            in_domain.append(lstm_bin["test"]["auroc"])
            ood.append(lstm_gen["binary"]["exp_b"]["auroc"])
        else:
            in_domain.append(lstm_3c["test"]["auroc"])
            ood.append(lstm_gen["3class"]["exp_b"]["auroc"])

        x = np.arange(len(models_all))
        w = 0.35
        bars1 = ax.bar(x - w/2, in_domain, w, label="In-Domain (Exp A)",
                       color="#4C72B0", edgecolor="white", alpha=0.9)
        bars2 = ax.bar(x + w/2, ood, w, label="OOD - irq_test",
                       color="#C44E52", edgecolor="white", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models_all], rotation=25, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{task_label} Classification")
        ax.set_ylabel("AUROC")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5, label="Random (0.5)")
        ax.legend(fontsize=8)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    fig.tight_layout()
    save(fig, "generalization_exp_b")


# ==============================================================================
# 8. GENERALIZATION EXP C - LOOCV per fold
# ==============================================================================
def fig_exp_c():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Generalization Experiment C: Leave-One-Workload-Out CV (LOOCV)",
                 fontsize=12, fontweight="bold")

    workloads = list(WORKLOAD_LABELS.keys())
    x = np.arange(len(workloads))
    width = 0.15

    # Binary
    ax = axes[0]
    for i, model in enumerate(MODELS):
        vals = [exp_c["binary"]["summary"][model]["per_fold"][wl] for wl in workloads]
        ax.bar(x + i * width, vals, width,
               label=MODEL_LABELS[model], color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    # LSTM LOOCV
    lstm_vals = [lstm_gen["binary"]["exp_c"]["per_fold"][wl]["auroc"] for wl in workloads]
    ax.bar(x + 4 * width, lstm_vals, width,
           label=MODEL_LABELS["lstm"], color=MODEL_COLORS["lstm"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads], rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Binary Classification")
    ax.set_ylabel("AUROC (held-out workload)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=8)

    # 3-class
    ax = axes[1]
    for i, model in enumerate(MODELS):
        vals = [exp_c["multiclass"]["summary"][model]["per_fold"][wl] for wl in workloads]
        ax.bar(x + i * width, vals, width,
               label=MODEL_LABELS[model], color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5)
    lstm_vals_3c = [lstm_gen["3class"]["exp_c"]["per_fold"][wl]["auroc"] for wl in workloads]
    ax.bar(x + 4 * width, lstm_vals_3c, width,
           label=MODEL_LABELS["lstm"], color=MODEL_COLORS["lstm"], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([WORKLOAD_LABELS[w] for w in workloads], rotation=25, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("3-Class Classification")
    ax.set_ylabel("AUROC (held-out workload)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, "generalization_exp_c")


# ==============================================================================
# 9. GENERALIZATION SUMMARY - radar/heatmap of LOOCV AUROC
# ==============================================================================
def fig_generalization_heatmap():
    workloads = list(WORKLOAD_LABELS.keys())
    models_all = MODELS + ["lstm"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LOOCV Generalization AUROC Heatmap", fontsize=13, fontweight="bold")

    for ax, (task_key, task_label) in zip(axes, [
        ("binary", "Binary"), ("multiclass", "3-Class"),
    ]):
        mat = []
        row_labels = []
        for model in MODELS:
            row = [exp_c[task_key]["summary"][model]["per_fold"][wl] for wl in workloads]
            mat.append(row)
            row_labels.append(MODEL_LABELS[model])
        if task_key == "binary":
            lstm_row = [lstm_gen["binary"]["exp_c"]["per_fold"][wl]["auroc"] for wl in workloads]
        else:
            lstm_row = [lstm_gen["3class"]["exp_c"]["per_fold"][wl]["auroc"] for wl in workloads]
        mat.append(lstm_row)
        row_labels.append(MODEL_LABELS["lstm"])

        mat = np.array(mat)
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=[WORKLOAD_LABELS[w] for w in workloads],
                    yticklabels=row_labels, ax=ax,
                    vmin=0.3, vmax=1.0, linewidths=0.5,
                    annot_kws={"size": 9})
        ax.set_title(f"{task_label} Classification")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    save(fig, "generalization_heatmap")


# ==============================================================================
# 10. METRICS COMPARISON TABLE - accuracy, precision, recall, F1, AUROC
# ==============================================================================
def fig_metrics_table():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Full Metrics Comparison - Exp A (Test Set)",
                 fontsize=13, fontweight="bold")

    metrics = ["accuracy", "precision", "recall", "f1_macro", "auroc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 (macro)", "AUROC"]

    for ax, (res, task_label) in zip(axes, [
        (bin_res, "Binary Classification"),
        (mc_res, "3-Class Classification"),
    ]):
        x = np.arange(len(metrics))
        width = 0.18
        for i, model in enumerate(MODELS):
            vals = [res["exp_a"][model][m] for m in metrics]
            ax.bar(x + i * width, vals, width,
                   label=MODEL_LABELS[model], color=MODEL_COLORS[model],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(metric_labels, rotation=20, ha="right")
        ax.set_ylim(0.6, 1.05)
        ax.set_title(task_label)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8, loc="lower right")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

    fig.tight_layout()
    save(fig, "metrics_comparison")


# ==============================================================================
# 11. FEATURE IMPORTANCE - Random Forest (top 20 aggregate features)
# ==============================================================================
def fig_feature_importance():
    importances = np.array(bin_res["exp_a"]["random_forest"]["feature_importances"])
    top_k = 20
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_imp = importances[top_idx]

    # Feature names: 13 features x 12 windows = 156 aggregate features
    OBS_FEATURE_NAMES = [
        "pc", "instr", "rs1", "rs2", "rd",
        "mem_addr", "mem_data", "reg_data",
        "ctrl_sig", "branch_taken", "trap",
        "alu_out", "csr",
    ]
    feat_names = []
    for win in range(12):
        for fn in OBS_FEATURE_NAMES:
            feat_names.append(f"win{win}_{fn}")

    labels_feat = [feat_names[i] if i < len(feat_names) else f"feat_{i}"
                   for i in top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Feature Importance - Random Forest (Aggregate Features)",
                 fontsize=13, fontweight="bold")

    # Binary
    ax = axes[0]
    colors = plt.cm.Blues_r(np.linspace(0.2, 0.7, top_k))
    bars = ax.barh(range(top_k), top_imp[::-1], color=colors[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(labels_feat[::-1], fontsize=8)
    ax.set_title(f"Top {top_k} Features (Binary RF)")
    ax.set_xlabel("Importance")

    # XGBoost comparison if available (binary has no feature_importances for xgb in JSON)
    # Instead, show multiclass RF importances
    mc_importances = np.array(mc_res["exp_a"]["random_forest"]["feature_importances"])
    mc_top_idx = np.argsort(mc_importances)[::-1][:top_k]
    mc_top_imp = mc_importances[mc_top_idx]
    mc_labels = [feat_names[i] if i < len(feat_names) else f"feat_{i}"
                 for i in mc_top_idx]

    ax = axes[1]
    colors2 = plt.cm.Oranges_r(np.linspace(0.2, 0.7, top_k))
    ax.barh(range(top_k), mc_top_imp[::-1], color=colors2[::-1])
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(mc_labels[::-1], fontsize=8)
    ax.set_title(f"Top {top_k} Features (3-Class RF)")
    ax.set_xlabel("Importance")

    fig.tight_layout()
    save(fig, "feature_importance")


# ==============================================================================
# 12. t-SNE VISUALIZATION
# ==============================================================================
def fig_tsne():
    print("  running t-SNE (may take 30s)...")
    scaler = StandardScaler()
    X = scaler.fit_transform(features_agg)
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, max_iter=1000)
    Z = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("t-SNE Embedding of Aggregate Features (N=3,150)",
                 fontsize=13, fontweight="bold")

    # Color by class
    ax = axes[0]
    for cls_id, (cls_label, color) in enumerate(zip(CLASS_LABELS, CLASS_COLORS)):
        mask = labels == cls_id
        ax.scatter(Z[mask, 0], Z[mask, 1], c=color, label=cls_label,
                   alpha=0.5, s=8, linewidths=0)
    ax.set_title("Colored by Fault Class")
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Color by workload
    ax = axes[1]
    wl_palette = sns.color_palette("tab10", n_colors=len(WORKLOAD_LABELS))
    for i, (wl, wl_label) in enumerate(WORKLOAD_LABELS.items()):
        mask_wl = (meta["workload"] == wl).values
        # Only fault samples for clarity
        mask = mask_wl & (labels != 0)
        ax.scatter(Z[mask, 0], Z[mask, 1], c=[wl_palette[i]], label=wl_label,
                   alpha=0.5, s=8, linewidths=0)
    ax.set_title("Colored by Workload (Fault Samples Only)")
    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    fig.tight_layout()
    save(fig, "tsne_visualization")


# ==============================================================================
# 13. LSTM vs BASELINES SUMMARY
# ==============================================================================
def fig_lstm_vs_baselines():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("BiLSTM vs. Baseline Models - Performance Summary",
                 fontsize=13, fontweight="bold")

    # In-domain AUROC
    ax = axes[0]
    all_models = MODELS + ["lstm"]
    bin_aurocs = [bin_res["exp_a"][m]["auroc"] for m in MODELS] + [lstm_bin["test"]["auroc"]]
    mc_aurocs = [mc_res["exp_a"][m]["auroc"] for m in MODELS] + [lstm_3c["test"]["auroc"]]
    x = np.arange(len(all_models))
    w = 0.35
    ax.bar(x - w/2, bin_aurocs, w, label="Binary", color="#4C72B0", edgecolor="white")
    ax.bar(x + w/2, mc_aurocs, w, label="3-Class", color="#DD8452", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in all_models], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0.75, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("In-Domain (Exp A)")
    ax.legend(fontsize=9)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    # Exp B AUROC (OOD)
    ax = axes[1]
    bin_b = [exp_b["binary"]["results"][m]["auroc"] for m in MODELS] + \
            [lstm_gen["binary"]["exp_b"]["auroc"]]
    mc_b = [exp_b["multiclass"]["results"][m]["auroc"] for m in MODELS] + \
           [lstm_gen["3class"]["exp_b"]["auroc"]]
    ax.bar(x - w/2, bin_b, w, label="Binary", color="#4C72B0", edgecolor="white")
    ax.bar(x + w/2, mc_b, w, label="3-Class", color="#DD8452", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in all_models], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC")
    ax.set_title("Exp B: Holdout Workload (irq_test)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5, label="Chance")
    ax.legend(fontsize=9)

    # Exp C LOOCV mean AUROC
    ax = axes[2]
    bin_c = [exp_c["binary"]["summary"][m]["auroc_mean"] for m in MODELS] + \
            [lstm_gen["binary"]["exp_c"]["auroc_mean"]]
    bin_c_std = [exp_c["binary"]["summary"][m]["auroc_std"] for m in MODELS] + \
                [lstm_gen["binary"]["exp_c"]["auroc_std"]]
    mc_c = [exp_c["multiclass"]["summary"][m]["auroc_mean"] for m in MODELS] + \
           [lstm_gen["3class"]["exp_c"]["auroc_mean"]]
    mc_c_std = [exp_c["multiclass"]["summary"][m]["auroc_std"] for m in MODELS] + \
               [lstm_gen["3class"]["exp_c"]["auroc_std"]]

    bars1 = ax.bar(x - w/2, bin_c, w, label="Binary", color="#4C72B0",
                   edgecolor="white", yerr=bin_c_std, capsize=3, error_kw={"elinewidth": 1})
    bars2 = ax.bar(x + w/2, mc_c, w, label="3-Class", color="#DD8452",
                   edgecolor="white", yerr=mc_c_std, capsize=3, error_kw={"elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in all_models], rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC (mean +/- std)")
    ax.set_title("Exp C: LOOCV (mean +/- std)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5, label="Chance")
    ax.legend(fontsize=9)

    fig.tight_layout()
    save(fig, "lstm_vs_baselines")


# ==============================================================================
# 14. AUROC BY CATEGORY (from metadata - infer from available data)
# ==============================================================================
def fig_auroc_by_category():
    # Compute per-category observable rate from metadata
    fault_meta = meta[meta["label"].isin([1, 2])].copy()
    categories = fault_meta["category"].unique().tolist()
    categories.sort()

    cat_obs_rate = fault_meta.groupby("category")["observable"].mean() * 100
    cat_counts = fault_meta.groupby("category").size()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fault Category Analysis", fontsize=13, fontweight="bold")

    # Observable rate by category
    ax = axes[0]
    cats = cat_obs_rate.index.tolist()
    rates = cat_obs_rate.values
    cnts = [cat_counts[c] for c in cats]
    colors_cat = sns.color_palette("husl", n_colors=len(cats))
    bars = ax.bar(cats, rates, color=colors_cat, edgecolor="white", linewidth=0.8)
    ax.set_title("Observable Rate by Fault Category")
    ax.set_ylabel("Observable Fault %")
    ax.set_ylim(0, max(rates) * 1.3)
    ax.tick_params(axis="x", rotation=30)
    for bar, rate, cnt in zip(bars, rates, cnts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{rate:.1f}%\nn={cnt}", ha="center", va="bottom", fontsize=9)

    # Sample count by category x workload heatmap
    ax = axes[1]
    pivot = fault_meta.pivot_table(index="category", columns="workload",
                                   values="observable", aggfunc="mean") * 100
    pivot = pivot[[w for w in WORKLOAD_LABELS.keys() if w in pivot.columns]]
    pivot.columns = [WORKLOAD_LABELS[c] for c in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax,
                vmin=0, vmax=100, linewidths=0.5,
                annot_kws={"size": 9})
    ax.set_title("Observable Rate (%) - Category x Workload")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    save(fig, "auroc_by_category")


# ==============================================================================
# 15. IN-DOMAIN vs OOD DEGRADATION SUMMARY
# ==============================================================================
def fig_degradation_summary():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("In-Domain -> Out-of-Domain AUROC Degradation (Binary, Exp B)",
                 fontsize=12, fontweight="bold")

    all_models = MODELS + ["lstm"]
    in_d = [bin_res["exp_a"][m]["auroc"] for m in MODELS] + [lstm_bin["test"]["auroc"]]
    ood = [exp_b["binary"]["results"][m]["auroc"] for m in MODELS] + \
          [lstm_gen["binary"]["exp_b"]["auroc"]]

    x = np.arange(len(all_models))
    for xi, model, ind, od in zip(x, all_models, in_d, ood):
        ax.plot([xi, xi], [ind, od], color="gray", linewidth=1.5, zorder=1)
        ax.scatter(xi, ind, color=MODEL_COLORS[model], s=120, zorder=2,
                   marker="o", label=f"{MODEL_LABELS[model]} (in-domain)" if xi == 0 else "")
        ax.scatter(xi, od, color=MODEL_COLORS[model], s=120, zorder=2,
                   marker="^", alpha=0.7)
        delta = od - ind
        ax.text(xi + 0.08, (ind + od) / 2, f"{delta:+.3f}",
                ha="left", va="center", fontsize=8.5, color="darkred" if delta < 0 else "green")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in all_models], rotation=20, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.1, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=8,
               label="In-Domain (Exp A)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None", markersize=8,
               label="OOD - irq_test (Exp B)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    save(fig, "ood_degradation")


# ==============================================================================
# RUN ALL
# ==============================================================================
if __name__ == "__main__":
    print("Generating all figures...")
    fig_dataset_overview()
    fig_confusion_binary()
    fig_confusion_multiclass()
    fig_confusion_combined()
    fig_auroc_by_model()
    fig_auroc_by_workload()
    fig_exp_b()
    fig_exp_c()
    fig_generalization_heatmap()
    fig_metrics_table()
    fig_feature_importance()
    fig_tsne()
    fig_lstm_vs_baselines()
    fig_auroc_by_category()
    fig_degradation_summary()
    print(f"\nAll figures saved to {FIG_DIR}")
