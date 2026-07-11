import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def load_capture(path):
    data = np.load(path, allow_pickle=False)
    clean = data["clean_traces"].astype(np.float32)
    fault = data["fault_traces"].astype(np.float32)
    target = data["target"].astype(int)
    bit = data["bit"].astype(int)
    polarity = data["pol"].astype(int)
    x = np.vstack([clean, fault])
    y = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(fault), dtype=int)]
    groups = np.r_[np.arange(len(clean)), 1000 + (target << 6) + (bit << 1) + polarity]
    return str(data["workload"]), x, y, groups


def fit_score(x_train, y_train, x_test, y_test):
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear")
    )
    model.fit(x_train, y_train)
    return float(roc_auc_score(y_test, model.decision_function(x_test)))


def grouped_within_score(x, y, groups):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros(len(y))
    for train, test in splitter.split(x, y, groups):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear")
        )
        model.fit(x[train], y[train])
        scores[test] = model.decision_function(x[test])
    return float(roc_auc_score(y, scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+")
    args = parser.parse_args()
    datasets = [load_capture(path) for path in args.captures]
    names = [item[0] for item in datasets]
    matrix = np.empty((len(names), len(names)))
    for i, (_, x_train, y_train, groups) in enumerate(datasets):
        for j, (_, x_test, y_test, _) in enumerate(datasets):
            matrix[i, j] = (
                grouped_within_score(x_train, y_train, groups)
                if i == j
                else fit_score(x_train, y_train, x_test, y_test)
            )
    off_diagonal = matrix[~np.eye(len(names), dtype=bool)]
    result = {
        "train_workloads": names,
        "test_workloads": names,
        "auroc_matrix": matrix.tolist(),
        "within_workload_grouped_cv": [float(matrix[i, i]) for i in range(len(names))],
        "cross_workload_mean": float(off_diagonal.mean()),
        "cross_workload_min": float(off_diagonal.min()),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results" / "cross_workload_detection.json").write_text(
        json.dumps(result, indent=2)
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Test workload")
    ax.set_ylabel("Training workload")
    ax.set_title("Cross-workload power-trace detection")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "cross_workload_detection.png", dpi=180)


if __name__ == "__main__":
    main()
