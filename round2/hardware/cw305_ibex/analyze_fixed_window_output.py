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


def grouped_auroc(data):
    clean = data["clean_traces"].astype(np.float32)
    fault = data["fault_traces"].astype(np.float32)
    target = data["target"].astype(int)
    bit = data["bit"].astype(int)
    polarity = data["pol"].astype(int)
    x = np.vstack([clean, fault])
    y = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(fault), dtype=int)]
    groups = np.r_[np.arange(len(clean)), 1000 + (target << 6) + (bit << 1) + polarity]
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
    workloads = {}
    for path in args.captures:
        data = np.load(path, allow_pickle=False)
        name = str(data["workload"])
        observable = data["observable"].astype(int)
        workloads[name] = {
            "architectural_result_divergence_rate": float(observable.mean()),
            "architectural_result_divergence_count": int(observable.sum()),
            "n_fault_trials": int(len(observable)),
            "power_trace_auroc_grouped_cv": grouped_auroc(data),
            "golden_result": int(data["golden_result"]),
        }
    result = {
        "observation_window_core_cycles": 600,
        "per_workload": workloads,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results" / "fixed_window_output.json").write_text(
        json.dumps(result, indent=2)
    )
    names = list(workloads)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        x - 0.2,
        [workloads[name]["architectural_result_divergence_rate"] for name in names],
        0.4,
        label="Architectural-result divergence rate",
    )
    ax.bar(
        x + 0.2,
        [workloads[name]["power_trace_auroc_grouped_cv"] for name in names],
        0.4,
        label="Power-trace AUROC",
    )
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fixed-window architectural output and power traces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "fixed_window_output.png", dpi=180)


if __name__ == "__main__":
    main()
