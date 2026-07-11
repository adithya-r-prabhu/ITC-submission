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
    observable = data["observable"].astype(int)
    workload = str(data["workload"])
    return clean, fault, target, bit, polarity, observable, workload


def grouped_scores(clean, fault, target, bit, polarity):
    length = min(clean.shape[1], fault.shape[1])
    x = np.vstack([clean[:, :length], fault[:, :length]])
    y = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(fault), dtype=int)]
    clean_groups = np.arange(len(clean), dtype=int)
    fault_groups = 1000 + (target << 6) + (bit << 1) + polarity
    groups = np.r_[clean_groups, fault_groups]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    scores = np.zeros(len(y))
    for train, test in splitter.split(x, y, groups):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, solver="liblinear")
        )
        model.fit(x[train], y[train])
        scores[test] = model.decision_function(x[test])
    return y, scores


def welch_t(a, b):
    denominator = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    return (a.mean(0) - b.mean(0)) / (denominator + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    args = parser.parse_args()
    clean, fault, target, bit, polarity, observable, workload = load_capture(
        args.capture
    )
    y, scores = grouped_scores(clean, fault, target, bit, polarity)
    unique_sites = np.unique(np.column_stack([target, bit, polarity]), axis=0)
    repetitions = len(fault) / len(unique_sites)
    rates = {
        "instruction": float(observable[target == 1].mean()),
        "load_data": float(observable[target == 2].mean()),
        "sa0": float(observable[polarity == 0].mean()),
        "sa1": float(observable[polarity == 1].mean()),
    }
    length = min(clean.shape[1], fault.shape[1])
    t_stat = welch_t(fault[:, :length], clean[:, :length])
    result = {
        "workload": workload,
        "n_clean_traces": int(len(clean)),
        "n_fault_trials": int(len(fault)),
        "n_unique_sites": int(len(unique_sites)),
        "repetitions_per_site": repetitions,
        "architectural_result_divergence_rate": float(observable.mean()),
        "architectural_result_divergence_count": int(observable.sum()),
        "divergence_rate_by_group": rates,
        "power_trace_auroc_grouped_cv": float(roc_auc_score(y, scores)),
        "tvla_max_abs_t": float(np.max(np.abs(t_stat))),
        "tvla_samples_above_4_5": int(np.sum(np.abs(t_stat) > 4.5)),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results" / f"{workload}_detection.json").write_text(
        json.dumps(result, indent=2)
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(list(rates), list(rates.values()), color="#4878a8")
    axes[0].set_ylabel("Architectural-result divergence rate")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].plot(clean.mean(0), label="clean", linewidth=0.8)
    axes[1].plot(fault.mean(0), label="fault enabled", linewidth=0.8)
    axes[1].set_xlabel("ADC sample")
    axes[1].set_ylabel("Mean normalized ADC value")
    axes[1].legend()
    fig.suptitle(f"CW305 Ibex capture: {workload}")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{workload}_detection.png", dpi=180)


if __name__ == "__main__":
    main()
