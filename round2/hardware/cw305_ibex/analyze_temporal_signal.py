import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def welch_t(a, b):
    denominator = np.sqrt(a.var(0, ddof=1) / len(a) + b.var(0, ddof=1) / len(b))
    return np.abs(a.mean(0) - b.mean(0)) / (denominator + 1e-12)


def sample_auroc(clean, fault):
    x = np.vstack([clean, fault])
    y = np.r_[np.zeros(len(clean), dtype=int), np.ones(len(fault), dtype=int)]
    values = np.array([roc_auc_score(y, x[:, index]) for index in range(x.shape[1])])
    return np.maximum(values, 1 - values)


def shortest_energy_window(values, fraction):
    target = values.sum() * fraction
    start = 0
    total = 0.0
    best = (0, len(values) - 1)
    for end, value in enumerate(values):
        total += value
        while start <= end and total - values[start] >= target:
            total -= values[start]
            start += 1
        if total >= target and end - start < best[1] - best[0]:
            best = (start, end)
    return {"start": best[0], "end": best[1], "length": best[1] - best[0] + 1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture")
    parser.add_argument("--fraction", type=float, default=0.9)
    args = parser.parse_args()
    data = np.load(args.capture, allow_pickle=False)
    clean = data["clean_traces"].astype(np.float64)
    fault = data["fault_traces"].astype(np.float64)
    workload = str(data["workload"])
    t_stat = welch_t(fault, clean)
    auroc = sample_auroc(clean, fault)
    result = {
        "workload": workload,
        "samples": int(clean.shape[1]),
        "tvla": {
            "max_abs_t": float(t_stat.max()),
            "peak_sample": int(t_stat.argmax()),
            "samples_above_4_5": int(np.sum(t_stat >= 4.5)),
            "shortest_energy_window": shortest_energy_window(t_stat**2, args.fraction),
        },
        "univariate_auroc": {
            "max": float(auroc.max()),
            "peak_sample": int(auroc.argmax()),
            "samples_above_0_9": int(np.sum(auroc >= 0.9)),
            "shortest_energy_window": shortest_energy_window(
                (auroc - 0.5) ** 2, args.fraction
            ),
        },
        "energy_fraction": args.fraction,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "results" / f"{workload}_temporal_signal.json").write_text(
        json.dumps(result, indent=2)
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].plot(t_stat)
    axes[0].axhline(4.5, color="black", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("TVLA |t|")
    axes[1].plot(auroc)
    axes[1].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("Univariate AUROC")
    axes[1].set_xlabel("ADC sample")
    fig.suptitle(f"Temporal signal distribution: {workload}")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / f"{workload}_temporal_signal.png", dpi=180)


if __name__ == "__main__":
    main()
