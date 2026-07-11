import argparse
import json
import re
import subprocess
from pathlib import Path
from shutil import which

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from compare_observation_channels import INT_SIGNALS, build_trace_buffer_features

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = SCRIPT_DIR.parent
RESULTS = REPO_ROOT / "results"
FIGS = REPO_ROOT / "reports" / "figures"
SEED = 42
NBUS = 156
SYNTH = REPO_ROOT / "dft_synth"


def find_yosys():
    candidates = [
        REPO_ROOT / "external" / "yosys" / "yosys",
        Path("/usr/bin/yosys"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    executable = which("yosys")
    return Path(executable) if executable else None


def gen_monitor(n, wbits=16, fbits=16, acc=40):
    rng = np.random.default_rng(42)
    weights = rng.integers(-(2 ** (wbits - 2)), 2 ** (wbits - 2), size=n)
    lines = [
        "module monitor (input wire clk, input wire rst, input wire start,",
        f"    input wire signed [{fbits - 1}:0] feature,",
        "    output reg done, output reg fault_flag);",
        f"  localparam N = {n};",
        f"  reg signed [{acc - 1}:0] accm; reg [8:0] idx; reg running;",
        f"  reg signed [{wbits - 1}:0] wrom [0:N-1];",
        f"  wire signed [{wbits - 1}:0] w; assign w = wrom[idx];",
        "  initial begin",
    ]
    for i, value in enumerate(weights):
        value = int(value)
        lines.append(
            f"    wrom[{i}] = {'-' if value < 0 else ''}{wbits}'sd{abs(value)};"
        )
    lines += [
        "  end",
        "  always @(posedge clk) begin",
        "    if (rst) begin accm<=0; idx<=0; running<=0; done<=0; fault_flag<=0;",
        "    end else if (start) begin accm<=0; idx<=0; running<=1; done<=0;",
        "    end else if (running) begin accm <= accm + w*feature;",
        "      if (idx==N-1) begin running<=0; done<=1;",
        "        fault_flag <= (accm + w*feature) > 0; end",
        "      else idx <= idx + 1;",
        "    end else done<=0;",
        "  end",
        "endmodule",
    ]
    return "\n".join(lines) + "\n"


def synth(yosys, top, verilog, tag):
    SYNTH.mkdir(exist_ok=True)
    vpath = SYNTH / f"{tag}.v"
    vpath.write_text(verilog)
    script = (
        f"read_verilog {vpath}; hierarchy -top {top}; proc; opt; techmap; "
        f"opt; synth -top {top} -flatten; stat"
    )
    result = subprocess.run(
        [str(yosys), "-p", script], capture_output=True, text=True, check=False
    )
    stat = result.stdout
    (SYNTH / f"{tag}_stat.txt").write_text(stat)
    section = stat.split("Printing statistics.")[-1]
    match = re.search(r"Number of cells:\s+(\d+)", section)
    cells = int(match.group(1)) if match else 0
    ff = sum(int(x) for x in re.findall(r"\b(\d+)\s+\$_S?DFF\w*", section))
    if cells == 0:
        cells = sum(int(x) for x in re.findall(r"^\s+(\d+)\s+\$_", section, re.M))
    return {"cells": cells, "ff": ff}


SIGSTAT_RTL = """
module sig_stat #(parameter W=32) (
    input wire clk, input wire rst, input wire en,
    input wire [W-1:0] x,
    output reg [W+5:0] s_sum, output reg [W-1:0] s_max, output reg [5:0] s_nz);
    always @(posedge clk) begin
        if (rst) begin s_sum<=0; s_max<=0; s_nz<=0; end
        else if (en) begin
            s_sum <= s_sum + x;
            if (x > s_max) s_max <= x;
            if (|x) s_nz <= s_nz + 1;
        end
    end
endmodule
"""


def sig_cols(Buf, j):

    return Buf[:, [j, 9 + j, 18 + j, 27 + j]]


def cv_auroc(X, y):
    pipe = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1, class_weight="balanced"
        ),
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return float(cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc").mean())


def test_auroc(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = RandomForestClassifier(
        n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced"
    ).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--itc-root", default=str(REPO_ROOT / "external" / "picorv32_fault_sim")
    )
    ap.add_argument("--yosys", default=None)
    args = ap.parse_args()

    _ = str(Path(args.itc_root) / "build" / "sim_fault_oracle")
    _ = Path(args.itc_root) / "src" / "firmware"
    cache = REPO_ROOT / ".cache" / "internal"
    cache.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(REPO_ROOT / "data_picorv32" / "metadata.csv")
    Xbus = np.load(REPO_ROOT / "data_picorv32" / "features_agg.npy")
    y = np.load(REPO_ROOT / "data_picorv32" / "labels.npy")
    _ = len(meta)

    print("  building per-signal internal features (reusing cache) ...")
    Buf, missing = build_trace_buffer_features(meta, args.itc_root, cache)
    print(f"  built ({missing} missing)")

    tr = (meta["split"] == "train").values & ((y == 0) | (y == 1))
    te = (meta["split"] == "test").values & ((y == 0) | (y == 1))
    ytr = (y[tr] == 1).astype(int)
    yte = (y[te] == 1).astype(int)

    remaining = list(range(len(INT_SIGNALS)))
    selected = []
    base_cv = cv_auroc(Xbus[tr], ytr)
    base_te = test_auroc(Xbus[tr], ytr, Xbus[te], yte)
    print(f"  bus-only: CV={base_cv:.4f} test={base_te:.4f}")
    steps = [
        {
            "step": 0,
            "added": "bus",
            "signals": [],
            "cv_auroc": round(base_cv, 4),
            "test_auroc": round(base_te, 4),
            "n_features": NBUS,
        }
    ]

    while remaining:
        best = (None, -1.0)
        for j in remaining:
            cols = np.hstack([sig_cols(Buf, k) for k in selected + [j]])
            Xc = np.hstack([Xbus, cols])
            a = cv_auroc(Xc[tr], ytr)
            if a > best[1]:
                best = (j, a)
        j = best[0]
        selected.append(j)
        remaining.remove(j)
        cols = np.hstack([sig_cols(Buf, k) for k in selected])
        Xs = np.hstack([Xbus, cols])
        te_a = test_auroc(Xs[tr], ytr, Xs[te], yte)
        steps.append(
            {
                "step": len(selected),
                "added": INT_SIGNALS[j],
                "signals": [INT_SIGNALS[k] for k in selected],
                "cv_auroc": round(best[1], 4),
                "test_auroc": round(te_a, 4),
                "n_features": NBUS + 4 * len(selected),
            }
        )
        print(
            f"  +{INT_SIGNALS[j]:9} ({len(selected)} sig)  CV={best[1]:.4f}  test={te_a:.4f}"
        )

    yosys = Path(args.yosys) if args.yosys else find_yosys()
    synthesis_cells_per_signal = None
    if yosys and yosys.exists():
        ss = synth(yosys, "sig_stat", SIGSTAT_RTL, "sig_stat_greedy")
        per_sig_cells = ss["cells"]
        for s in steps:
            k = len(s["signals"])
            mon = gen_monitor(s["n_features"])
            m = synth(yosys, "monitor", mon, f"monitor_g{s['n_features']}")
            s["extract_cells"] = per_sig_cells * k
            s["monitor_cells"] = m["cells"]
            s["total_cells"] = per_sig_cells * k + m["cells"]
        synthesis_cells_per_signal = per_sig_cells

        knee, knee_eff = None, -1
        for a, b in zip(steps[:-1], steps[1:]):
            dcost = (b["total_cells"] - a["total_cells"]) / 1000.0
            eff = (b["cv_auroc"] - a["cv_auroc"]) / dcost if dcost > 0 else 0
            if eff > knee_eff:
                knee_eff, knee = eff, b["added"]
    else:
        knee, knee_eff = None, None

    cv_gain = steps[-1]["cv_auroc"] - steps[0]["cv_auroc"]
    k95 = next(
        (
            s["step"]
            for s in steps
            if cv_gain > 0 and (s["cv_auroc"] - steps[0]["cv_auroc"]) >= 0.95 * cv_gain
        ),
        steps[-1]["step"],
    )
    full_gain = steps[-1]["test_auroc"] - steps[0]["test_auroc"]
    k95_gain_test = steps[k95]["test_auroc"] - steps[0]["test_auroc"]

    out = {
        "selection_protocol": "greedy forward selection and budget selection on five-fold training-split cross-validation",
        "bus_features": NBUS,
        "synthesis_cells_per_signal": synthesis_cells_per_signal,
        "ordering": [s["added"] for s in steps[1:]],
        "steps": steps,
        "signals_for_95pct_of_gain": int(k95),
        "k95_selection_curve": "train_cv",
        "k95_test_auroc": steps[k95]["test_auroc"],
        "k95_test_gain": round(k95_gain_test, 4),
        "full_trace_buffer_gain": round(full_gain, 4),
        "knee_signal": knee,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "observation_signal_selection.json").write_text(
        json.dumps(out, indent=2)
    )
    print("  wrote results/observation_signal_selection.json")

    FIGS.mkdir(parents=True, exist_ok=True)
    have_cost = "total_cells" in steps[-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [s.get("total_cells", s["n_features"]) for s in steps]
    ys = [s["test_auroc"] for s in steps]
    ax.plot(xs, ys, "-o", color="#2a9d8f", zorder=3)
    for s, x, yv in zip(steps, xs, ys):
        lab = "bus" if s["step"] == 0 else f"+{s['added']}"
        ax.annotate(
            lab, (x, yv), textcoords="offset points", xytext=(6, -3), fontsize=8
        )
    if k95:
        ax.axvline(
            xs[k95],
            ls="--",
            color="#c0392b",
            alpha=0.7,
            label=f"95% of gain @ {k95} signals (chosen on train CV)",
        )
    ax.set(
        xlabel="DFT cost (gate-equivalents, synthesised)"
        if have_cost
        else "feature count",
        ylabel="Held-out-site test AUROC",
        title="Greedy per-signal observability insertion",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "observation_signal_selection.png", dpi=130)
    print("  wrote reports/figures/observation_signal_selection.png")


if __name__ == "__main__":
    main()
