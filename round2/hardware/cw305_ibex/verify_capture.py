import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def inspect(path):
    data = np.load(path, allow_pickle=False)
    clean = data["clean_traces"].astype(np.float64)
    fault = data["fault_traces"].astype(np.float64)
    traces = np.vstack([clean, fault])
    values = np.unique(traces[: min(50, len(traces))])
    differences = np.diff(values)
    positive = differences[differences > 1e-9]
    step = float(positive.min()) if len(positive) else 0.0
    repeated = sum(
        np.array_equal(traces[0], traces[index])
        for index in range(1, min(50, len(traces)))
    )
    return {
        "workload": str(data["workload"]),
        "clean_traces": int(len(clean)),
        "fault_traces": int(len(fault)),
        "samples": int(traces.shape[1]),
        "minimum": float(traces.min()),
        "maximum": float(traces.max()),
        "adc_step_estimate": step,
        "adc_step_matches_10_bit_grid": abs(step - 1 / 1024) < 1e-5,
        "identical_to_first_trace_in_first_50": int(repeated),
        "mean_trace_variance": float(traces.var(axis=0).mean()),
        "required_fields_present": all(
            field in data.files
            for field in (
                "clean_traces",
                "fault_traces",
                "target",
                "bit",
                "pol",
                "observable",
                "workload",
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("captures", nargs="+")
    args = parser.parse_args()
    result = {Path(path).name: inspect(path) for path in args.captures}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "capture_validation.json").write_text(
        json.dumps(result, indent=2)
    )
    for name, values in result.items():
        print(name, values)


if __name__ == "__main__":
    main()
