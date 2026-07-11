import numpy as np

BOOT_N = 2000
BOOT_SEED = 42


def make_binary_groups(meta, y_binary):

    fs = np.asarray(meta["fault_site"])
    wl = np.asarray(meta["workload"])
    fc = np.asarray(meta["fault_cycle"])
    yb = np.asarray(y_binary)
    return np.array(
        [
            str(int(f)) if lab == 1 else f"c_{w}_{c}"
            for f, lab, w, c in zip(fs, yb, wl, fc)
        ]
    )


def _cluster_members(groups):
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [[] for _ in range(len(uniq))]
    for row, g in enumerate(inv):
        members[g].append(row)
    return uniq, [np.asarray(m) for m in members]


def cluster_bootstrap(stat_fn, groups, n_boot=BOOT_N, seed=BOOT_SEED):

    uniq, members = _cluster_members(groups)
    rng = np.random.default_rng(seed)
    k = len(uniq)
    vals = []
    for _ in range(n_boot):
        picks = rng.integers(0, k, k)
        idx = np.concatenate([members[p] for p in picks])
        v = stat_fn(idx)
        if v is not None:
            vals.append(v)
    return np.asarray(vals)


def cluster_bootstrap_auroc_ci(y_true, y_score, groups, n_boot=BOOT_N, seed=BOOT_SEED):

    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    def stat(idx):
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            return None
        return roc_auc_score(yt, y_score[idx])

    vals = cluster_bootstrap(stat, groups, n_boot=n_boot, seed=seed)
    if len(vals) == 0:
        return {
            "lo": float("nan"),
            "hi": float("nan"),
            "median": float("nan"),
            "se": float("nan"),
            "n_valid_boot": 0,
        }
    return {
        "lo": float(np.percentile(vals, 2.5)),
        "hi": float(np.percentile(vals, 97.5)),
        "median": float(np.median(vals)),
        "se": float(np.std(vals)),
        "n_valid_boot": int(len(vals)),
    }
