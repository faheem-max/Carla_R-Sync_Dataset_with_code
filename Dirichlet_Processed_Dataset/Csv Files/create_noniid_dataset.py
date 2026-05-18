"""
Non-IID Dataset Creation via Dirichlet Partitioning
R-Sync Paper — 7 Physical AI Nodes

Node mapping (CARLA towns):
  client_1  → Town01  → target 1233 samples
  client_2  → Town02  → target 1586 samples
  client_3  → Town03  → target 1730 samples
  client_4  → Town04  → target 1811 samples
  client_5  → Town05  → target 1972 samples
  client_6  → Town06  → target 2717 samples
  client_10 → Town10HD → target 3136 samples

Dataset sizes are monotonically non-decreasing:
  |D1| ≤ |D2| ≤ ... ≤ |D7|
"""

import pandas as pd
import numpy as np
from scipy.stats import dirichlet

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

files = [
    "client_1_dataset.csv",    # Town01  → Node v1
    "client_2_dataset.csv",    # Town02  → Node v2
    "client_3_dataset.csv",    # Town03  → Node v3
    "client_4_dataset.csv",    # Town04  → Node v4
    "client_5_dataset.csv",    # Town05  → Node v5
    "client_6_dataset.csv",    # Town06  → Node v6
    "client_10_dataset.csv",   # Town10HD → Node v10
]

# Target samples — monotonically non-decreasing
# Matches paper Table: |Dk| = 1233, 1586, 1730, 1811, 1972, 2717, 3136
target_samples = [1233, 1586, 1730, 1811, 1972, 2717, 3136]

n_bins      = 10     # quantile bins for speed distribution
alpha       = 0.5    # Dirichlet concentration: smaller = stronger Non-IID
                     # 0.3 = very strong Non-IID
                     # 0.5 = strong Non-IID (paper setting)
                     # 1.0 = moderate Non-IID
random_seed = 42
target_col  = "speed_kmh"   # target variable column name

# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(random_seed)

# Verify monotonically non-decreasing
assert all(target_samples[i] <= target_samples[i+1]
           for i in range(len(target_samples)-1)), \
    "target_samples must be monotonically non-decreasing"

print("=" * 85)
print("Non-IID Dirichlet Dataset Creation — R-Sync Paper")
print(f"  alpha={alpha}  n_bins={n_bins}  seed={random_seed}")
print("=" * 85)
print(f"{'Node':<12} {'Town':<10} {'Original':>9} → {'Target':>7} → "
      f"{'Final':>7}  {'Keep%':>6}  Dirichlet p-vector")
print("-" * 85)

town_names = [
    "Town01", "Town02", "Town03", "Town04",
    "Town05", "Town06", "Town10HD"
]

results = []

for i, (file_path, target_n, town) in enumerate(
    zip(files, target_samples, town_names)
):
    client_id = file_path.split("_")[1]  # 1, 2, 3, 4, 5, 6, 10

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  WARNING: {file_path} not found — skipping")
        continue

    if target_col not in df.columns:
        print(f"  WARNING: '{target_col}' column not found in {file_path}")
        continue

    original_n = len(df)

    # Step 1: Speed quantile binning (equal-population bins)
    df["speed_bin"] = pd.qcut(
        df[target_col], q=n_bins, labels=False, duplicates="drop"
    )
    actual_bins = df["speed_bin"].nunique()

    # Step 2: Dirichlet proportion sampling
    rng = np.random.default_rng(random_seed + i)
    p   = dirichlet.rvs(
        alpha * np.ones(actual_bins), random_state=rng
    )[0]
    p = p / p.sum()   # normalise to sum = 1

    # Step 3: Per-bin sampling without replacement
    kept = []
    for b in range(actual_bins):
        bin_df = df[df["speed_bin"] == b]
        n_keep = int(target_n * p[b])
        n_keep = min(n_keep, len(bin_df))   # cannot exceed bin size
        if n_keep > 0:
            kept.append(
                bin_df.sample(n=n_keep, random_state=rng)
            )

    if not kept:
        print(f"  WARNING: No samples retained for client_{client_id}")
        continue

    df_nonIID = (
        pd.concat(kept)
        .sample(frac=1, random_state=rng)
        .reset_index(drop=True)
    )
    df_nonIID = df_nonIID.drop(columns=["speed_bin"])

    final_n  = len(df_nonIID)
    keep_pct = final_n / original_n * 100

    # Save
    out_name = (
        f"client_{client_id}_nonIID"
        f"_alpha{alpha}_n{final_n}.csv"
    )
    df_nonIID.to_csv(out_name, index=False)

    results.append({
        "client":    f"client_{client_id}",
        "town":      town,
        "original":  original_n,
        "target":    target_n,
        "final":     final_n,
        "keep_pct":  round(keep_pct, 1),
    })

    print(
        f"client_{client_id:<4}  {town:<10} {original_n:>9,} → "
        f"{target_n:>7,} → {final_n:>7,}  {keep_pct:5.1f}%  "
        f"{np.round(p, 3)}"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 85)
print("SUMMARY")
print("=" * 85)

if results:
    sizes = [r["final"] for r in results]

    # Verify monotonically non-decreasing
    is_mono = all(sizes[i] <= sizes[i+1] for i in range(len(sizes)-1))

    print(f"{'Node':<12} {'Town':<10} {'Final |Dk|':>10}")
    print("-" * 35)
    for r in results:
        print(f"{r['client']:<12} {r['town']:<10} {r['final']:>10,}")
    print("-" * 35)
    print(f"{'Total':<22} {sum(sizes):>10,}")
    print()
    print(f"Monotonically non-decreasing: {'✓ YES' if is_mono else '✗ NO'}")
    print(f"alpha (Dirichlet):            {alpha}")
    print(f"n_bins:                       {n_bins}")
    print(f"random_seed:                  {random_seed}")

print()
print("✅ Done! Non-IID client files saved.")
print("   Format: client_X_nonIID_alpha0.5_nYYYY.csv")
