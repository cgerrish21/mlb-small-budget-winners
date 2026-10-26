"""
Angels-specific diagnosis: where do they fall on every metric
vs. the efficient winner benchmark?

Output:
  output/angels_diagnosis.csv
  output/angels_vs_benchmark.csv
  Console report
"""

import os
import sys
import pandas as pd
import numpy as np

from config import DATA_DIR, OUTPUT_DIR


def percentile_rank(value, all_values):
    """Compute percentile rank of value within all_values."""
    all_values = all_values.dropna()
    if len(all_values) == 0 or pd.isna(value):
        return np.nan
    return (all_values < value).sum() / len(all_values) * 100


def main():
    master_path = os.path.join(DATA_DIR, "master.csv")
    if not os.path.exists(master_path):
        print(f"ERROR: {master_path} not found. Run build_features.py first.", flush=True)
        sys.exit(1)

    df = pd.read_csv(master_path)
    rankings_path = os.path.join(OUTPUT_DIR, "metric_rankings.csv")
    if os.path.exists(rankings_path):
        rankings = pd.read_csv(rankings_path)
    else:
        rankings = None

    # Filter groups
    angels = df[df["team"] == "LAA"].sort_values("Season")
    ew = df[df["group"] == "efficient_winner"]
    cl = df[df["group"] == "cheap_loser"]

    print("=" * 90)
    print("  ANGELS DIAGNOSIS: 2022-2026")
    print("=" * 90)
    print(f"\n  Angels team-seasons: {len(angels)}")
    print(f"  Angels groups: {angels['group'].value_counts().to_dict()}")
    print(f"  Angels W-L by year:")
    for _, row in angels.iterrows():
        print(f"    {int(row['Season'])}: {int(row['W'])}-{int(row['L'])} "
              f"({row['W_pct']:.3f}) | ${row['payroll_mm']:.0f}M | {row['group']}")

    # Use top metrics from rankings, or a default set
    if rankings is not None:
        top_metrics = rankings.head(15)["column"].tolist()
    else:
        top_metrics = [
            "homegrown_war_pct", "war_gini", "players_2plus_war", "top3_war_share",
            "bat_wRC+", "pit_FIP", "fld_Def", "war_per_10mm",
            "total_team_war", "premium_war_ratio", "roster_turnover_pct",
            "bat_WAR", "pit_WAR",
        ]

    # --- Angels vs. Benchmark ---
    print("\n" + "=" * 90)
    print("  ANGELS vs. EFFICIENT WINNER BENCHMARK")
    print("=" * 90)

    diagnosis_rows = []
    print(f"\n{'Metric':<30} {'Angels Avg':>12} {'EW Avg':>12} {'Gap':>10} {'Pctl':>8}")
    print("-" * 75)

    for col in top_metrics:
        if col not in df.columns:
            continue

        all_vals = pd.to_numeric(df[col], errors="coerce")
        ang_vals = pd.to_numeric(angels[col], errors="coerce")
        ew_vals = pd.to_numeric(ew[col], errors="coerce")

        ang_mean = ang_vals.mean()
        ew_mean = ew_vals.mean()
        gap = ang_mean - ew_mean
        pctl = percentile_rank(ang_mean, all_vals)

        # Get display name from rankings if available
        display = col
        if rankings is not None and col in rankings["column"].values:
            display = rankings[rankings["column"] == col]["metric"].values[0]

        diagnosis_rows.append({
            "metric": display,
            "column": col,
            "angels_mean": round(ang_mean, 3) if not np.isnan(ang_mean) else np.nan,
            "ew_mean": round(ew_mean, 3) if not np.isnan(ew_mean) else np.nan,
            "gap": round(gap, 3) if not np.isnan(gap) else np.nan,
            "percentile": round(pctl, 1) if not np.isnan(pctl) else np.nan,
        })

        if not np.isnan(ang_mean):
            print(f"{display:<30} {ang_mean:>12.3f} {ew_mean:>12.3f} "
                  f"{gap:>+10.3f} {pctl:>7.1f}%")

    # Save diagnosis
    diag_df = pd.DataFrame(diagnosis_rows)
    diag_df.to_csv(os.path.join(OUTPUT_DIR, "angels_diagnosis.csv"), index=False)
    print(f"\nSaved angels_diagnosis.csv")

    # --- Year-by-year Angels detail ---
    print("\n" + "=" * 90)
    print("  ANGELS YEAR-BY-YEAR TRAJECTORY")
    print("=" * 90)

    key_cols = [c for c in ["homegrown_war_pct", "war_gini", "players_2plus_war",
                            "total_team_war", "war_per_10mm", "bat_OPS", "pit_ERA",
                            "pit_FIP", "bat_WAR", "pit_WAR"] if c in df.columns]

    if key_cols:
        print(f"\n{'Year':<6}", end="")
        for c in key_cols:
            label = c.replace("homegrown_war_pct", "HG%").replace("war_gini", "Gini") \
                     .replace("players_2plus_war", "2+WAR").replace("total_team_war", "tWAR") \
                     .replace("war_per_10mm", "W/$10M").replace("bat_OPS", "OPS") \
                     .replace("pit_ERA", "ERA").replace("pit_FIP", "FIP") \
                     .replace("bat_WAR", "bWAR").replace("pit_WAR", "pWAR")
            print(f"{label:>10}", end="")
        print()
        print("-" * (6 + 10 * len(key_cols)))

        for _, row in angels.iterrows():
            print(f"{int(row['Season']):<6}", end="")
            for c in key_cols:
                val = row[c] if c in row.index else np.nan
                if pd.isna(val):
                    print(f"{'N/A':>10}", end="")
                else:
                    print(f"{val:>10.3f}", end="")
            print()

    # --- White Sox Trajectory (comparison turnaround story) ---
    chw = df[df["team"] == "CHW"].sort_values("Season")
    if len(chw) > 0:
        print("\n" + "=" * 90)
        print("  WHITE SOX TRAJECTORY (turnaround comparison)")
        print("=" * 90)

        print(f"\n  White Sox W-L by year:")
        for _, row in chw.iterrows():
            print(f"    {int(row['Season'])}: {int(row['W'])}-{int(row['L'])} "
                  f"({row['W_pct']:.3f}) | ${row['payroll_mm']:.0f}M | {row['group']}")

        if key_cols:
            print(f"\n{'Year':<6}", end="")
            for c in key_cols:
                label = c.replace("homegrown_war_pct", "HG%").replace("war_gini", "Gini") \
                         .replace("players_2plus_war", "2+WAR").replace("total_team_war", "tWAR") \
                         .replace("war_per_10mm", "W/$10M").replace("bat_OPS", "OPS") \
                         .replace("pit_ERA", "ERA").replace("pit_FIP", "FIP") \
                         .replace("bat_WAR", "bWAR").replace("pit_WAR", "pWAR")
                print(f"{label:>10}", end="")
            print()
            print("-" * (6 + 10 * len(key_cols)))

            for _, row in chw.iterrows():
                print(f"{int(row['Season']):<6}", end="")
                for c in key_cols:
                    val = row[c] if c in row.index else np.nan
                    if pd.isna(val):
                        print(f"{'N/A':>10}", end="")
                    else:
                        print(f"{val:>10.3f}", end="")
                print()

    # --- Benchmark comparison table ---
    benchmark_rows = []
    for col in top_metrics:
        if col not in df.columns:
            continue
        for _, row in angels.iterrows():
            val = pd.to_numeric(row.get(col), errors="coerce") if col in row.index else np.nan
            benchmark_rows.append({
                "Season": int(row["Season"]),
                "team": "LAA",
                "metric": col,
                "value": val,
                "ew_mean": pd.to_numeric(ew[col], errors="coerce").mean(),
            })

    bench_df = pd.DataFrame(benchmark_rows)
    bench_df.to_csv(os.path.join(OUTPUT_DIR, "angels_vs_benchmark.csv"), index=False)
    print(f"\nSaved angels_vs_benchmark.csv")

    # --- Key Takeaways ---
    print("\n" + "=" * 90)
    print("  KEY TAKEAWAYS")
    print("=" * 90)

    # Find worst gaps
    if diagnosis_rows:
        sorted_gaps = sorted(diagnosis_rows,
                             key=lambda x: x["percentile"] if x["percentile"] is not np.nan else 50)
        print("\n  Angels' worst metrics (lowest percentile rank):")
        for row in sorted_gaps[:5]:
            if row["percentile"] is not np.nan:
                print(f"    {row['metric']}: {row['percentile']:.0f}th percentile "
                      f"(Angels: {row['angels_mean']:.3f}, EW benchmark: {row['ew_mean']:.3f})")

        print("\n  Angels' best metrics (highest percentile rank):")
        for row in sorted_gaps[-3:]:
            if row["percentile"] is not np.nan:
                print(f"    {row['metric']}: {row['percentile']:.0f}th percentile "
                      f"(Angels: {row['angels_mean']:.3f}, EW benchmark: {row['ew_mean']:.3f})")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
