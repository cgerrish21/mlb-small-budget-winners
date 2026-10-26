"""
Main comparative analysis: cheap winners vs. cheap losers.
Computes group means, Hedges' g effect sizes (bias-corrected for small
samples), ranks metrics by discriminating power, and applies
Benjamini-Hochberg FDR correction for multiple comparisons.

Output:
  output/group_comparison.csv
  output/metric_rankings.csv
  output/correlation_matrix.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

from config import DATA_DIR, OUTPUT_DIR


def hedges_g(group1, group2):
    """
    Compute Hedges' g effect size between two groups.
    This is Cohen's d with a bias correction factor (J) for small samples.
    Returns (g, ci_lo, ci_hi) where ci is the 95% confidence interval.
    """
    g1 = group1.dropna()
    g2 = group2.dropna()
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan, np.nan
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan, np.nan, np.nan
    d = (g1.mean() - g2.mean()) / pooled_std
    # Hedges' correction factor J (approximation)
    df = n1 + n2 - 2
    j = 1 - (3 / (4 * df - 1))
    g = d * j
    # SE of Hedges' g
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * df))
    ci_lo = g - 1.96 * se
    ci_hi = g + 1.96 * se
    return g, ci_lo, ci_hi


def mann_whitney_p(group1, group2):
    """Compute Mann-Whitney U test p-value."""
    g1 = group1.dropna()
    g2 = group2.dropna()
    if len(g1) < 3 or len(g2) < 3:
        return np.nan
    try:
        _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        return p
    except Exception:
        return np.nan


def benjamini_hochberg(p_values):
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.
    Returns adjusted p-values (q-values).
    """
    p_arr = np.array(p_values, dtype=float)
    n = len(p_arr)
    valid = ~np.isnan(p_arr)
    q_values = np.full(n, np.nan)
    if valid.sum() == 0:
        return q_values

    # Sort valid p-values
    valid_idx = np.where(valid)[0]
    valid_p = p_arr[valid_idx]
    sorted_order = np.argsort(valid_p)
    sorted_p = valid_p[sorted_order]

    # BH adjustment
    m = len(sorted_p)
    adjusted = np.zeros(m)
    adjusted[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * m / (i + 1))
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original positions
    for i, orig_idx in enumerate(sorted_order):
        q_values[valid_idx[orig_idx]] = adjusted[i]

    return q_values


# Metrics to analyze.
# Format: (column_name, display_name, higher_is_better, is_redundant)
# is_redundant flags metrics that are mechanical combinations of other
# metrics in this list. They're kept for reference but excluded from
# the multiple comparison correction count to avoid inflating FWER.
ANALYSIS_METRICS = [
    # Roster construction (from bWAR)
    ("homegrown_war_pct", "Homegrown WAR %", True, False),
    ("war_gini", "WAR Gini (concentration)", False, False),
    ("players_2plus_war", "Players with 2+ WAR", True, False),
    ("top3_war_share", "Top-3 WAR Share", False, False),
    ("roster_size_positive_war", "Players with Positive WAR", True, False),
    ("weighted_age", "PA-Weighted Age", None, False),
    ("median_age", "Median Age", None, True),  # redundant with weighted_age
    ("roster_turnover_pct", "Roster Turnover %", None, False),

    # Team batting (BRef aggregated)
    ("bat_OPS", "OPS", True, True),  # OBP + SLG, mechanical
    ("bat_OBP", "OBP", True, False),
    ("bat_SLG", "SLG", True, False),
    ("bat_ISO", "ISO", True, True),  # SLG - AVG, mechanical
    ("bat_K%", "Team K%", False, False),
    ("bat_BB%", "Team BB%", True, False),
    ("bat_BABIP", "BABIP", None, False),
    ("bat_HR", "Home Runs", True, False),
    ("bat_SB", "Stolen Bases", True, False),
    ("bat_WAR", "Batting WAR (bWAR)", True, False),

    # Team pitching (BRef aggregated)
    ("pit_ERA", "Team ERA", False, False),
    ("pit_FIP", "Team FIP", False, False),
    ("pit_WHIP", "WHIP", False, True),  # (BB+H)/IP, near-mechanical
    ("pit_K/9", "K/9", True, True),  # redundant with pit_K%
    ("pit_BB/9", "BB/9", False, True),  # redundant with pit_BB%
    ("pit_HR/9", "HR/9 Allowed", False, False),
    ("pit_K%", "Pitching K%", True, False),
    ("pit_BB%", "Pitching BB%", False, False),
    ("pit_WAR", "Pitching WAR (bWAR)", True, False),

    # Statcast batting
    ("sc_bat_avg_hit_speed", "Exit Velo (batting)", True, False),
    ("sc_bat_hard_hit_percent", "Hard Hit % (batting)", True, True),  # correlated with exit velo
    ("sc_bat_brl_percent", "Barrel % (batting)", True, False),

    # Statcast pitching
    ("sc_pit_avg_hit_speed", "Exit Velo Allowed", False, False),
    ("sc_pit_hard_hit_percent", "Hard Hit % Allowed", False, True),  # correlated with exit velo
    ("sc_pit_brl_percent", "Barrel % Allowed", False, False),

    # Efficiency
    ("total_team_war", "Total Team WAR", True, False),
    ("war_per_10mm", "WAR per $10M", True, False),
    ("W", "Wins", True, True),  # near-tautological (defines the groups)
]


def main():
    # Load master dataset
    master_path = os.path.join(DATA_DIR, "master.csv")
    if not os.path.exists(master_path):
        print(f"ERROR: {master_path} not found. Run build_features.py first.", flush=True)
        sys.exit(1)

    df = pd.read_csv(master_path)
    print(f"Loaded master dataset: {len(df)} rows, {len(df.columns)} columns\n", flush=True)

    # Define comparison groups
    ew = df[df["group"] == "efficient_winner"]
    cl = df[df["group"] == "cheap_loser"]
    bw = df[df["group"] == "big_budget_winner"]
    el = df[df["group"] == "expensive_loser"]

    print(f"Group sizes:")
    print(f"  Efficient winners (low payroll, good record):  {len(ew)}")
    print(f"  Cheap losers (low payroll, bad record):        {len(cl)}")
    print(f"  Big budget winners (high payroll, good record): {len(bw)}")
    print(f"  Expensive losers (high payroll, bad record):   {len(el)}")
    print()

    # Statistical power warning
    min_group = min(len(ew), len(cl))
    if min_group < 15:
        print(f"  WARNING: Smallest group has {min_group} observations.")
        print(f"  Statistical power is limited. Effect sizes should be")
        print(f"  interpreted with caution; CI widths reflect uncertainty.\n")

    # --- Group Comparison ---
    print("=" * 100)
    print("  GROUP COMPARISON: Efficient Winners vs. Cheap Losers")
    print("  Using Hedges' g (bias-corrected for small samples)")
    print("=" * 100)

    comparison_rows = []
    ranking_rows = []
    raw_p_values = []  # for BH correction
    non_redundant_indices = []  # track which tests count for FDR

    for idx, (col, display, higher_better, is_redundant) in enumerate(ANALYSIS_METRICS):
        if col not in df.columns:
            continue

        ew_vals = pd.to_numeric(ew[col], errors="coerce")
        cl_vals = pd.to_numeric(cl[col], errors="coerce")
        bw_vals = pd.to_numeric(bw[col], errors="coerce")
        el_vals = pd.to_numeric(el[col], errors="coerce")
        all_vals = pd.to_numeric(df[col], errors="coerce")

        g, ci_lo, ci_hi = hedges_g(ew_vals, cl_vals)
        p = mann_whitney_p(ew_vals, cl_vals)

        row = {
            "metric": display,
            "column": col,
            "higher_is_better": higher_better,
            "is_redundant": is_redundant,
            "ew_mean": round(ew_vals.mean(), 3) if ew_vals.notna().sum() > 0 else np.nan,
            "ew_median": round(ew_vals.median(), 3) if ew_vals.notna().sum() > 0 else np.nan,
            "cl_mean": round(cl_vals.mean(), 3) if cl_vals.notna().sum() > 0 else np.nan,
            "cl_median": round(cl_vals.median(), 3) if cl_vals.notna().sum() > 0 else np.nan,
            "bw_mean": round(bw_vals.mean(), 3) if bw_vals.notna().sum() > 0 else np.nan,
            "el_mean": round(el_vals.mean(), 3) if el_vals.notna().sum() > 0 else np.nan,
            "all_mean": round(all_vals.mean(), 3) if all_vals.notna().sum() > 0 else np.nan,
            "hedges_g": round(g, 3) if not np.isnan(g) else np.nan,
            "g_ci_lo": round(ci_lo, 3) if not np.isnan(ci_lo) else np.nan,
            "g_ci_hi": round(ci_hi, 3) if not np.isnan(ci_hi) else np.nan,
            "p_value": round(p, 4) if not np.isnan(p) else np.nan,
            "ew_n": ew_vals.notna().sum(),
            "cl_n": cl_vals.notna().sum(),
        }
        comparison_rows.append(row)
        raw_p_values.append(p)
        if not is_redundant:
            non_redundant_indices.append(len(comparison_rows) - 1)

    # Apply Benjamini-Hochberg FDR correction on non-redundant tests only
    non_redundant_p = [raw_p_values[i] for i in non_redundant_indices]
    adjusted_p = benjamini_hochberg(non_redundant_p)
    # Map adjusted p-values back
    for j, orig_idx in enumerate(non_redundant_indices):
        comparison_rows[orig_idx]["q_value"] = (
            round(adjusted_p[j], 4) if not np.isnan(adjusted_p[j]) else np.nan
        )
    # Redundant metrics get q_value = NaN (not part of FDR correction)
    for i, row in enumerate(comparison_rows):
        if "q_value" not in row:
            row["q_value"] = np.nan

    n_independent = sum(1 for p in non_redundant_p if not np.isnan(p))
    print(f"\n  FDR correction: {n_independent} independent tests "
          f"(excluding {len(comparison_rows) - len(non_redundant_indices)} redundant metrics)\n")

    # Build ranking rows
    for row in comparison_rows:
        g_val = row["hedges_g"]
        if pd.notna(g_val):
            ranking_rows.append({
                "metric": row["metric"],
                "column": row["column"],
                "hedges_g": row["hedges_g"],
                "abs_g": round(abs(row["hedges_g"]), 3),
                "g_ci_lo": row["g_ci_lo"],
                "g_ci_hi": row["g_ci_hi"],
                "p_value": row["p_value"],
                "q_value": row["q_value"],
                "is_redundant": row["is_redundant"],
                "direction": "EW higher" if row["hedges_g"] > 0 else "CL higher",
                "ew_mean": row["ew_mean"],
                "cl_mean": row["cl_mean"],
                "gap": round(row["ew_mean"] - row["cl_mean"], 3)
                       if pd.notna(row["ew_mean"]) and pd.notna(row["cl_mean"]) else np.nan,
            })

    # Print top differentiators (non-redundant only)
    ranking_df = pd.DataFrame(ranking_rows).sort_values("abs_g", ascending=False)
    non_redundant_df = ranking_df[~ranking_df["is_redundant"]]
    print(f"\n{'Rank':<5} {'Metric':<30} {'Hedges g':>10} {'95% CI':>18} "
          f"{'EW Mean':>10} {'CL Mean':>10} {'p':>8} {'q(BH)':>8}")
    print("-" * 105)
    for i, (_, row) in enumerate(non_redundant_df.head(20).iterrows()):
        ci_str = f"[{row['g_ci_lo']:.2f}, {row['g_ci_hi']:.2f}]"
        q_str = f"{row['q_value']:.4f}" if pd.notna(row['q_value']) else "   N/A"
        sig = "*" if pd.notna(row['q_value']) and row['q_value'] < 0.10 else " "
        print(f"{i+1:<5} {row['metric']:<30} {row['hedges_g']:>10.3f} {ci_str:>18} "
              f"{row['ew_mean']:>10.3f} {row['cl_mean']:>10.3f} "
              f"{row['p_value']:>8.4f} {q_str:>8}{sig}")

    print(f"\n  * = q < 0.10 (survives FDR correction at 10% level)")

    # Save comparison
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "group_comparison.csv"), index=False)
    print(f"\nSaved group_comparison.csv ({len(comp_df)} metrics)")

    # Save rankings (all metrics, including redundant)
    ranking_df.to_csv(os.path.join(OUTPUT_DIR, "metric_rankings.csv"), index=False)
    print(f"Saved metric_rankings.csv ({len(ranking_df)} metrics)")

    # --- Correlation Matrix ---
    # Exclude redundant metrics from the correlation matrix
    print("\n\nComputing correlation matrix...", flush=True)
    non_redundant_cols = [col for col, _, _, redundant in ANALYSIS_METRICS
                          if col in df.columns and not redundant]
    corr_data = df[non_redundant_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = corr_data.corr()
    corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix.csv"))
    print(f"Saved correlation_matrix.csv ({len(non_redundant_cols)} x {len(non_redundant_cols)})")

    # --- Key findings summary ---
    print("\n" + "=" * 100)
    print("  KEY FINDINGS")
    print("=" * 100)

    # Only report non-redundant metrics that survive FDR
    sig_metrics = non_redundant_df[
        non_redundant_df["q_value"].notna() & (non_redundant_df["q_value"] < 0.10)
    ]
    top_metrics = non_redundant_df.head(8)

    if len(sig_metrics) > 0:
        print(f"\n{len(sig_metrics)} metrics survive FDR correction at q < 0.10:")
        for i, (_, row) in enumerate(sig_metrics.iterrows()):
            direction = "higher" if row["hedges_g"] > 0 else "lower"
            print(f"  {i+1}. {row['metric']}: EW is {direction} "
                  f"(g={row['hedges_g']:.2f}, "
                  f"CI=[{row['g_ci_lo']:.2f},{row['g_ci_hi']:.2f}], "
                  f"q={row['q_value']:.4f})")
    else:
        print(f"\n  No metrics survive FDR correction at q < 0.10.")
        print(f"  This likely reflects limited statistical power (n={min_group} per group).")
        print(f"  The top effect sizes below are suggestive but not statistically confirmed.")

    print(f"\nTop 8 metrics by effect size (non-redundant):")
    for i, (_, row) in enumerate(top_metrics.iterrows()):
        direction = "higher" if row["hedges_g"] > 0 else "lower"
        q_str = f"q={row['q_value']:.3f}" if pd.notna(row['q_value']) else "q=N/A"
        print(f"  {i+1}. {row['metric']}: EW is {direction} "
              f"(g={row['hedges_g']:.2f}, {q_str})")

    # Efficient winner profile
    print(f"\n  Efficient Winner Profile (group means):")
    for _, row in top_metrics.iterrows():
        if pd.notna(row['ew_mean']) and pd.notna(row['cl_mean']):
            print(f"    {row['metric']}: {row['ew_mean']:.3f} vs CL {row['cl_mean']:.3f}")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
