"""
deep_dive_sticky.py
Which reliever Statcast metrics are "sticky" (predictive year-over-year)?

For each metric, we compute:
  1. Same-year correlation with WAR (r_same)
  2. Year-over-year autocorrelation of the metric itself (r_sticky) --
     does a pitcher's year-N value predict their year-N+1 value?
  3. Predictive correlation: year-N metric -> year-N+1 WAR (r_pred)

A useful scouting metric should score high on both #2 (sticky/repeatable)
AND #3 (actually predicts future WAR). A metric that correlates with
same-year WAR but isn't sticky is just describing what already happened.
"""

import os, warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

from pybaseball import (
    statcast_pitcher_arsenal_stats,
    statcast_pitcher_pitch_arsenal,
)

import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
YEARS = list(range(2022, 2027))
RELIEVER_GS_FRAC = 0.15
CACHE_FILE = os.path.join(DATA_DIR, "reliever_statcast_full_cache.csv")


# ── Load bWAR and identify relievers ────────────────────────────────
def load_relievers():
    df = pd.read_csv(os.path.join(DATA_DIR, "bwar_pitching.csv"))
    df = df[df["year_ID"].isin(YEARS)].copy()
    df["mlb_ID"] = pd.to_numeric(df["mlb_ID"], errors="coerce")
    df = df.dropna(subset=["mlb_ID"])
    df["mlb_ID"] = df["mlb_ID"].astype(int)
    df["G"] = pd.to_numeric(df["G"], errors="coerce").fillna(0)
    df["GS"] = pd.to_numeric(df["GS"], errors="coerce").fillna(0)
    df["WAR"] = pd.to_numeric(df["WAR"], errors="coerce").fillna(0)
    mask = (
        (df["G"] >= 15)
        & ((df["GS"] == 0) | (df["GS"] / df["G"] < RELIEVER_GS_FRAC))
    )
    rel = df[mask].copy()

    # Aggregate multi-stint rows (traded relievers have separate rows per team).
    # Sum WAR and G across stints; keep first team and name.
    rel = (
        rel.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg({
            "name_common": "first",
            "team": "first",
            "WAR": "sum",
            "G": "sum",
            "GS": "sum",
        })
    )
    return rel[["mlb_ID", "year_ID", "name_common", "team", "WAR", "G", "GS"]].copy()


# ── Pull or load Statcast ───────────────────────────────────────────
def pull_all_statcast():
    """Pull arsenal stats, velocity, spin, movement for all years."""
    if os.path.exists(CACHE_FILE):
        print(f"  Loading cache: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE)

    print("  Pulling from Statcast API (will cache when done)...")

    # Arsenal stats (whiff%, K%, wOBA, hard hit, etc.)
    arsenal_frames = []
    for y in YEARS:
        try:
            df = statcast_pitcher_arsenal_stats(y, minPA=10)
            df["Season"] = y
            arsenal_frames.append(df)
            print(f"    Arsenal {y}: {len(df)} rows")
        except Exception as e:
            print(f"    Arsenal {y}: FAILED - {e}")
        time.sleep(1.5)

    arsenal = pd.concat(arsenal_frames, ignore_index=True) if arsenal_frames else pd.DataFrame()

    # Velocity by pitch type
    velo_frames = []
    for y in YEARS:
        try:
            df = statcast_pitcher_pitch_arsenal(y, minP=50, arsenal_type="avg_speed")
            df["Season"] = y
            velo_frames.append(df)
            print(f"    Velocity {y}: {len(df)} rows")
        except Exception as e:
            print(f"    Velocity {y}: FAILED - {e}")
        time.sleep(1.5)

    velocity = pd.concat(velo_frames, ignore_index=True) if velo_frames else pd.DataFrame()

    # Spin by pitch type
    spin_frames = []
    for y in YEARS:
        try:
            df = statcast_pitcher_pitch_arsenal(y, minP=50, arsenal_type="avg_spin")
            df["Season"] = y
            spin_frames.append(df)
            print(f"    Spin {y}: {len(df)} rows")
        except Exception as e:
            print(f"    Spin {y}: FAILED - {e}")
        time.sleep(1.5)

    spin = pd.concat(spin_frames, ignore_index=True) if spin_frames else pd.DataFrame()

    # Build per-pitcher-year aggregated dataset
    result = build_pitcher_metrics(arsenal, velocity, spin)
    result.to_csv(CACHE_FILE, index=False)
    print(f"  Cached to {CACHE_FILE} ({len(result)} rows)")
    return result


def weighted_mean(group, col, weight_col="pitches"):
    d = group[[col, weight_col]].dropna()
    if d[weight_col].sum() == 0:
        return np.nan
    return np.average(d[col], weights=d[weight_col])


def build_pitcher_metrics(arsenal, velocity, spin):
    """Aggregate per-pitch-type Statcast into per-pitcher-year rows."""
    if arsenal.empty:
        return pd.DataFrame()

    # Aggregate arsenal stats weighted by pitches thrown
    agg_cols = ["whiff_percent", "woba", "hard_hit_percent", "k_percent",
                "put_away", "run_value_per_100", "ba", "slg", "est_ba",
                "est_slg", "est_woba"]

    arsenal_agg = (
        arsenal.groupby(["player_id", "Season"])
        .apply(
            lambda g: pd.Series({
                "total_pitches": g["pitches"].sum(),
                "whiff_pct": weighted_mean(g, "whiff_percent"),
                "woba_against": weighted_mean(g, "woba"),
                "hard_hit_pct": weighted_mean(g, "hard_hit_percent"),
                "k_pct": weighted_mean(g, "k_percent"),
                "put_away_pct": weighted_mean(g, "put_away"),
                "rv_per100": weighted_mean(g, "run_value_per_100"),
                "ba_against": weighted_mean(g, "ba"),
                "slg_against": weighted_mean(g, "slg"),
                "est_ba": weighted_mean(g, "est_ba") if "est_ba" in g.columns else np.nan,
                "est_slg": weighted_mean(g, "est_slg") if "est_slg" in g.columns else np.nan,
                "est_woba": weighted_mean(g, "est_woba"),
                "n_pitch_types": g["pitch_type"].nunique(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Pitch mix percentages
    pitch_type_map = {
        "FF": "ff", "SI": "si", "FC": "fc", "SL": "sl",
        "CH": "ch", "CU": "cu", "FS": "fs", "ST": "st",
        "SV": "sv", "KC": "kc",
    }

    pitch_usage = (
        arsenal[arsenal["pitch_type"].isin(pitch_type_map.keys())]
        .assign(pt=lambda d: d["pitch_type"].map(pitch_type_map))
    )

    pitch_mix = (
        pitch_usage.groupby(["player_id", "Season"])
        .apply(
            lambda g: pd.Series({
                f"pct_{v}": g.loc[g["pt"] == v, "pitch_usage"].sum()
                for v in pitch_type_map.values()
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Derived mix categories
    pitch_mix["pct_fastball"] = pitch_mix.get("pct_ff", pd.Series(0)).fillna(0) + pitch_mix.get("pct_si", pd.Series(0)).fillna(0)
    pitch_mix["pct_breaking"] = (
        pitch_mix.get("pct_sl", pd.Series(0)).fillna(0)
        + pitch_mix.get("pct_cu", pd.Series(0)).fillna(0)
        + pitch_mix.get("pct_st", pd.Series(0)).fillna(0)
        + pitch_mix.get("pct_sv", pd.Series(0)).fillna(0)
        + pitch_mix.get("pct_kc", pd.Series(0)).fillna(0)
    )
    pitch_mix["pct_offspeed"] = pitch_mix.get("pct_ch", pd.Series(0)).fillna(0) + pitch_mix.get("pct_fs", pd.Series(0)).fillna(0)

    # Per-pitch-type whiff rates (not aggregated, per pitch type)
    pitch_whiff = (
        arsenal[arsenal["pitch_type"].isin(pitch_type_map.keys())]
        .assign(pt=lambda d: d["pitch_type"].map(pitch_type_map))
        .groupby(["player_id", "Season"])
        .apply(
            lambda g: pd.Series({
                f"whiff_{v}": g.loc[g["pt"] == v, "whiff_percent"].values[0]
                if len(g.loc[g["pt"] == v]) > 0 else np.nan
                for v in pitch_type_map.values()
            }),
            include_groups=False,
        )
        .reset_index()
    )

    # Velocity
    if not velocity.empty:
        vel = velocity.rename(columns={"pitcher": "player_id"}).copy()
        vel_cols = ["player_id", "Season"] + [c for c in vel.columns
                                               if c.endswith("_avg_speed") or c == "n_total"]
        vel = vel[vel_cols].copy()
        # Add max velocity across all pitch types
        speed_cols = [c for c in vel.columns if c.endswith("_avg_speed")]
        if speed_cols:
            vel["max_velo"] = vel[speed_cols].max(axis=1)
            # Velo differential: max pitch speed minus slowest pitch speed
            vel["velo_spread"] = vel[speed_cols].max(axis=1) - vel[speed_cols].min(axis=1)
    else:
        vel = pd.DataFrame()

    # Spin
    if not spin.empty:
        spn = spin.rename(columns={"pitcher": "player_id"}).copy()
        spn_cols = ["player_id", "Season"] + [c for c in spn.columns if c.endswith("_avg_spin")]
        spn = spn[spn_cols].copy()
    else:
        spn = pd.DataFrame()

    # Merge everything
    result = arsenal_agg.copy()
    for extra in [pitch_mix, pitch_whiff]:
        if not extra.empty:
            result = result.merge(extra, on=["player_id", "Season"], how="left")
    if not vel.empty:
        result = result.merge(vel, on=["player_id", "Season"], how="left")
    if not spn.empty:
        result = result.merge(spn, on=["player_id", "Season"], how="left")

    return result


# ── Analysis ────────────────────────────────────────────────────────
def run_stickiness_analysis(merged):
    """
    For every numeric metric, compute:
      r_same:   correlation with same-year WAR
      r_sticky: year-N metric -> year-N+1 metric (autocorrelation)
      r_pred:   year-N metric -> year-N+1 WAR
    """
    print("\n" + "=" * 90)
    print("  STICKINESS ANALYSIS: WHICH METRICS PREDICT FUTURE RELIEVER WAR?")
    print("=" * 90)

    # Numeric columns to test (exclude IDs/names/year)
    skip = {"player_id", "Season", "mlb_ID", "year_ID", "name_common", "team",
            "WAR", "G", "GS", "IPouts", "group", "total_pitches", "n_total"}
    metric_cols = [c for c in merged.columns
                   if c not in skip and merged[c].dtype in [np.float64, np.int64, float, int]]

    # Build year-over-year pairs
    pairs = merged.merge(
        merged,
        left_on=["player_id"],
        right_on=["player_id"],
        suffixes=("_n", "_n1"),
    )
    pairs = pairs[pairs["Season_n1"] == pairs["Season_n"] + 1].copy()
    print(f"\n  Year-over-year pitcher-pairs: {len(pairs)}")
    print(f"  Unique pitchers with consecutive seasons: {pairs['player_id'].nunique()}")

    rows = []
    for col in metric_cols:
        col_n = f"{col}_n"
        col_n1 = f"{col}_n1"

        # 1. Same-year: metric vs WAR (using year N data)
        sub_same = merged[["WAR", col]].dropna()
        if len(sub_same) >= 30:
            r_same, p_same = scipy_stats.pearsonr(sub_same["WAR"], sub_same[col])
        else:
            r_same, p_same = np.nan, np.nan

        # 2. Stickiness: year-N metric vs year-N+1 metric
        if col_n in pairs.columns and col_n1 in pairs.columns:
            sub_sticky = pairs[[col_n, col_n1]].dropna()
            if len(sub_sticky) >= 30:
                r_sticky, p_sticky = scipy_stats.pearsonr(sub_sticky[col_n], sub_sticky[col_n1])
            else:
                r_sticky, p_sticky = np.nan, np.nan
        else:
            r_sticky, p_sticky = np.nan, np.nan

        # 3. Predictive: year-N metric vs year-N+1 WAR
        war_n1 = "WAR_n1"
        if col_n in pairs.columns and war_n1 in pairs.columns:
            sub_pred = pairs[[col_n, war_n1]].dropna()
            if len(sub_pred) >= 30:
                r_pred, p_pred = scipy_stats.pearsonr(sub_pred[col_n], sub_pred[war_n1])
            else:
                r_pred, p_pred = np.nan, np.nan
        else:
            r_pred, p_pred = np.nan, np.nan

        # Composite score: geometric mean of |r_sticky| and |r_pred|
        if not np.isnan(r_sticky) and not np.isnan(r_pred):
            composite = np.sqrt(abs(r_sticky) * abs(r_pred))
            # Keep the sign of r_pred for direction
            if r_pred < 0:
                composite = -composite
        else:
            composite = np.nan

        rows.append({
            "metric": col,
            "r_same": round(r_same, 3) if not np.isnan(r_same) else np.nan,
            "p_same": round(p_same, 4) if not np.isnan(p_same) else np.nan,
            "r_sticky": round(r_sticky, 3) if not np.isnan(r_sticky) else np.nan,
            "p_sticky": round(p_sticky, 4) if not np.isnan(p_sticky) else np.nan,
            "r_pred": round(r_pred, 3) if not np.isnan(r_pred) else np.nan,
            "p_pred": round(p_pred, 4) if not np.isnan(p_pred) else np.nan,
            "composite": round(composite, 3) if not np.isnan(composite) else np.nan,
            "n_same": len(sub_same),
            "n_pairs": len(sub_pred) if col_n in pairs.columns else 0,
        })

    results = pd.DataFrame(rows).dropna(subset=["r_same"])

    # Sort by composite score (most useful for scouting)
    results = results.sort_values("composite", key=abs, ascending=False)

    # Print results
    print("\n  FULL RESULTS (sorted by |composite| = sqrt(|r_sticky| * |r_pred|)):\n")
    print("  r_same   = same-year correlation with WAR")
    print("  r_sticky = year-over-year autocorrelation (is the metric repeatable?)")
    print("  r_pred   = year-N metric -> year-N+1 WAR (does it predict future WAR?)")
    print("  composite = sqrt(|r_sticky| * |r_pred|) -- useful scouting metrics score high on both\n")

    display = results[["metric", "r_same", "r_sticky", "r_pred", "composite", "n_same", "n_pairs"]].copy()
    print(display.to_string(index=False))

    # Highlight the best scouting metrics
    print("\n" + "-" * 90)
    print("  TOP 10 MOST USEFUL SCOUTING METRICS (highest |composite|):")
    print("-" * 90)
    top10 = results.head(10)
    for _, row in top10.iterrows():
        direction = "higher = better WAR" if row["r_pred"] > 0 else "lower = better WAR"
        sticky_label = "very sticky" if abs(row["r_sticky"]) > 0.5 else "moderately sticky" if abs(row["r_sticky"]) > 0.3 else "weakly sticky"
        print(f"\n  {row['metric']}")
        print(f"    Same-year r={row['r_same']:.3f} | Sticky r={row['r_sticky']:.3f} ({sticky_label}) | Predictive r={row['r_pred']:.3f} ({direction})")
        print(f"    Composite: {row['composite']:.3f} | n={row['n_pairs']} pairs")

    # Save to CSV
    outpath = os.path.join(OUTPUT_DIR, "reliever_metric_stickiness.csv")
    results.to_csv(outpath, index=False)
    print(f"\n  Saved to {outpath}")

    return results


def compare_sticky_vs_not(merged, results):
    """Show the practical difference: use sticky metrics to identify future good relievers."""
    print("\n" + "=" * 90)
    print("  PRACTICAL TEST: CAN STICKY METRICS IDENTIFY NEXT YEAR'S GOOD RELIEVERS?")
    print("=" * 90)

    # Build year-over-year pairs
    pairs = merged.merge(
        merged,
        left_on=["player_id"],
        right_on=["player_id"],
        suffixes=("_n", "_n1"),
    )
    pairs = pairs[pairs["Season_n1"] == pairs["Season_n"] + 1].copy()

    # Get top 3 sticky+predictive metrics
    top_metrics = results.head(5)["metric"].tolist()
    # Filter to metrics that exist as _n columns
    top_metrics = [m for m in top_metrics if f"{m}_n" in pairs.columns][:3]

    if len(top_metrics) < 2:
        print("  Not enough valid metrics for this analysis.")
        return

    print(f"\n  Using top sticky metrics: {top_metrics}")

    for metric in top_metrics:
        col_n = f"{metric}_n"
        war_n1 = "WAR_n1"

        sub = pairs[[col_n, war_n1, "name_common_n", "Season_n"]].dropna()
        if len(sub) < 50:
            continue

        # Quintile analysis
        sub["quintile"] = pd.qcut(sub[col_n], 5, labels=["Q1 (worst)", "Q2", "Q3", "Q4", "Q5 (best)"])

        # For metrics where lower is better (woba, ba, hard hit, etc.), reverse quintile labels
        r_sign = results.loc[results["metric"] == metric, "r_pred"].values[0]
        if r_sign < 0:
            sub["quintile"] = pd.qcut(sub[col_n], 5, labels=["Q5 (best)", "Q4", "Q3", "Q2", "Q1 (worst)"])

        quint = sub.groupby("quintile", observed=True).agg(
            next_yr_WAR=("WAR_n1", "mean"),
            n=(war_n1, "count"),
        ).round(3)

        print(f"\n  {metric}: Year-N quintile -> Year-N+1 avg WAR")
        print(quint.to_string())

        # Monotonicity check
        vals = quint["next_yr_WAR"].values
        monotonic = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        print(f"  Monotonic trend? {'Yes' if monotonic else 'No'}")


def main():
    print("=" * 90)
    print("  RELIEVER METRIC STICKINESS ANALYSIS")
    print("  Which Statcast metrics are repeatable AND predict future reliever WAR?")
    print("=" * 90)

    # Load relievers
    print("\n[1/3] Loading relievers...")
    relievers = load_relievers()
    print(f"  Relievers: {len(relievers)} pitcher-seasons, {relievers['mlb_ID'].nunique()} unique pitchers")

    # Pull Statcast
    print("\n[2/3] Loading Statcast data...")
    statcast = pull_all_statcast()
    print(f"  Statcast rows: {len(statcast)}")

    # Merge
    print("\n[3/3] Merging...")
    merged = relievers.merge(
        statcast,
        left_on=["mlb_ID", "year_ID"],
        right_on=["player_id", "Season"],
        how="inner",
    )
    print(f"  Merged reliever-Statcast rows: {len(merged)}")
    print(f"  Unique relievers: {merged['mlb_ID'].nunique()}")
    print(f"  Seasons: {sorted(merged['Season'].unique())}")

    # Rename for consistency
    if "Season" not in merged.columns and "year_ID" in merged.columns:
        merged["Season"] = merged["year_ID"]

    # Run stickiness analysis
    results = run_stickiness_analysis(merged)

    # Practical test
    compare_sticky_vs_not(merged, results)

    # Survivorship bias caveat
    print("\n" + "=" * 90)
    print("  METHODOLOGICAL NOTE: SURVIVORSHIP BIAS")
    print("=" * 90)
    print("""
  The year-over-year analysis only includes relievers who appeared in BOTH
  year N and year N+1 (i.e., who stayed in the majors). Pitchers who were
  bad in year N and got cut or demoted are excluded from the predictive
  correlations. This likely UNDERSTATES the true predictive power of K%,
  whiff%, and put-away%, because the missing data points are
  disproportionately low-skill pitchers who produced zero WAR the next year.

  The reported r_pred values should be treated as conservative lower bounds.
""")

    print("\nDone.")


if __name__ == "__main__":
    main()
