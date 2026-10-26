"""
Feature engineering: merge all data sources and compute derived features.
Input: all CSVs from pull scripts + payroll
Output: data/master.csv — one row per team-season (150 rows)
"""

import os
import sys
import pandas as pd
import numpy as np

from config import (
    YEARS, DATA_DIR, OUTPUT_DIR, ALL_TEAMS, PREMIUM_POSITIONS,
    classify_team, standardize_team,
)


def load_csv(name):
    """Load a CSV from the data directory, return None if missing."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {name} not found", flush=True)
        return None
    df = pd.read_csv(path)
    print(f"  Loaded {name}: {len(df)} rows, {len(df.columns)} cols", flush=True)
    return df


def gini_coefficient(values):
    """
    Compute Gini coefficient of a list of values.
    Handles negative values by shifting all values up so the minimum is 0.
    This preserves the relative spread while making Gini computable.
    """
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    # Shift negative values: Gini requires non-negative inputs
    # Shifting preserves relative dispersion
    if values.min() < 0:
        values = values - values.min()
    values = np.sort(values)
    n = len(values)
    total = np.sum(values)
    if total == 0:
        return 0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * total) / (n * total)


def compute_homegrown_war(bwar_bat, bwar_pit, years):
    """
    For each player, find their debut team (earliest appearance in full
    bWAR history, not just analysis years). Homegrown = debut team
    matches current team.

    Uses full historical bWAR data so veterans who debuted before the
    analysis window are correctly attributed to their actual debut team.
    For players with multiple stints in the same earliest year, uses the
    first stint (lowest stint_ID if available, or first row).
    """
    frames = []
    for df, label in [(bwar_bat, "bat"), (bwar_pit, "pit")]:
        if df is None:
            continue
        cols = ["team", "year_ID", "player_ID", "WAR"]
        # Use stint_ID for ordering within a season if available
        if "stint_ID" in df.columns:
            cols.append("stint_ID")
        subset = df[cols].copy()
        rename = {"year_ID": "Season", "player_ID": "player_id"}
        subset = subset.rename(columns=rename)
        subset["WAR"] = pd.to_numeric(subset["WAR"], errors="coerce").fillna(0)
        if "stint_ID" in subset.columns:
            subset["stint_ID"] = pd.to_numeric(subset["stint_ID"], errors="coerce").fillna(1)
        else:
            subset["stint_ID"] = 1
        frames.append(subset)

    if not frames:
        return None

    all_players = pd.concat(frames, ignore_index=True)

    # Find each player's debut team using full history:
    # Sort by season then stint_ID, take the first row per player
    debut = (all_players.sort_values(["Season", "stint_ID"])
             .groupby("player_id")
             .first()
             .reset_index()[["player_id", "team"]])
    debut.columns = ["player_id", "debut_team"]

    # Only merge analysis years for the output
    analysis = all_players[all_players["Season"].isin(years)].copy()
    analysis = analysis.merge(debut, on="player_id", how="left")
    analysis["is_homegrown"] = analysis["team"] == analysis["debut_team"]

    results = []
    for (team, season), group in analysis.groupby(["team", "Season"]):
        total_war = group["WAR"].sum()
        homegrown_war = group[group["is_homegrown"]]["WAR"].sum()
        pct = homegrown_war / total_war if total_war > 0 else np.nan
        # Report raw pct (can exceed 1.0 when non-homegrown WAR is negative;
        # this is informative, not a bug)
        results.append({
            "team": team,
            "Season": season,
            "homegrown_war": round(homegrown_war, 1),
            "total_player_war": round(total_war, 1),
            "homegrown_war_pct": round(pct, 3) if not np.isnan(pct) else np.nan,
        })

    return pd.DataFrame(results)


def compute_war_distribution(bwar_bat, bwar_pit, years):
    """
    WAR distribution metrics:
    - Gini coefficient
    - Count of 2+ WAR players
    - Top-3 WAR share
    """
    frames = []
    for df in [bwar_bat, bwar_pit]:
        if df is None:
            continue
        subset = df[["team", "year_ID", "player_ID", "WAR"]].copy()
        subset.columns = ["team", "Season", "player_id", "WAR"]
        subset["WAR"] = pd.to_numeric(subset["WAR"], errors="coerce").fillna(0)
        frames.append(subset)

    if not frames:
        return None

    all_players = pd.concat(frames, ignore_index=True)
    # Sum WAR by player-team-season (handle multiple stints)
    agg = (all_players[all_players["Season"].isin(years)]
           .groupby(["team", "Season", "player_id"])["WAR"]
           .sum()
           .reset_index())

    results = []
    for (team, season), group in agg.groupby(["team", "Season"]):
        wars = group["WAR"].values
        total = wars.sum()
        n_above_2 = (wars >= 2.0).sum()
        top3 = np.sort(wars)[-3:].sum() if len(wars) >= 3 else wars.sum()
        top3_share = top3 / total if total > 0 else np.nan
        # Cap at 1.0 (can exceed when bottom players have negative WAR)
        if not np.isnan(top3_share):
            top3_share = min(top3_share, 1.0)
        # Use ALL WAR values for Gini (including negative)
        # Gini function handles negative values via shifting
        gini = gini_coefficient(wars) if len(wars) > 1 else np.nan
        pos_wars = wars[wars > 0]

        results.append({
            "team": team,
            "Season": season,
            "war_gini": round(gini, 3) if not np.isnan(gini) else np.nan,
            "players_2plus_war": int(n_above_2),
            "top3_war_share": round(top3_share, 3) if not np.isnan(top3_share) else np.nan,
            "total_roster_war": round(total, 1),
            "roster_size_positive_war": int(len(pos_wars)),
        })

    return pd.DataFrame(results)


def compute_age_profile(bwar_bat, bwar_pit, years):
    """WAR-weighted average age per team-season."""
    frames = []
    for df in [bwar_bat, bwar_pit]:
        if df is None:
            continue
        # bwar data doesn't have Age directly, but we can compute from birth year
        # or skip if not available. BRef individual data has Age.
        # For now, skip age from bwar and compute from BRef individual data.
        pass

    return None  # Will compute from BRef data below


def compute_age_from_bref(player_bat_bref, player_pit_bref, years):
    """Compute age profile from BRef individual stats."""
    # Multi-team summary row indicators to filter out
    multi_tm = {"2TM", "3TM", "4TM", "5TM", "2 Tms", "3 Tms", "4 Tms", "5 Tms"}
    frames = []
    for df in [player_bat_bref, player_pit_bref]:
        if df is None:
            continue
        if "Age" not in df.columns or "team" not in df.columns:
            continue
        if "PA" in df.columns:
            vol_col = "PA"
        elif "IP" in df.columns:
            vol_col = "IP"
        elif "G" in df.columns:
            vol_col = "G"
        else:
            continue
        subset = df[df["Season"].isin(years)].copy()
        # Filter out multi-team summary rows (team = "2TM", "3TM", etc.)
        if "Tm" in subset.columns:
            subset = subset[~subset["Tm"].astype(str).str.strip().isin(multi_tm)]
        subset = subset[["team", "Season", "Age", vol_col]].copy()
        subset["Age"] = pd.to_numeric(subset["Age"], errors="coerce")
        subset[vol_col] = pd.to_numeric(subset[vol_col], errors="coerce").fillna(0)
        subset = subset[subset["Age"].notna() & (subset[vol_col] > 0)]
        frames.append(subset.rename(columns={vol_col: "volume"}))

    if not frames:
        return None

    all_p = pd.concat(frames, ignore_index=True)
    results = []
    for (team, season), group in all_p.groupby(["team", "Season"]):
        if group["volume"].sum() > 0:
            war_weighted_age = np.average(group["Age"], weights=group["volume"])
        else:
            war_weighted_age = group["Age"].mean()
        results.append({
            "team": team,
            "Season": season,
            "weighted_age": round(war_weighted_age, 1),
            "median_age": round(group["Age"].median(), 1),
        })

    return pd.DataFrame(results)


def compute_positional_war(bwar_bat, years):
    """
    WAR at premium positions (C, SS, CF, 2B) vs. non-premium.
    bwar_bat doesn't have position directly, but we can use player_ID
    to determine primary position from the data.
    """
    # bwar_bat doesn't have a position column, so we skip this for now
    # unless we can get position data from another source.
    # The 'pitcher' column in bwar_bat tells us if they're a pitcher.
    return None


def compute_roster_turnover(bwar_bat, bwar_pit, years):
    """Year-over-year roster turnover using player IDs."""
    frames = []
    for df in [bwar_bat, bwar_pit]:
        if df is None:
            continue
        subset = df[["team", "year_ID", "player_ID", "PA" if "PA" in df.columns else "G"]].copy()
        vol_col = "PA" if "PA" in df.columns else "G"
        subset.columns = ["team", "Season", "player_id", "volume"]
        subset["volume"] = pd.to_numeric(subset["volume"], errors="coerce").fillna(1)
        frames.append(subset)

    if not frames:
        return None

    all_players = pd.concat(frames, ignore_index=True)

    results = []
    for year in years:
        if year == min(years):
            continue
        prev_year = year - 1

        for team in ALL_TEAMS:
            curr = all_players[(all_players["team"] == team) & (all_players["Season"] == year)]
            prev = all_players[(all_players["team"] == team) & (all_players["Season"] == prev_year)]

            if len(curr) == 0:
                continue

            prev_players = set(prev["player_id"].unique())
            curr_copy = curr.copy()
            curr_copy["is_new"] = ~curr_copy["player_id"].isin(prev_players)
            total_vol = curr_copy["volume"].sum()
            new_vol = curr_copy[curr_copy["is_new"]]["volume"].sum()
            turnover = new_vol / total_vol if total_vol > 0 else np.nan

            results.append({
                "team": team,
                "Season": year,
                "roster_turnover_pct": round(turnover, 3) if not np.isnan(turnover) else np.nan,
                "new_players": int(curr_copy["is_new"].sum()),
                "total_players": len(curr_copy),
            })

    return pd.DataFrame(results)


def compute_team_bwar(bwar_bat, bwar_pit, years):
    """Sum bWAR by team-season for batting and pitching."""
    results = []

    for year in years:
        for team in ALL_TEAMS:
            bat_war = 0
            pit_war = 0

            if bwar_bat is not None:
                mask = (bwar_bat["team"] == team) & (bwar_bat["year_ID"] == year)
                bat_war = bwar_bat.loc[mask, "WAR"].sum()

            if bwar_pit is not None:
                mask = (bwar_pit["team"] == team) & (bwar_pit["year_ID"] == year)
                pit_war = bwar_pit.loc[mask, "WAR"].sum()

            results.append({
                "team": team,
                "Season": year,
                "bat_WAR": round(bat_war, 1),
                "pit_WAR": round(pit_war, 1),
                "total_team_war": round(bat_war + pit_war, 1),
            })

    return pd.DataFrame(results)


def main():
    print("Building master feature dataset...\n", flush=True)

    # Load all data sources
    print("Loading data...", flush=True)
    team_bat = load_csv("team_batting.csv")
    team_pit = load_csv("team_pitching.csv")
    bwar_bat = load_csv("bwar_batting.csv")
    bwar_pit = load_csv("bwar_pitching.csv")
    player_bat_bref = load_csv("player_batting.csv")
    player_pit_bref = load_csv("player_pitching.csv")
    sc_bat = load_csv("statcast_batting_agg.csv")
    sc_pit = load_csv("statcast_pitching_agg.csv")
    standings = load_csv("standings.csv")
    payroll = load_csv("payroll.csv")

    if standings is None or payroll is None:
        print("\nERROR: standings and payroll are required.", flush=True)
        sys.exit(1)

    # Start with standings + payroll as the base
    print("\nBuilding base (standings + payroll)...", flush=True)
    master = standings.merge(payroll, on=["team", "Season"], how="inner")
    print(f"  Base: {len(master)} team-seasons", flush=True)

    # --- Merge team batting ---
    if team_bat is not None:
        bat_cols = ["team", "Season"]
        for c in ["AVG", "OBP", "SLG", "OPS", "ISO", "K%", "BB%", "BABIP",
                   "HR", "SB", "R", "H", "PA", "SO"]:
            if c in team_bat.columns:
                bat_cols.append(c)
        tb = team_bat[bat_cols].copy()
        # Prefix to avoid collision with pitching
        rename = {}
        for c in tb.columns:
            if c not in ["team", "Season"]:
                rename[c] = f"bat_{c}"
        tb = tb.rename(columns=rename)
        master = master.merge(tb, on=["team", "Season"], how="left")
        print(f"  After team batting merge: {len(master)} rows", flush=True)

    # --- Merge team pitching ---
    if team_pit is not None:
        pit_cols = ["team", "Season"]
        for c in ["ERA", "FIP", "WHIP", "K/9", "BB/9", "HR/9", "K%", "BB%", "H/9",
                   "IP", "SO", "BB", "HR", "ER", "R"]:
            if c in team_pit.columns:
                pit_cols.append(c)
        tp = team_pit[pit_cols].copy()
        rename = {}
        for c in tp.columns:
            if c not in ["team", "Season"]:
                rename[c] = f"pit_{c}"
        tp = tp.rename(columns=rename)
        master = master.merge(tp, on=["team", "Season"], how="left")
        print(f"  After team pitching merge: {len(master)} rows", flush=True)

    # --- Merge Statcast batting ---
    if sc_bat is not None:
        rename = {c: f"sc_bat_{c}" for c in sc_bat.columns if c not in ["team", "Season"]}
        sc_bat_r = sc_bat.rename(columns=rename)
        master = master.merge(sc_bat_r, on=["team", "Season"], how="left")
        print(f"  After Statcast batting merge: {len(master)} rows", flush=True)

    # --- Merge Statcast pitching ---
    if sc_pit is not None:
        rename = {c: f"sc_pit_{c}" for c in sc_pit.columns if c not in ["team", "Season"]}
        sc_pit_r = sc_pit.rename(columns=rename)
        master = master.merge(sc_pit_r, on=["team", "Season"], how="left")
        print(f"  After Statcast pitching merge: {len(master)} rows", flush=True)

    # --- Team WAR from bWAR ---
    years = list(YEARS)
    print("\nComputing team WAR from bWAR...", flush=True)
    team_war = compute_team_bwar(bwar_bat, bwar_pit, years)
    if team_war is not None:
        master = master.merge(team_war, on=["team", "Season"], how="left")
        print(f"  After team WAR merge: {len(master)} rows", flush=True)

    # --- Compute derived features ---
    print("\nComputing derived features...", flush=True)

    # Homegrown WAR
    print("  Homegrown WAR...", flush=True)
    hg = compute_homegrown_war(bwar_bat, bwar_pit, years)
    if hg is not None:
        master = master.merge(hg, on=["team", "Season"], how="left")

    # WAR distribution
    print("  WAR distribution (Gini, depth)...", flush=True)
    wd = compute_war_distribution(bwar_bat, bwar_pit, years)
    if wd is not None:
        master = master.merge(wd, on=["team", "Season"], how="left")

    # Age profile (from BRef individual data)
    print("  Age profile...", flush=True)
    ap = compute_age_from_bref(player_bat_bref, player_pit_bref, years)
    if ap is not None:
        master = master.merge(ap, on=["team", "Season"], how="left")

    # Roster turnover
    print("  Roster turnover...", flush=True)
    rt = compute_roster_turnover(bwar_bat, bwar_pit, years)
    if rt is not None:
        master = master.merge(rt, on=["team", "Season"], how="left")

    # --- Compute run differential from batting R and pitching R ---
    if "bat_R" in master.columns and "pit_R" in master.columns:
        master["run_diff"] = (pd.to_numeric(master["bat_R"], errors="coerce") -
                              pd.to_numeric(master["pit_R"], errors="coerce"))
        print(f"  Computed run_diff from batting R - pitching R", flush=True)

    # --- Classify teams ---
    print("\nClassifying teams...", flush=True)
    master["group"] = master.apply(
        lambda r: classify_team(r["team"], r["Season"], r["payroll_rank_pct"], r["W"]),
        axis=1
    )
    print(f"\n  Group counts:")
    print(master["group"].value_counts().to_string())

    # --- WAR per dollar (efficiency) ---
    if "total_team_war" in master.columns:
        master["war_per_10mm"] = (master["total_team_war"] /
                                  (master["payroll_mm"] / 10)).round(3)

    # Save
    out = os.path.join(DATA_DIR, "master.csv")
    master.to_csv(out, index=False)
    print(f"\nSaved master dataset: {len(master)} rows, {len(master.columns)} columns")
    print(f"  -> {out}", flush=True)

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Teams: {master['team'].nunique()}")
    print(f"  Seasons: {sorted(master['Season'].unique())}")
    print(f"  Total rows: {len(master)}")
    print(f"  Columns: {len(master.columns)}")

    # Show a few key stats for validation
    print(f"\n--- Validation ---")
    for team in ["MIL", "TBR", "CLE", "LAA", "MIA", "PIT"]:
        t = master[master["team"] == team]
        if len(t) > 0:
            avg_war = t["total_team_war"].mean() if "total_team_war" in t.columns else "N/A"
            avg_hg = t["homegrown_war_pct"].mean() if "homegrown_war_pct" in t.columns else "N/A"
            avg_w = t["W"].mean()
            print(f"  {team}: avg W={avg_w:.0f}, avg tWAR={avg_war:.1f}, "
                  f"avg HG%={avg_hg:.3f}" if isinstance(avg_war, float) else
                  f"  {team}: avg W={avg_w:.0f}, tWAR=N/A, HG%=N/A")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
