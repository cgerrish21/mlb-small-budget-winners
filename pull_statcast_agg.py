"""
Pull pre-aggregated Statcast metrics (exit velo, barrel rate, etc.)
at the player level, then aggregate to team level using bWAR team mappings.

Output: data/statcast_batting_agg.csv, data/statcast_pitching_agg.csv
"""

import os
import time
import pandas as pd
import numpy as np
from pybaseball import (
    statcast_batter_exitvelo_barrels,
    statcast_pitcher_exitvelo_barrels,
)

from config import YEARS, DATA_DIR, standardize_team


def pull_statcast_data(func, name, years):
    """Pull pre-aggregated Statcast data for multiple years."""
    frames = []
    for year in years:
        print(f"  {name} {year}...", flush=True)
        try:
            df = func(year)
            if df is not None and len(df) > 0:
                df["Season"] = year
                frames.append(df)
                print(f"    got {len(df)} players", flush=True)
            else:
                print(f"    no data", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
        time.sleep(1)

    if not frames:
        print(f"  No {name} data collected!", flush=True)
        return None

    return pd.concat(frames, ignore_index=True)


def build_player_team_map():
    """
    Build a player_id -> (team, season) mapping from bWAR data.
    Uses mlb_ID from bWAR to match Statcast player_id.
    For players on multiple teams, uses the team where they had more WAR.
    """
    bwar_bat_path = os.path.join(DATA_DIR, "bwar_batting.csv")
    bwar_pit_path = os.path.join(DATA_DIR, "bwar_pitching.csv")

    frames = []
    for path in [bwar_bat_path, bwar_pit_path]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "mlb_ID" in df.columns and "team" in df.columns:
                subset = df[["mlb_ID", "team", "year_ID", "WAR"]].copy()
                subset.columns = ["player_id", "team", "Season", "WAR"]
                subset["player_id"] = pd.to_numeric(subset["player_id"], errors="coerce")
                subset["WAR"] = pd.to_numeric(subset["WAR"], errors="coerce").fillna(0)
                frames.append(subset)

    if not frames:
        print("  WARNING: no bWAR data for player-team mapping", flush=True)
        return {}

    all_p = pd.concat(frames, ignore_index=True)
    # Normalize team codes (e.g., ATH -> OAK for relocated Athletics)
    all_p["team"] = all_p["team"].apply(
        lambda t: standardize_team(t) if isinstance(t, str) else t
    )
    # For each player-season, pick the team with highest WAR
    best = all_p.loc[all_p.groupby(["player_id", "Season"])["WAR"].idxmax()]
    mapping = {}
    for _, row in best.iterrows():
        pid = row["player_id"]
        season = row["Season"]
        if pd.notna(pid):
            mapping[(int(pid), int(season))] = row["team"]

    print(f"  Built player-team map: {len(mapping)} entries", flush=True)
    return mapping


def aggregate_to_team(df, player_team_map, metrics, weight_col="attempts"):
    """
    Aggregate player-level Statcast data to team level
    using the player-team mapping from bWAR.
    """
    # Map players to teams
    df = df.copy()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["team"] = df.apply(
        lambda r: player_team_map.get(
            (int(r["player_id"]), int(r["Season"])) if pd.notna(r["player_id"]) else (0, 0),
            None
        ),
        axis=1
    )

    # Drop unmapped players
    unmapped = df["team"].isna().sum()
    if unmapped > 0:
        print(f"  {unmapped} players could not be mapped to teams (dropped)", flush=True)
    df = df[df["team"].notna()]

    # Find available metrics
    available = [m for m in metrics if m in df.columns]
    if not available:
        print(f"  WARNING: none of {metrics} found in columns", flush=True)
        return None

    # Compute team-level weighted averages
    results = []
    for (team, season), group in df.groupby(["team", "Season"]):
        row = {"team": team, "Season": int(season)}
        weights = pd.to_numeric(group[weight_col], errors="coerce") if weight_col in group.columns else None

        for m in available:
            vals = pd.to_numeric(group[m], errors="coerce")
            mask = vals.notna()
            if weights is not None:
                w = weights[mask]
                v = vals[mask]
                if w.sum() > 0 and len(v) > 0:
                    row[m] = np.average(v, weights=w)
                else:
                    row[m] = np.nan
            else:
                row[m] = vals.mean()

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("Pulling pre-aggregated Statcast data...", flush=True)
    years = list(YEARS)

    # Build player-team mapping from bWAR data
    print("\nBuilding player-team mapping...", flush=True)
    player_team_map = build_player_team_map()

    if not player_team_map:
        print("ERROR: cannot map players to teams. Run pull_player_stats.py first.", flush=True)
        return

    metrics = ["avg_hit_speed", "max_hit_speed", "anglesweetspotpercent",
               "ev50", "brl_percent", "hard_hit_percent",
               "avg_hit_angle", "avg_distance", "ev95percent"]

    # Batter exit velo / barrels
    print("\n--- Batter Exit Velo & Barrels ---", flush=True)
    bat_sc = pull_statcast_data(statcast_batter_exitvelo_barrels, "batter_statcast", years)
    if bat_sc is not None:
        bat_agg = aggregate_to_team(bat_sc, player_team_map, metrics, weight_col="attempts")
        if bat_agg is not None:
            out = os.path.join(DATA_DIR, "statcast_batting_agg.csv")
            bat_agg.to_csv(out, index=False)
            print(f"  Saved {len(bat_agg)} rows to {out}", flush=True)

    # Pitcher exit velo / barrels
    print("\n--- Pitcher Exit Velo & Barrels ---", flush=True)
    pit_sc = pull_statcast_data(statcast_pitcher_exitvelo_barrels, "pitcher_statcast", years)
    if pit_sc is not None:
        pit_agg = aggregate_to_team(pit_sc, player_team_map, metrics, weight_col="attempts")
        if pit_agg is not None:
            out = os.path.join(DATA_DIR, "statcast_pitching_agg.csv")
            pit_agg.to_csv(out, index=False)
            print(f"  Saved {len(pit_agg)} rows to {out}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
