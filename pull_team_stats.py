"""
Pull team-level batting and pitching stats by aggregating individual
player data from Baseball Reference (bref).

FanGraphs' legacy API is blocked by Cloudflare, so we use BRef individual
stats and aggregate to team level. We also compute FIP ourselves.

Output: data/team_batting.csv, data/team_pitching.csv
"""

import os
import time
import pandas as pd
import numpy as np
from pybaseball import batting_stats_bref, pitching_stats_bref

from config import YEARS, DATA_DIR, standardize_team

# Multi-team summary row indicators from BRef
# These rows aggregate a traded player's full-season totals and must be excluded
MULTI_TEAM_INDICATORS = {"2TM", "3TM", "4TM", "5TM", "2 Tms", "3 Tms", "4 Tms", "5 Tms"}


def pull_bref_data(func, name, years):
    """Pull individual stats from BRef for multiple years."""
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
        time.sleep(2)

    if not frames:
        print(f"  No {name} data collected!", flush=True)
        return None

    return pd.concat(frames, ignore_index=True)


def aggregate_team_batting(player_df):
    """Aggregate individual batting stats to team level."""
    df = player_df.copy()

    # Filter out non-MLB
    df = df[df["Lev"].str.contains("Maj", na=False)].copy()

    # Filter out multi-team summary rows (e.g., "2TM", "3TM")
    # These are full-season totals for traded players; we want per-team rows only
    df = df[~df["Tm"].astype(str).str.strip().isin(MULTI_TEAM_INDICATORS)].copy()

    # Use league info to disambiguate city names
    df["team"] = df.apply(
        lambda r: standardize_team(str(r["Tm"]), league=str(r.get("Lev", ""))),
        axis=1
    )

    # Convert numeric columns first
    for col in ["PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP",
                "SH", "SF", "SB", "CS", "R", "RBI", "GDP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Handle multi-team players: BRef gives comma-separated team names.
    # Split into individual rows, dividing stats equally.
    expanded = []
    for _, row in df.iterrows():
        tm = str(row["Tm"])
        if "," in tm:
            teams = [t.strip() for t in tm.split(",")]
            # Parse leagues from Lev column (e.g., "Maj-AL,Maj-NL")
            lev = str(row.get("Lev", ""))
            leagues = [l.strip() for l in lev.split(",")]
            n_teams = len(teams)
            for idx, t in enumerate(teams):
                new_row = row.copy()
                new_row["Tm"] = t
                lg = leagues[idx] if idx < len(leagues) else leagues[-1] if leagues else ""
                new_row["team"] = standardize_team(t, league=lg)
                for col in ["PA", "AB", "H", "2B", "3B", "HR", "BB", "SO",
                            "HBP", "SH", "SF", "SB", "CS", "R", "RBI", "GDP"]:
                    if col in new_row.index:
                        new_row[col] = row[col] / n_teams
                expanded.append(new_row)
        else:
            expanded.append(row)

    df = pd.DataFrame(expanded)

    # Aggregate to team-season
    team_stats = df.groupby(["team", "Season"]).agg({
        "PA": "sum", "AB": "sum", "H": "sum", "2B": "sum", "3B": "sum",
        "HR": "sum", "BB": "sum", "SO": "sum", "HBP": "sum",
        "SH": "sum", "SF": "sum", "SB": "sum", "CS": "sum",
        "R": "sum", "RBI": "sum", "GDP": "sum",
    }).reset_index()

    # Compute rate stats
    ts = team_stats
    ts["1B"] = ts["H"] - ts["2B"] - ts["3B"] - ts["HR"]
    ts["AVG"] = (ts["H"] / ts["AB"]).round(3)
    ts["OBP"] = ((ts["H"] + ts["BB"] + ts["HBP"]) /
                 (ts["AB"] + ts["BB"] + ts["HBP"] + ts["SF"])).round(3)
    ts["SLG"] = ((ts["1B"] + 2*ts["2B"] + 3*ts["3B"] + 4*ts["HR"]) / ts["AB"]).round(3)
    ts["OPS"] = (ts["OBP"] + ts["SLG"]).round(3)
    ts["ISO"] = (ts["SLG"] - ts["AVG"]).round(3)
    ts["K%"] = (ts["SO"] / ts["PA"]).round(3)
    ts["BB%"] = (ts["BB"] / ts["PA"]).round(3)
    ts["BABIP"] = ((ts["H"] - ts["HR"]) /
                   (ts["AB"] - ts["SO"] - ts["HR"] + ts["SF"])).round(3)

    return ts


def aggregate_team_pitching(player_df):
    """Aggregate individual pitching stats to team level."""
    df = player_df.copy()
    df = df[df["Lev"].str.contains("Maj", na=False)].copy()

    # Filter out multi-team summary rows
    df = df[~df["Tm"].astype(str).str.strip().isin(MULTI_TEAM_INDICATORS)].copy()

    # Use league info to disambiguate city names
    df["team"] = df.apply(
        lambda r: standardize_team(str(r["Tm"]), league=str(r.get("Lev", ""))),
        axis=1
    )

    # Convert IP (handles fractional innings like 6.1 = 6 1/3)
    def parse_ip(ip_str):
        try:
            ip = float(ip_str)
            whole = int(ip)
            frac = ip - whole
            return whole + (frac * 10 / 3)
        except (ValueError, TypeError):
            return 0

    for col in ["IP", "H", "R", "ER", "BB", "SO", "HR", "HBP", "BF"]:
        if col in df.columns:
            if col == "IP":
                df[col] = df[col].apply(parse_ip)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Handle multi-team pitchers (same approach as batting)
    expanded = []
    for _, row in df.iterrows():
        tm = str(row["Tm"])
        if "," in tm:
            teams = [t.strip() for t in tm.split(",")]
            lev = str(row.get("Lev", ""))
            leagues = [l.strip() for l in lev.split(",")]
            n_teams = len(teams)
            for idx, t in enumerate(teams):
                new_row = row.copy()
                new_row["Tm"] = t
                lg = leagues[idx] if idx < len(leagues) else leagues[-1] if leagues else ""
                new_row["team"] = standardize_team(t, league=lg)
                for col in ["IP", "H", "R", "ER", "BB", "SO", "HR", "HBP", "BF"]:
                    if col in new_row.index:
                        new_row[col] = row[col] / n_teams
                expanded.append(new_row)
        else:
            expanded.append(row)

    df = pd.DataFrame(expanded)

    team_stats = df.groupby(["team", "Season"]).agg({
        "IP": "sum", "H": "sum", "R": "sum", "ER": "sum",
        "BB": "sum", "SO": "sum", "HR": "sum", "HBP": "sum", "BF": "sum",
    }).reset_index()

    ts = team_stats
    ts["ERA"] = (9 * ts["ER"] / ts["IP"]).round(2)
    ts["WHIP"] = ((ts["BB"] + ts["H"]) / ts["IP"]).round(3)
    ts["K/9"] = (9 * ts["SO"] / ts["IP"]).round(2)
    ts["BB/9"] = (9 * ts["BB"] / ts["IP"]).round(2)
    ts["HR/9"] = (9 * ts["HR"] / ts["IP"]).round(2)
    ts["K%"] = (ts["SO"] / ts["BF"]).round(3)
    ts["BB%"] = (ts["BB"] / ts["BF"]).round(3)
    ts["H/9"] = (9 * ts["H"] / ts["IP"]).round(2)

    # FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + cFIP
    # Compute year-specific FIP constant: lgERA - lgFIP_component
    for season in ts["Season"].unique():
        mask = ts["Season"] == season
        season_data = ts[mask]
        lg_ip = season_data["IP"].sum()
        if lg_ip > 0:
            lg_era = 9 * season_data["ER"].sum() / lg_ip
            lg_fip_component = (13 * season_data["HR"].sum() +
                                3 * (season_data["BB"].sum() + season_data["HBP"].sum()) -
                                2 * season_data["SO"].sum()) / lg_ip
            cfip = lg_era - lg_fip_component
        else:
            cfip = 3.10  # fallback
        ts.loc[mask, "FIP"] = ((13*season_data["HR"] + 3*(season_data["BB"]+season_data["HBP"]) -
                                2*season_data["SO"]) / season_data["IP"] + cfip).round(2)

    return ts


def main():
    print("Pulling team stats via Baseball Reference individual data...", flush=True)
    years = list(YEARS)

    # Player batting
    print("\n--- Individual Batting (BRef) ---", flush=True)
    bat_raw = pull_bref_data(batting_stats_bref, "batting", years)
    if bat_raw is not None:
        print(f"\n  Aggregating to team level...", flush=True)
        team_bat = aggregate_team_batting(bat_raw)
        out = os.path.join(DATA_DIR, "team_batting.csv")
        team_bat.to_csv(out, index=False)
        print(f"  Saved {len(team_bat)} rows to {out}", flush=True)

        # Also save individual batting for homegrown/WAR calculations
        bat_raw["team"] = bat_raw["Tm"].apply(standardize_team)
        out2 = os.path.join(DATA_DIR, "player_batting.csv")
        bat_raw.to_csv(out2, index=False)
        print(f"  Saved {len(bat_raw)} player batting rows to {out2}", flush=True)

    # Player pitching
    print("\n--- Individual Pitching (BRef) ---", flush=True)
    pit_raw = pull_bref_data(pitching_stats_bref, "pitching", years)
    if pit_raw is not None:
        print(f"\n  Aggregating to team level...", flush=True)
        team_pit = aggregate_team_pitching(pit_raw)
        out = os.path.join(DATA_DIR, "team_pitching.csv")
        team_pit.to_csv(out, index=False)
        print(f"  Saved {len(team_pit)} rows to {out}", flush=True)

        # Also save individual pitching
        pit_raw["team"] = pit_raw["Tm"].apply(standardize_team)
        out2 = os.path.join(DATA_DIR, "player_pitching.csv")
        pit_raw.to_csv(out2, index=False)
        print(f"  Saved {len(pit_raw)} player pitching rows to {out2}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
