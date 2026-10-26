"""
Angel Stadium Park Analysis: Pitch-level Statcast comparison.

Same-pitcher controlled comparisons at Angel Stadium vs. road,
plus Angels pitchers home vs. away, and visiting pitchers at
Angel Stadium vs. their other road starts.

Uses cached 2024 Statcast data from the Coors project.
"""

import os
import sys
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
COORS_DIR = os.path.join(os.path.dirname(__file__), "..", "nba-expected-assists", "coors")

PITCH_TYPES = {
    "FF": "4-Seam Fastball",
    "SI": "Sinker",
    "SL": "Slider",
    "CU": "Curveball",
    "CH": "Changeup",
    "FC": "Cutter",
    "ST": "Sweeper",
    "FS": "Splitter",
}

SWING_EVENTS = [
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "foul_bunt", "hit_into_play", "hit_into_play_no_out",
    "hit_into_play_score", "missed_bunt",
]

METRICS = {
    "release_speed": "Velo (mph)",
    "release_spin_rate": "Spin Rate (rpm)",
    "pfx_x": "Horiz Break (in)",
    "pfx_z": "Vert Break (in)",
    "release_extension": "Extension (ft)",
}


def load_statcast():
    """Load 2024 Statcast data (from Coors cache or pull fresh)."""
    cached = os.path.join(COORS_DIR, "statcast_2024.csv")
    local = os.path.join(DATA_DIR, "statcast_2024.csv")

    if os.path.exists(local):
        print(f"Loading from {local}...", flush=True)
        return pd.read_csv(local, low_memory=False)
    elif os.path.exists(cached):
        print(f"Loading from Coors cache: {cached}...", flush=True)
        return pd.read_csv(cached, low_memory=False)
    else:
        print("No cached data found. Pulling 2024 season...", flush=True)
        from pybaseball import statcast, cache
        cache.enable()
        months = [
            ("2024-03-20", "2024-04-30"),
            ("2024-05-01", "2024-05-31"),
            ("2024-06-01", "2024-06-30"),
            ("2024-07-01", "2024-07-31"),
            ("2024-08-01", "2024-08-31"),
            ("2024-09-01", "2024-09-30"),
            ("2024-10-01", "2024-10-31"),
        ]
        frames = []
        for start, end in months:
            print(f"  {start} to {end}...", flush=True)
            try:
                chunk = statcast(start_dt=start, end_dt=end)
                if chunk is not None and len(chunk) > 0:
                    frames.append(chunk)
                    print(f"  got {len(chunk)} pitches", flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)
        if not frames:
            print("No data pulled.", flush=True)
            sys.exit(1)
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(local, index=False)
        print(f"Saved {len(df)} pitches to {local}", flush=True)
        return df


def prep(df):
    """Filter to standard pitch types, tag Angel Stadium games."""
    df = df[df["pitch_type"].isin(PITCH_TYPES.keys())].copy()
    df["pitch_name"] = df["pitch_type"].map(PITCH_TYPES)
    df["at_anaheim"] = df["home_team"] == "LAA"
    df["is_angels_pitcher"] = df["pitcher_team"] == "LAA" if "pitcher_team" in df.columns else False

    # Determine pitcher's team from home/away logic
    # In Statcast, inning_topbot: "Top" = away batting, home pitching
    #                              "Bot" = home batting, away pitching
    df["pitcher_is_home"] = df["inning_topbot"] == "Top"
    df["pitcher_team_derived"] = np.where(
        df["pitcher_is_home"], df["home_team"], df["away_team"]
    )
    df["is_angels_pitcher"] = df["pitcher_team_derived"] == "LAA"
    return df


def compute_whiff_rate(subset):
    """Compute whiff rate from a pitch-level DataFrame."""
    swings = subset[subset["description"].isin(SWING_EVENTS)]
    whiffs = subset[subset["description"].str.startswith("swinging_strike", na=False)]
    if len(swings) == 0:
        return np.nan
    return len(whiffs) / len(swings)


def compute_csw(subset):
    """Called Strike + Whiff rate."""
    called = (subset["description"] == "called_strike").sum()
    whiffs = subset["description"].str.startswith("swinging_strike", na=False).sum()
    if len(subset) == 0:
        return np.nan
    return (called + whiffs) / len(subset)


def compute_outcomes(subset):
    """Compute key outcome metrics for a pitch subset."""
    bip = subset[subset["launch_speed"].notna()]
    return {
        "whiff_rate": compute_whiff_rate(subset),
        "csw": compute_csw(subset),
        "called_strike_pct": (subset["description"] == "called_strike").mean() if len(subset) > 0 else np.nan,
        "exit_velo": bip["launch_speed"].mean() if len(bip) > 0 else np.nan,
        "launch_angle": bip["launch_angle"].mean() if len(bip) > 0 else np.nan,
        "hard_hit_pct": (bip["launch_speed"] >= 95).mean() if len(bip) > 0 else np.nan,
        "barrel_pct": (bip["barrel"].fillna(0) == 1).mean() if "barrel" in bip.columns and len(bip) > 0 else np.nan,
        "gb_pct": (bip["bb_type"] == "ground_ball").mean() if "bb_type" in bip.columns and len(bip) > 0 else np.nan,
        "fb_pct": (bip["bb_type"] == "fly_ball").mean() if "bb_type" in bip.columns and len(bip) > 0 else np.nan,
        "n_pitches": len(subset),
        "n_bip": len(bip),
    }


def controlled_comparison(df, label, filter_fn):
    """
    Same-pitcher controlled comparison.
    filter_fn(df) -> subset of pitchers to analyze.
    Compare their pitches at Angel Stadium vs. elsewhere.
    """
    subset = filter_fn(df)
    pitchers_at_ana = set(subset[subset["at_anaheim"]]["pitcher"].unique())
    pitchers_away = set(subset[~subset["at_anaheim"]]["pitcher"].unique())
    common = pitchers_at_ana & pitchers_away
    subset = subset[subset["pitcher"].isin(common)].copy()

    print(f"\n{'=' * 80}")
    print(f"  {label}")
    print(f"  {len(common)} pitchers with data at Angel Stadium AND elsewhere")
    print(f"{'=' * 80}")

    # --- Pitch characteristics ---
    print(f"\n--- PITCH CHARACTERISTICS ---")
    char_rows = []
    for pt_code, pt_name in PITCH_TYPES.items():
        pt_sub = subset[subset["pitch_type"] == pt_code]
        home = pt_sub[pt_sub["at_anaheim"]]
        road = pt_sub[~pt_sub["at_anaheim"]]
        if len(home) < 50 or len(road) < 200:
            continue

        row = {"pitch_type": pt_name, "n_home": len(home), "n_road": len(road)}
        print(f"\n  {pt_name} (n={len(home)} home, {len(road)} road)")
        for col, lbl in METRICS.items():
            h = home[col].mean()
            r = road[col].mean()
            d = h - r
            row[f"{lbl}_home"] = round(h, 2)
            row[f"{lbl}_road"] = round(r, 2)
            row[f"{lbl}_delta"] = round(d, 2)
            sign = "+" if d > 0 else ""
            print(f"    {lbl:22s}  Home: {h:8.2f}  Road: {r:8.2f}  Delta: {sign}{d:.2f}")
        char_rows.append(row)

    # --- Outcomes ---
    print(f"\n--- OUTCOMES ---")
    outcome_rows = []
    for pt_code, pt_name in PITCH_TYPES.items():
        pt_sub = subset[subset["pitch_type"] == pt_code]
        home = pt_sub[pt_sub["at_anaheim"]]
        road = pt_sub[~pt_sub["at_anaheim"]]
        if len(home) < 50:
            continue

        h_out = compute_outcomes(home)
        r_out = compute_outcomes(road)

        print(f"\n  {pt_name}")
        for key in ["whiff_rate", "csw", "called_strike_pct", "exit_velo", "launch_angle", "hard_hit_pct", "gb_pct", "fb_pct"]:
            h_val = h_out[key]
            r_val = r_out[key]
            if pd.isna(h_val) or pd.isna(r_val):
                continue
            d = h_val - r_val
            sign = "+" if d > 0 else ""
            fmt = ".1f" if key in ("exit_velo", "launch_angle") else ".3f"
            print(f"    {key:20s}  Home: {h_val:{fmt}}  Road: {r_val:{fmt}}  Delta: {sign}{d:{fmt}}")

        outcome_rows.append({
            "pitch_type": pt_name,
            **{f"{k}_home": round(h_out[k], 4) if not pd.isna(h_out[k]) else np.nan for k in h_out},
            **{f"{k}_road": round(r_out[k], 4) if not pd.isna(r_out[k]) else np.nan for k in r_out},
        })

    return char_rows, outcome_rows


def angels_vs_league(df):
    """Compare Angels pitching staff vs. league average."""
    print(f"\n{'=' * 80}")
    print("  ANGELS PITCHING STAFF vs LEAGUE AVERAGE (2024)")
    print(f"{'=' * 80}")

    for pt_code, pt_name in PITCH_TYPES.items():
        pt_sub = df[df["pitch_type"] == pt_code]
        angels = pt_sub[pt_sub["is_angels_pitcher"]]
        league = pt_sub[~pt_sub["is_angels_pitcher"]]
        if len(angels) < 100:
            continue

        print(f"\n  {pt_name} (Angels: {len(angels)}, League: {len(league)})")
        for col, lbl in METRICS.items():
            a = angels[col].mean()
            l = league[col].mean()
            d = a - l
            sign = "+" if d > 0 else ""
            print(f"    {lbl:22s}  Angels: {a:8.2f}  League: {l:8.2f}  Delta: {sign}{d:.2f}")

        a_out = compute_outcomes(angels)
        l_out = compute_outcomes(league)
        for key in ["whiff_rate", "csw", "exit_velo", "hard_hit_pct", "gb_pct", "barrel_pct"]:
            a_val = a_out[key]
            l_val = l_out[key]
            if pd.isna(a_val) or pd.isna(l_val):
                continue
            d = a_val - l_val
            sign = "+" if d > 0 else ""
            fmt = ".1f" if key == "exit_velo" else ".3f"
            print(f"    {key:22s}  Angels: {a_val:{fmt}}  League: {l_val:{fmt}}  Delta: {sign}{d:{fmt}}")


def pitch_mix_comparison(df):
    """Compare Angels pitch mix vs. league average."""
    print(f"\n{'=' * 80}")
    print("  PITCH MIX: ANGELS vs LEAGUE AVERAGE")
    print(f"{'=' * 80}")

    angels = df[df["is_angels_pitcher"]]
    league = df[~df["is_angels_pitcher"]]
    total_a = len(angels)
    total_l = len(league)

    print(f"\n  {'Pitch Type':20s} {'Angels %':>10s} {'League %':>10s} {'Delta':>10s}")
    print(f"  {'-'*50}")
    for pt_code, pt_name in PITCH_TYPES.items():
        a_pct = (angels["pitch_type"] == pt_code).sum() / total_a * 100
        l_pct = (league["pitch_type"] == pt_code).sum() / total_l * 100
        d = a_pct - l_pct
        sign = "+" if d > 0 else ""
        print(f"  {pt_name:20s} {a_pct:9.1f}% {l_pct:9.1f}% {sign}{d:8.1f}%")


def angels_pitcher_breakdown(df):
    """Individual Angels pitcher analysis."""
    print(f"\n{'=' * 80}")
    print("  INDIVIDUAL ANGELS PITCHERS (min 500 pitches)")
    print(f"{'=' * 80}")

    angels = df[df["is_angels_pitcher"]].copy()
    pitcher_counts = angels.groupby("player_name").size()
    qualified = pitcher_counts[pitcher_counts >= 500].index

    rows = []
    for name in sorted(qualified):
        p = angels[angels["player_name"] == name]
        out = compute_outcomes(p)
        bip = p[p["launch_speed"].notna()]

        # Pitch mix
        total = len(p)
        mix = {}
        for pt_code, pt_name in PITCH_TYPES.items():
            pct = (p["pitch_type"] == pt_code).sum() / total * 100
            if pct > 1:
                mix[pt_code] = round(pct, 1)

        row = {
            "pitcher": name,
            "pitches": total,
            "whiff_rate": round(out["whiff_rate"], 3) if not pd.isna(out["whiff_rate"]) else np.nan,
            "csw": round(out["csw"], 3) if not pd.isna(out["csw"]) else np.nan,
            "exit_velo": round(out["exit_velo"], 1) if not pd.isna(out["exit_velo"]) else np.nan,
            "hard_hit_pct": round(out["hard_hit_pct"], 3) if not pd.isna(out["hard_hit_pct"]) else np.nan,
            "gb_pct": round(out["gb_pct"], 3) if not pd.isna(out["gb_pct"]) else np.nan,
            "barrel_pct": round(out["barrel_pct"], 3) if not pd.isna(out["barrel_pct"]) else np.nan,
            "avg_velo": round(p["release_speed"].mean(), 1),
            "mix": mix,
        }
        rows.append(row)

        print(f"\n  {name} ({total} pitches)")
        print(f"    Avg Velo: {row['avg_velo']} | Whiff: {row['whiff_rate']} | CSW: {row['csw']}")
        print(f"    Exit Velo: {row['exit_velo']} | Hard Hit: {row['hard_hit_pct']} | GB%: {row['gb_pct']} | Barrel%: {row['barrel_pct']}")
        mix_str = ", ".join(f"{PITCH_TYPES.get(k,k)} {v}%" for k, v in sorted(mix.items(), key=lambda x: -x[1]))
        print(f"    Mix: {mix_str}")

    return rows


def home_away_splits(df):
    """Angels pitchers: home vs away performance."""
    print(f"\n{'=' * 80}")
    print("  ANGELS PITCHERS: HOME vs AWAY SPLITS (min 200 pitches each)")
    print(f"{'=' * 80}")

    angels = df[df["is_angels_pitcher"]].copy()
    for name in sorted(angels["player_name"].unique()):
        p = angels[angels["player_name"] == name]
        home = p[p["at_anaheim"]]
        away = p[~p["at_anaheim"]]
        if len(home) < 200 or len(away) < 200:
            continue

        h_out = compute_outcomes(home)
        a_out = compute_outcomes(away)

        print(f"\n  {name} (Home: {len(home)}, Away: {len(away)})")
        for key in ["whiff_rate", "csw", "exit_velo", "hard_hit_pct", "gb_pct", "barrel_pct"]:
            h = h_out[key]
            a = a_out[key]
            if pd.isna(h) or pd.isna(a):
                continue
            d = h - a
            sign = "+" if d > 0 else ""
            fmt = ".1f" if key == "exit_velo" else ".3f"
            print(f"    {key:20s}  Home: {h:{fmt}}  Away: {a:{fmt}}  Delta: {sign}{d:{fmt}}")


def main():
    print("Loading Statcast data...", flush=True)
    df = load_statcast()
    print(f"Loaded {len(df)} pitches", flush=True)

    df = prep(df)
    print(f"After filtering: {len(df)} pitches", flush=True)
    print(f"At Angel Stadium: {df['at_anaheim'].sum()} ({df['at_anaheim'].mean():.1%})", flush=True)
    print(f"Angels pitchers: {df['is_angels_pitcher'].sum()} ({df['is_angels_pitcher'].mean():.1%})", flush=True)

    # 1. Park effects: all pitchers who threw at Angel Stadium vs their road data
    controlled_comparison(
        df,
        "ALL PITCHERS: Angel Stadium vs Road (same-pitcher controlled)",
        lambda d: d[d["pitcher"].isin(
            set(d[d["at_anaheim"]]["pitcher"].unique()) &
            set(d[~d["at_anaheim"]]["pitcher"].unique())
        )]
    )

    # 2. Angels pitchers only: home vs road
    controlled_comparison(
        df,
        "ANGELS PITCHERS ONLY: Home vs Road",
        lambda d: d[d["is_angels_pitcher"]]
    )

    # 3. Visiting pitchers only: at Angel Stadium vs their other games
    controlled_comparison(
        df,
        "VISITING PITCHERS ONLY: At Angel Stadium vs Elsewhere",
        lambda d: d[~d["is_angels_pitcher"]]
    )

    # 4. Angels staff vs league average
    angels_vs_league(df)

    # 5. Pitch mix comparison
    pitch_mix_comparison(df)

    # 6. Individual pitcher breakdown
    pitcher_rows = angels_pitcher_breakdown(df)

    # 7. Home/away splits per pitcher
    home_away_splits(df)

    # Save pitcher data
    if pitcher_rows:
        pitcher_df = pd.DataFrame(pitcher_rows)
        pitcher_df.to_csv(os.path.join(OUTPUT_DIR, "angels_pitchers_2024.csv"), index=False)
        print(f"\nSaved pitcher data to {OUTPUT_DIR}/angels_pitchers_2024.csv")

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
