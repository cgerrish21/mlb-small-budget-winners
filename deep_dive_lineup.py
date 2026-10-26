"""
deep_dive_lineup.py
What does the lineup composition look like on efficient winner teams?

Is it 9 Chandler Simpsons (speed/contact only)? Or a mix of player types?
Classifies individual hitters into archetypes and compares roster construction
between efficient winners, cheap losers, and big-budget winners.

Data: individual bWAR + BRef batting stats + Statcast exit velo/barrels
"""

import os, warnings, time
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
YEARS = list(range(2022, 2027))
CACHE_FILE = os.path.join(DATA_DIR, "statcast_batter_individual_cache.csv")


# ── Load and merge individual batting data ──────────────────────────
def load_hitters():
    """Merge bWAR, BRef batting stats, and team groups into one dataset."""
    # bWAR for WAR values
    bwar = pd.read_csv(os.path.join(DATA_DIR, "bwar_batting.csv"))
    bwar = bwar[bwar["year_ID"].isin(YEARS)].copy()
    bwar["mlb_ID"] = pd.to_numeric(bwar["mlb_ID"], errors="coerce")
    bwar = bwar.dropna(subset=["mlb_ID"])
    bwar["mlb_ID"] = bwar["mlb_ID"].astype(int)
    bwar["WAR"] = pd.to_numeric(bwar["WAR"], errors="coerce").fillna(0)
    bwar["PA"] = pd.to_numeric(bwar["PA"], errors="coerce").fillna(0)

    # Filter to position players with meaningful playing time
    bwar = bwar[bwar["PA"] >= 100].copy()

    # Aggregate multi-stint rows
    bwar = (
        bwar.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg({
            "name_common": "first",
            "team": "first",
            "WAR": "sum",
            "PA": "sum",
        })
    )

    # BRef individual batting stats
    stats = pd.read_csv(os.path.join(DATA_DIR, "player_batting.csv"))
    stats["mlbID"] = pd.to_numeric(stats["mlbID"], errors="coerce")
    stats = stats.dropna(subset=["mlbID"])
    stats["mlbID"] = stats["mlbID"].astype(int)
    stats["Season"] = pd.to_numeric(stats["Season"], errors="coerce")

    # Filter out multi-team summary rows
    multi_team = {"2TM", "3TM", "4TM", "5TM", "2 Tms", "3 Tms", "4 Tms", "5 Tms", "TOT"}
    stats = stats[~stats["Tm"].isin(multi_team)].copy()

    # Aggregate multi-stint BRef stats (sum counting stats, recalculate rates)
    count_cols = ["PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SB", "CS",
                  "R", "RBI", "SH", "SF", "IBB", "GDP"]
    for c in count_cols:
        stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0)

    stats_agg = stats.groupby(["mlbID", "Season"], as_index=False)[count_cols].sum()
    stats_agg["1B"] = stats_agg["H"] - stats_agg["2B"] - stats_agg["3B"] - stats_agg["HR"]

    # Recalculate rate stats
    stats_agg["BA"] = np.where(stats_agg["AB"] > 0, stats_agg["H"] / stats_agg["AB"], 0)
    stats_agg["OBP"] = np.where(
        stats_agg["PA"] > 0,
        (stats_agg["H"] + stats_agg["BB"] + stats_agg["HBP"]) /
        (stats_agg["AB"] + stats_agg["BB"] + stats_agg["HBP"] + stats_agg["SF"]),
        0
    )
    stats_agg["SLG"] = np.where(
        stats_agg["AB"] > 0,
        (stats_agg["1B"] + 2*stats_agg["2B"] + 3*stats_agg["3B"] + 4*stats_agg["HR"]) / stats_agg["AB"],
        0
    )
    stats_agg["OPS"] = stats_agg["OBP"] + stats_agg["SLG"]
    stats_agg["ISO"] = stats_agg["SLG"] - stats_agg["BA"]
    stats_agg["K_pct"] = np.where(stats_agg["PA"] > 0, stats_agg["SO"] / stats_agg["PA"] * 100, 0)
    stats_agg["BB_pct"] = np.where(stats_agg["PA"] > 0, stats_agg["BB"] / stats_agg["PA"] * 100, 0)
    stats_agg["HR_per_PA"] = np.where(stats_agg["PA"] > 0, stats_agg["HR"] / stats_agg["PA"], 0)
    stats_agg["SB_per_PA"] = np.where(stats_agg["PA"] > 0, stats_agg["SB"] / stats_agg["PA"], 0)

    # Merge bWAR + BRef stats
    merged = bwar.merge(
        stats_agg,
        left_on=["mlb_ID", "year_ID"],
        right_on=["mlbID", "Season"],
        how="inner",
        suffixes=("_bwar", "_bref"),
    )

    # Use bWAR PA (more reliable) -- rename to avoid confusion
    merged["PA"] = merged["PA_bwar"]

    # Team groups from master
    master = pd.read_csv(os.path.join(DATA_DIR, "master.csv"))
    master = master[["Season", "team", "group"]].copy()
    merged = merged.merge(master, left_on=["year_ID", "team"], right_on=["Season", "team"], how="left")

    print(f"  Hitters (PA >= 100): {len(merged)}")
    print(f"  With group assignment: {merged['group'].notna().sum()}")
    print(f"  Years: {sorted(merged['year_ID'].unique())}")

    return merged


# ── Pull individual Statcast exit velo/barrels ──────────────────────
def pull_statcast_batters():
    """Pull individual batter Statcast (exit velo, barrel rate, hard hit)."""
    if os.path.exists(CACHE_FILE):
        print(f"  Loading cache: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE)

    from pybaseball import statcast_batter_exitvelo_barrels

    print("  Pulling from Statcast API...")
    frames = []
    for y in YEARS:
        try:
            df = statcast_batter_exitvelo_barrels(y, minBBE=50)
            df["Season"] = y
            frames.append(df)
            print(f"    {y}: {len(df)} batters")
        except Exception as e:
            print(f"    {y}: FAILED - {e}")
        time.sleep(1.5)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(CACHE_FILE, index=False)
    print(f"  Cached to {CACHE_FILE} ({len(result)} rows)")
    return result


# ── Classify hitter archetypes ──────────────────────────────────────
def classify_hitter(row, lg_medians):
    """
    Classify a hitter into an archetype based on their profile
    relative to league medians for that season.
    """
    yr = row["year_ID"]
    med = lg_medians.get(yr, lg_medians[2024])  # fallback

    high_power = row["ISO"] > med["ISO"] and row["HR_per_PA"] > med["HR_per_PA"]
    high_speed = row["SB_per_PA"] > med["SB_per_PA"] * 1.5  # well above median
    low_k = row["K_pct"] < med["K_pct"]
    high_obp = row["OBP"] > med["OBP"]
    high_k = row["K_pct"] > med["K_pct"] * 1.15  # 15%+ above median

    if high_power and high_speed:
        return "power_speed"
    elif high_power and high_k:
        return "three_true_outcomes"  # HR, BB, K
    elif high_power:
        return "power"
    elif high_speed and low_k:
        return "speed_contact"
    elif high_speed:
        return "speed"
    elif low_k and high_obp:
        return "contact_obp"
    elif low_k:
        return "contact"
    elif high_obp:
        return "obp"
    else:
        return "generic"


def compute_league_medians(df):
    """Compute league median rates per season for archetype classification."""
    medians = {}
    for yr in YEARS:
        yr_data = df[df["year_ID"] == yr]
        medians[yr] = {
            "ISO": yr_data["ISO"].median(),
            "HR_per_PA": yr_data["HR_per_PA"].median(),
            "SB_per_PA": yr_data["SB_per_PA"].median(),
            "K_pct": yr_data["K_pct"].median(),
            "OBP": yr_data["OBP"].median(),
            "BB_pct": yr_data["BB_pct"].median(),
        }
    return medians


# ── Analysis functions ──────────────────────────────────────────────
def print_header(title):
    print("\n" + "=" * 85)
    print(f"  {title}")
    print("=" * 85)


def archetype_distribution(df):
    """How are archetypes distributed across team groups?"""
    print_header("LINEUP ARCHETYPE DISTRIBUTION BY TEAM GROUP")

    groups = ["efficient_winner", "cheap_loser", "big_budget_winner"]
    for g in groups:
        sub = df[df["group"] == g]
        n_teams = sub.groupby(["team", "year_ID"]).ngroups
        total = len(sub)

        dist = sub["archetype"].value_counts(normalize=True).round(3) * 100
        print(f"\n  {g} ({total} player-seasons across {n_teams} team-seasons):")
        for arch, pct in dist.items():
            count = (sub["archetype"] == arch).sum()
            bar = "#" * int(pct)
            print(f"    {arch:25s}  {pct:5.1f}%  (n={count:3d})  {bar}")


def war_by_archetype(df):
    """How much WAR does each archetype produce, by group?"""
    print_header("WAR CONTRIBUTION BY ARCHETYPE AND TEAM GROUP")

    groups = ["efficient_winner", "cheap_loser", "big_budget_winner"]

    # WAR share by archetype per group
    for g in groups:
        sub = df[df["group"] == g]
        total_war = sub["WAR"].sum()
        arch_war = sub.groupby("archetype")["WAR"].agg(["sum", "mean", "count"])
        arch_war["war_share"] = (arch_war["sum"] / total_war * 100).round(1)
        arch_war["mean"] = arch_war["mean"].round(2)
        arch_war = arch_war.sort_values("sum", ascending=False)
        arch_war.columns = ["total_WAR", "avg_WAR", "n", "WAR_share%"]

        print(f"\n  {g} (total WAR: {total_war:.1f}):")
        print(arch_war.to_string())


def lineup_balance(df):
    """Are efficient winners more balanced or more concentrated in archetype?"""
    print_header("LINEUP BALANCE: ARCHETYPE DIVERSITY PER TEAM-SEASON")

    def team_diversity(group):
        """Shannon entropy of archetype distribution within a team-season."""
        counts = group["archetype"].value_counts(normalize=True)
        entropy = -np.sum(counts * np.log2(counts + 1e-10))
        return entropy

    team_stats = (
        df.groupby(["team", "year_ID", "group"])
        .apply(
            lambda g: pd.Series({
                "n_hitters": len(g),
                "n_archetypes": g["archetype"].nunique(),
                "entropy": team_diversity(g),
                "total_WAR": g["WAR"].sum(),
                "pct_power": (g["archetype"].isin(["power", "three_true_outcomes", "power_speed"])).mean() * 100,
                "pct_contact_speed": (g["archetype"].isin(["contact", "contact_obp", "speed_contact", "speed"])).mean() * 100,
                "pct_speed_any": (g["archetype"].isin(["speed", "speed_contact", "power_speed"])).mean() * 100,
                "avg_ISO": g["ISO"].mean(),
                "avg_K_pct": g["K_pct"].mean(),
                "avg_SB_per_PA": g["SB_per_PA"].mean(),
                "avg_OBP": g["OBP"].mean(),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    groups = ["efficient_winner", "cheap_loser", "big_budget_winner"]
    compare_cols = ["n_hitters", "n_archetypes", "entropy", "total_WAR",
                    "pct_power", "pct_contact_speed", "pct_speed_any",
                    "avg_ISO", "avg_K_pct", "avg_SB_per_PA", "avg_OBP"]

    rows = []
    for col in compare_cols:
        row = {"metric": col}
        for g in groups:
            vals = team_stats.loc[team_stats["group"] == g, col]
            row[g] = f"{vals.mean():.2f}"
        rows.append(row)

    result = pd.DataFrame(rows)
    print("\n  Team-season averages:")
    print(result.to_string(index=False))

    # Stat tests: EW vs CL
    print("\n  Efficient Winner vs Cheap Loser (t-tests):")
    for col in compare_cols:
        ew = team_stats.loc[team_stats["group"] == "efficient_winner", col].dropna()
        cl = team_stats.loc[team_stats["group"] == "cheap_loser", col].dropna()
        if len(ew) >= 5 and len(cl) >= 5:
            t, p = scipy_stats.ttest_ind(ew, cl, equal_var=False)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {col:25s}  EW={ew.mean():.3f}  CL={cl.mean():.3f}  diff={ew.mean()-cl.mean():+.3f}  p={p:.4f} {sig}")

    return team_stats


def roster_profile_examples(df):
    """Show actual roster profiles of specific efficient winner teams."""
    print_header("EXAMPLE ROSTERS: EFFICIENT WINNER LINEUP COMPOSITION")

    # Pick a few iconic EW seasons
    examples = [
        ("CLE", 2024, "Cleveland 2024 (92W, $79M)"),
        ("TBR", 2023, "Tampa Bay 2023 (99W, $82M)"),
        ("BAL", 2023, "Baltimore 2023 (101W, $54M)"),
        ("CHW", 2026, "White Sox 2026 (84W, $85M)"),
    ]

    for team, year, label in examples:
        sub = df[(df["team"] == team) & (df["year_ID"] == year)].sort_values("WAR", ascending=False)
        if len(sub) == 0:
            continue

        print(f"\n  {label}")
        cols = ["name_common", "WAR", "PA", "archetype", "OBP", "ISO", "K_pct", "SB"]
        available = [c for c in cols if c in sub.columns]
        display = sub[available].head(12).copy()
        for c in ["OBP", "ISO"]:
            if c in display.columns:
                display[c] = display[c].round(3)
        if "K_pct" in display.columns:
            display["K_pct"] = display["K_pct"].round(1)
        print(display.to_string(index=False))

        # Summary
        arch_dist = sub["archetype"].value_counts()
        print(f"    Archetypes: {dict(arch_dist)}")


def power_vs_contact_war(df):
    """Is WAR from power hitters vs contact hitters different by group?"""
    print_header("WHERE DOES THE WAR COME FROM? POWER vs CONTACT vs SPEED")

    groups = ["efficient_winner", "cheap_loser", "big_budget_winner"]

    # Broader categories
    df["broad_type"] = df["archetype"].map({
        "power": "power",
        "three_true_outcomes": "power",
        "power_speed": "power_speed",
        "speed_contact": "speed_contact",
        "speed": "speed_contact",
        "contact_obp": "contact",
        "contact": "contact",
        "obp": "contact",
        "generic": "generic",
    })

    # Per team-season: WAR share by broad type
    team_type_war = (
        df.groupby(["team", "year_ID", "group", "broad_type"])["WAR"]
        .sum()
        .reset_index()
    )
    team_total = df.groupby(["team", "year_ID", "group"])["WAR"].sum().reset_index(name="total_WAR")
    team_type_war = team_type_war.merge(team_total, on=["team", "year_ID", "group"])
    team_type_war["war_share"] = team_type_war["WAR"] / team_type_war["total_WAR"] * 100

    # Average WAR share per broad type, by group
    print("\n  Average % of team batting WAR by hitter type:")
    for g in groups:
        sub = team_type_war[team_type_war["group"] == g]
        pivot = sub.groupby("broad_type")["war_share"].mean().round(1)
        print(f"\n  {g}:")
        for bt, share in pivot.sort_values(ascending=False).items():
            bar = "#" * int(share / 2)
            print(f"    {bt:20s}  {share:5.1f}%  {bar}")


def statcast_by_archetype(df, statcast):
    """Add Statcast quality-of-contact data if available."""
    if statcast.empty:
        print("\n  (No individual Statcast data available)")
        return df

    print_header("QUALITY OF CONTACT BY ARCHETYPE (Statcast)")

    # Statcast has player_id column
    sc = statcast.rename(columns={"player_id": "mlb_ID"}).copy()
    sc["mlb_ID"] = pd.to_numeric(sc["mlb_ID"], errors="coerce")
    sc = sc.dropna(subset=["mlb_ID"])
    sc["mlb_ID"] = sc["mlb_ID"].astype(int)

    # Key Statcast columns
    sc_cols = ["mlb_ID", "Season"]
    for c in ["avg_hit_speed", "max_hit_speed", "brl_percent", "ev95percent",
              "anglesweetspotpercent", "avg_distance"]:
        if c in sc.columns:
            sc_cols.append(c)

    sc_sub = sc[sc_cols].copy()

    merged = df.merge(sc_sub, left_on=["mlb_ID", "year_ID"],
                       right_on=["mlb_ID", "Season"], how="left")

    # Compare by archetype
    if "avg_hit_speed" in merged.columns:
        arch_sc = merged.groupby("archetype").agg({
            "avg_hit_speed": "mean",
            "brl_percent": "mean",
            "ev95percent": "mean",
            "WAR": "mean",
        }).round(2)
        arch_sc = arch_sc.sort_values("WAR", ascending=False)
        print("\n  By archetype (mean values):")
        print(arch_sc.to_string())

        # By group + broad type
        if "broad_type" in merged.columns:
            print("\n  By group + hitter type:")
            for g in ["efficient_winner", "cheap_loser"]:
                sub = merged[merged["group"] == g]
                bt_sc = sub.groupby("broad_type").agg({
                    "avg_hit_speed": "mean",
                    "brl_percent": "mean",
                    "WAR": ["mean", "count"],
                }).round(2)
                print(f"\n  {g}:")
                print(bt_sc.to_string())

    return merged


def the_punchline(df, team_stats):
    """What's the actual finding?"""
    print_header("THE PUNCHLINE: WHAT DOES THE IDEAL CHEAP LINEUP LOOK LIKE?")

    ew = team_stats[team_stats["group"] == "efficient_winner"]
    cl = team_stats[team_stats["group"] == "cheap_loser"]
    bb = team_stats[team_stats["group"] == "big_budget_winner"]

    print(f"""
  Efficient winners vs cheap losers (team-season averages):

                          Eff.Winners   Cheap Losers   Big-Budget Winners
  % Power hitters         {ew['pct_power'].mean():.1f}%          {cl['pct_power'].mean():.1f}%          {bb['pct_power'].mean():.1f}%
  % Contact/Speed         {ew['pct_contact_speed'].mean():.1f}%          {cl['pct_contact_speed'].mean():.1f}%          {bb['pct_contact_speed'].mean():.1f}%
  % Any speed             {ew['pct_speed_any'].mean():.1f}%          {cl['pct_speed_any'].mean():.1f}%          {bb['pct_speed_any'].mean():.1f}%
  Avg team ISO            {ew['avg_ISO'].mean():.3f}          {cl['avg_ISO'].mean():.3f}          {bb['avg_ISO'].mean():.3f}
  Avg team K%             {ew['avg_K_pct'].mean():.1f}%          {cl['avg_K_pct'].mean():.1f}%          {bb['avg_K_pct'].mean():.1f}%
  Avg team OBP            {ew['avg_OBP'].mean():.3f}          {cl['avg_OBP'].mean():.3f}          {bb['avg_OBP'].mean():.3f}
  Avg SB/PA               {ew['avg_SB_per_PA'].mean():.4f}         {cl['avg_SB_per_PA'].mean():.4f}         {bb['avg_SB_per_PA'].mean():.4f}
  Archetype diversity     {ew['n_archetypes'].mean():.1f}            {cl['n_archetypes'].mean():.1f}            {bb['n_archetypes'].mean():.1f}
""")


# ── Main ────────────────────────────────────────────────────────────
def main():
    print("=" * 85)
    print("  LINEUP COMPOSITION DEEP DIVE")
    print("  What mix of hitter types do efficient winners actually roster?")
    print("=" * 85)

    print("\n[1/4] Loading hitter data...")
    df = load_hitters()

    print("\n[2/4] Pulling individual Statcast data...")
    statcast = pull_statcast_batters()
    print(f"  Statcast rows: {len(statcast)}")

    print("\n[3/4] Classifying hitter archetypes...")
    lg_medians = compute_league_medians(df)
    print("  League medians by year:")
    for yr, med in lg_medians.items():
        print(f"    {yr}: ISO={med['ISO']:.3f}  HR/PA={med['HR_per_PA']:.4f}  "
              f"SB/PA={med['SB_per_PA']:.4f}  K%={med['K_pct']:.1f}  OBP={med['OBP']:.3f}")

    df["archetype"] = df.apply(lambda r: classify_hitter(r, lg_medians), axis=1)
    print(f"\n  Overall archetype distribution:")
    print(df["archetype"].value_counts().to_string())

    print("\n[4/4] Running analyses...")
    archetype_distribution(df)
    war_by_archetype(df)
    team_stats = lineup_balance(df)
    power_vs_contact_war(df)
    roster_profile_examples(df)
    df = statcast_by_archetype(df, statcast)
    the_punchline(df, team_stats)

    # Save
    outpath = os.path.join(OUTPUT_DIR, "lineup_archetypes.csv")
    save_cols = ["name_common", "mlb_ID", "year_ID", "team", "group", "WAR", "PA",
                 "archetype", "OBP", "SLG", "ISO", "K_pct", "BB_pct", "HR", "SB",
                 "HR_per_PA", "SB_per_PA"]
    df[[c for c in save_cols if c in df.columns]].to_csv(outpath, index=False)
    print(f"\n  Saved to {outpath}")
    print("\nDone.")


if __name__ == "__main__":
    main()
