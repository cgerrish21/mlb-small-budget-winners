"""
target_finder.py
Identify specific acquisition targets for a team building the efficient winner way.

Uses the project's findings to screen for:
  Position players: defense + contact/OBP + speed, affordable
  Relievers: K% + whiff% + put-away%, affordable

Each player is tagged with an acquirability tier based on surplus value
(WAR * $8M/WAR market rate vs actual salary), age, and team context:
  - Untouchable: young franchise cornerstones, elite players on cheap deals
  - Difficult: stars on contenders, high-surplus young players
  - Realistic: fair-value contracts, older players, middling teams
  - Available: on losing teams, approaching free agency, salary dump candidates

Outputs ranked target lists for the 2027 Angels offseason.
"""

import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use 2026 as the scouting year (most recent full season)
SCOUT_YEAR = 2026
# Also look at 2025 for two-year sample
YEARS = [2025, 2026]

# Rough market rate: ~$8M per WAR on the open market
DOLLARS_PER_WAR = 8_000_000


def print_header(title):
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def estimate_acquirability(war, salary, age, team_w):
    """
    Estimate how realistic it is to acquire a player via trade or free agency.

    Returns one of: 'Untouchable', 'Difficult', 'Realistic', 'Available'

    Factors:
    - Surplus value = (WAR * $8M/WAR) - salary. High surplus = team has no
      incentive to trade.
    - Age: younger players have more years of control, harder to pry loose.
    - Team record: losing teams are more likely to sell.
    - Salary: mega-contracts on bad teams are salary dump candidates.
    """
    if salary <= 0:
        est_salary = 750_000  # league minimum for pre-arb
    else:
        est_salary = salary

    market_value = max(war, 0) * DOLLARS_PER_WAR
    surplus = market_value - est_salary

    # ── Untouchable: young franchise cornerstones ──
    # Young, elite, cheap = maximum team control and value. Nobody trades these.
    if age <= 26 and war >= 5.0 and est_salary < 10_000_000:
        return "Untouchable"

    # Elite producers on contenders with team-friendly deals
    if team_w >= 85 and war >= 4.0 and est_salary < 12_000_000:
        return "Untouchable"

    # ── Difficult: high-value players teams would hate to lose ──
    # Stars on contenders with significant surplus
    if team_w >= 85 and war >= 3.0 and surplus > 15_000_000:
        return "Difficult"

    # Young + cheap + productive on ANY team (including rebuilders).
    # A 25-year-old putting up 3+ WAR on a pre-arb deal is core-building
    # talent. Bad teams don't sell these guys, they build around them.
    # Oakland, Detroit, etc. are accumulating this type of player on purpose.
    if age <= 25 and war >= 3.0 and est_salary < 5_000_000:
        return "Difficult"

    # Pre-arb or arb-eligible players with high surplus on any team.
    # Even bad teams hold onto cheap, productive young players.
    if age <= 28 and war >= 2.0 and est_salary < 5_000_000 and surplus > 15_000_000:
        return "Difficult"

    # Very high surplus on non-contenders (franchise player on a middling team)
    if surplus > 35_000_000:
        return "Difficult"

    # ── Available: salary dumps and expiring veterans ──
    # Mega-contracts on bad teams = salary dump candidates
    if est_salary >= 20_000_000 and team_w <= 78:
        return "Available"

    # Older players on non-contenders (approaching free agency, less future value)
    if age >= 30 and team_w <= 80:
        return "Available"

    # Overpaid veterans on losing teams (team wants payroll relief)
    if est_salary >= 10_000_000 and team_w <= 75 and war < 3.0:
        return "Available"

    # ── Realistic: gettable with the right package ──
    # Mid-tier players on bad teams (not core pieces but useful players)
    if team_w <= 75 and war >= 1.0 and war < 3.0 and age >= 27:
        return "Realistic"

    # On bad-to-middling teams, not young/cheap enough to be untouchable
    if team_w <= 78 and est_salary >= 5_000_000:
        return "Realistic"

    # Fair-value or overpaid contracts anywhere
    if surplus < 10_000_000:
        return "Realistic"

    # Older players anywhere (less team control remaining)
    if age >= 29:
        return "Realistic"

    # Default: moderate-to-high surplus, probably building block for their team
    if surplus > 20_000_000:
        return "Difficult"

    return "Realistic"


# ── Load position player data ──────────────────────────────────────
def load_position_players():
    """Load bWAR + BRef stats for position players in scouting years."""
    # bWAR
    bwar = pd.read_csv(os.path.join(DATA_DIR, "bwar_batting.csv"))
    bwar = bwar[bwar["year_ID"].isin(YEARS)].copy()
    bwar["mlb_ID"] = pd.to_numeric(bwar["mlb_ID"], errors="coerce")
    bwar = bwar.dropna(subset=["mlb_ID"])
    bwar["mlb_ID"] = bwar["mlb_ID"].astype(int)
    for c in ["WAR", "PA", "G", "runs_above_avg_off", "runs_above_avg_def", "salary"]:
        bwar[c] = pd.to_numeric(bwar[c], errors="coerce").fillna(0)

    bwar = bwar[bwar["PA"] >= 100].copy()

    # Aggregate multi-stint
    bwar = (
        bwar.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg({
            "name_common": "first",
            "team": "first",
            "WAR": "sum",
            "PA": "sum",
            "G": "sum",
            "runs_above_avg_off": "sum",
            "runs_above_avg_def": "sum",
            "salary": "max",
        })
    )

    # BRef individual stats
    stats = pd.read_csv(os.path.join(DATA_DIR, "player_batting.csv"))
    stats["mlbID"] = pd.to_numeric(stats["mlbID"], errors="coerce")
    stats = stats.dropna(subset=["mlbID"])
    stats["mlbID"] = stats["mlbID"].astype(int)
    stats["Season"] = pd.to_numeric(stats["Season"], errors="coerce")
    stats["Age"] = pd.to_numeric(stats["Age"], errors="coerce")

    # Filter multi-team rows
    multi = {"2TM", "3TM", "4TM", "5TM", "2 Tms", "3 Tms", "4 Tms", "5 Tms", "TOT"}
    stats = stats[~stats["Tm"].isin(multi)].copy()

    count_cols = ["PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SB", "CS",
                  "R", "RBI", "SH", "SF", "IBB", "GDP"]
    for c in count_cols:
        stats[c] = pd.to_numeric(stats[c], errors="coerce").fillna(0)

    stats_agg = stats.groupby(["mlbID", "Season"], as_index=False).agg(
        {**{c: "sum" for c in count_cols}, "Age": "first"}
    )
    stats_agg["1B"] = stats_agg["H"] - stats_agg["2B"] - stats_agg["3B"] - stats_agg["HR"]
    stats_agg["BA"] = np.where(stats_agg["AB"] > 0, stats_agg["H"] / stats_agg["AB"], 0)
    stats_agg["OBP"] = np.where(
        stats_agg["PA"] > 0,
        (stats_agg["H"] + stats_agg["BB"] + stats_agg["HBP"]) /
        (stats_agg["AB"] + stats_agg["BB"] + stats_agg["HBP"] + stats_agg["SF"]),
        0,
    )
    stats_agg["SLG"] = np.where(
        stats_agg["AB"] > 0,
        (stats_agg["1B"] + 2*stats_agg["2B"] + 3*stats_agg["3B"] + 4*stats_agg["HR"]) / stats_agg["AB"],
        0,
    )
    stats_agg["ISO"] = stats_agg["SLG"] - stats_agg["BA"]
    stats_agg["K_pct"] = np.where(stats_agg["PA"] > 0, stats_agg["SO"] / stats_agg["PA"] * 100, 0)
    stats_agg["BB_pct"] = np.where(stats_agg["PA"] > 0, stats_agg["BB"] / stats_agg["PA"] * 100, 0)
    stats_agg["SB_per_PA"] = np.where(stats_agg["PA"] > 0, stats_agg["SB"] / stats_agg["PA"], 0)

    # Merge
    merged = bwar.merge(
        stats_agg,
        left_on=["mlb_ID", "year_ID"],
        right_on=["mlbID", "Season"],
        how="inner",
        suffixes=("_bwar", "_bref"),
    )
    merged["PA"] = merged["PA_bwar"]

    # Team groups
    master = pd.read_csv(os.path.join(DATA_DIR, "master.csv"))[["Season", "team", "group", "W"]]
    merged = merged.merge(master, left_on=["year_ID", "team"], right_on=["Season", "team"], how="left")

    return merged


# ── Load reliever data ──────────────────────────────────────────────
def load_relievers():
    """Load bWAR relievers + Statcast arsenal data for scouting years."""
    bwar = pd.read_csv(os.path.join(DATA_DIR, "bwar_pitching.csv"))
    bwar = bwar[bwar["year_ID"].isin(YEARS)].copy()
    bwar["mlb_ID"] = pd.to_numeric(bwar["mlb_ID"], errors="coerce")
    bwar = bwar.dropna(subset=["mlb_ID"])
    bwar["mlb_ID"] = bwar["mlb_ID"].astype(int)
    for c in ["WAR", "G", "GS", "salary"]:
        bwar[c] = pd.to_numeric(bwar[c], errors="coerce").fillna(0)

    mask = (bwar["G"] >= 15) & ((bwar["GS"] == 0) | (bwar["GS"] / bwar["G"] < 0.15))
    rel = bwar[mask].copy()

    # Aggregate multi-stint
    rel = (
        rel.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg({"name_common": "first", "team": "first", "WAR": "sum", "G": "sum",
               "GS": "sum", "salary": "max"})
    )

    # Statcast arsenal cache
    cache_path = os.path.join(DATA_DIR, "reliever_statcast_full_cache.csv")
    if os.path.exists(cache_path):
        sc = pd.read_csv(cache_path)
        sc = sc[sc["Season"].isin(YEARS)].copy()
        rel = rel.merge(sc, left_on=["mlb_ID", "year_ID"],
                         right_on=["player_id", "Season"], how="inner")

    # BRef pitching for age
    pb = pd.read_csv(os.path.join(DATA_DIR, "player_pitching.csv"))
    pb["mlbID"] = pd.to_numeric(pb["mlbID"], errors="coerce")
    pb["Age"] = pd.to_numeric(pb["Age"], errors="coerce")
    pb = pb.dropna(subset=["mlbID"])
    pb["mlbID"] = pb["mlbID"].astype(int)
    age_map = pb.groupby("mlbID")["Age"].max().to_dict()
    rel["Age"] = rel["mlb_ID"].map(age_map)

    # Team groups
    master = pd.read_csv(os.path.join(DATA_DIR, "master.csv"))[["Season", "team", "group", "W"]]
    rel = rel.merge(master, left_on=["year_ID", "team"], right_on=["Season", "team"], how="left")

    return rel


# ── Screening functions ─────────────────────────────────────────────
def _add_tiers(df):
    """Add acquirability tier to a dataframe with WAR, salary, Age, W columns."""
    df["tier"] = df.apply(
        lambda r: estimate_acquirability(r["WAR"], r["salary"], r["Age"], r.get("W", 81)),
        axis=1,
    )
    tier_order = {"Available": 0, "Realistic": 1, "Difficult": 2, "Untouchable": 3}
    df["_tier_rank"] = df["tier"].map(tier_order)
    return df


def _fmt_salary(x):
    return f"${x/1e6:.1f}M" if x > 0 else "pre-arb"


def screen_position_players(df):
    """Find position players who fit the efficient winner profile."""
    print_header("POSITION PLAYER TARGETS: DEFENSE + CONTACT/OBP")

    # Focus on most recent year for current ability
    recent = df[df["year_ID"] == SCOUT_YEAR].copy()
    print(f"  {SCOUT_YEAR} position players (PA >= 100): {len(recent)}")

    # Exclude Angels (can't trade with yourself)
    recent = recent[recent["team"] != "LAA"].copy()

    # League medians for context
    med_obp = recent["OBP"].median()
    med_k = recent["K_pct"].median()
    med_def = recent["runs_above_avg_def"].median()
    print(f"  League medians: OBP={med_obp:.3f}, K%={med_k:.1f}, Def runs={med_def:.1f}")

    # ── SCREEN 1: Good glove + gets on base + doesn't strike out ──
    screen1 = recent[
        (recent["runs_above_avg_def"] >= 5) &
        (recent["OBP"] >= med_obp) &
        (recent["K_pct"] <= med_k * 1.15) &
        (recent["PA"] >= 250)
    ].copy()
    screen1["score"] = (
        screen1["runs_above_avg_def"] / 10 +
        (screen1["OBP"] - med_obp) * 100 +
        (med_k - screen1["K_pct"]) / 5 +
        screen1["SB_per_PA"] * 50
    )
    screen1 = _add_tiers(screen1)
    screen1 = screen1.sort_values(["_tier_rank", "score"], ascending=[True, False])

    cols = ["name_common", "team", "Age", "WAR", "PA", "OBP", "ISO", "K_pct",
            "SB", "runs_above_avg_def", "salary", "tier"]

    # Show realistic targets
    realistic1 = screen1[screen1["tier"].isin(["Available", "Realistic"])]
    print(f"\n  SCREEN 1: Good glove + OBP + low K% — REALISTIC TARGETS ({len(realistic1)} players)")
    show = realistic1[cols].head(25).copy()
    show["salary"] = show["salary"].apply(_fmt_salary)
    show["OBP"] = show["OBP"].round(3)
    show["ISO"] = show["ISO"].round(3)
    show["K_pct"] = show["K_pct"].round(1)
    show.rename(columns={"runs_above_avg_def": "def_runs"}, inplace=True)
    print(show.to_string(index=False))

    # Show untouchables for context
    untouchable1 = screen1[screen1["tier"].isin(["Untouchable", "Difficult"])]
    if len(untouchable1) > 0:
        print(f"\n  (For context: {len(untouchable1)} players fit the profile but are Untouchable/Difficult)")
        show_u = untouchable1[["name_common", "team", "Age", "WAR", "salary", "tier"]].head(15).copy()
        show_u["salary"] = show_u["salary"].apply(_fmt_salary)
        print(show_u.to_string(index=False))

    # ── SCREEN 2: Young + cheap + solid all-around ─────────────────
    screen2 = recent[
        (recent["Age"] <= 28) &
        (recent["WAR"] >= 1.0) &
        (recent["runs_above_avg_def"] >= 0) &
        (recent["PA"] >= 300)
    ].copy()
    screen2 = _add_tiers(screen2)
    screen2 = screen2.sort_values(["_tier_rank", "WAR"], ascending=[True, False])

    realistic2 = screen2[screen2["tier"].isin(["Available", "Realistic"])]
    print(f"\n  SCREEN 2: Age <= 28, WAR >= 1.0, good defense — REALISTIC ({len(realistic2)} players)")
    show2 = realistic2[["name_common", "team", "Age", "WAR", "PA", "OBP", "ISO",
                          "K_pct", "SB", "runs_above_avg_def", "salary", "W", "tier"]].head(25).copy()
    show2["salary"] = show2["salary"].apply(_fmt_salary)
    show2["OBP"] = show2["OBP"].round(3)
    show2["ISO"] = show2["ISO"].round(3)
    show2["K_pct"] = show2["K_pct"].round(1)
    show2.rename(columns={"runs_above_avg_def": "def_runs", "W": "team_W"}, inplace=True)
    print(show2.to_string(index=False))

    untouchable2 = screen2[screen2["tier"].isin(["Untouchable", "Difficult"])]
    if len(untouchable2) > 0:
        print(f"\n  (Excluded {len(untouchable2)} Untouchable/Difficult: "
              + ", ".join(untouchable2["name_common"].head(10).tolist()) + ")")

    # ── SCREEN 3: On non-contenders (trade targets) ────────────────
    screen3 = recent[
        (recent["W"] <= 78) &
        (recent["WAR"] >= 1.5) &
        (recent["Age"] <= 30)
    ].copy()
    screen3 = _add_tiers(screen3)
    screen3 = screen3.sort_values(["_tier_rank", "WAR"], ascending=[True, False])

    realistic3 = screen3[screen3["tier"].isin(["Available", "Realistic"])]
    print(f"\n  SCREEN 3: On losing teams (W <= 78), WAR >= 1.5 — REALISTIC ({len(realistic3)} players)")
    show3 = realistic3[["name_common", "team", "Age", "WAR", "PA", "OBP", "ISO",
                          "K_pct", "SB", "runs_above_avg_def", "salary", "W", "tier"]].head(25).copy()
    show3["salary"] = show3["salary"].apply(_fmt_salary)
    show3["OBP"] = show3["OBP"].round(3)
    show3["ISO"] = show3["ISO"].round(3)
    show3["K_pct"] = show3["K_pct"].round(1)
    show3.rename(columns={"runs_above_avg_def": "def_runs", "W": "team_W"}, inplace=True)
    print(show3.to_string(index=False))

    untouchable3 = screen3[screen3["tier"].isin(["Untouchable", "Difficult"])]
    if len(untouchable3) > 0:
        print(f"\n  (Excluded {len(untouchable3)} Untouchable/Difficult: "
              + ", ".join(untouchable3["name_common"].head(10).tolist()) + ")")

    return screen1, screen2, screen3


def screen_relievers(df):
    """Find relievers who fit the efficient winner bullpen profile."""
    print_header("RELIEVER TARGETS: K% + WHIFF% + PUT-AWAY%")

    recent = df[df["year_ID"] == SCOUT_YEAR].copy()
    print(f"  {SCOUT_YEAR} relievers with Statcast data: {len(recent)}")

    # Exclude Angels
    recent = recent[recent["team"] != "LAA"].copy()

    # League medians
    has_stats = "k_pct" in recent.columns
    if not has_stats:
        print("  No Statcast data available for relievers.")
        return None, None

    med_k = recent["k_pct"].median()
    med_whiff = recent["whiff_pct"].median()
    med_pa = recent["put_away_pct"].median() if "put_away_pct" in recent.columns else 0
    print(f"  League medians: K%={med_k:.1f}, Whiff%={med_whiff:.1f}, Put-Away%={med_pa:.1f}")

    # ── SCREEN 1: Elite bat-missers ────────────────────────────────
    screen1 = recent[
        (recent["k_pct"] >= recent["k_pct"].quantile(0.65)) &
        (recent["whiff_pct"] >= recent["whiff_pct"].quantile(0.65)) &
        (recent["G"] >= 25)
    ].copy()

    # Composite scouting score based on our stickiness findings
    # Weight by predictive power: K% (0.229), put-away (0.189), whiff (0.184)
    screen1["scout_score"] = (
        (screen1["k_pct"] - med_k) / recent["k_pct"].std() * 0.229 +
        (screen1["whiff_pct"] - med_whiff) / recent["whiff_pct"].std() * 0.184 +
        ((screen1["put_away_pct"] - med_pa) / recent["put_away_pct"].std() * 0.189
         if "put_away_pct" in screen1.columns else 0)
    )
    screen1 = _add_tiers(screen1)
    screen1 = screen1.sort_values(["_tier_rank", "scout_score"], ascending=[True, False])

    cols = ["name_common", "team", "Age", "WAR", "G", "k_pct", "whiff_pct",
            "put_away_pct", "scout_score", "salary", "tier"]
    avail = [c for c in cols if c in screen1.columns]

    realistic1 = screen1[screen1["tier"].isin(["Available", "Realistic"])]
    print(f"\n  SCREEN 1: Elite bat-missers — REALISTIC TARGETS ({len(realistic1)} relievers)")
    show = realistic1[avail].head(25).copy()
    show["salary"] = show["salary"].apply(_fmt_salary)
    for c in ["k_pct", "whiff_pct", "put_away_pct", "scout_score"]:
        if c in show.columns:
            show[c] = show[c].round(1)
    print(show.to_string(index=False))

    untouchable1 = screen1[screen1["tier"].isin(["Untouchable", "Difficult"])]
    if len(untouchable1) > 0:
        print(f"\n  (Excluded {len(untouchable1)} Untouchable/Difficult: "
              + ", ".join(untouchable1["name_common"].head(10).tolist()) + ")")

    # ── SCREEN 2: Young + cheap + misses bats ─────────────────────
    screen2 = recent[
        (recent["Age"] <= 28) &
        (recent["k_pct"] >= med_k) &
        (recent["whiff_pct"] >= med_whiff) &
        (recent["G"] >= 20)
    ].copy()
    screen2["scout_score"] = (
        (screen2["k_pct"] - med_k) / recent["k_pct"].std() * 0.229 +
        (screen2["whiff_pct"] - med_whiff) / recent["whiff_pct"].std() * 0.184
    )
    screen2 = _add_tiers(screen2)
    screen2 = screen2.sort_values(["_tier_rank", "scout_score"], ascending=[True, False])

    realistic2 = screen2[screen2["tier"].isin(["Available", "Realistic"])]
    print(f"\n  SCREEN 2: Age <= 28, misses bats — REALISTIC ({len(realistic2)} relievers)")
    show2 = realistic2[avail].head(25).copy()
    show2["salary"] = show2["salary"].apply(_fmt_salary)
    for c in ["k_pct", "whiff_pct", "put_away_pct", "scout_score"]:
        if c in show2.columns:
            show2[c] = show2[c].round(1)
    print(show2.to_string(index=False))

    untouchable2 = screen2[screen2["tier"].isin(["Untouchable", "Difficult"])]
    if len(untouchable2) > 0:
        print(f"\n  (Excluded {len(untouchable2)} Untouchable/Difficult: "
              + ", ".join(untouchable2["name_common"].head(10).tolist()) + ")")

    # ── SCREEN 3: On bad teams (trade targets) ────────────────────
    screen3 = recent[
        (recent["W"] <= 78) &
        (recent["k_pct"] >= med_k) &
        (recent["WAR"] >= 0.3) &
        (recent["Age"] <= 30)
    ].copy()
    screen3["scout_score"] = (
        (screen3["k_pct"] - med_k) / recent["k_pct"].std() * 0.229 +
        (screen3["whiff_pct"] - med_whiff) / recent["whiff_pct"].std() * 0.184
    )
    screen3 = _add_tiers(screen3)
    screen3 = screen3.sort_values(["_tier_rank", "scout_score"], ascending=[True, False])

    cols_trade = ["name_common", "team", "Age", "WAR", "G", "k_pct", "whiff_pct",
                  "put_away_pct", "scout_score", "salary", "W", "tier"]
    avail_trade = [c for c in cols_trade if c in screen3.columns]

    realistic3 = screen3[screen3["tier"].isin(["Available", "Realistic"])]
    print(f"\n  SCREEN 3: On losing teams, misses bats — REALISTIC ({len(realistic3)} relievers)")
    show3 = realistic3[avail_trade].head(25).copy()
    show3["salary"] = show3["salary"].apply(_fmt_salary)
    show3.rename(columns={"W": "team_W"}, inplace=True)
    for c in ["k_pct", "whiff_pct", "put_away_pct", "scout_score"]:
        if c in show3.columns:
            show3[c] = show3[c].round(1)
    print(show3.to_string(index=False))

    untouchable3 = screen3[screen3["tier"].isin(["Untouchable", "Difficult"])]
    if len(untouchable3) > 0:
        print(f"\n  (Excluded {len(untouchable3)} Untouchable/Difficult: "
              + ", ".join(untouchable3["name_common"].head(10).tolist()) + ")")

    return screen1, screen2


def angels_current_roster(pos_df, rel_df):
    """Show where the current Angels roster falls on these metrics."""
    print_header("CURRENT ANGELS ROSTER: WHERE THEY STAND")

    # Position players
    angels_pos = pos_df[(pos_df["team"] == "LAA") & (pos_df["year_ID"] == SCOUT_YEAR)].copy()
    angels_pos = angels_pos.sort_values("WAR", ascending=False)

    print(f"\n  Angels {SCOUT_YEAR} position players ({len(angels_pos)}):")
    cols = ["name_common", "Age", "WAR", "PA", "OBP", "ISO", "K_pct", "SB",
            "runs_above_avg_def"]
    show = angels_pos[cols].copy()
    show["OBP"] = show["OBP"].round(3)
    show["ISO"] = show["ISO"].round(3)
    show["K_pct"] = show["K_pct"].round(1)
    show.rename(columns={"runs_above_avg_def": "def_runs"}, inplace=True)
    print(show.to_string(index=False))

    total_def = angels_pos["runs_above_avg_def"].sum()
    good = (angels_pos["runs_above_avg_def"] >= 5).sum()
    bad = (angels_pos["runs_above_avg_def"] <= -5).sum()
    avg_k = angels_pos["K_pct"].mean()
    avg_obp = angels_pos["OBP"].mean()
    print(f"\n  Team totals: def_runs={total_def:+.0f}, good gloves={good}, bad gloves={bad}")
    print(f"  Avg K%={avg_k:.1f}, avg OBP={avg_obp:.3f}")

    # Relievers
    if rel_df is not None:
        angels_rel = rel_df[(rel_df["team"] == "LAA") & (rel_df["year_ID"] == SCOUT_YEAR)].copy()
        angels_rel = angels_rel.sort_values("WAR", ascending=False)

        has_stats = "k_pct" in angels_rel.columns
        if has_stats:
            print(f"\n  Angels {SCOUT_YEAR} relievers ({len(angels_rel)}):")
            cols = ["name_common", "Age", "WAR", "G", "k_pct", "whiff_pct", "put_away_pct"]
            avail = [c for c in cols if c in angels_rel.columns]
            show_r = angels_rel[avail].copy()
            for c in ["k_pct", "whiff_pct", "put_away_pct"]:
                if c in show_r.columns:
                    show_r[c] = show_r[c].round(1)
            print(show_r.to_string(index=False))


def main():
    print("=" * 90)
    print("  TARGET FINDER: PLAYERS WHO FIT THE EFFICIENT WINNER PROFILE")
    print(f"  Scouting year: {SCOUT_YEAR} | Focus: Angels 2027 offseason targets")
    print("=" * 90)

    print("\n[1/3] Loading position player data...")
    pos = load_position_players()
    print(f"  Position players: {len(pos)}")

    print("\n[2/3] Loading reliever data...")
    rel = load_relievers()
    print(f"  Relievers: {len(rel)}")

    print("\n[3/3] Screening targets...")
    angels_current_roster(pos, rel)
    pos_screens = screen_position_players(pos)
    rel_screens = screen_relievers(rel)

    print("\nDone.")


if __name__ == "__main__":
    main()
