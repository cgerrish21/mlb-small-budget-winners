"""
Deep-dive analysis: HOW efficient winners build differently.

Goes beyond "develop more good players" to answer:
1. What offensive profile do efficient winners have?
2. How is their pitching WAR split between starters and relievers?
3. What does their WAR distribution shape look like (depth vs stars)?
4. What % of their good players are homegrown vs acquired?
5. What age profile do their contributors have?
6. Where positionally is the WAR coming from (batting vs pitching split)?

Groups compared:
  - efficient_winner: bottom-third payroll, 85+ W or playoff
  - cheap_loser: bottom-third payroll, below thresholds
  - big_budget_winner: top-third payroll, winners
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PYTHON = sys.executable

# ── Load data ──────────────────────────────────────────────────────────
print("=" * 80)
print("DEEP DIVE: How Efficient Winners Build Differently")
print("=" * 80)
print()

master = pd.read_csv(os.path.join(DATA_DIR, "master.csv"))
bwar_bat = pd.read_csv(os.path.join(DATA_DIR, "bwar_batting.csv"), encoding="utf-8")
bwar_pit = pd.read_csv(os.path.join(DATA_DIR, "bwar_pitching.csv"), encoding="utf-8")
player_bat = pd.read_csv(os.path.join(DATA_DIR, "player_batting.csv"), encoding="utf-8")
player_pit = pd.read_csv(os.path.join(DATA_DIR, "player_pitching.csv"), encoding="utf-8")

# Coerce numerics
bwar_bat["WAR"] = pd.to_numeric(bwar_bat["WAR"], errors="coerce").fillna(0)
bwar_bat["PA"] = pd.to_numeric(bwar_bat["PA"], errors="coerce").fillna(0)
bwar_bat["G"] = pd.to_numeric(bwar_bat["G"], errors="coerce").fillna(0)
bwar_bat["mlb_ID"] = pd.to_numeric(bwar_bat["mlb_ID"], errors="coerce")
bwar_bat["year_ID"] = pd.to_numeric(bwar_bat["year_ID"], errors="coerce")
bwar_bat["stint_ID"] = pd.to_numeric(bwar_bat["stint_ID"], errors="coerce").fillna(1)

bwar_pit["WAR"] = pd.to_numeric(bwar_pit["WAR"], errors="coerce").fillna(0)
bwar_pit["G"] = pd.to_numeric(bwar_pit["G"], errors="coerce").fillna(0)
bwar_pit["GS"] = pd.to_numeric(bwar_pit["GS"], errors="coerce").fillna(0)
bwar_pit["mlb_ID"] = pd.to_numeric(bwar_pit["mlb_ID"], errors="coerce")
bwar_pit["year_ID"] = pd.to_numeric(bwar_pit["year_ID"], errors="coerce")
bwar_pit["stint_ID"] = pd.to_numeric(bwar_pit["stint_ID"], errors="coerce").fillna(1)

# Filter to analysis years
YEARS = list(range(2022, 2027))
bwar_bat_recent = bwar_bat[bwar_bat["year_ID"].isin(YEARS)].copy()
bwar_pit_recent = bwar_pit[bwar_pit["year_ID"].isin(YEARS)].copy()

# Build team-season group lookup from master
group_lookup = master.set_index(["team", "Season"])["group"].to_dict()

# Assign group to bWAR rows
bwar_bat_recent["group"] = bwar_bat_recent.apply(
    lambda r: group_lookup.get((r["team"], r["year_ID"]), None), axis=1
)
bwar_pit_recent["group"] = bwar_pit_recent.apply(
    lambda r: group_lookup.get((r["team"], r["year_ID"]), None), axis=1
)

# Focus groups
FOCUS_GROUPS = ["efficient_winner", "cheap_loser", "big_budget_winner"]
GROUP_LABELS = {
    "efficient_winner": "Efficient Winners",
    "cheap_loser": "Cheap Losers",
    "big_budget_winner": "Big-Budget Winners",
}

# Count team-seasons per group
group_counts = master["group"].value_counts()
print("Team-seasons per group:")
for g in FOCUS_GROUPS:
    print(f"  {GROUP_LABELS[g]:25s}: {group_counts.get(g, 0)}")
print()


def print_header(title):
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def print_table(rows, headers, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for r in rows:
                max_w = max(max_w, len(str(r[i])))
            col_widths.append(max_w + 2)

    # Header
    header_line = ""
    for i, h in enumerate(headers):
        header_line += str(h).ljust(col_widths[i])
    print(header_line)
    print("-" * sum(col_widths))

    # Rows
    for r in rows:
        line = ""
        for i, val in enumerate(r):
            line += str(val).ljust(col_widths[i])
        print(line)


# ══════════════════════════════════════════════════════════════════════
# 1. BATTING PROFILE: offensive identity differences
# ══════════════════════════════════════════════════════════════════════
print_header("1. BATTING PROFILE — Offensive Identity")

# Use master.csv team-level batting stats (rate stats already computed)
# Normalize per-PA where possible
bat_cols = {
    "bat_OPS": "OPS",
    "bat_OBP": "OBP",
    "bat_SLG": "SLG",
    "bat_ISO": "ISO (raw power)",
    "bat_K%": "K%",
    "bat_BB%": "BB%",
    "bat_BABIP": "BABIP",
    "bat_AVG": "AVG",
}

# Also compute HR/PA and SB/PA from raw counts
master["HR_per_PA"] = pd.to_numeric(master["bat_HR"], errors="coerce") / pd.to_numeric(master["bat_PA"], errors="coerce")
master["SB_per_PA"] = pd.to_numeric(master["bat_SB"], errors="coerce") / pd.to_numeric(master["bat_PA"], errors="coerce")
master["SO_per_PA"] = pd.to_numeric(master["bat_SO"], errors="coerce") / pd.to_numeric(master["bat_PA"], errors="coerce")
master["BB_per_PA_raw"] = pd.to_numeric(master["bat_SO"], errors="coerce") / pd.to_numeric(master["bat_PA"], errors="coerce")

# Actually compute BB/PA correctly
master["BB_per_PA_raw"] = None  # placeholder, use bat_BB% instead

# Build comparison table
rate_metrics = [
    ("bat_OPS", "OPS"),
    ("bat_OBP", "OBP"),
    ("bat_SLG", "SLG"),
    ("bat_ISO", "ISO (power)"),
    ("bat_K%", "K%"),
    ("bat_BB%", "BB%"),
    ("bat_BABIP", "BABIP"),
    ("HR_per_PA", "HR / PA"),
    ("SB_per_PA", "SB / PA"),
]

rows = []
for col, label in rate_metrics:
    vals = {}
    for g in FOCUS_GROUPS:
        subset = master[master["group"] == g]
        v = pd.to_numeric(subset[col], errors="coerce").mean()
        vals[g] = v

    # Compute difference: efficient_winner vs cheap_loser
    diff_ew_cl = vals["efficient_winner"] - vals["cheap_loser"]
    # Also compute: efficient_winner vs big_budget_winner
    diff_ew_bb = vals["efficient_winner"] - vals["big_budget_winner"]

    rows.append([
        label,
        f"{vals['efficient_winner']:.4f}",
        f"{vals['cheap_loser']:.4f}",
        f"{vals['big_budget_winner']:.4f}",
        f"{diff_ew_cl:+.4f}",
        f"{diff_ew_bb:+.4f}",
    ])

print("Rate stats (team-level averages across team-seasons):")
print()
print_table(
    rows,
    ["Metric", "Eff. Winners", "Cheap Losers", "Big-$ Winners", "EW-CL diff", "EW-BB diff"],
    [18, 14, 14, 14, 12, 12],
)

print()
print("Interpretation guide: positive EW-CL diff = efficient winners are HIGHER.")
print("                      negative EW-BB diff = efficient winners are LOWER than big-budget.")

# Deeper: HR totals, SB totals
print()
print("Counting stats per season (team totals):")
count_metrics = [
    ("bat_HR", "HR"),
    ("bat_SB", "SB"),
    ("bat_R", "Runs"),
    ("bat_H", "Hits"),
    ("bat_PA", "PA"),
    ("bat_SO", "SO"),
]

rows2 = []
for col, label in count_metrics:
    vals = {}
    for g in FOCUS_GROUPS:
        subset = master[master["group"] == g]
        v = pd.to_numeric(subset[col], errors="coerce").mean()
        vals[g] = v
    rows2.append([
        label,
        f"{vals['efficient_winner']:.0f}",
        f"{vals['cheap_loser']:.0f}",
        f"{vals['big_budget_winner']:.0f}",
    ])

print()
print_table(
    rows2,
    ["Stat", "Eff. Winners", "Cheap Losers", "Big-$ Winners"],
    [10, 14, 14, 14],
)

# ══════════════════════════════════════════════════════════════════════
# 2. PITCHING COMPOSITION: Starter WAR vs Reliever WAR
# ══════════════════════════════════════════════════════════════════════
print_header("2. PITCHING COMPOSITION — Starter WAR vs Reliever WAR")

# Classify each pitcher-season as starter or reliever based on GS/G ratio
bwar_pit_recent["is_starter"] = bwar_pit_recent["GS"] >= (bwar_pit_recent["G"] * 0.5)
bwar_pit_recent["role"] = bwar_pit_recent["is_starter"].map({True: "SP", False: "RP"})

pit_comp_rows = []
for g in FOCUS_GROUPS:
    subset = bwar_pit_recent[bwar_pit_recent["group"] == g]
    n_teamseasons = master[master["group"] == g].shape[0]

    sp_war = subset[subset["role"] == "SP"]["WAR"].sum()
    rp_war = subset[subset["role"] == "RP"]["WAR"].sum()
    total_pwar = sp_war + rp_war

    sp_count = subset[subset["role"] == "SP"].shape[0]
    rp_count = subset[subset["role"] == "RP"].shape[0]

    # Per team-season averages
    sp_war_avg = sp_war / n_teamseasons if n_teamseasons > 0 else 0
    rp_war_avg = rp_war / n_teamseasons if n_teamseasons > 0 else 0
    total_avg = total_pwar / n_teamseasons if n_teamseasons > 0 else 0

    sp_share = sp_war / total_pwar * 100 if total_pwar > 0 else 0
    rp_share = rp_war / total_pwar * 100 if total_pwar > 0 else 0

    pit_comp_rows.append([
        GROUP_LABELS[g],
        f"{sp_war_avg:.1f}",
        f"{rp_war_avg:.1f}",
        f"{total_avg:.1f}",
        f"{sp_share:.1f}%",
        f"{rp_share:.1f}%",
        f"{sp_count / n_teamseasons:.1f}",
        f"{rp_count / n_teamseasons:.1f}",
    ])

print("Per team-season averages:")
print()
print_table(
    pit_comp_rows,
    ["Group", "SP WAR", "RP WAR", "Total pWAR", "SP %", "RP %", "SP count", "RP count"],
    [22, 10, 10, 12, 8, 8, 10, 10],
)

# Also look at SP depth: how many starters with 2+ WAR per team-season
print()
print("Starter depth (starters with 2+ WAR):")
sp_depth_rows = []
for g in FOCUS_GROUPS:
    subset = bwar_pit_recent[(bwar_pit_recent["group"] == g) & (bwar_pit_recent["role"] == "SP")]
    n_ts = master[master["group"] == g].shape[0]

    # Count SP with 2+ WAR per team-season
    sp_good = subset[subset["WAR"] >= 2.0]
    n_good_sp = sp_good.shape[0]

    # Count SP with 4+ WAR (aces)
    sp_ace = subset[subset["WAR"] >= 4.0]
    n_ace = sp_ace.shape[0]

    sp_depth_rows.append([
        GROUP_LABELS[g],
        f"{n_good_sp / n_ts:.2f}",
        f"{n_ace / n_ts:.2f}",
        f"{sp_good['WAR'].mean():.2f}" if len(sp_good) > 0 else "N/A",
    ])

print()
print_table(
    sp_depth_rows,
    ["Group", "SP 2+ WAR / team-yr", "SP 4+ WAR / team-yr", "Avg WAR of 2+ SP"],
    [22, 22, 22, 20],
)

# Bullpen value: how many relievers with 1+ WAR
print()
print("Bullpen contributors (relievers with 1+ WAR):")
rp_depth_rows = []
for g in FOCUS_GROUPS:
    subset = bwar_pit_recent[(bwar_pit_recent["group"] == g) & (bwar_pit_recent["role"] == "RP")]
    n_ts = master[master["group"] == g].shape[0]

    rp_good = subset[subset["WAR"] >= 1.0]
    rp_solid = subset[subset["WAR"] >= 0.5]

    rp_depth_rows.append([
        GROUP_LABELS[g],
        f"{rp_good.shape[0] / n_ts:.2f}",
        f"{rp_solid.shape[0] / n_ts:.2f}",
        f"{subset['WAR'].sum() / n_ts:.2f}",
    ])

print()
print_table(
    rp_depth_rows,
    ["Group", "RP 1+ WAR / team-yr", "RP 0.5+ WAR / team-yr", "Total RP WAR / team-yr"],
    [22, 22, 24, 24],
)


# ══════════════════════════════════════════════════════════════════════
# 3. WAR DISTRIBUTION SHAPE: depth vs stars
# ══════════════════════════════════════════════════════════════════════
print_header("3. WAR DISTRIBUTION — Stars vs Depth")

# Combine batting and pitching WAR per player per team-season
bat_player = bwar_bat_recent[["team", "year_ID", "player_ID", "WAR", "group"]].copy()
bat_player = bat_player.rename(columns={"WAR": "bat_war"})
pit_player = bwar_pit_recent[["team", "year_ID", "player_ID", "WAR", "group"]].copy()
pit_player = pit_player.rename(columns={"WAR": "pit_war"})

# Sum batting WAR by player-team-season (handle multiple stints)
bat_agg = bat_player.groupby(["team", "year_ID", "player_ID", "group"]).agg(
    bat_war=("bat_war", "sum")
).reset_index()

pit_agg = pit_player.groupby(["team", "year_ID", "player_ID", "group"]).agg(
    pit_war=("pit_war", "sum")
).reset_index()

# Merge batting and pitching WAR per player
combined = bat_agg.merge(
    pit_agg,
    on=["team", "year_ID", "player_ID", "group"],
    how="outer"
)
combined["bat_war"] = combined["bat_war"].fillna(0)
combined["pit_war"] = combined["pit_war"].fillna(0)
combined["total_war"] = combined["bat_war"] + combined["pit_war"]

# Fill group for pitchers-only (who might not be in batting)
combined["group"] = combined.apply(
    lambda r: r["group"] if pd.notna(r["group"]) else group_lookup.get((r["team"], r["year_ID"]), None),
    axis=1
)

# WAR tier breakdown per team-season
print("Among players with 2+ WAR: tier breakdown per team-season:")
print()

tier_rows = []
for g in FOCUS_GROUPS:
    subset = combined[combined["group"] == g]
    n_ts = master[master["group"] == g].shape[0]

    # Tiers
    war_2_3 = subset[(subset["total_war"] >= 2.0) & (subset["total_war"] < 3.0)]
    war_3_4 = subset[(subset["total_war"] >= 3.0) & (subset["total_war"] < 4.0)]
    war_4_5 = subset[(subset["total_war"] >= 4.0) & (subset["total_war"] < 5.0)]
    war_5_6 = subset[(subset["total_war"] >= 5.0) & (subset["total_war"] < 6.0)]
    war_6plus = subset[subset["total_war"] >= 6.0]
    war_2plus = subset[subset["total_war"] >= 2.0]
    war_4plus = subset[subset["total_war"] >= 4.0]

    tier_rows.append([
        GROUP_LABELS[g],
        f"{war_2_3.shape[0] / n_ts:.2f}",
        f"{war_3_4.shape[0] / n_ts:.2f}",
        f"{war_4_5.shape[0] / n_ts:.2f}",
        f"{war_5_6.shape[0] / n_ts:.2f}",
        f"{war_6plus.shape[0] / n_ts:.2f}",
        f"{war_2plus.shape[0] / n_ts:.2f}",
        f"{war_4plus.shape[0] / n_ts:.2f}",
    ])

print_table(
    tier_rows,
    ["Group", "2-3 WAR", "3-4 WAR", "4-5 WAR", "5-6 WAR", "6+ WAR", "Total 2+", "Total 4+"],
    [22, 10, 10, 10, 10, 10, 12, 12],
)

# Total WAR from different player tiers
print()
print("WAR contribution by player tier (per team-season):")
print()

war_by_tier_rows = []
for g in FOCUS_GROUPS:
    subset = combined[combined["group"] == g]
    n_ts = master[master["group"] == g].shape[0]

    # WAR from different tiers
    below_0 = subset[subset["total_war"] < 0]["total_war"].sum() / n_ts
    tier_0_1 = subset[(subset["total_war"] >= 0) & (subset["total_war"] < 1)]["total_war"].sum() / n_ts
    tier_1_2 = subset[(subset["total_war"] >= 1) & (subset["total_war"] < 2)]["total_war"].sum() / n_ts
    tier_2_4 = subset[(subset["total_war"] >= 2) & (subset["total_war"] < 4)]["total_war"].sum() / n_ts
    tier_4plus = subset[subset["total_war"] >= 4]["total_war"].sum() / n_ts
    total = subset["total_war"].sum() / n_ts

    war_by_tier_rows.append([
        GROUP_LABELS[g],
        f"{below_0:+.1f}",
        f"{tier_0_1:.1f}",
        f"{tier_1_2:.1f}",
        f"{tier_2_4:.1f}",
        f"{tier_4plus:.1f}",
        f"{total:.1f}",
    ])

print_table(
    war_by_tier_rows,
    ["Group", "< 0 WAR", "0-1 WAR", "1-2 WAR", "2-4 WAR", "4+ WAR", "Total"],
    [22, 10, 10, 10, 10, 10, 10],
)

# Key ratio: what fraction of total team WAR comes from 4+ WAR players?
print()
print("Share of total team WAR by player tier:")
print()

share_rows = []
for g in FOCUS_GROUPS:
    subset = combined[combined["group"] == g]
    total = subset["total_war"].sum()
    if total <= 0:
        continue

    below_0 = subset[subset["total_war"] < 0]["total_war"].sum()
    tier_0_2 = subset[(subset["total_war"] >= 0) & (subset["total_war"] < 2)]["total_war"].sum()
    tier_2_4 = subset[(subset["total_war"] >= 2) & (subset["total_war"] < 4)]["total_war"].sum()
    tier_4plus = subset[subset["total_war"] >= 4]["total_war"].sum()

    share_rows.append([
        GROUP_LABELS[g],
        f"{below_0 / total * 100:.1f}%",
        f"{tier_0_2 / total * 100:.1f}%",
        f"{tier_2_4 / total * 100:.1f}%",
        f"{tier_4plus / total * 100:.1f}%",
    ])

print_table(
    share_rows,
    ["Group", "< 0 WAR share", "0-2 WAR share", "2-4 WAR share", "4+ WAR share"],
    [22, 16, 16, 16, 16],
)

# ══════════════════════════════════════════════════════════════════════
# 4. PLAYER ACQUISITION: homegrown vs acquired among 2+ WAR contributors
# ══════════════════════════════════════════════════════════════════════
print_header("4. PLAYER ACQUISITION — Homegrown vs Acquired (2+ WAR players)")

# Determine debut team for each player from full bWAR history
all_bat = bwar_bat[["player_ID", "year_ID", "team", "stint_ID"]].copy()
all_bat["year_ID"] = pd.to_numeric(all_bat["year_ID"], errors="coerce")
all_bat["stint_ID"] = pd.to_numeric(all_bat["stint_ID"], errors="coerce").fillna(1)

all_pit = bwar_pit[["player_ID", "year_ID", "team", "stint_ID"]].copy()
all_pit["year_ID"] = pd.to_numeric(all_pit["year_ID"], errors="coerce")
all_pit["stint_ID"] = pd.to_numeric(all_pit["stint_ID"], errors="coerce").fillna(1)

all_appearances = pd.concat([all_bat, all_pit], ignore_index=True)
debut_team = (
    all_appearances.sort_values(["year_ID", "stint_ID"])
    .groupby("player_ID")
    .first()
    .reset_index()[["player_ID", "team"]]
)
debut_team.columns = ["player_ID", "debut_team"]

# Merge debut team onto combined player WAR
combined_with_debut = combined.merge(debut_team, on="player_ID", how="left")
combined_with_debut["is_homegrown"] = combined_with_debut["team"] == combined_with_debut["debut_team"]

# Focus on 2+ WAR players
good_players = combined_with_debut[combined_with_debut["total_war"] >= 2.0].copy()

acq_rows = []
for g in FOCUS_GROUPS:
    subset = good_players[good_players["group"] == g]
    n_ts = master[master["group"] == g].shape[0]
    total = len(subset)
    homegrown = subset["is_homegrown"].sum()
    acquired = total - homegrown

    hg_war = subset[subset["is_homegrown"]]["total_war"].sum()
    acq_war = subset[~subset["is_homegrown"]]["total_war"].sum()
    total_war = hg_war + acq_war

    acq_rows.append([
        GROUP_LABELS[g],
        f"{total / n_ts:.1f}",
        f"{homegrown / n_ts:.1f}",
        f"{acquired / n_ts:.1f}",
        f"{homegrown / total * 100:.1f}%" if total > 0 else "N/A",
        f"{hg_war / total_war * 100:.1f}%" if total_war > 0 else "N/A",
    ])

print("Among players with 2+ WAR (per team-season):")
print()
print_table(
    acq_rows,
    ["Group", "Total 2+ WAR", "Homegrown", "Acquired", "HG % (count)", "HG % (WAR)"],
    [22, 14, 12, 12, 14, 14],
)

# Also do this for ALL players (not just 2+ WAR)
print()
print("All players — homegrown WAR share:")
print()
all_acq_rows = []
for g in FOCUS_GROUPS:
    subset = combined_with_debut[combined_with_debut["group"] == g]
    n_ts = master[master["group"] == g].shape[0]

    hg_war = subset[subset["is_homegrown"]]["total_war"].sum() / n_ts
    acq_war = subset[~subset["is_homegrown"]]["total_war"].sum() / n_ts
    total_war_avg = hg_war + acq_war

    # Homegrown count
    hg_count = subset[subset["is_homegrown"]].shape[0] / n_ts
    acq_count = subset[~subset["is_homegrown"]].shape[0] / n_ts

    all_acq_rows.append([
        GROUP_LABELS[g],
        f"{hg_war:.1f}",
        f"{acq_war:.1f}",
        f"{total_war_avg:.1f}",
        f"{hg_war / total_war_avg * 100:.1f}%" if total_war_avg > 0 else "N/A",
        f"{hg_count:.1f}",
        f"{acq_count:.1f}",
    ])

print_table(
    all_acq_rows,
    ["Group", "HG WAR/yr", "Acq WAR/yr", "Total WAR/yr", "HG WAR %", "HG players/yr", "Acq players/yr"],
    [22, 12, 12, 14, 10, 14, 14],
)

# Deeper: What WAR tier are the homegrown players in?
print()
print("Homegrown player WAR tier breakdown (per team-season, among 2+ WAR):")
print()
hg_tier_rows = []
for g in FOCUS_GROUPS:
    subset = good_players[(good_players["group"] == g) & (good_players["is_homegrown"])]
    n_ts = master[master["group"] == g].shape[0]

    war_2_3 = subset[(subset["total_war"] >= 2) & (subset["total_war"] < 3)].shape[0] / n_ts
    war_3_4 = subset[(subset["total_war"] >= 3) & (subset["total_war"] < 4)].shape[0] / n_ts
    war_4plus = subset[subset["total_war"] >= 4].shape[0] / n_ts

    hg_tier_rows.append([
        GROUP_LABELS[g],
        f"{war_2_3:.2f}",
        f"{war_3_4:.2f}",
        f"{war_4plus:.2f}",
    ])

print_table(
    hg_tier_rows,
    ["Group", "HG 2-3 WAR/yr", "HG 3-4 WAR/yr", "HG 4+ WAR/yr"],
    [22, 16, 16, 16],
)

# And for acquired players
print()
print("Acquired player WAR tier breakdown (per team-season, among 2+ WAR):")
print()
acq_tier_rows = []
for g in FOCUS_GROUPS:
    subset = good_players[(good_players["group"] == g) & (~good_players["is_homegrown"])]
    n_ts = master[master["group"] == g].shape[0]

    war_2_3 = subset[(subset["total_war"] >= 2) & (subset["total_war"] < 3)].shape[0] / n_ts
    war_3_4 = subset[(subset["total_war"] >= 3) & (subset["total_war"] < 4)].shape[0] / n_ts
    war_4plus = subset[subset["total_war"] >= 4].shape[0] / n_ts

    acq_tier_rows.append([
        GROUP_LABELS[g],
        f"{war_2_3:.2f}",
        f"{war_3_4:.2f}",
        f"{war_4plus:.2f}",
    ])

print_table(
    acq_tier_rows,
    ["Group", "Acq 2-3 WAR/yr", "Acq 3-4 WAR/yr", "Acq 4+ WAR/yr"],
    [22, 16, 16, 16],
)


# ══════════════════════════════════════════════════════════════════════
# 5. AGE PROFILE OF CONTRIBUTORS
# ══════════════════════════════════════════════════════════════════════
print_header("5. AGE PROFILE — Are contributors younger or older?")

# Link bWAR to player_batting/player_pitching via mlb_ID for age
# Build age lookup from player_batting and player_pitching
age_lookup = {}

for df, id_col in [(player_bat, "mlbID"), (player_pit, "mlbID")]:
    df["Age_num"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Season_num"] = pd.to_numeric(df["Season"], errors="coerce")
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    for _, row in df.iterrows():
        if pd.notna(row["Age_num"]) and pd.notna(row[id_col]) and pd.notna(row["Season_num"]):
            key = (int(row[id_col]), int(row["Season_num"]))
            age_lookup[key] = row["Age_num"]

print(f"Age lookup entries: {len(age_lookup)}")
print()

# Assign age to combined player data
combined_with_debut["mlb_ID_int"] = combined_with_debut.apply(
    lambda r: None, axis=1  # placeholder
)

# Need to get mlb_ID from bWAR data. Merge it.
bat_ids = bwar_bat_recent[["player_ID", "mlb_ID", "year_ID", "team"]].drop_duplicates()
pit_ids = bwar_pit_recent[["player_ID", "mlb_ID", "year_ID", "team"]].drop_duplicates()
all_ids = pd.concat([bat_ids, pit_ids], ignore_index=True).drop_duplicates(subset=["player_ID", "year_ID", "team"])

combined_with_age = combined_with_debut.merge(
    all_ids[["player_ID", "mlb_ID", "year_ID", "team"]],
    on=["player_ID", "year_ID", "team"],
    how="left"
)

# Look up age
def get_age(row):
    if pd.notna(row.get("mlb_ID")):
        key = (int(row["mlb_ID"]), int(row["year_ID"]))
        return age_lookup.get(key, np.nan)
    return np.nan

combined_with_age["age"] = combined_with_age.apply(get_age, axis=1)

# How many ages did we find?
n_with_age = combined_with_age["age"].notna().sum()
n_total = len(combined_with_age)
print(f"Players with age data: {n_with_age} / {n_total} ({n_with_age/n_total*100:.1f}%)")
print()

# Age profile of 2+ WAR contributors
good_with_age = combined_with_age[(combined_with_age["total_war"] >= 2.0) & (combined_with_age["age"].notna())]

age_rows = []
for g in FOCUS_GROUPS:
    subset = good_with_age[good_with_age["group"] == g]
    if len(subset) == 0:
        continue

    mean_age = subset["age"].mean()
    median_age = subset["age"].median()

    # Age brackets
    young = subset[subset["age"] < 25].shape[0]
    prime = subset[(subset["age"] >= 25) & (subset["age"] <= 29)].shape[0]
    aging = subset[(subset["age"] >= 30) & (subset["age"] <= 33)].shape[0]
    old = subset[subset["age"] > 33].shape[0]
    total = len(subset)

    age_rows.append([
        GROUP_LABELS[g],
        f"{mean_age:.1f}",
        f"{median_age:.1f}",
        f"{young/total*100:.1f}%",
        f"{prime/total*100:.1f}%",
        f"{aging/total*100:.1f}%",
        f"{old/total*100:.1f}%",
        f"{total}",
    ])

print("Age of 2+ WAR contributors:")
print()
print_table(
    age_rows,
    ["Group", "Mean Age", "Median", "<25", "25-29", "30-33", "34+", "N"],
    [22, 10, 9, 9, 9, 9, 9, 6],
)

# WAR-weighted age for 2+ WAR contributors
print()
print("WAR-weighted age of 2+ WAR contributors (how old is the production?):")
print()
wt_age_rows = []
for g in FOCUS_GROUPS:
    subset = good_with_age[good_with_age["group"] == g]
    if len(subset) == 0:
        continue
    wt_age = np.average(subset["age"], weights=subset["total_war"])
    wt_age_rows.append([GROUP_LABELS[g], f"{wt_age:.1f}"])

print_table(wt_age_rows, ["Group", "WAR-weighted age"], [22, 18])

# All contributors (any WAR)
print()
print("Age of ALL contributors with positive WAR:")
all_positive = combined_with_age[(combined_with_age["total_war"] > 0) & (combined_with_age["age"].notna())]

all_age_rows = []
for g in FOCUS_GROUPS:
    subset = all_positive[all_positive["group"] == g]
    if len(subset) == 0:
        continue
    n_ts = master[master["group"] == g].shape[0]

    mean_age = subset["age"].mean()
    young = subset[subset["age"] < 25].shape[0]
    prime = subset[(subset["age"] >= 25) & (subset["age"] <= 29)].shape[0]
    aging = subset[(subset["age"] >= 30) & (subset["age"] <= 33)].shape[0]
    old = subset[subset["age"] > 33].shape[0]
    total = len(subset)

    all_age_rows.append([
        GROUP_LABELS[g],
        f"{mean_age:.1f}",
        f"{young/total*100:.1f}%",
        f"{prime/total*100:.1f}%",
        f"{aging/total*100:.1f}%",
        f"{old/total*100:.1f}%",
    ])

print()
print_table(
    all_age_rows,
    ["Group", "Mean Age", "<25", "25-29", "30-33", "34+"],
    [22, 10, 9, 9, 9, 9],
)


# ══════════════════════════════════════════════════════════════════════
# 6. POSITION DISTRIBUTION: Batting WAR vs Pitching WAR split
# ══════════════════════════════════════════════════════════════════════
print_header("6. POSITIONAL WAR — Where does the value come from?")

# Since we don't have specific defensive positions, we can analyze:
# a) Batting WAR vs Pitching WAR split (from master.csv)
# b) Offensive WAR vs Defensive WAR from bWAR (runs_above_avg_off vs runs_above_avg_def)
# c) Position players who are pitchers flagged in bWAR

print("Batting WAR vs Pitching WAR split (from master.csv):")
print()

pos_rows = []
for g in FOCUS_GROUPS:
    subset = master[master["group"] == g]
    bat_war = pd.to_numeric(subset["bat_WAR"], errors="coerce").mean()
    pit_war = pd.to_numeric(subset["pit_WAR"], errors="coerce").mean()
    total_war = bat_war + pit_war
    bat_share = bat_war / total_war * 100 if total_war > 0 else 0
    pit_share = pit_war / total_war * 100 if total_war > 0 else 0

    pos_rows.append([
        GROUP_LABELS[g],
        f"{bat_war:.1f}",
        f"{pit_war:.1f}",
        f"{total_war:.1f}",
        f"{bat_share:.1f}%",
        f"{pit_share:.1f}%",
    ])

print_table(
    pos_rows,
    ["Group", "Bat WAR", "Pit WAR", "Total WAR", "Bat %", "Pit %"],
    [22, 10, 10, 12, 10, 10],
)

# Offensive vs Defensive run value from bWAR batting
print()
print("Offensive vs Defensive value (bWAR batting, runs above avg):")
print()

bwar_bat_recent["runs_above_avg_off"] = pd.to_numeric(bwar_bat_recent["runs_above_avg_off"], errors="coerce").fillna(0)
bwar_bat_recent["runs_above_avg_def"] = pd.to_numeric(bwar_bat_recent["runs_above_avg_def"], errors="coerce").fillna(0)

off_def_rows = []
for g in FOCUS_GROUPS:
    subset = bwar_bat_recent[bwar_bat_recent["group"] == g]
    n_ts = master[master["group"] == g].shape[0]

    off_runs = subset["runs_above_avg_off"].sum() / n_ts
    def_runs = subset["runs_above_avg_def"].sum() / n_ts
    total_runs = off_runs + def_runs

    off_def_rows.append([
        GROUP_LABELS[g],
        f"{off_runs:+.1f}",
        f"{def_runs:+.1f}",
        f"{total_runs:+.1f}",
    ])

print_table(
    off_def_rows,
    ["Group", "Offensive RAA/yr", "Defensive RAA/yr", "Total RAA/yr"],
    [22, 18, 18, 18],
)

# Position player (non-pitcher) WAR breakdown by WAR tier
print()
print("Position player WAR vs pitcher WAR in bWAR batting data:")
print("  (pitcher='Y' in bwar_batting means player is a pitcher)")
print()

bwar_bat_recent["is_pitcher"] = bwar_bat_recent["pitcher"] == "Y"

pp_rows = []
for g in FOCUS_GROUPS:
    subset = bwar_bat_recent[bwar_bat_recent["group"] == g]
    n_ts = master[master["group"] == g].shape[0]

    pos_war = subset[~subset["is_pitcher"]]["WAR"].sum() / n_ts
    pit_bat_war = subset[subset["is_pitcher"]]["WAR"].sum() / n_ts
    pos_count = subset[~subset["is_pitcher"]].shape[0] / n_ts
    pit_count = subset[subset["is_pitcher"]].shape[0] / n_ts

    pp_rows.append([
        GROUP_LABELS[g],
        f"{pos_war:.1f}",
        f"{pit_bat_war:.1f}",
        f"{pos_count:.1f}",
        f"{pit_count:.1f}",
    ])

print_table(
    pp_rows,
    ["Group", "Pos player bWAR/yr", "Pitcher bWAR/yr", "Pos players/yr", "Pitchers batting/yr"],
    [22, 20, 18, 16, 20],
)


# ══════════════════════════════════════════════════════════════════════
# 7. SYNTHESIS: The efficient winner blueprint
# ══════════════════════════════════════════════════════════════════════
print_header("7. SYNTHESIS — The Efficient Winner Blueprint")

# Compute key differentiating metrics
ew = master[master["group"] == "efficient_winner"]
cl = master[master["group"] == "cheap_loser"]
bb = master[master["group"] == "big_budget_winner"]

print("KEY DIFFERENTIATORS (efficient_winner vs cheap_loser):")
print()

# Pitching comparison
ew_era = pd.to_numeric(ew["pit_ERA"], errors="coerce").mean()
cl_era = pd.to_numeric(cl["pit_ERA"], errors="coerce").mean()
bb_era = pd.to_numeric(bb["pit_ERA"], errors="coerce").mean()
ew_fip = pd.to_numeric(ew["pit_FIP"], errors="coerce").mean()
cl_fip = pd.to_numeric(cl["pit_FIP"], errors="coerce").mean()
bb_fip = pd.to_numeric(bb["pit_FIP"], errors="coerce").mean()

print("  Pitching:")
print(f"    ERA:  Eff.Winners {ew_era:.2f} | Cheap Losers {cl_era:.2f} | Big-$ {bb_era:.2f}")
print(f"    FIP:  Eff.Winners {ew_fip:.2f} | Cheap Losers {cl_fip:.2f} | Big-$ {bb_fip:.2f}")
print()

# WAR efficiency
ew_warper10 = pd.to_numeric(ew["war_per_10mm"], errors="coerce").mean()
cl_warper10 = pd.to_numeric(cl["war_per_10mm"], errors="coerce").mean()
bb_warper10 = pd.to_numeric(bb["war_per_10mm"], errors="coerce").mean()
print("  WAR efficiency:")
print(f"    WAR per $10M:  Eff.Winners {ew_warper10:.2f} | Cheap Losers {cl_warper10:.2f} | Big-$ {bb_warper10:.2f}")
print()

# Homegrown
ew_hg = pd.to_numeric(ew["homegrown_war_pct"], errors="coerce").mean()
cl_hg = pd.to_numeric(cl["homegrown_war_pct"], errors="coerce").mean()
bb_hg = pd.to_numeric(bb["homegrown_war_pct"], errors="coerce").mean()
print("  Homegrown WAR %:")
print(f"    Eff.Winners {ew_hg:.3f} | Cheap Losers {cl_hg:.3f} | Big-$ {bb_hg:.3f}")
print()

# Roster turnover
ew_turn = pd.to_numeric(ew["roster_turnover_pct"], errors="coerce").mean()
cl_turn = pd.to_numeric(cl["roster_turnover_pct"], errors="coerce").mean()
bb_turn = pd.to_numeric(bb["roster_turnover_pct"], errors="coerce").mean()
print("  Roster turnover %:")
print(f"    Eff.Winners {ew_turn:.3f} | Cheap Losers {cl_turn:.3f} | Big-$ {bb_turn:.3f}")
print()

# 2+ WAR players
ew_2war = pd.to_numeric(ew["players_2plus_war"], errors="coerce").mean()
cl_2war = pd.to_numeric(cl["players_2plus_war"], errors="coerce").mean()
bb_2war = pd.to_numeric(bb["players_2plus_war"], errors="coerce").mean()
print("  2+ WAR players:")
print(f"    Eff.Winners {ew_2war:.1f} | Cheap Losers {cl_2war:.1f} | Big-$ {bb_2war:.1f}")
print()

# Gini
ew_gini = pd.to_numeric(ew["war_gini"], errors="coerce").mean()
cl_gini = pd.to_numeric(cl["war_gini"], errors="coerce").mean()
bb_gini = pd.to_numeric(bb["war_gini"], errors="coerce").mean()
print("  WAR Gini (inequality — higher = more concentrated):")
print(f"    Eff.Winners {ew_gini:.3f} | Cheap Losers {cl_gini:.3f} | Big-$ {bb_gini:.3f}")
print()

# Final summary: the actual roster construction recipe
print()
print("-" * 80)
print("ROSTER CONSTRUCTION RECIPE: How efficient winners differ from cheap losers")
print("-" * 80)
print()

# Compute the specific numbers for the summary
ew_pit_data = bwar_pit_recent[bwar_pit_recent["group"] == "efficient_winner"]
cl_pit_data = bwar_pit_recent[bwar_pit_recent["group"] == "cheap_loser"]
ew_n = master[master["group"] == "efficient_winner"].shape[0]
cl_n = master[master["group"] == "cheap_loser"].shape[0]

ew_sp_2plus = ew_pit_data[(ew_pit_data["GS"] >= ew_pit_data["G"] * 0.5) & (ew_pit_data["WAR"] >= 2.0)].shape[0] / ew_n
cl_sp_2plus = cl_pit_data[(cl_pit_data["GS"] >= cl_pit_data["G"] * 0.5) & (cl_pit_data["WAR"] >= 2.0)].shape[0] / cl_n

ew_rp_1plus = ew_pit_data[(ew_pit_data["GS"] < ew_pit_data["G"] * 0.5) & (ew_pit_data["WAR"] >= 1.0)].shape[0] / ew_n
cl_rp_1plus = cl_pit_data[(cl_pit_data["GS"] < cl_pit_data["G"] * 0.5) & (cl_pit_data["WAR"] >= 1.0)].shape[0] / cl_n

ew_good_players_data = combined_with_debut[(combined_with_debut["group"] == "efficient_winner") & (combined_with_debut["total_war"] >= 2.0)]
cl_good_players_data = combined_with_debut[(combined_with_debut["group"] == "cheap_loser") & (combined_with_debut["total_war"] >= 2.0)]
ew_hg_pct_2plus = ew_good_players_data["is_homegrown"].mean() * 100 if len(ew_good_players_data) > 0 else 0
cl_hg_pct_2plus = cl_good_players_data["is_homegrown"].mean() * 100 if len(cl_good_players_data) > 0 else 0

ew_4plus = combined[(combined["group"] == "efficient_winner") & (combined["total_war"] >= 4.0)].shape[0] / ew_n
cl_4plus = combined[(combined["group"] == "cheap_loser") & (combined["total_war"] >= 4.0)].shape[0] / cl_n
ew_2_3 = combined[(combined["group"] == "efficient_winner") & (combined["total_war"] >= 2.0) & (combined["total_war"] < 4.0)].shape[0] / ew_n
cl_2_3 = combined[(combined["group"] == "cheap_loser") & (combined["total_war"] >= 2.0) & (combined["total_war"] < 4.0)].shape[0] / cl_n

# Age summary
ew_age_data = good_with_age[good_with_age["group"] == "efficient_winner"]
cl_age_data = good_with_age[good_with_age["group"] == "cheap_loser"]
ew_prime_pct = ((ew_age_data["age"] >= 25) & (ew_age_data["age"] <= 29)).mean() * 100 if len(ew_age_data) > 0 else 0
cl_prime_pct = ((cl_age_data["age"] >= 25) & (cl_age_data["age"] <= 29)).mean() * 100 if len(cl_age_data) > 0 else 0
ew_young_pct = (ew_age_data["age"] < 25).mean() * 100 if len(ew_age_data) > 0 else 0
cl_young_pct = (cl_age_data["age"] < 25).mean() * 100 if len(cl_age_data) > 0 else 0

print(f"  1) PITCHING IS THE EDGE")
print(f"     SP with 2+ WAR:   {ew_sp_2plus:.1f}/yr vs {cl_sp_2plus:.1f}/yr (cheap losers)")
print(f"     RP with 1+ WAR:   {ew_rp_1plus:.1f}/yr vs {cl_rp_1plus:.1f}/yr")
print(f"     Team ERA:         {ew_era:.2f} vs {cl_era:.2f}")
print()
print(f"  2) DEPTH, NOT JUST STARS")
print(f"     4+ WAR players:   {ew_4plus:.1f}/yr vs {cl_4plus:.1f}/yr")
print(f"     2-4 WAR players:  {ew_2_3:.1f}/yr vs {cl_2_3:.1f}/yr")
print(f"     WAR Gini:         {ew_gini:.3f} vs {cl_gini:.3f} (lower = more spread)")
print()
print(f"  3) HOMEGROWN PIPELINE")
print(f"     HG% of 2+ WAR:   {ew_hg_pct_2plus:.1f}% vs {cl_hg_pct_2plus:.1f}%")
print(f"     HG WAR share:    {ew_hg:.3f} vs {cl_hg:.3f}")
print(f"     Roster turnover:  {ew_turn:.3f} vs {cl_turn:.3f}")
print()
print(f"  4) AGE PROFILE")
print(f"     Pre-prime (<25):  {ew_young_pct:.1f}% vs {cl_young_pct:.1f}% of 2+ WAR players")
print(f"     Prime (25-29):    {ew_prime_pct:.1f}% vs {cl_prime_pct:.1f}% of 2+ WAR players")
print()


# ══════════════════════════════════════════════════════════════════════
# 8. TEAM-BY-TEAM: Who are the efficient winners?
# ══════════════════════════════════════════════════════════════════════
print_header("8. TEAM-BY-TEAM — Efficient Winner Seasons")

ew_seasons = master[master["group"] == "efficient_winner"].sort_values(["team", "Season"])

team_rows = []
for _, row in ew_seasons.iterrows():
    team_rows.append([
        f"{row['team']} {int(row['Season'])}",
        f"{int(row['W'])}W",
        f"${int(row['payroll_mm'])}M",
        f"{row['total_team_war']:.1f}",
        f"{row['players_2plus_war']:.0f}",
        f"{row['homegrown_war_pct']:.3f}" if pd.notna(row["homegrown_war_pct"]) else "N/A",
        f"{row['war_per_10mm']:.2f}",
    ])

print_table(
    team_rows,
    ["Team-Season", "W", "Payroll", "Team WAR", "2+ WAR", "HG %", "WAR/$10M"],
    [16, 6, 8, 10, 8, 8, 10],
)

print()
print("=" * 80)
print("END OF DEEP DIVE")
print("=" * 80)
