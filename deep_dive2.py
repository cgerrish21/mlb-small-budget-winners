"""
Deep Dive 2: Pushback Analysis

Three questions that stress-test the efficient-winner findings:

1. IS "CUT FASTER" CIRCULAR?
   Bad teams need to give young players reps. Is negative WAR on cheap losers
   coming from development time for young players (justified) or from veterans
   who should be cut?

2. HOW IS "HOMEGROWN" CALCULATED, AND IS IT MEANINGFUL?
   Methodological audit of the compute_homegrown_war function.

3. DEEPER DIVE ON CONTACT / SMALL-BALL OFFENSIVE IDENTITY
   Do efficient winners actually score fewer runs? Is low K% a team-wide
   pattern or driven by one team (CLE)? What % of their wins come from
   pitching vs offense?
"""

import os
import sys
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ── Load data ─────────────────────────────────────────────────────────
master = pd.read_csv(os.path.join(DATA_DIR, "master.csv"))
bwar_bat = pd.read_csv(os.path.join(DATA_DIR, "bwar_batting.csv"), encoding="utf-8")
bwar_pit = pd.read_csv(os.path.join(DATA_DIR, "bwar_pitching.csv"), encoding="utf-8")
player_bat = pd.read_csv(os.path.join(DATA_DIR, "player_batting.csv"), encoding="utf-8")
player_pit = pd.read_csv(os.path.join(DATA_DIR, "player_pitching.csv"), encoding="utf-8")

# Coerce numerics
for col in ["WAR", "PA", "G", "mlb_ID", "year_ID", "stint_ID"]:
    bwar_bat[col] = pd.to_numeric(bwar_bat[col], errors="coerce")
bwar_bat["WAR"] = bwar_bat["WAR"].fillna(0)
bwar_bat["PA"] = bwar_bat["PA"].fillna(0)
bwar_bat["stint_ID"] = bwar_bat["stint_ID"].fillna(1)

for col in ["WAR", "G", "GS", "mlb_ID", "year_ID", "stint_ID"]:
    bwar_pit[col] = pd.to_numeric(bwar_pit[col], errors="coerce")
bwar_pit["WAR"] = bwar_pit["WAR"].fillna(0)
bwar_pit["stint_ID"] = bwar_pit["stint_ID"].fillna(1)

YEARS = list(range(2022, 2027))
bwar_bat_recent = bwar_bat[bwar_bat["year_ID"].isin(YEARS)].copy()
bwar_pit_recent = bwar_pit[bwar_pit["year_ID"].isin(YEARS)].copy()

# Group lookup
group_lookup = master.set_index(["team", "Season"])["group"].to_dict()
bwar_bat_recent["group"] = bwar_bat_recent.apply(
    lambda r: group_lookup.get((r["team"], r["year_ID"]), None), axis=1
)
bwar_pit_recent["group"] = bwar_pit_recent.apply(
    lambda r: group_lookup.get((r["team"], r["year_ID"]), None), axis=1
)

FOCUS_GROUPS = ["efficient_winner", "cheap_loser", "big_budget_winner"]
GROUP_LABELS = {
    "efficient_winner": "Efficient Winners",
    "cheap_loser": "Cheap Losers",
    "big_budget_winner": "Big-Budget Winners",
}

# ── Build age lookup ──────────────────────────────────────────────────
age_map = {}
for df, id_col in [(player_bat, "mlbID"), (player_pit, "mlbID")]:
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df["Season"] = pd.to_numeric(df["Season"], errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    for _, row in df[df["Age"].notna()].iterrows():
        if pd.notna(row[id_col]) and pd.notna(row["Season"]):
            key = (int(row[id_col]), int(row["Season"]))
            if key not in age_map:
                age_map[key] = row["Age"]

def get_age(mlb_id, year):
    if pd.notna(mlb_id):
        return age_map.get((int(mlb_id), int(year)), np.nan)
    return np.nan

# Assign ages
bwar_bat_recent["age"] = bwar_bat_recent.apply(
    lambda r: get_age(r["mlb_ID"], r["year_ID"]), axis=1
)
bwar_pit_recent["age"] = bwar_pit_recent.apply(
    lambda r: get_age(r["mlb_ID"], r["year_ID"]), axis=1
)


def print_header(title):
    print()
    print("=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print()


def print_table(rows, headers, col_widths=None):
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for r in rows:
                if i < len(r):
                    max_w = max(max_w, len(str(r[i])))
            col_widths.append(max_w + 2)
    header_line = ""
    for i, h in enumerate(headers):
        header_line += str(h).ljust(col_widths[i])
    print(header_line)
    print("-" * sum(col_widths))
    for r in rows:
        line = ""
        for i, val in enumerate(r):
            line += str(val).ljust(col_widths[i])
        print(line)


print("=" * 90)
print("DEEP DIVE 2: Pushback Analysis")
print("=" * 90)
print()
print(f"Team-seasons: efficient_winner={len(master[master['group']=='efficient_winner'])}, "
      f"cheap_loser={len(master[master['group']=='cheap_loser'])}, "
      f"big_budget_winner={len(master[master['group']=='big_budget_winner'])}")
print()


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 1: IS "CUT FASTER" CIRCULAR?
# ═══════════════════════════════════════════════════════════════════════════════
print_header("QUESTION 1: IS 'CUT FASTER' ACTUALLY CIRCULAR?")

print("""The claim: efficient winners limit below-replacement playing time.
The pushback: bad teams NEED to give young players development reps.
The test: among negative-WAR players, are cheap losers giving time to
YOUNG players (development, justified) or OLD players (inertia, wasteful)?
""")

# ── BATTERS with negative WAR ────────────────────────────────────────
print("-" * 90)
print("  BATTERS with negative WAR (2022-2026)")
print("-" * 90)
print()

neg_bat = bwar_bat_recent[
    (bwar_bat_recent["WAR"] < 0) &
    (bwar_bat_recent["group"].isin(FOCUS_GROUPS)) &
    (bwar_bat_recent["age"].notna())
].copy()

# Age buckets
neg_bat["age_bucket"] = pd.cut(
    neg_bat["age"],
    bins=[0, 24.999, 29.999, 99],
    labels=["<25 (dev)", "25-29 (prime)", "30+ (vet)"]
)

print("Count of negative-WAR batter-seasons by age bucket:")
print()
rows = []
for g in FOCUS_GROUPS:
    sub = neg_bat[neg_bat["group"] == g]
    n_ts = len(master[master["group"] == g])
    total_n = len(sub)
    for bucket in ["<25 (dev)", "25-29 (prime)", "30+ (vet)"]:
        bucket_sub = sub[sub["age_bucket"] == bucket]
        count = len(bucket_sub)
        pct = count / total_n * 100 if total_n > 0 else 0
        avg_war = bucket_sub["WAR"].mean() if len(bucket_sub) > 0 else 0
        total_war = bucket_sub["WAR"].sum()
        avg_pa = bucket_sub["PA"].mean() if len(bucket_sub) > 0 else 0
        total_pa = bucket_sub["PA"].sum()
        rows.append([
            GROUP_LABELS[g], bucket,
            f"{count}", f"{pct:.1f}%",
            f"{avg_war:.2f}", f"{total_war:.1f}",
            f"{avg_pa:.0f}", f"{total_pa:.0f}",
            f"{count/n_ts:.1f}/yr"
        ])

print_table(
    rows,
    ["Group", "Age Bucket", "Count", "% of neg", "Avg WAR", "Total WAR", "Avg PA", "Total PA", "Per yr"],
    [22, 16, 8, 9, 10, 11, 9, 11, 9]
)

# ── Summary comparison: per team-season ──
print()
print("Per team-season summary (negative-WAR batters only):")
print()
summary_rows = []
for g in FOCUS_GROUPS:
    sub = neg_bat[neg_bat["group"] == g]
    n_ts = len(master[master["group"] == g])

    young = sub[sub["age_bucket"] == "<25 (dev)"]
    prime = sub[sub["age_bucket"] == "25-29 (prime)"]
    vet = sub[sub["age_bucket"] == "30+ (vet)"]

    summary_rows.append([
        GROUP_LABELS[g],
        f"{young['WAR'].sum()/n_ts:.1f}",
        f"{young['PA'].sum()/n_ts:.0f}",
        f"{prime['WAR'].sum()/n_ts:.1f}",
        f"{prime['PA'].sum()/n_ts:.0f}",
        f"{vet['WAR'].sum()/n_ts:.1f}",
        f"{vet['PA'].sum()/n_ts:.0f}",
        f"{sub['WAR'].sum()/n_ts:.1f}",
        f"{sub['PA'].sum()/n_ts:.0f}",
    ])

print_table(
    summary_rows,
    ["Group", "<25 WAR", "<25 PA", "25-29 WAR", "25-29 PA", "30+ WAR", "30+ PA", "Total WAR", "Total PA"],
    [22, 10, 9, 11, 10, 10, 9, 11, 10]
)

# ── PITCHERS with negative WAR ──────────────────────────────────────
print()
print("-" * 90)
print("  PITCHERS with negative WAR (2022-2026)")
print("-" * 90)
print()

neg_pit = bwar_pit_recent[
    (bwar_pit_recent["WAR"] < 0) &
    (bwar_pit_recent["group"].isin(FOCUS_GROUPS)) &
    (bwar_pit_recent["age"].notna())
].copy()

neg_pit["age_bucket"] = pd.cut(
    neg_pit["age"],
    bins=[0, 24.999, 29.999, 99],
    labels=["<25 (dev)", "25-29 (prime)", "30+ (vet)"]
)

rows_p = []
for g in FOCUS_GROUPS:
    sub = neg_pit[neg_pit["group"] == g]
    n_ts = len(master[master["group"] == g])
    total_n = len(sub)
    for bucket in ["<25 (dev)", "25-29 (prime)", "30+ (vet)"]:
        bucket_sub = sub[sub["age_bucket"] == bucket]
        count = len(bucket_sub)
        pct = count / total_n * 100 if total_n > 0 else 0
        avg_war = bucket_sub["WAR"].mean() if len(bucket_sub) > 0 else 0
        total_war = bucket_sub["WAR"].sum()
        avg_g = bucket_sub["G"].mean() if len(bucket_sub) > 0 else 0
        total_g = bucket_sub["G"].sum()
        rows_p.append([
            GROUP_LABELS[g], bucket,
            f"{count}", f"{pct:.1f}%",
            f"{avg_war:.2f}", f"{total_war:.1f}",
            f"{avg_g:.0f}", f"{total_g:.0f}",
            f"{count/n_ts:.1f}/yr"
        ])

print_table(
    rows_p,
    ["Group", "Age Bucket", "Count", "% of neg", "Avg WAR", "Total WAR", "Avg G", "Total G", "Per yr"],
    [22, 16, 8, 9, 10, 11, 8, 10, 9]
)

# Per team-season pitching summary
print()
print("Per team-season summary (negative-WAR pitchers only):")
print()
summary_rows_p = []
for g in FOCUS_GROUPS:
    sub = neg_pit[neg_pit["group"] == g]
    n_ts = len(master[master["group"] == g])

    young = sub[sub["age_bucket"] == "<25 (dev)"]
    prime = sub[sub["age_bucket"] == "25-29 (prime)"]
    vet = sub[sub["age_bucket"] == "30+ (vet)"]

    summary_rows_p.append([
        GROUP_LABELS[g],
        f"{young['WAR'].sum()/n_ts:.1f}",
        f"{young['G'].sum()/n_ts:.0f}",
        f"{prime['WAR'].sum()/n_ts:.1f}",
        f"{prime['G'].sum()/n_ts:.0f}",
        f"{vet['WAR'].sum()/n_ts:.1f}",
        f"{vet['G'].sum()/n_ts:.0f}",
        f"{sub['WAR'].sum()/n_ts:.1f}",
        f"{sub['G'].sum()/n_ts:.0f}",
    ])

print_table(
    summary_rows_p,
    ["Group", "<25 WAR", "<25 G", "25-29 WAR", "25-29 G", "30+ WAR", "30+ G", "Total WAR", "Total G"],
    [22, 10, 8, 11, 9, 10, 8, 11, 10]
)

# ── COMBINED (batting + pitching) negative WAR ──────────────────────
print()
print("-" * 90)
print("  COMBINED NEGATIVE WAR (batting + pitching, per team-season)")
print("-" * 90)
print()

combined_neg_rows = []
for g in FOCUS_GROUPS:
    n_ts = len(master[master["group"] == g])
    bat_sub = neg_bat[neg_bat["group"] == g]
    pit_sub = neg_pit[neg_pit["group"] == g]

    young_war = (bat_sub[bat_sub["age_bucket"] == "<25 (dev)"]["WAR"].sum() +
                 pit_sub[pit_sub["age_bucket"] == "<25 (dev)"]["WAR"].sum()) / n_ts
    prime_war = (bat_sub[bat_sub["age_bucket"] == "25-29 (prime)"]["WAR"].sum() +
                 pit_sub[pit_sub["age_bucket"] == "25-29 (prime)"]["WAR"].sum()) / n_ts
    vet_war = (bat_sub[bat_sub["age_bucket"] == "30+ (vet)"]["WAR"].sum() +
               pit_sub[pit_sub["age_bucket"] == "30+ (vet)"]["WAR"].sum()) / n_ts
    total_neg = young_war + prime_war + vet_war

    combined_neg_rows.append([
        GROUP_LABELS[g],
        f"{young_war:.1f}",
        f"{prime_war:.1f}",
        f"{vet_war:.1f}",
        f"{total_neg:.1f}",
    ])

print("Total negative WAR per team-season by age bucket (batting + pitching combined):")
print()
print_table(
    combined_neg_rows,
    ["Group", "<25 neg WAR", "25-29 neg WAR", "30+ neg WAR", "Total neg WAR"],
    [22, 14, 16, 14, 16]
)

# ── Key delta: vet negative WAR (the wasteful part) ──────────────────
print()
ew_bat_vet = neg_bat[(neg_bat["group"] == "efficient_winner") & (neg_bat["age_bucket"] == "30+ (vet)")]
cl_bat_vet = neg_bat[(neg_bat["group"] == "cheap_loser") & (neg_bat["age_bucket"] == "30+ (vet)")]
ew_pit_vet = neg_pit[(neg_pit["group"] == "efficient_winner") & (neg_pit["age_bucket"] == "30+ (vet)")]
cl_pit_vet = neg_pit[(neg_pit["group"] == "cheap_loser") & (neg_pit["age_bucket"] == "30+ (vet)")]
ew_n = len(master[master["group"] == "efficient_winner"])
cl_n = len(master[master["group"] == "cheap_loser"])

ew_vet_total = (ew_bat_vet["WAR"].sum() + ew_pit_vet["WAR"].sum()) / ew_n
cl_vet_total = (cl_bat_vet["WAR"].sum() + cl_pit_vet["WAR"].sum()) / cl_n
ew_young_total = (neg_bat[(neg_bat["group"] == "efficient_winner") & (neg_bat["age_bucket"] == "<25 (dev)")]["WAR"].sum() +
                  neg_pit[(neg_pit["group"] == "efficient_winner") & (neg_pit["age_bucket"] == "<25 (dev)")]["WAR"].sum()) / ew_n
cl_young_total = (neg_bat[(neg_bat["group"] == "cheap_loser") & (neg_bat["age_bucket"] == "<25 (dev)")]["WAR"].sum() +
                  neg_pit[(neg_pit["group"] == "cheap_loser") & (neg_pit["age_bucket"] == "<25 (dev)")]["WAR"].sum()) / cl_n

print("KEY FINDING:")
print(f"  Vet (30+) negative WAR per team-season:")
print(f"    Efficient winners:   {ew_vet_total:.1f} WAR")
print(f"    Cheap losers:        {cl_vet_total:.1f} WAR")
print(f"    Delta:               {cl_vet_total - ew_vet_total:.1f} WAR (losers waste more on vets)")
print()
print(f"  Young (<25) negative WAR per team-season:")
print(f"    Efficient winners:   {ew_young_total:.1f} WAR")
print(f"    Cheap losers:        {cl_young_total:.1f} WAR")
print(f"    Delta:               {cl_young_total - ew_young_total:.1f} WAR")
print()

# Also: PA/G given to 30+ negative-WAR players
ew_vet_pa = ew_bat_vet["PA"].sum() / ew_n
cl_vet_pa = cl_bat_vet["PA"].sum() / cl_n
ew_vet_g = ew_pit_vet["G"].sum() / ew_n
cl_vet_g = cl_pit_vet["G"].sum() / cl_n

print(f"  PA given to 30+ negative-WAR batters per team-season:")
print(f"    Efficient winners:   {ew_vet_pa:.0f} PA")
print(f"    Cheap losers:        {cl_vet_pa:.0f} PA")
print()
print(f"  Games given to 30+ negative-WAR pitchers per team-season:")
print(f"    Efficient winners:   {ew_vet_g:.0f} G")
print(f"    Cheap losers:        {cl_vet_g:.0f} G")

# Verdict
print()
print("VERDICT:")
vet_delta = cl_vet_total - ew_vet_total
young_delta = cl_young_total - ew_young_total
if abs(vet_delta) > abs(young_delta):
    print("  The negative WAR gap is driven MORE by veterans than by young development.")
    print("  Cheap losers are not just developing players; they are also giving too many")
    print("  reps to aging players who are below replacement level.")
    print(f"  Vet gap: {vet_delta:.1f} WAR | Young gap: {young_delta:.1f} WAR")
else:
    print("  The negative WAR gap is driven MORE by young player development than vets.")
    print("  This supports the pushback: cheap losers absorb more negative WAR because")
    print("  they are investing in player development, which is justified.")
    print(f"  Young gap: {young_delta:.1f} WAR | Vet gap: {vet_delta:.1f} WAR")
print()
print("  NOTE: Both groups have some wasteful vet playing time and some justified")
print("  development time. The question is the RATIO. See breakdown above.")


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 2: HOW IS "HOMEGROWN" CALCULATED, AND IS IT MEANINGFUL?
# ═══════════════════════════════════════════════════════════════════════════════
print_header("QUESTION 2: IS 'HOMEGROWN' MEANINGFUL? (Methodological Audit)")

print("""HOW compute_homegrown_war() WORKS (from build_features.py):

1. It loads the FULL bWAR history (all years back to 1871, not just 2022-2026).
2. For each player_ID, it finds the EARLIEST row by (year_ID, stint_ID).
3. The 'team' column on that earliest row = the player's "debut team."
4. If a player's current team matches their debut team, they are "homegrown."

THE PROBLEM: What "debut team" actually means.

bWAR tracks MLB appearances only. A player's debut team is the team where
they first appeared in the majors. This is NOT necessarily the team that
drafted them or developed them.

SCENARIO: Player drafted by Team A, spends 3 years in Team A's minors,
traded to Team B as a minor leaguer, debuts with Team B.
  -> compute_homegrown_war() calls this player "homegrown" for Team B.
  -> Team A gets ZERO credit for drafting and developing them.

This means "homegrown" conflates two different things:
  a) Actually drafting and developing a player through your own system
  b) Acquiring a pre-MLB player (via trade/IFA) and debuting them

For efficient winners like Tampa Bay and Cleveland, this matters because
they are known for acquiring minor league talent via trade (e.g., trading
veterans at the deadline for prospects). Those acquired prospects then
debut with TBR/CLE and get counted as "homegrown."
""")

# Let's quantify the issue: find players who appear with one team in
# earlier years of bwar data, then switch. Actually, we can't distinguish
# traded-as-minor-leaguers from the data. But we CAN show the limitation.

# Show example: players who debuted 2022-2026 with efficient winner teams
# How many of them had NO prior MLB stint with another team?
# (This is necessarily 100% since debut = first MLB appearance by definition)
# The real question is: how many were drafted by a different team?
# We can't answer that from bWAR data alone.

print("QUANTIFYING THE ISSUE:")
print()

# Find all debut-in-analysis-window players for each group
all_appearances = pd.concat([
    bwar_bat[["player_ID", "year_ID", "team", "stint_ID"]].copy(),
    bwar_pit[["player_ID", "year_ID", "team", "stint_ID"]].copy(),
], ignore_index=True)
all_appearances["year_ID"] = pd.to_numeric(all_appearances["year_ID"], errors="coerce")
all_appearances["stint_ID"] = pd.to_numeric(all_appearances["stint_ID"], errors="coerce").fillna(1)

debut_df = (
    all_appearances.sort_values(["year_ID", "stint_ID"])
    .groupby("player_ID")
    .first()
    .reset_index()[["player_ID", "team", "year_ID"]]
)
debut_df.columns = ["player_ID", "debut_team", "debut_year"]

# For each group, look at "homegrown" players (debut_team == current team)
# and see how many debuted recently (2022-2026) vs long ago
analysis_bat = bwar_bat_recent[["player_ID", "team", "year_ID", "WAR", "group"]].copy()
analysis_pit = bwar_pit_recent[["player_ID", "team", "year_ID", "WAR", "group"]].copy()
analysis_all = pd.concat([analysis_bat, analysis_pit], ignore_index=True)
analysis_all = analysis_all.merge(debut_df, on="player_ID", how="left")
analysis_all["is_homegrown"] = analysis_all["team"] == analysis_all["debut_team"]

# Aggregate per player-team-season
player_agg = (
    analysis_all.groupby(["player_ID", "team", "year_ID", "group", "debut_team", "debut_year", "is_homegrown"])
    .agg(total_war=("WAR", "sum"))
    .reset_index()
)

print("Among 'homegrown' players with 2+ WAR: when did they debut?")
print()

hg_good = player_agg[(player_agg["is_homegrown"]) & (player_agg["total_war"] >= 2.0)]

for g in FOCUS_GROUPS:
    sub = hg_good[hg_good["group"] == g]
    n_ts = len(master[master["group"] == g])
    if len(sub) == 0:
        continue
    recent_debut = sub[sub["debut_year"] >= 2022]
    older_debut = sub[sub["debut_year"] < 2022]
    print(f"  {GROUP_LABELS[g]}:")
    print(f"    Total HG 2+ WAR player-seasons:  {len(sub)} ({len(sub)/n_ts:.1f}/yr)")
    print(f"    Debuted 2022-2026:               {len(recent_debut)} ({len(recent_debut)/len(sub)*100:.1f}%)")
    print(f"    Debuted before 2022:             {len(older_debut)} ({len(older_debut)/len(sub)*100:.1f}%)")
    print()

print()
print("SPECIFIC EXAMPLES: 'homegrown' players on efficient winner teams")
print("who might have been acquired as minor leaguers (debut year = analysis year):")
print()

ew_hg = hg_good[hg_good["group"] == "efficient_winner"].copy()
ew_hg = ew_hg.merge(
    bwar_bat[["player_ID", "name_common"]].drop_duplicates(subset="player_ID"),
    on="player_ID", how="left"
)
# Also try pitching names
pit_names = bwar_pit[["player_ID", "name_common"]].drop_duplicates(subset="player_ID")
ew_hg["name_common"] = ew_hg["name_common"].fillna(
    ew_hg["player_ID"].map(pit_names.set_index("player_ID")["name_common"])
)

ew_hg_sorted = ew_hg.sort_values("total_war", ascending=False)
print("Top 20 'homegrown' 2+ WAR players on efficient winner teams:")
print()
rows_ex = []
for _, r in ew_hg_sorted.head(20).iterrows():
    rows_ex.append([
        r["name_common"] if pd.notna(r["name_common"]) else r["player_ID"],
        r["team"],
        f"{int(r['year_ID'])}",
        f"{r['total_war']:.1f}",
        f"{int(r['debut_year'])}",
        "Recent" if r["debut_year"] >= 2020 else "Established",
    ])

print_table(
    rows_ex,
    ["Player", "Team", "Season", "WAR", "MLB Debut", "Status"],
    [22, 6, 8, 6, 11, 14]
)

print()
print("METHODOLOGICAL CONCLUSION:")
print()
print("  The 'homegrown' metric measures 'debuted in your MLB uniform,' not")
print("  'drafted and developed in your system.' For teams like TBR and CLE")
print("  that aggressively acquire minor league talent via trades, this metric")
print("  OVERSTATES their development pipeline and UNDERSTATES their trade")
print("  acquisition skill.")
print()
print("  This is not necessarily wrong -- debuting a player is still a talent")
print("  evaluation act -- but it should be labeled 'debuted with team' rather")
print("  than 'homegrown.' The distinction matters for prescriptive advice:")
print("  'develop your own talent' vs 'acquire pre-debut talent cheaply' are")
print("  different strategies, but both show up as 'homegrown' in this metric.")
print()
print("  TO FIX: Would need draft data (e.g., from Baseball Reference draft")
print("  pages or Chadwick Bureau) to separate truly drafted-and-developed")
print("  players from acquired-as-minor-leaguers.")


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION 3: DEEPER DIVE ON CONTACT / SMALL-BALL OFFENSIVE IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════
print_header("QUESTION 3: CONTACT / SMALL-BALL OFFENSIVE IDENTITY")

# ── 3a: Full offensive profile comparison ─────────────────────────────
print("3a. FULL OFFENSIVE PROFILE COMPARISON")
print()

# Compute derived metrics
master["G"] = pd.to_numeric(master["W"], errors="coerce") + pd.to_numeric(master["L"], errors="coerce")
master["R_per_G"] = pd.to_numeric(master["bat_R"], errors="coerce") / master["G"]
master["RA_per_G"] = pd.to_numeric(master["pit_R"], errors="coerce") / master["G"]
master["HR_per_PA"] = pd.to_numeric(master["bat_HR"], errors="coerce") / pd.to_numeric(master["bat_PA"], errors="coerce")
master["SB_per_PA"] = pd.to_numeric(master["bat_SB"], errors="coerce") / pd.to_numeric(master["bat_PA"], errors="coerce")
master["K_per_PA"] = pd.to_numeric(master["bat_SO"], errors="coerce") / pd.to_numeric(master["bat_PA"], errors="coerce")

metrics = [
    ("bat_OBP", "OBP", ".3f"),
    ("bat_SLG", "SLG", ".3f"),
    ("bat_ISO", "ISO", ".3f"),
    ("bat_BABIP", "BABIP", ".3f"),
    ("bat_BB%", "BB%", ".3f"),
    ("bat_K%", "K%", ".3f"),
    ("HR_per_PA", "HR/PA", ".4f"),
    ("SB_per_PA", "SB/PA", ".4f"),
    ("R_per_G", "R/G (scored)", ".2f"),
    ("RA_per_G", "RA/G (allowed)", ".2f"),
]

# Compute league average (all teams) for context
rows_off = []
for col, label, fmt in metrics:
    vals = {}
    for g in FOCUS_GROUPS:
        subset = master[master["group"] == g]
        v = pd.to_numeric(subset[col], errors="coerce").mean()
        vals[g] = v
    lg_avg = pd.to_numeric(master[col], errors="coerce").mean()

    rows_off.append([
        label,
        f"{vals['efficient_winner']:{fmt}}",
        f"{vals['cheap_loser']:{fmt}}",
        f"{vals['big_budget_winner']:{fmt}}",
        f"{lg_avg:{fmt}}",
        f"{vals['efficient_winner'] - lg_avg:+{fmt}}",
    ])

print_table(
    rows_off,
    ["Metric", "Eff. Winners", "Cheap Losers", "Big-$ Winners", "Lg Avg", "EW vs Lg"],
    [18, 14, 14, 16, 10, 12]
)

# ── 3b: Run production comparison ────────────────────────────────────
print()
print("3b. RUN PRODUCTION: Do efficient winners score fewer runs?")
print()

run_rows = []
for g in FOCUS_GROUPS:
    sub = master[master["group"] == g]
    r_scored = sub["R_per_G"].mean()
    r_allowed = sub["RA_per_G"].mean()
    run_diff = (r_scored - r_allowed)
    total_r = pd.to_numeric(sub["bat_R"], errors="coerce").mean()
    total_ra = pd.to_numeric(sub["pit_R"], errors="coerce").mean()
    wins = sub["W"].mean()

    run_rows.append([
        GROUP_LABELS[g],
        f"{r_scored:.2f}",
        f"{r_allowed:.2f}",
        f"{run_diff:+.2f}",
        f"{total_r:.0f}",
        f"{total_ra:.0f}",
        f"{total_r - total_ra:+.0f}",
        f"{wins:.0f}",
    ])

# Also add league average
lg_r = master["R_per_G"].mean()
lg_ra = master["RA_per_G"].mean()
run_rows.append([
    "LEAGUE AVERAGE",
    f"{lg_r:.2f}",
    f"{lg_ra:.2f}",
    f"{lg_r - lg_ra:+.2f}",
    f"{pd.to_numeric(master['bat_R'], errors='coerce').mean():.0f}",
    f"{pd.to_numeric(master['pit_R'], errors='coerce').mean():.0f}",
    f"{pd.to_numeric(master['bat_R'], errors='coerce').mean() - pd.to_numeric(master['pit_R'], errors='coerce').mean():+.0f}",
    f"{master['W'].mean():.0f}",
])

print_table(
    run_rows,
    ["Group", "R/G", "RA/G", "Diff/G", "Total R", "Total RA", "Run Diff", "Avg W"],
    [22, 8, 8, 9, 10, 10, 10, 8]
)

# ── 3c: Offense vs Pitching contribution ─────────────────────────────
print()
print("3c. WHERE DO THE WINS COME FROM: Offense vs Pitching?")
print()

# Method 1: bat_WAR vs pit_WAR ratio
print("Method 1: WAR split (bat_WAR vs pit_WAR)")
print()

war_split_rows = []
for g in FOCUS_GROUPS:
    sub = master[master["group"] == g]
    bat_war = pd.to_numeric(sub["bat_WAR"], errors="coerce").mean()
    pit_war = pd.to_numeric(sub["pit_WAR"], errors="coerce").mean()
    total_war = bat_war + pit_war
    bat_pct = bat_war / total_war * 100 if total_war > 0 else 0
    pit_pct = pit_war / total_war * 100 if total_war > 0 else 0

    war_split_rows.append([
        GROUP_LABELS[g],
        f"{bat_war:.1f}",
        f"{pit_war:.1f}",
        f"{total_war:.1f}",
        f"{bat_pct:.1f}%",
        f"{pit_pct:.1f}%",
    ])

print_table(
    war_split_rows,
    ["Group", "Bat WAR", "Pit WAR", "Total WAR", "Bat %", "Pit %"],
    [22, 10, 10, 12, 10, 10]
)

# Method 2: runs above/below league average
print()
print("Method 2: Runs scored/allowed vs league average per game")
print()

lg_rpg = master["R_per_G"].mean()
lg_rapg = master["RA_per_G"].mean()

runs_method_rows = []
for g in FOCUS_GROUPS:
    sub = master[master["group"] == g]
    avg_rpg = sub["R_per_G"].mean()
    avg_rapg = sub["RA_per_G"].mean()
    off_edge = avg_rpg - lg_rpg     # positive = score more than avg
    pit_edge = lg_rapg - avg_rapg   # positive = allow fewer than avg (note reversal)
    total_edge = off_edge + pit_edge

    off_share = off_edge / total_edge * 100 if total_edge > 0 else 0
    pit_share = pit_edge / total_edge * 100 if total_edge > 0 else 0

    runs_method_rows.append([
        GROUP_LABELS[g],
        f"{off_edge:+.2f}",
        f"{pit_edge:+.2f}",
        f"{total_edge:+.2f}",
        f"{off_share:.1f}%" if total_edge > 0 else "N/A",
        f"{pit_share:.1f}%" if total_edge > 0 else "N/A",
    ])

print_table(
    runs_method_rows,
    ["Group", "Off edge (R/G)", "Pit edge (R/G)", "Total edge", "Off share", "Pit share"],
    [22, 16, 16, 13, 12, 12]
)
print()
print("  Off edge = team R/G minus league avg R/G (positive = better offense)")
print("  Pit edge = league avg RA/G minus team RA/G (positive = better pitching)")

# ── 3d: Is low K% driven by Cleveland? ───────────────────────────────
print()
print("-" * 90)
print("3d. IS LOW K% DRIVEN BY CLEVELAND?")
print("-" * 90)
print()
print("K% for each of the 14 efficient winner team-seasons:")
print()

ew_seasons = master[master["group"] == "efficient_winner"].sort_values(["bat_K%"])

kpct_rows = []
for _, row in ew_seasons.iterrows():
    team = row["team"]
    season = int(row["Season"])
    kpct = pd.to_numeric(row["bat_K%"], errors="coerce")
    bb_pct = pd.to_numeric(row["bat_BB%"], errors="coerce")
    babip = pd.to_numeric(row["bat_BABIP"], errors="coerce")
    obp = pd.to_numeric(row["bat_OBP"], errors="coerce")
    slg = pd.to_numeric(row["bat_SLG"], errors="coerce")
    wins = int(row["W"])

    # Compute league rank for that season
    season_data = master[master["Season"] == season].sort_values("bat_K%")
    rank = list(season_data["team"]).index(team) + 1

    kpct_rows.append([
        f"{team} {season}",
        f"{wins}W",
        f"{kpct:.3f}",
        f"#{rank}/30",
        f"{bb_pct:.3f}",
        f"{babip:.3f}",
        f"{obp:.3f}",
        f"{slg:.3f}",
    ])

print_table(
    kpct_rows,
    ["Team-Season", "W", "K%", "K% Rank", "BB%", "BABIP", "OBP", "SLG"],
    [14, 6, 8, 10, 8, 8, 8, 8]
)

# Stats
ew_kpct_values = pd.to_numeric(ew_seasons["bat_K%"], errors="coerce")
lg_kpct = pd.to_numeric(master["bat_K%"], errors="coerce").mean()
cle_seasons = ew_seasons[ew_seasons["team"] == "CLE"]
non_cle = ew_seasons[ew_seasons["team"] != "CLE"]

print()
print(f"  League avg K%:           {lg_kpct:.3f}")
print(f"  Eff. winner avg K%:      {ew_kpct_values.mean():.3f}")
print(f"  CLE avg K%:              {pd.to_numeric(cle_seasons['bat_K%'], errors='coerce').mean():.3f}")
print(f"  Non-CLE eff winners K%:  {pd.to_numeric(non_cle['bat_K%'], errors='coerce').mean():.3f}")
print()

# How many EW team-seasons are below league average K%?
below_lg = (ew_kpct_values < lg_kpct).sum()
print(f"  Eff winner team-seasons with K% below lg avg: {below_lg}/{len(ew_kpct_values)}")
print()

if pd.to_numeric(non_cle["bat_K%"], errors="coerce").mean() < lg_kpct:
    print("  VERDICT: Low K% is NOT just a Cleveland effect. Even excluding CLE's")
    print(f"  4 seasons, the remaining {len(non_cle)} efficient winner seasons average")
    print(f"  {pd.to_numeric(non_cle['bat_K%'], errors='coerce').mean():.3f} K%, still below league avg of {lg_kpct:.3f}.")
else:
    print("  VERDICT: Low K% IS largely a Cleveland effect. Excluding CLE,")
    print(f"  the remaining {len(non_cle)} efficient winner seasons average")
    print(f"  {pd.to_numeric(non_cle['bat_K%'], errors='coerce').mean():.3f} K%, AT or ABOVE league avg of {lg_kpct:.3f}.")

print()

# ── 3e: Broader team-by-team offensive profile ───────────────────────
print()
print("-" * 90)
print("3e. TEAM-BY-TEAM OFFENSIVE IDENTITY (all 14 efficient winner seasons)")
print("-" * 90)
print()

ew_sorted = master[master["group"] == "efficient_winner"].sort_values(["team", "Season"])
profile_rows = []
for _, row in ew_sorted.iterrows():
    team = row["team"]
    season = int(row["Season"])
    rpg = row["R_per_G"]
    rapg = row["RA_per_G"]
    kpct = pd.to_numeric(row["bat_K%"], errors="coerce")
    bb_pct = pd.to_numeric(row["bat_BB%"], errors="coerce")
    hr_per_pa = pd.to_numeric(row["bat_HR"], errors="coerce") / pd.to_numeric(row["bat_PA"], errors="coerce")
    sb = pd.to_numeric(row["bat_SB"], errors="coerce")
    iso = pd.to_numeric(row["bat_ISO"], errors="coerce")
    bat_war = pd.to_numeric(row["bat_WAR"], errors="coerce")
    pit_war = pd.to_numeric(row["pit_WAR"], errors="coerce")
    total_war = bat_war + pit_war
    pit_share = pit_war / total_war * 100 if total_war > 0 else 0

    profile_rows.append([
        f"{team} {season}",
        f"{int(row['W'])}W",
        f"{rpg:.2f}",
        f"{rapg:.2f}",
        f"{kpct:.3f}",
        f"{iso:.3f}",
        f"{int(sb)}",
        f"{bat_war:.1f}",
        f"{pit_war:.1f}",
        f"{pit_share:.0f}%",
    ])

print_table(
    profile_rows,
    ["Team-Season", "W", "R/G", "RA/G", "K%", "ISO", "SB", "bWAR", "pWAR", "Pit%"],
    [14, 6, 7, 7, 8, 8, 5, 7, 7, 6]
)

# Compute how many are "pitching-led" (pit% > 50%) vs "offense-led"
pit_led = sum(1 for _, r in ew_sorted.iterrows()
              if pd.to_numeric(r["pit_WAR"], errors="coerce") >
              pd.to_numeric(r["bat_WAR"], errors="coerce"))
print()
print(f"  Pitching-led (pit_WAR > bat_WAR): {pit_led}/{len(ew_sorted)} team-seasons")
print(f"  Offense-led:                      {len(ew_sorted) - pit_led}/{len(ew_sorted)} team-seasons")

# ── 3f: Pitching quality comparison ──────────────────────────────────
print()
print()
print("-" * 90)
print("3f. PITCHING QUALITY (team-level stats)")
print("-" * 90)
print()

pit_metrics = [
    ("pit_ERA", "ERA", ".2f"),
    ("pit_FIP", "FIP", ".2f"),
    ("pit_WHIP", "WHIP", ".3f"),
    ("pit_K/9", "K/9", ".2f"),
    ("pit_BB/9", "BB/9", ".2f"),
    ("pit_HR/9", "HR/9", ".2f"),
    ("pit_K%", "Pit K%", ".3f"),
    ("pit_BB%", "Pit BB%", ".3f"),
]

pit_rows = []
for col, label, fmt in pit_metrics:
    vals = {}
    for g in FOCUS_GROUPS:
        subset = master[master["group"] == g]
        v = pd.to_numeric(subset[col], errors="coerce").mean()
        vals[g] = v
    lg = pd.to_numeric(master[col], errors="coerce").mean()

    pit_rows.append([
        label,
        f"{vals['efficient_winner']:{fmt}}",
        f"{vals['cheap_loser']:{fmt}}",
        f"{vals['big_budget_winner']:{fmt}}",
        f"{lg:{fmt}}",
    ])

print_table(
    pit_rows,
    ["Metric", "Eff. Winners", "Cheap Losers", "Big-$ Winners", "Lg Avg"],
    [12, 14, 14, 16, 10]
)


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════════
print_header("SYNTHESIS: Answers to the Three Pushback Questions")

print("""
QUESTION 1: Is "cut faster" circular?
""")
print(f"  Partially. Cheap losers DO give more development time to young players,")
print(f"  but they ALSO waste more time on declining veterans.")
print(f"  Combined negative WAR by age bucket (per team-season):")
for r in combined_neg_rows:
    print(f"    {r[0]:25s}: <25={r[1]:>6s} | 25-29={r[2]:>6s} | 30+={r[3]:>6s} | Total={r[4]:>6s}")

print()
print("""
QUESTION 2: Is "homegrown" meaningful?
""")
print(f"  The metric measures 'debuted in your uniform,' not 'drafted and developed")
print(f"  in your system.' This conflates internal development with acquiring pre-MLB")
print(f"  talent via trade. For teams like TBR and CLE, this overstates their farm")
print(f"  system output and understates their trade acquisition skill.")
print(f"  The finding (efficient winners have higher homegrown WAR %) is still")
print(f"  directionally useful but should be reframed as 'higher proportion of WAR")
print(f"  from players who debuted with the team' rather than 'better at developing")
print(f"  drafted players.'")

print()
print("""
QUESTION 3: Contact / small-ball identity?
""")

ew_data = master[master["group"] == "efficient_winner"]
bb_data = master[master["group"] == "big_budget_winner"]

ew_rpg = ew_data["R_per_G"].mean()
bb_rpg = bb_data["R_per_G"].mean()
ew_rapg = ew_data["RA_per_G"].mean()
bb_rapg = bb_data["RA_per_G"].mean()
ew_bat_war_avg = pd.to_numeric(ew_data["bat_WAR"], errors="coerce").mean()
ew_pit_war_avg = pd.to_numeric(ew_data["pit_WAR"], errors="coerce").mean()
bb_bat_war_avg = pd.to_numeric(bb_data["bat_WAR"], errors="coerce").mean()
bb_pit_war_avg = pd.to_numeric(bb_data["pit_WAR"], errors="coerce").mean()

print(f"  a) Efficient winners score {ew_rpg:.2f} R/G vs big-budget {bb_rpg:.2f} R/G")
print(f"     ({ew_rpg - bb_rpg:+.2f} R/G difference, or ~{(ew_rpg - bb_rpg) * 162:+.0f} runs/season)")
print()
print(f"  b) But they allow {ew_rapg:.2f} RA/G vs big-budget {bb_rapg:.2f} RA/G")
print(f"     ({bb_rapg - ew_rapg:+.2f} RA/G edge for efficient winners)")
print()
print(f"  c) WAR split: Eff winners {ew_bat_war_avg:.1f} bWAR / {ew_pit_war_avg:.1f} pWAR")
print(f"                Big-budget  {bb_bat_war_avg:.1f} bWAR / {bb_pit_war_avg:.1f} pWAR")
print(f"     -> WAR split is similar ({ew_pit_war_avg/(ew_bat_war_avg+ew_pit_war_avg)*100:.1f}% pit vs {bb_pit_war_avg/(bb_bat_war_avg+bb_pit_war_avg)*100:.1f}% pit)")
ew_off_edge = ew_rpg - lg_rpg
ew_pit_edge = lg_rapg - ew_rapg
ew_total_edge = ew_off_edge + ew_pit_edge
ew_off_share = ew_off_edge / ew_total_edge * 100 if ew_total_edge > 0 else 0
print(f"     -> vs league average: EW's above-average wins come {ew_off_share:.0f}% from offense,")
print(f"        only {100 - ew_off_share:.0f}% from pitching. They outscore the league (+{ew_off_edge:.2f} R/G)")
print(f"        but barely out-pitch it (+{ew_pit_edge:.2f} RA/G saved).")
print(f"     -> vs big-budget: EW score ~{(ew_rpg - bb_rpg) * 162:+.0f} runs/season but also allow")
print(f"        ~{(ew_rapg - bb_rapg) * 162:+.0f} more runs. Their offense is good-not-great;")
print(f"        their pitching is adequate. They succeed with FEWER resources by")
print(f"        putting the ball in play (low K%), manufacturing runs, and keeping")
print(f"        pitching costs down while still being competent.")
print()

non_cle_kpct = pd.to_numeric(
    master[(master["group"] == "efficient_winner") & (master["team"] != "CLE")]["bat_K%"],
    errors="coerce"
).mean()
cle_kpct = pd.to_numeric(
    master[(master["group"] == "efficient_winner") & (master["team"] == "CLE")]["bat_K%"],
    errors="coerce"
).mean()

print(f"  d) CLE effect on K%: CLE avg {cle_kpct:.3f} | Others avg {non_cle_kpct:.3f} | Lg avg {lg_kpct:.3f}")
if non_cle_kpct < lg_kpct:
    print(f"     Low K% is a BROAD pattern, not just CLE.")
    print(f"     {below_lg} of {len(ew_kpct_values)} EW seasons are below league avg K%.")
else:
    print(f"     Low K% is LARGELY a CLE effect. Remove CLE and the pattern weakens.")
    print(f"     Only {below_lg} of {len(ew_kpct_values)} EW seasons are below league avg K%.")

print()
print("=" * 90)
print("END OF DEEP DIVE 2")
print("=" * 90)
