# What Separates Cheap Winners from Cheap Losers in MLB?

A data-driven analysis of roster construction strategies across 150 MLB team-seasons (2022-2026). The question: when low-payroll teams make the playoffs (Brewers, Rays, Guardians, 2026 White Sox), what are they doing differently than low-payroll teams that lose (Angels, Marlins, Pirates, A's)?

## Key Findings

### 1. The Bullpen Is the Biggest Hidden Edge

Efficient winners get **40% of their pitching WAR from relievers**, compared to 31.5% for both cheap losers and big-budget winners. The bullpen WAR gap (3.3 WAR/year) is nearly as large as the rotation gap (2.9 WAR/year). This is the Rays/Guardians model: convert starters to high-leverage relief roles, use openers, treat the bullpen as a strategic weapon rather than an afterthought.

### 2. Contact and Speed Over Power

Efficient winners have a distinct offensive identity that differs from big-budget teams:

- **Lowest K% of any group** (21.3% vs 22.5% league avg). 10 of 14 efficient winner team-seasons are below league average.
- **17% more stolen bases per PA** than league average
- Below-average home run rate and ISO
- Average OBP

They don't out-slug anyone. They put the ball in play and run. This makes economic sense: contact skills and speed are cheaper to acquire than 40-homer power bats.

### 3. Stars AND Depth, Not One or the Other

Efficient winners average 2.4 players at 4+ WAR per year (cheap losers: 0.9) AND 5.9 players in the 2-4 WAR tier (cheap losers: 4.3). The "stars vs. depth" framing is a false dichotomy. These teams develop elite talent and supplement with solid contributors.

### The Angels Problem

The Angels are the inverse of the efficient winner profile on every dimension:

| Metric | Angels (avg) | Efficient Winners | Percentile |
|--------|:---:|:---:|:---:|
| Wins | 68.6 | 89.9 | 15th |
| WAR per $10M | 1.47 | 5.28 | 18th |
| Team K% (batting) | 25.5% | 21.3% | 93rd (bad) |
| Pitching BB% | 9.7% | 8.1% | 91st (bad) |

High-strikeout offense, walk-heavy pitching staff, no bullpen depth. The opposite of what works on a budget.

## Methodology

### Data Sources
- **Baseball Reference** via pybaseball: individual batting/pitching stats, bWAR, standings
- **Statcast** via pybaseball: exit velocity, barrel rate, hard hit rate
- **Payroll**: compiled from Spotrac/Cot's Baseball Contracts (approximate opening day payrolls)

### Team Classification
Each of 150 team-seasons (30 teams x 5 years) is classified by payroll tier (bottom/middle/top third) and performance (playoff appearance or 85+ wins):

| Group | Criteria | Count |
|-------|----------|:-----:|
| Efficient Winner | Bottom-third payroll + playoffs/85W | 14 |
| Cheap Loser | Bottom-third payroll + no playoffs | 36 |
| Big-Budget Winner | Top-third payroll + playoffs/85W | 30 |
| Expensive Loser | Top-third payroll + no playoffs | 20 |

### Statistical Approach
- **Hedges' g** effect sizes (bias-corrected for small samples) with 95% confidence intervals
- **Benjamini-Hochberg FDR correction** for multiple comparisons (27 independent tests)
- 15 of 27 non-redundant metrics survive FDR correction at q < 0.10

### Known Limitations
- **"Homegrown" is really "debuted with team."** bWAR tracks MLB appearances, not draft history. A minor leaguer traded pre-debut counts as homegrown for the acquiring team. This conflates drafting-and-developing with acquiring pre-debut talent.
- **Small group sizes.** 14 efficient winner team-seasons limits statistical power. Effect sizes should be interpreted with confidence interval widths in mind.
- **2026 payroll data is estimated** from preseason projections. Standings and playoff data are actual.
- **FanGraphs data unavailable.** The FanGraphs legacy API is blocked by Cloudflare, so wRC+, Def, and fielding metrics are not included. Analysis uses BRef-sourced stats and bWAR.

## Charts

| Chart | Description |
|-------|-------------|
| `scatter_homegrown_war.png` | Wins vs. homegrown WAR %, colored by payroll tier |
| `scatter_war_gini.png` | Wins vs. WAR concentration (Gini coefficient) |
| `group_bar_comparison.png` | Top 8 differentiating metrics by group |
| `radar_angels_vs_ew.png` | Angels profile vs. efficient winner benchmark |
| `heatmap_correlations.png` | Correlation matrix of key metrics |
| `timeline_white_sox.png` | White Sox 2022-2026 trajectory (41-121 to playoff team) |
| `timeline_angels.png` | Angels 2022-2026 trajectory |

## Pipeline

```
pull_standings.py ──────→ data/standings.csv
compile_payroll.py ─────→ data/payroll.csv
pull_player_stats.py ───→ data/bwar_batting.csv, data/bwar_pitching.csv
pull_team_stats.py ─────→ data/team_batting.csv, data/team_pitching.csv
pull_statcast_agg.py ───→ data/statcast_batting_agg.csv, statcast_pitching_agg.csv
                                    ↓
                          build_features.py → data/master.csv (150 rows, 75 columns)
                                    ↓
                    ┌───────────────┼───────────────┐
                analyze.py    angels_report.py   visualize.py
                    ↓               ↓                 ↓
             output/*.csv    output/angels_*.csv  output/*.png
```

Scripts 1-5 pull from external APIs (Baseball Reference, Statcast) and can run in parallel. `build_features.py` depends on all of them. The analysis scripts depend on `build_features.py`.

### Running It

```bash
pip install pybaseball pandas numpy scipy matplotlib seaborn

# Data pulls (slow, hits external APIs)
python pull_standings.py
python compile_payroll.py
python pull_player_stats.py
python pull_team_stats.py
python pull_statcast_agg.py

# Feature engineering + analysis
python build_features.py
python analyze.py
python angels_report.py
python visualize.py

# Deep dives
python deep_dive.py
python deep_dive2.py
```

`bwar_batting.csv` and `bwar_pitching.csv` are gitignored (18MB combined, full historical database). Run `pull_player_stats.py` to regenerate them.

## The Efficient Winner Seasons

| Team | Year | W | Payroll | WAR | WAR/$10M |
|------|:----:|:---:|:---:|:---:|:---:|
| BAL | 2023 | 101 | $54M | 43.3 | 8.02 |
| BAL | 2024 | 91 | $55M | 43.8 | 7.96 |
| TBR | 2023 | 99 | $82M | 53.1 | 6.48 |
| CLE | 2022 | 92 | $68M | 41.2 | 6.06 |
| CLE | 2026 | 85 | $65M | 40.8 | 6.28 |
| TBR | 2026 | 98 | $72M | 41.3 | 5.74 |
| TBR | 2022 | 86 | $86M | 40.6 | 4.72 |
| CLE | 2024 | 92 | $79M | 36.5 | 4.62 |
| KCR | 2024 | 86 | $94M | 41.3 | 4.39 |
| MIA | 2023 | 84 | $78M | 34.0 | 4.36 |
| CHW | 2026 | 84 | $85M | 36.6 | 4.31 |
| ARI | 2024 | 89 | $101M | 39.4 | 3.90 |
| ARI | 2023 | 84 | $92M | 33.9 | 3.69 |
| CLE | 2025 | 88 | $72M | 24.5 | 3.40 |
