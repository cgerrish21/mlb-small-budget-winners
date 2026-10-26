"""Quick summary of reliever metric stickiness results."""
import pandas as pd, numpy as np
from scipy import stats

# Reload
bwar = pd.read_csv('data/bwar_pitching.csv')
bwar = bwar[bwar['year_ID'].isin(range(2022,2027))].copy()
bwar['mlb_ID'] = pd.to_numeric(bwar['mlb_ID'], errors='coerce')
bwar = bwar.dropna(subset=['mlb_ID'])
bwar['mlb_ID'] = bwar['mlb_ID'].astype(int)
bwar['G'] = pd.to_numeric(bwar['G'], errors='coerce').fillna(0)
bwar['GS'] = pd.to_numeric(bwar['GS'], errors='coerce').fillna(0)
bwar['WAR'] = pd.to_numeric(bwar['WAR'], errors='coerce').fillna(0)
mask = (bwar['G']>=15) & ((bwar['GS']==0) | (bwar['GS']/bwar['G']<0.15))
rel = bwar[mask][['mlb_ID','year_ID','name_common','team','WAR','G','GS']].copy()

# Aggregate multi-stint rows (traded relievers have separate rows per team)
rel = (
    rel.groupby(['mlb_ID', 'year_ID'], as_index=False)
    .agg({
        'name_common': 'first',
        'team': 'first',
        'WAR': 'sum',
        'G': 'sum',
        'GS': 'sum',
    })
)

sc = pd.read_csv('data/reliever_statcast_full_cache.csv')
merged = rel.merge(sc, left_on=['mlb_ID','year_ID'], right_on=['player_id','Season'], how='inner')

print("=" * 80)
print("  SUMMARY: RELIEVER METRIC STICKINESS RANKINGS")
print("=" * 80)

categories = {
    'ff_avg_speed': ('Fastball Velocity', 'input'),
    'si_avg_speed': ('Sinker Velocity', 'input'),
    'sl_avg_speed': ('Slider Velocity', 'input'),
    'cu_avg_speed': ('Curveball Velocity', 'input'),
    'ch_avg_speed': ('Changeup Velocity', 'input'),
    'fc_avg_speed': ('Cutter Velocity', 'input'),
    'fs_avg_speed': ('Splitter Velocity', 'input'),
    'st_avg_speed': ('Sweeper Velocity', 'input'),
    'max_velo':     ('Max Velocity', 'input'),
    'velo_spread':  ('Velo Spread (max-min)', 'input'),
    'ff_avg_spin':  ('4-Seam Spin Rate', 'input'),
    'si_avg_spin':  ('Sinker Spin Rate', 'input'),
    'sl_avg_spin':  ('Slider Spin Rate', 'input'),
    'cu_avg_spin':  ('Curveball Spin Rate', 'input'),
    'ch_avg_spin':  ('Changeup Spin Rate', 'input'),
    'fc_avg_spin':  ('Cutter Spin Rate', 'input'),
    'fs_avg_spin':  ('Splitter Spin Rate', 'input'),
    'st_avg_spin':  ('Sweeper Spin Rate', 'input'),
    'n_pitch_types': ('# Pitch Types', 'input'),
    'pct_ff':       ('% 4-Seam FB', 'mix'),
    'pct_si':       ('% Sinker', 'mix'),
    'pct_fc':       ('% Cutter', 'mix'),
    'pct_sl':       ('% Slider', 'mix'),
    'pct_cu':       ('% Curveball', 'mix'),
    'pct_ch':       ('% Changeup', 'mix'),
    'pct_fs':       ('% Splitter', 'mix'),
    'pct_st':       ('% Sweeper', 'mix'),
    'pct_fastball':  ('% All Fastballs', 'mix'),
    'pct_breaking':  ('% All Breaking', 'mix'),
    'pct_offspeed':  ('% All Offspeed', 'mix'),
    'whiff_pct':    ('Overall Whiff %', 'approach'),
    'whiff_ff':     ('4-Seam Whiff %', 'approach'),
    'whiff_si':     ('Sinker Whiff %', 'approach'),
    'whiff_sl':     ('Slider Whiff %', 'approach'),
    'whiff_cu':     ('Curveball Whiff %', 'approach'),
    'whiff_ch':     ('Changeup Whiff %', 'approach'),
    'whiff_fc':     ('Cutter Whiff %', 'approach'),
    'whiff_st':     ('Sweeper Whiff %', 'approach'),
    'whiff_fs':     ('Splitter Whiff %', 'approach'),
    'k_pct':        ('K %', 'approach'),
    'put_away_pct': ('Put-Away %', 'approach'),
    'woba_against':  ('wOBA Against', 'outcome'),
    'est_woba':      ('Expected wOBA', 'outcome'),
    'ba_against':    ('BA Against', 'outcome'),
    'est_ba':        ('Expected BA', 'outcome'),
    'slg_against':   ('SLG Against', 'outcome'),
    'est_slg':       ('Expected SLG', 'outcome'),
    'hard_hit_pct':  ('Hard Hit %', 'outcome'),
    'rv_per100':     ('Run Value / 100', 'outcome'),
}

pairs = merged.merge(merged, on='player_id', suffixes=('_n','_n1'))
pairs = pairs[pairs['Season_n1'] == pairs['Season_n'] + 1]

print(f"\n  Relievers: {len(merged)} pitcher-seasons, {merged['player_id'].nunique()} unique")
print(f"  Year-over-year pairs: {len(pairs)}")

rows = []
for metric, (label, cat) in categories.items():
    col_n = f"{metric}_n"
    war_n1 = "WAR_n1"

    sub_same = merged[['WAR', metric]].dropna()
    if len(sub_same) < 30:
        continue
    r_same, _ = stats.pearsonr(sub_same['WAR'], sub_same[metric])

    if col_n in pairs.columns and f"{metric}_n1" in pairs.columns:
        sub_s = pairs[[col_n, f"{metric}_n1"]].dropna()
        r_sticky = stats.pearsonr(sub_s[col_n], sub_s[f"{metric}_n1"])[0] if len(sub_s)>=30 else np.nan
    else:
        r_sticky = np.nan

    if col_n in pairs.columns:
        sub_p = pairs[[col_n, war_n1]].dropna()
        r_pred = stats.pearsonr(sub_p[col_n], sub_p[war_n1])[0] if len(sub_p)>=30 else np.nan
        n_pred = len(sub_p)
    else:
        r_pred, n_pred = np.nan, 0

    rows.append({
        'Category': cat, 'Metric': label, 'col': metric,
        'r_same': round(r_same, 3),
        'r_sticky': round(r_sticky, 3) if not np.isnan(r_sticky) else np.nan,
        'r_pred': round(r_pred, 3) if not np.isnan(r_pred) else np.nan,
        'n_pairs': n_pred,
    })

df = pd.DataFrame(rows)

for cat, cat_label in [('input','PITCH CHARACTERISTICS (scourable)'),
                        ('mix','PITCH MIX'),
                        ('approach','APPROACH/RESULTS (partially scourable)'),
                        ('outcome','OUTCOME METRICS (descriptive)')]:
    subset = df[df['Category']==cat].sort_values('r_pred', key=abs, ascending=False)
    print(f"\n--- {cat_label} ---")
    print(f"{'Metric':30s} {'r_same':>8s} {'r_sticky':>8s} {'r_pred':>8s} {'n_pairs':>8s}")
    for _, r in subset.iterrows():
        sticky_str = f"{r['r_sticky']:.3f}" if not np.isnan(r['r_sticky']) else "  N/A"
        pred_str = f"{r['r_pred']:.3f}" if not np.isnan(r['r_pred']) else "  N/A"
        flag = ""
        if not np.isnan(r['r_pred']) and abs(r['r_pred']) >= 0.15: flag = " <--"
        if not np.isnan(r['r_pred']) and abs(r['r_pred']) >= 0.20: flag = " <-- **"
        print(f"{r['Metric']:30s} {r['r_same']:8.3f} {sticky_str:>8s} {pred_str:>8s} {r['n_pairs']:8.0f}{flag}")

print()
print("=" * 80)
print("  KEY FINDINGS")
print("=" * 80)

# Quintile breakdown for the 3 best predictive metrics
print("\n  QUINTILE ANALYSIS (Year-N metric quintile -> Year-N+1 avg WAR):")
for metric, label in [('k_pct', 'K%'), ('whiff_pct', 'Whiff%'), ('put_away_pct', 'Put-Away%'),
                       ('ff_avg_speed', 'FB Velo'), ('est_woba', 'xwOBA')]:
    col_n = f"{metric}_n"
    war_n1 = "WAR_n1"
    sub = pairs[[col_n, war_n1]].dropna()
    if len(sub) < 50:
        continue
    r, p = stats.pearsonr(sub[col_n], sub[war_n1])

    if r > 0:
        sub['Q'] = pd.qcut(sub[col_n], 5, labels=['Q1(worst)','Q2','Q3','Q4','Q5(best)'])
    else:
        sub['Q'] = pd.qcut(sub[col_n], 5, labels=['Q5(best)','Q4','Q3','Q2','Q1(worst)'])

    quint = sub.groupby('Q', observed=True)[war_n1].agg(['mean','count']).round(3)

    best_war = quint.loc[quint.index.str.contains('Q5'), 'mean'].values[0]
    worst_war = quint.loc[quint.index.str.contains('Q1'), 'mean'].values[0]
    gap = best_war - worst_war

    print(f"\n  {label} (r_pred={r:.3f}):")
    for idx, row in quint.iterrows():
        bar = "#" * int(row['mean'] * 15)
        print(f"    {idx:12s}  WAR={row['mean']:+.3f}  n={int(row['count']):3d}  {bar}")
    print(f"    Best-vs-worst gap: {gap:+.3f} WAR")

print("\n" + "=" * 80)
print("  BOTTOM LINE")
print("=" * 80)
print("""
  The three metrics that are both STICKY (repeatable year-to-year) and
  PREDICTIVE (correlated with next-year WAR) for relievers:

  1. K% (strikeout rate)
     - Stickiness: r=0.560 (moderately sticky)
     - Predictive: r=0.229 (strongest single predictor of next-yr WAR)
     - Top quintile averages 1.04 WAR next year vs 0.40 for bottom quintile
     - This is a TRUE skill that persists and matters.

  2. Whiff% (swing-and-miss rate)
     - Stickiness: r=0.659 (very sticky)
     - Predictive: r=0.184
     - Clean monotonic quintile trend (0.36 -> 0.36 -> 0.47 -> 0.67 -> 0.80)
     - Stickier than K% but slightly less predictive. Underlying swing-and-miss
       ability is more stable year-to-year than the K% outcome.

  3. Put-Away% (2-strike put-away rate)
     - Stickiness: r=0.344 (moderate)
     - Predictive: r=0.189
     - Top quintile: 0.89 WAR vs 0.34 for bottom quintile
     - Measures ability to finish at-bats. Less sticky than whiff%,
       but nearly as predictive as K%.

  What does NOT predict future WAR:
  - Hard hit % (r_pred=-0.026, basically zero)
  - Pitch mix / pitch type usage (all |r_pred| < 0.10)
  - Outcome metrics like wOBA, BA, run value (high same-year r but
    r_sticky < 0.25 and r_pred < 0.20 -- they describe luck + skill
    combined, and the luck part doesn't repeat)

  Velocity is a special case: extremely sticky (r=0.940) because it is
  a physical trait, but only weakly predictive (r=0.167). Fast pitchers
  are slightly more likely to be good next year, but the signal is modest.
  You would not draft a reliever for velocity alone.

  METHODOLOGICAL NOTE -- SURVIVORSHIP BIAS:
  The year-over-year analysis only includes relievers who appeared in BOTH
  year N and year N+1 (i.e., who stayed in the majors). Pitchers who were
  bad in year N and got cut or demoted are excluded. This likely UNDERSTATES
  the true predictive power of K%, whiff%, and put-away%, because the
  missing data points are disproportionately low-skill pitchers who produced
  zero WAR the next year. The reported r_pred values are conservative.
""")
