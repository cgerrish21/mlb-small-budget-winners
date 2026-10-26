"""
Generate publication-quality charts for the GitHub README.

Charts:
1. Scatter: wins vs. homegrown WAR % (colored by payroll tier)
2. Scatter: wins vs. WAR Gini coefficient
3. Bar: group means for top differentiating metrics
4. Radar: Angels profile vs. efficient winner average
5. Heatmap: correlation matrix of key metrics
6. Timeline: White Sox metric trajectory 2022-2026
7. Timeline: Angels metric trajectory 2022-2026

Output: output/*.png
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from config import DATA_DIR, OUTPUT_DIR

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# Color palette
COLORS = {
    "efficient_winner": "#2ecc71",   # green
    "cheap_loser": "#e74c3c",        # red
    "big_budget_winner": "#3498db",  # blue
    "expensive_loser": "#e67e22",    # orange
    "middle_winner": "#9b59b6",      # purple
    "middle_loser": "#95a5a6",       # gray
}

TIER_COLORS = {
    "bottom": "#e74c3c",
    "middle": "#f39c12",
    "top": "#3498db",
}

ANGELS_COLOR = "#BA0021"  # Angels red
CHW_COLOR = "#27251F"     # White Sox black


def load_data():
    master_path = os.path.join(DATA_DIR, "master.csv")
    if not os.path.exists(master_path):
        print(f"ERROR: {master_path} not found. Run build_features.py first.", flush=True)
        sys.exit(1)
    return pd.read_csv(master_path)


def plot_scatter_homegrown(df):
    """Scatter: wins vs. homegrown WAR % colored by payroll tier."""
    if "homegrown_war_pct" not in df.columns or "payroll_tier" not in df.columns:
        print("  Skipping homegrown scatter (column missing)", flush=True)
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for tier, color in TIER_COLORS.items():
        mask = df["payroll_tier"] == tier
        sub = df[mask]
        ax.scatter(sub["homegrown_war_pct"] * 100, sub["W"],
                   c=color, alpha=0.6, s=40, label=f"{tier.title()} third payroll")

    # Highlight Angels
    angels = df[df["team"] == "LAA"]
    ax.scatter(angels["homegrown_war_pct"] * 100, angels["W"],
               c=ANGELS_COLOR, s=120, marker="*", zorder=5, label="Angels")
    for _, row in angels.iterrows():
        ax.annotate(f"LAA {int(row['Season'])}",
                    (row["homegrown_war_pct"] * 100, row["W"]),
                    fontsize=8, ha="left", va="bottom", color=ANGELS_COLOR)

    ax.set_xlabel("Homegrown WAR %")
    ax.set_ylabel("Wins")
    ax.set_title("Wins vs. Homegrown WAR % (2022-2026, colored by payroll tier)")
    ax.legend(loc="lower right", fontsize=9)

    out = os.path.join(OUTPUT_DIR, "scatter_homegrown_war.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}", flush=True)


def plot_scatter_gini(df):
    """Scatter: wins vs. WAR Gini coefficient."""
    if "war_gini" not in df.columns:
        print("  Skipping Gini scatter (column missing)", flush=True)
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for group, color in COLORS.items():
        mask = df["group"] == group
        sub = df[mask]
        if len(sub) == 0:
            continue
        label = group.replace("_", " ").title()
        ax.scatter(sub["war_gini"], sub["W"],
                   c=color, alpha=0.6, s=40, label=label)

    # Highlight Angels
    angels = df[df["team"] == "LAA"]
    ax.scatter(angels["war_gini"], angels["W"],
               c=ANGELS_COLOR, s=120, marker="*", zorder=5, label="Angels")
    for _, row in angels.iterrows():
        ax.annotate(f"LAA {int(row['Season'])}",
                    (row["war_gini"], row["W"]),
                    fontsize=8, ha="left", va="bottom", color=ANGELS_COLOR)

    ax.set_xlabel("WAR Gini Coefficient (higher = more concentrated)")
    ax.set_ylabel("Wins")
    ax.set_title("Wins vs. WAR Distribution (Gini), 2022-2026")
    ax.legend(loc="upper right", fontsize=8)

    out = os.path.join(OUTPUT_DIR, "scatter_war_gini.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}", flush=True)


def plot_group_bars(df):
    """Bar chart: group means for top differentiating metrics."""
    rankings_path = os.path.join(OUTPUT_DIR, "metric_rankings.csv")
    if not os.path.exists(rankings_path):
        print("  Skipping group bars (metric_rankings.csv missing)", flush=True)
        return

    rankings = pd.read_csv(rankings_path)
    # Filter out redundant metrics if the column exists
    if "is_redundant" in rankings.columns:
        rankings = rankings[~rankings["is_redundant"]]
    top = rankings.head(8)

    groups = ["efficient_winner", "cheap_loser", "big_budget_winner", "expensive_loser"]
    group_labels = ["Efficient\nWinners", "Cheap\nLosers", "Big Budget\nWinners", "Expensive\nLosers"]

    n_metrics = len(top)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (_, mrow) in enumerate(top.iterrows()):
        if i >= len(axes):
            break
        ax = axes[i]
        col = mrow["column"]
        if col not in df.columns:
            ax.set_visible(False)
            continue

        means = []
        colors = []
        for g in groups:
            vals = pd.to_numeric(df[df["group"] == g][col], errors="coerce")
            means.append(vals.mean())
            colors.append(COLORS[g])

        bars = ax.bar(range(len(groups)), means, color=colors, alpha=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(group_labels, fontsize=8)
        ax.set_title(mrow["metric"], fontsize=10, fontweight="bold")

        # Add value labels
        for bar, val in zip(bars, means):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Differentiating Metrics by Group (2022-2026)", fontsize=14, y=1.02)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "group_bar_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}", flush=True)


def plot_radar(df):
    """Radar chart: Angels profile vs. efficient winner average."""
    rankings_path = os.path.join(OUTPUT_DIR, "metric_rankings.csv")
    if not os.path.exists(rankings_path):
        print("  Skipping radar (metric_rankings.csv missing)", flush=True)
        return

    rankings = pd.read_csv(rankings_path)
    if "is_redundant" in rankings.columns:
        rankings = rankings[~rankings["is_redundant"]]
    top = rankings.head(8)
    cols = [c for c in top["column"].tolist() if c in df.columns]
    labels = [top[top["column"] == c]["metric"].values[0] for c in cols]

    if len(cols) < 3:
        print("  Skipping radar (not enough metrics)", flush=True)
        return

    ew = df[df["group"] == "efficient_winner"]
    angels = df[df["team"] == "LAA"]

    # Normalize to 0-1 scale using all data
    ew_vals = []
    ang_vals = []
    for col in cols:
        all_v = pd.to_numeric(df[col], errors="coerce")
        mn, mx = all_v.min(), all_v.max()
        rng = mx - mn if mx != mn else 1

        ew_mean = pd.to_numeric(ew[col], errors="coerce").mean()
        ang_mean = pd.to_numeric(angels[col], errors="coerce").mean()

        ew_vals.append((ew_mean - mn) / rng)
        ang_vals.append((ang_mean - mn) / rng)

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    ew_vals += ew_vals[:1]
    ang_vals += ang_vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, ew_vals, "o-", color="#2ecc71", linewidth=2, label="Efficient Winners")
    ax.fill(angles, ew_vals, alpha=0.15, color="#2ecc71")
    ax.plot(angles, ang_vals, "o-", color=ANGELS_COLOR, linewidth=2, label="Angels (avg 2022-2026)")
    ax.fill(angles, ang_vals, alpha=0.15, color=ANGELS_COLOR)

    ax.set_xticks(angles[:-1])
    # Wrap long labels
    wrapped = [l.replace(" ", "\n") if len(l) > 15 else l for l in labels]
    ax.set_xticklabels(wrapped, fontsize=9)
    ax.set_title("Angels vs. Efficient Winners\n(normalized 0-1 scale)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    out = os.path.join(OUTPUT_DIR, "radar_angels_vs_ew.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}", flush=True)


def plot_heatmap(df):
    """Heatmap: correlation matrix of key metrics."""
    corr_path = os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
    if not os.path.exists(corr_path):
        print("  Skipping heatmap (correlation_matrix.csv missing)", flush=True)
        return

    corr = pd.read_csv(corr_path, index_col=0)

    # Select a manageable subset
    key_cols = [c for c in [
        "homegrown_war_pct", "war_gini", "players_2plus_war", "top3_war_share",
        "bat_OPS", "pit_ERA", "pit_FIP", "total_team_war", "war_per_10mm",
        "bat_WAR", "pit_WAR", "W",
    ] if c in corr.columns]

    if len(key_cols) < 3:
        print("  Skipping heatmap (not enough metrics)", flush=True)
        return

    sub = corr.loc[key_cols, key_cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(key_cols)))
    ax.set_yticks(range(len(key_cols)))
    short_labels = [c.replace("homegrown_war_pct", "HG WAR%").replace("war_gini", "Gini")
                     .replace("players_2plus_war", "2+WAR").replace("top3_war_share", "Top3 Share")
                     .replace("bat_OPS", "OPS").replace("pit_ERA", "ERA").replace("pit_FIP", "FIP")
                     .replace("total_team_war", "tWAR").replace("war_per_10mm", "WAR/$10M")
                     .replace("bat_WAR", "bWAR").replace("pit_WAR", "pWAR")
                    for c in key_cols]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short_labels, fontsize=9)

    # Add correlation values
    for i in range(len(key_cols)):
        for j in range(len(key_cols)):
            val = sub.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_title("Correlation Matrix: Key Efficiency Metrics")

    out = os.path.join(OUTPUT_DIR, "heatmap_correlations.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}", flush=True)


def plot_timeline(df, team, team_color, team_name, filename):
    """Timeline of key metrics for a single team."""
    team_data = df[df["team"] == team].sort_values("Season")
    if len(team_data) == 0:
        print(f"  Skipping {team_name} timeline (no data)", flush=True)
        return

    metrics = [
        ("W", "Wins"),
        ("homegrown_war_pct", "Homegrown WAR %"),
        ("players_2plus_war", "Players 2+ WAR"),
        ("war_gini", "WAR Gini"),
        ("total_team_war", "Total Team WAR"),
        ("war_per_10mm", "WAR per $10M"),
    ]

    available = [(col, label) for col, label in metrics if col in df.columns]
    if len(available) < 2:
        print(f"  Skipping {team_name} timeline (not enough metrics)", flush=True)
        return

    n = len(available)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (col, label) in enumerate(available):
        if i >= len(axes):
            break
        ax = axes[i]
        years = team_data["Season"].values
        vals = pd.to_numeric(team_data[col], errors="coerce").values

        ax.plot(years, vals, "o-", color=team_color, linewidth=2, markersize=8)

        # Add EW benchmark line
        ew = df[df["group"] == "efficient_winner"]
        for yr in years:
            ew_yr = pd.to_numeric(ew[ew["Season"] == yr][col], errors="coerce")
            if len(ew_yr) > 0:
                ew_mean = ew_yr.mean()
                ax.axhline(y=ew_mean, color="#2ecc71", linestyle="--", alpha=0.5)

        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xticks(years)
        ax.set_xticklabels([str(int(y)) for y in years], fontsize=9)

        # Annotate values
        for x, y in zip(years, vals):
            if not np.isnan(y):
                fmt = ".0f" if col in ("W", "players_2plus_war") else ".2f"
                ax.annotate(f"{y:{fmt}}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=team_color)

    # Hide unused
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=team_color, linewidth=2, label=team_name),
        Line2D([0], [0], color="#2ecc71", linestyle="--", linewidth=1, label="Efficient Winner avg"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10)

    fig.suptitle(f"{team_name} Trajectory: 2022-2026", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}", flush=True)


def main():
    print("Generating visualizations...\n", flush=True)
    df = load_data()

    plot_scatter_homegrown(df)
    plot_scatter_gini(df)
    plot_group_bars(df)
    plot_radar(df)
    plot_heatmap(df)
    plot_timeline(df, "CHW", CHW_COLOR, "White Sox", "timeline_white_sox.png")
    plot_timeline(df, "LAA", ANGELS_COLOR, "Angels", "timeline_angels.png")

    print("\nAll visualizations complete.", flush=True)


if __name__ == "__main__":
    main()
