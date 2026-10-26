"""
Pull individual player WAR data from Baseball Reference (bWAR).
Uses pybaseball's bwar_bat() and bwar_pitch() which access the
complete historical bWAR database.

We keep ALL historical data (not just recent years) so that
build_features.py can accurately determine each player's debut team.
A 2018+ cutoff would misattribute debut teams for veterans.

Output: data/bwar_batting.csv, data/bwar_pitching.csv
"""

import os
import pandas as pd
from pybaseball import bwar_bat, bwar_pitch

from config import DATA_DIR, standardize_team


def main():
    print("Pulling bWAR data...", flush=True)

    # Batting WAR -- keep full history for debut tracking
    print("\n--- Batting WAR (bwar_bat) ---", flush=True)
    bat = bwar_bat()
    print(f"  Full database: {len(bat)} rows", flush=True)

    bat["team"] = bat["team_ID"].apply(standardize_team)

    out = os.path.join(DATA_DIR, "bwar_batting.csv")
    bat.to_csv(out, index=False)
    print(f"  Saved to {out}", flush=True)

    # Pitching WAR -- keep full history for debut tracking
    print("\n--- Pitching WAR (bwar_pitch) ---", flush=True)
    pit = bwar_pitch()
    print(f"  Full database: {len(pit)} rows", flush=True)

    pit["team"] = pit["team_ID"].apply(standardize_team)

    out = os.path.join(DATA_DIR, "bwar_pitching.csv")
    pit.to_csv(out, index=False)
    print(f"  Saved to {out}", flush=True)

    # Summary (just analysis years)
    print(f"\n--- Summary (analysis years) ---", flush=True)
    for year in range(2022, 2027):
        bat_yr = bat[bat["year_ID"] == year]
        pit_yr = pit[pit["year_ID"] == year]
        bat_war = bat_yr["WAR"].sum()
        pit_war = pit_yr["WAR"].sum()
        print(f"  {year}: {len(bat_yr)} batters ({bat_war:.1f} bWAR), "
              f"{len(pit_yr)} pitchers ({pit_war:.1f} pWAR)", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
