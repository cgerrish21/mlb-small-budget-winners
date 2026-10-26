"""
Pull W-L records from Baseball Reference via pybaseball standings().
Tag playoff teams and division winners.

Note: pybaseball's standings() only returns Tm, W, L, W-L%, GB from
the division tables. Run differential is computed in build_features.py
from team batting/pitching data.

Output: data/standings.csv
"""

import os
import time
import pandas as pd
from pybaseball import standings

from config import (
    YEARS, DATA_DIR, PLAYOFF_TEAMS, DIVISION_WINNERS,
    standardize_team,
)


def main():
    print("Pulling standings from Baseball Reference...", flush=True)
    all_rows = []

    for year in YEARS:
        print(f"  {year}...", flush=True)
        try:
            # standings() returns a list of DataFrames (one per division)
            divs = standings(year)
            for div_df in divs:
                for _, row in div_df.iterrows():
                    # The team name column is typically 'Tm' or the first column
                    team_name = None
                    for col in ["Tm", "Name", "Team"]:
                        if col in row.index:
                            team_name = row[col]
                            break
                    if team_name is None:
                        team_name = row.iloc[0]

                    team_abbr = standardize_team(str(team_name))

                    # Extract W, L
                    w = int(row.get("W", 0))
                    l = int(row.get("L", 0))
                    wpct = w / (w + l) if (w + l) > 0 else 0

                    playoffs = team_abbr in PLAYOFF_TEAMS.get(year, [])
                    div_winner = team_abbr in DIVISION_WINNERS.get(year, [])

                    all_rows.append({
                        "Season": year,
                        "team": team_abbr,
                        "W": w,
                        "L": l,
                        "W_pct": round(wpct, 3),
                        "made_playoffs": playoffs,
                        "division_winner": div_winner,
                    })

            print(f"    got {len([r for r in all_rows if r['Season'] == year])} teams",
                  flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
        time.sleep(1)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out = os.path.join(DATA_DIR, "standings.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}", flush=True)
    else:
        print("\nNo standings data collected!", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
